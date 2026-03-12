"""Train CaLM-MoE: frozen attention + trainable MoE FFN + GatedHead.

Layers 0..moe_start-1: entirely frozen (dense FFN).
Layers moe_start..11: attention frozen, MoE FFN + ffn_ln trainable.
GatedHead: trainable.

No LoRA. The capacity comes from MoE experts, not attention adapters.

Usage:
    # Single GPU
    python -m proteinqc.cli.train_moe \
        --moe-dir models/calm_moe \
        --baked-train data/baked/eu8/train.pt \
        --baked-val data/baked/eu8/val.pt \
        --output models/heads/moe_gated_v1

    # Distributed (2 nodes)
    torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 \
        --master_addr=192.168.100.10 --master_port=29500 \
        -m proteinqc.cli.train_moe [args...]
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from proteinqc.data.dataset import (
    MAX_SEQ_LEN,
    _bucket_pad,
    collate_binary,
    token_budget_batcher,
)
from proteinqc.distributed import (
    all_reduce_mean,
    barrier,
    cleanup_distributed,
    get_world_size,
    is_main_process,
    setup_distributed,
    unwrap,
    wrap_ddp,
)
from proteinqc.models.calm_encoder import CaLMEncoder
from proteinqc.models.classification_heads import GatedHead
from proteinqc.models.moe_layer import MoETransformerLayer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ── Focal loss ────────────────────────────────────────────────────

class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none",
        )
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


# ── MoE Classifier ───────────────────────────────────────────────

class MoEClassifier(nn.Module):
    """Encoder (with MoE layers) + GatedHead."""

    def __init__(self, encoder: CaLMEncoder, head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        cls_emb = self.encoder(input_ids, attention_mask)
        return self.head(cls_emb).squeeze(-1)


# ── Model building ───────────────────────────────────────────────

def build_moe_encoder(
    calm_dir: Path,
    moe_dir: Path,
    position_type: str = "alibi",
) -> CaLMEncoder:
    """Build CaLM encoder with MoE layers from upcycled checkpoint.

    1. Create standard CaLMEncoder (loads pretrained weights).
    2. Load moe_config.json to know which layers are MoE.
    3. Replace those layers with MoETransformerLayer.
    4. Load upcycled MoE weights.
    """
    moe_config_path = moe_dir / "moe_config.json"
    moe_weights_path = moe_dir / "model_moe.pt"

    with open(moe_config_path) as f:
        moe_config = json.load(f)

    moe_start = moe_config["moe_start"]
    num_experts = moe_config["num_experts"]
    top_k = moe_config["top_k"]
    expert_inter = moe_config.get(
        "expert_intermediate_size", moe_config.get("intermediate_size", 3072),
    )

    # Build base encoder (loads pretrained dense weights)
    encoder = CaLMEncoder(calm_dir, freeze=False, position_type=position_type)
    cfg = encoder.config

    # Replace MoE layers
    for i in range(moe_start, cfg["num_hidden_layers"]):
        moe_layer = MoETransformerLayer(
            hidden_size=cfg["hidden_size"],
            num_heads=cfg["num_attention_heads"],
            intermediate_size=expert_inter,
            layer_norm_eps=cfg["layer_norm_eps"],
            attn_dropout=cfg["attention_dropout"],
            hidden_dropout=cfg["hidden_dropout"],
            num_experts=num_experts,
            top_k=top_k,
        )
        encoder.layers[i] = moe_layer

    # Load upcycled weights (has expert copies)
    moe_state = torch.load(moe_weights_path, weights_only=True, map_location="cpu")
    missing, unexpected = encoder.load_state_dict(moe_state, strict=False)

    real_missing = [k for k in missing if "router" not in k]
    if real_missing:
        print(f"WARNING: missing keys: {real_missing[:10]}")

    return encoder


def freeze_for_moe_training(
    encoder: CaLMEncoder,
    moe_start: int,
) -> tuple[list[torch.nn.Parameter], int]:
    """Freeze everything except MoE FFN + ffn_ln in MoE layers."""
    for param in encoder.parameters():
        param.requires_grad = False

    trainable = []
    for i in range(moe_start, len(encoder.layers)):
        layer = encoder.layers[i]
        for param in layer.moe_ffn.parameters():
            param.requires_grad = True
            trainable.append(param)
        for param in layer.ffn_ln.parameters():
            param.requires_grad = True
            trainable.append(param)

    count = sum(p.numel() for p in trainable)
    return trainable, count


# ── Forward helpers ───────────────────────────────────────────────

def batched_forward(
    samples: list[dict],
    model: nn.Module,
    device: torch.device,
    token_budget: int = 16_384,
    max_batch: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw = unwrap(model)
    n = len(samples)
    sorted_indices = sorted(range(n), key=lambda i: len(samples[i]["input_ids"]))
    all_logits = torch.zeros(n, device=device)

    i = 0
    while i < n:
        max_pad = _bucket_pad(len(samples[sorted_indices[i]]["input_ids"]))
        bs = 1
        while i + bs < n and bs < max_batch:
            next_pad = _bucket_pad(len(samples[sorted_indices[i + bs]]["input_ids"]))
            new_max = max(max_pad, next_pad)
            if new_max * (bs + 1) > token_budget:
                break
            max_pad = new_max
            bs += 1

        batch_samples = [samples[sorted_indices[j]] for j in range(i, i + bs)]
        batch = collate_binary(batch_samples)
        ids = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)

        with torch.no_grad():
            logits = raw(ids, mask)

        for j in range(bs):
            all_logits[sorted_indices[i + j]] = logits[j]
        i += bs

    return all_logits, torch.sigmoid(all_logits)


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray,
) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score, f1_score, matthews_corrcoef, roc_auc_score,
    )
    m: dict[str, float] = {}
    m["ACC"] = float(accuracy_score(y_true, y_pred) * 100)
    m["MCC"] = float(matthews_corrcoef(y_true, y_pred) * 100)
    m["F1"] = float(f1_score(y_true, y_pred, average="macro") * 100)
    try:
        m["AUC"] = float(roc_auc_score(y_true, y_prob) * 100)
    except ValueError:
        m["AUC"] = float("nan")
    return m


# ── Token budget probing ─────────────────────────────────────────

def probe_token_budget(
    model: nn.Module,
    device: torch.device,
    start: int = 8192,
    max_budget: int = 1_048_576,
    safety: float = 0.85,
) -> int:
    """Find max token budget via forward+backward probing on GPU.

    Doubles budget from *start* until peak memory exceeds 45% of total
    GPU memory, then returns *safety* fraction of the last successful
    budget (rounded down to 1024).

    Never probes to OOM — on UMA systems the memory ratchet means a
    failed allocation permanently reduces available memory.
    """
    import gc

    max_sl = 1026
    vocab = 131
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1 << 30)

    print(f"\nProbing token budget (forward + backward)...")
    print(f"  GPU total: {total_gb:.1f}GB")
    best = 0
    budget = start

    was_training = model.training
    model.train()

    while budget <= max_budget:
        bs = max(1, budget // max_sl)
        sl = min(budget, max_sl)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        ids = mask = labels = logits = loss = None
        ok = False
        peak_gb = 0.0
        try:
            ids = torch.randint(0, vocab, (bs, sl), device=device)
            mask = torch.ones(bs, sl, dtype=torch.bool, device=device)
            labels = torch.zeros(bs, device=device)
            logits = model(ids, mask)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            peak_gb = torch.cuda.max_memory_allocated() / (1 << 30)
            ok = True
        except torch.cuda.OutOfMemoryError:
            pass

        model.zero_grad(set_to_none=True)
        del ids, mask, labels, logits, loss
        gc.collect()
        torch.cuda.empty_cache()

        if ok:
            print(f"  budget={budget:>8,}  bs={bs:>4}  sl={sl:>4}  "
                  f"peak={peak_gb:.1f}GB ({peak_gb/total_gb:.0%})  OK")
            best = budget
            # Stop before OOM: if next 2x step would likely exceed GPU.
            # On UMA, probing to OOM ratchets memory permanently.
            if peak_gb > total_gb * 0.45:
                print(f"  (next 2x would exceed ~90% GPU — stopping)")
                break
            budget *= 2
        else:
            print(f"  budget={budget:>8,}  bs={bs:>4}  sl={sl:>4}  OOM")
            break

    if not was_training:
        model.eval()

    if best == 0:
        fallback = start // 2
        print(f"  WARNING: even {start:,} OOMs — using {fallback:,}")
        return fallback

    final = max(1024, (int(best * safety) // 1024) * 1024)
    print(f"\n  Max OK: {best:,}")
    print(f"  Using:  {final:,} ({safety:.0%} safety margin)")
    return final


# ── Training loop ─────────────────────────────────────────────────

def train_moe(
    model: nn.Module,
    train_samples: list[dict],
    val_samples: list[dict],
    device: torch.device,
    moe_start: int,
    token_budget: int = 16_384,
    max_batch: int = 512,
    epochs: int = 4,
    moe_lr: float = 1e-4,
    head_lr: float = 1e-3,
    grad_accum: int = 4,
    patience: int = 3,
    log_interval: int = 100,
    focal_gamma: float = 2.0,
    output_dir: Path | None = None,
) -> list[dict]:
    raw = unwrap(model)
    world_size = get_world_size()

    # Separate param groups: experts, routers (higher LR), head
    moe_params = []
    router_params = []
    for i in range(moe_start, len(raw.encoder.layers)):
        layer = raw.encoder.layers[i]
        for name, param in layer.moe_ffn.named_parameters():
            if not param.requires_grad:
                continue
            if "router" in name:
                router_params.append(param)
            else:
                moe_params.append(param)
        for param in layer.ffn_ln.parameters():
            if param.requires_grad:
                moe_params.append(param)

    head_params = list(raw.head.parameters())
    all_trainable = moe_params + router_params + head_params

    fused = device.type == "cuda"
    optimizer = torch.optim.AdamW([
        {"params": moe_params, "lr": moe_lr},
        {"params": router_params, "lr": moe_lr * 10},
        {"params": head_params, "lr": head_lr},
    ], weight_decay=1e-4, fused=fused)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n_pos = sum(s["label"] for s in train_samples)
    n_neg = len(train_samples) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    criterion = FocalBCEWithLogitsLoss(gamma=focal_gamma, pos_weight=pos_weight)

    log: list[dict] = []
    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    wait = 0

    n_train = len(train_samples)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    for epoch in range(epochs):
        model.train()
        rng = np.random.default_rng(42 + epoch)
        perm = rng.permutation(n_train)
        my_indices = perm[rank::world_size]
        my_samples = [train_samples[i] for i in my_indices]

        epoch_loss = 0.0
        epoch_balance_loss = 0.0
        n_steps = 0
        n_tok = 0
        opt_step_count = 0
        epoch_t0 = time.monotonic()
        optimizer.zero_grad()

        for batch in token_budget_batcher(
            my_samples, token_budget, max_batch, collate_binary,
        ):
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, dtype=torch.float32,
                                        non_blocking=True)

            logits = model(ids, mask)
            loss = criterion(logits, labels)

            # Balance loss from all MoE layers
            balance_loss = torch.tensor(0.0, device=device)
            for i in range(moe_start, len(raw.encoder.layers)):
                layer = raw.encoder.layers[i]
                balance_loss = balance_loss + layer.moe_ffn.get_balance_loss()
            loss = loss + balance_loss

            if hasattr(raw.head, "get_balance_loss"):
                loss = loss + raw.head.get_balance_loss()

            loss = loss / grad_accum
            loss.backward()

            epoch_loss += loss.item() * grad_accum
            epoch_balance_loss += balance_loss.item()
            n_steps += 1
            n_tok += ids.numel()

            if n_steps % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(all_trainable, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                opt_step_count += 1

                if (log_interval > 0 and opt_step_count % log_interval == 0
                        and is_main_process()):
                    elapsed = time.monotonic() - epoch_t0
                    avg = epoch_loss / n_steps
                    avg_bal = epoch_balance_loss / n_steps

                    last_moe = raw.encoder.layers[-1].moe_ffn
                    load_str = ""
                    stats = last_moe.get_load_stats()
                    if stats:
                        fracs = [f"{v:.2f}" for v in stats.values()]
                        load_str = f"  load=[{','.join(fracs)}]"

                    print(
                        f"    [{epoch+1}/{epochs}] step {opt_step_count}  "
                        f"loss={avg:.4f}  bal={avg_bal:.4f}{load_str}  "
                        f"tok={n_tok:,}  {elapsed:.0f}s",
                        flush=True,
                    )

        # Final gradient step
        if n_steps % grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(all_trainable, max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()

        # All-reduce avg loss
        avg_loss = epoch_loss / max(n_steps, 1)
        loss_tensor = torch.tensor([avg_loss], device=device)
        all_reduce_mean(loss_tensor)
        avg_loss = loss_tensor.item()

        # Validation — all ranks run to avoid NCCL timeout during long eval
        barrier()
        raw.eval()
        val_logits, val_probs = batched_forward(
            val_samples, model, device, token_budget, max_batch,
        )
        val_labels_t = torch.tensor(
            [s["label"] for s in val_samples],
            dtype=torch.float32, device=device,
        )
        val_loss = criterion(val_logits, val_labels_t).item()
        val_preds = (val_probs.cpu().numpy() > 0.5).astype(int)
        val_true = np.array([s["label"] for s in val_samples])
        val_acc = float((val_preds == val_true).mean() * 100)

        entry = {
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "micro_batches": n_steps,
            "opt_steps": max(n_steps // grad_accum, 1),
            "tokens": n_tok,
        }
        log.append(entry)
        if is_main_process():
            print(
                f"  Epoch {epoch+1}/{epochs}: "
                f"train_loss={avg_loss:.4f}  val_loss={val_loss:.4f}  "
                f"val_acc={val_acc:.1f}%  tok={n_tok:,}",
            )

        # Checkpoint best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_dict: dict[str, torch.Tensor] = {}
            for i in range(moe_start, len(raw.encoder.layers)):
                layer = raw.encoder.layers[i]
                for k, v in layer.moe_ffn.state_dict().items():
                    save_dict[f"layers.{i}.moe_ffn.{k}"] = v.cpu().clone()
                for k, v in layer.ffn_ln.state_dict().items():
                    save_dict[f"layers.{i}.ffn_ln.{k}"] = v.cpu().clone()
            save_dict["head"] = {
                k: v.cpu().clone() for k, v in raw.head.state_dict().items()
            }
            best_state = save_dict

            if output_dir is not None and is_main_process():
                output_dir.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, output_dir / "moe_checkpoint.pt")
                print(f"  Checkpoint saved (epoch {epoch+1}, "
                      f"val_acc={val_acc:.1f}%)")
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if is_main_process():
                    print(f"  Early stopping at epoch {epoch+1}")
                break

        barrier()

    # Restore best
    if best_state is not None:
        for k, v in best_state.items():
            if k == "head":
                raw.head.load_state_dict({
                    kk: vv.to(device) for kk, vv in v.items()
                })
            else:
                parts = k.split(".")
                layer_idx = int(parts[1])
                layer = raw.encoder.layers[layer_idx]
                submod_name = parts[2]
                rest = ".".join(parts[3:])
                submod = getattr(layer, submod_name)
                current = submod.state_dict()
                current[rest] = v.to(device)
                submod.load_state_dict(current)

    return log


# ── CLI ───────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    dist_info = setup_distributed(backend="nccl")
    is_dist = dist_info is not None

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda") if torch.cuda.is_available() else (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    with open(args.moe_dir / "moe_config.json") as f:
        moe_config = json.load(f)

    moe_start = moe_config["moe_start"]
    num_experts = moe_config["num_experts"]
    top_k = moe_config["top_k"]
    position_type = moe_config["position_type"]

    if is_main_process():
        print("=" * 70)
        print("  CaLM-MoE Training")
        print("=" * 70)
        print(f"Device: {device}")
        if is_dist:
            print(f"Distributed: {dist_info['world_size']} nodes")
        print(f"MoE layers: {moe_start}-11, "
              f"{num_experts} experts, top-{top_k}")
        print(f"Position: {position_type}")
        print(f"Epochs: {args.epochs}, MoE LR: {args.moe_lr}, "
              f"Head LR: {args.head_lr}")

    if is_main_process():
        print(f"\nBuilding CaLM-MoE encoder...")
    encoder = build_moe_encoder(args.calm_dir, args.moe_dir, position_type)

    moe_trainable, n_moe = freeze_for_moe_training(encoder, moe_start)
    if is_main_process():
        total = sum(p.numel() for p in encoder.parameters())
        print(f"  Total encoder params: {total:,}")
        print(f"  Trainable MoE params: {n_moe:,}")

    head = GatedHead(encoder.hidden_size, 256, 0.1)
    n_head = sum(p.numel() for p in head.parameters())
    if is_main_process():
        print(f"  Head params: {n_head:,}")
        print(f"  Total trainable: {n_moe + n_head:,}")

    model = MoEClassifier(encoder, head)

    if args.compile and device.type == "cuda":
        if is_main_process():
            print("\ntorch.compile (dynamic=None)...")
        if position_type == "alibi":
            encoder._ensure_alibi(MAX_SEQ_LEN, device)
        else:
            encoder._ensure_rope(1026, device)
        model = torch.compile(model, dynamic=None)

    model = wrap_ddp(model, device, find_unused_params=True)
    if is_dist:
        from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
        model.register_comm_hook(
            state=None, hook=default_hooks.bf16_compress_hook,
        )

    # Auto-detect token budget via GPU memory probing
    token_budget = args.token_budget
    max_batch = args.max_batch
    if args.probe_budget and device.type == "cuda":
        token_budget = probe_token_budget(model, device)
        max_batch = max(max_batch, token_budget // 16)
        if is_main_process():
            print(f"Token budget: {token_budget:,}")
            print(f"Max batch: {max_batch:,}")

    if is_main_process():
        print(f"\nLoading baked data...")
    tr_samples = torch.load(args.baked_train, weights_only=False)
    val_samples = torch.load(args.baked_val, weights_only=False)
    if is_main_process():
        n_pos = sum(1 for s in tr_samples if s["label"] == 1)
        n_neg = len(tr_samples) - n_pos
        print(f"  Train: {len(tr_samples):,} ({n_pos:,} cod / {n_neg:,} nc)")
        print(f"  Val:   {len(val_samples):,}")

    barrier()

    t0 = time.time()
    log = train_moe(
        model, tr_samples, val_samples, device,
        moe_start=moe_start,
        token_budget=token_budget,
        max_batch=max_batch,
        epochs=args.epochs,
        moe_lr=args.moe_lr,
        head_lr=args.head_lr,
        grad_accum=args.grad_accum,
        patience=args.patience,
        log_interval=args.log_interval,
        focal_gamma=args.focal_gamma,
        output_dir=args.output,
    )
    train_time = time.time() - t0

    if is_main_process():
        print(f"\nTraining completed in {train_time:.0f}s ({len(log)} epochs)")

        args.output.mkdir(parents=True, exist_ok=True)

        with open(args.output / "training_log.json", "w") as f:
            json.dump({
                "architecture": "moe",
                "head": "gated",
                "position_type": position_type,
                "moe_start": moe_start,
                "num_experts": num_experts,
                "top_k": top_k,
                "epochs_run": len(log),
                "train_time_sec": train_time,
                "n_moe_params": n_moe,
                "n_head_params": n_head,
                "world_size": get_world_size(),
                "token_budget": token_budget,
                "max_batch": max_batch,
                "budget_probed": args.probe_budget,
                "moe_lr": args.moe_lr,
                "head_lr": args.head_lr,
                "compiled": args.compile,
                "log": log,
            }, f, indent=2)

        shutil.copy2(
            args.moe_dir / "moe_config.json",
            args.output / "moe_config.json",
        )

        print(f"\nSaved: {args.output / 'moe_checkpoint.pt'}")
        print(f"Config: {args.output / 'training_log.json'}")

    barrier()
    if is_dist:
        cleanup_distributed()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train CaLM-MoE: frozen attn + MoE FFN + GatedHead",
    )
    p.add_argument("--calm-dir", type=Path,
                   default=PROJECT_ROOT / "models" / "calm")
    p.add_argument("--moe-dir", type=Path,
                   default=PROJECT_ROOT / "models" / "calm_moe")
    p.add_argument("--baked-train", type=Path, required=True)
    p.add_argument("--baked-val", type=Path, required=True)
    p.add_argument("--output", type=Path,
                   default=PROJECT_ROOT / "models" / "heads" / "moe_gated_v1")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--moe-lr", type=float, default=1e-4)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--token-budget", type=int, default=16_384)
    p.add_argument("--max-batch", type=int, default=512)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--compile", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--probe-budget", action="store_true",
                   help="Auto-detect token budget via GPU memory probing")
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main()
