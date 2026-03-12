"""Train MoE head on frozen CaLM encoder (no LoRA).

Frozen CaLM (85.75M) -> [CLS] -> MoEHead (4 experts, top-1, ~700K params).
Only the MoE head is trainable. No LoRA, no encoder fine-tuning.

Usage:
    python -m proteinqc.cli.train_moe_head \
        --baked-train data/baked/eu8/train.pt \
        --baked-val data/baked/eu8/val.pt \
        --output models/heads/moe_head_v1

    # Distributed
    torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 \
        --master_addr=IP --master_port=29500 \
        -m proteinqc.cli.train_moe_head [args...]
"""

from __future__ import annotations

import argparse
import json
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
from proteinqc.models.classification_heads import MoEHead

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


# ── Classifier wrapper ────────────────────────────────────────────

class FrozenEncoderClassifier(nn.Module):
    """Frozen encoder + trainable head."""

    def __init__(self, encoder: CaLMEncoder, head: MoEHead):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        cls_emb = self.encoder(input_ids, attention_mask)
        return self.head(cls_emb).squeeze(-1)


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


# ── Training loop ─────────────────────────────────────────────────

def train_moe_head(
    model: nn.Module,
    train_samples: list[dict],
    val_samples: list[dict],
    device: torch.device,
    token_budget: int = 16_384,
    max_batch: int = 512,
    epochs: int = 6,
    lr: float = 5e-4,
    grad_accum: int = 4,
    patience: int = 3,
    log_interval: int = 100,
    focal_gamma: float = 2.0,
    output_dir: Path | None = None,
) -> list[dict]:
    raw = unwrap(model)
    world_size = get_world_size()

    head_params = list(raw.head.parameters())
    fused = device.type == "cuda"
    optimizer = torch.optim.AdamW(
        head_params, lr=lr, weight_decay=1e-4, fused=fused,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n_pos = sum(s["label"] for s in train_samples)
    n_neg = len(train_samples) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    criterion = FocalBCEWithLogitsLoss(gamma=focal_gamma, pos_weight=pos_weight)

    log: list[dict] = []
    best_val_loss = float("inf")
    best_head_state: dict[str, torch.Tensor] | None = None
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

            balance_loss = raw.head.get_balance_loss()
            loss = loss + balance_loss

            loss = loss / grad_accum
            loss.backward()

            epoch_loss += loss.item() * grad_accum
            epoch_balance_loss += balance_loss.item()
            n_steps += 1
            n_tok += ids.numel()

            if n_steps % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(head_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                opt_step_count += 1

                if (log_interval > 0 and opt_step_count % log_interval == 0
                        and is_main_process()):
                    elapsed = time.monotonic() - epoch_t0
                    avg = epoch_loss / n_steps
                    avg_bal = epoch_balance_loss / n_steps

                    load_str = ""
                    stats = raw.head.get_load_stats()
                    if stats:
                        fracs = [f"{v:.2f}" for v in stats.values()]
                        load_str = f"  load=[{','.join(fracs)}]"

                    print(
                        f"    [{epoch+1}/{epochs}] step {opt_step_count}  "
                        f"loss={avg:.4f}  bal={avg_bal:.4f}{load_str}  "
                        f"tok={n_tok:,}  {elapsed:.0f}s",
                        flush=True,
                    )

            if device.type == "mps":
                torch.mps.empty_cache()

        # Final gradient step
        if n_steps % grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(head_params, max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()

        # All-reduce avg loss
        avg_loss = epoch_loss / max(n_steps, 1)
        loss_tensor = torch.tensor([avg_loss], device=device)
        all_reduce_mean(loss_tensor)
        avg_loss = loss_tensor.item()

        # Validation
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

        # Expert load on val set
        load_str = ""
        stats = raw.head.get_load_stats()
        if stats:
            fracs = [f"{v:.2f}" for v in stats.values()]
            load_str = f"  load=[{','.join(fracs)}]"

        # Per-expert internal gate stats
        expert_gate_stats = {}
        if hasattr(raw.head, "get_expert_gate_stats"):
            expert_gate_stats = raw.head.get_expert_gate_stats()

        entry = {
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "micro_batches": n_steps,
            "opt_steps": max(n_steps // grad_accum, 1),
            "tokens": n_tok,
            "expert_load": stats,
            "expert_gates": expert_gate_stats,
        }
        log.append(entry)
        if is_main_process():
            print(
                f"  Epoch {epoch+1}/{epochs}: "
                f"train_loss={avg_loss:.4f}  val_loss={val_loss:.4f}  "
                f"val_acc={val_acc:.1f}%{load_str}  tok={n_tok:,}",
            )

        # Checkpoint best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_head_state = {
                k: v.cpu().clone() for k, v in raw.head.state_dict().items()
            }
            if output_dir is not None and is_main_process():
                output_dir.mkdir(parents=True, exist_ok=True)
                torch.save(best_head_state, output_dir / "moe_head.pt")
                print(f"  Checkpoint saved (epoch {epoch+1}, val_acc={val_acc:.1f}%)")
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if is_main_process():
                    print(f"  Early stopping at epoch {epoch+1}")
                break

        barrier()

    # Restore best
    if best_head_state is not None:
        raw.head.load_state_dict({
            k: v.to(device) for k, v in best_head_state.items()
        })

    return log


# ── CLI ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train MoE head on frozen CaLM (no LoRA)",
    )
    p.add_argument("--calm-dir", type=Path,
                   default=PROJECT_ROOT / "models" / "calm")
    p.add_argument("--baked-train", type=Path, required=True)
    p.add_argument("--baked-val", type=Path, required=True)
    p.add_argument("--output", type=Path,
                   default=PROJECT_ROOT / "models" / "heads" / "moe_head_v1")
    p.add_argument("--num-experts", type=int, default=2)
    p.add_argument("--expert-hidden", type=int, default=394)
    p.add_argument("--top-k", type=int, default=1)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--token-budget", type=int, default=16_384)
    p.add_argument("--max-batch", type=int, default=512)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--balance-coeff", type=float, default=0.01)
    p.add_argument("--position-type", choices=["rotary", "alibi"],
                   default="alibi")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


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

    if is_main_process():
        print("=" * 70)
        print("  CaLM Frozen + MoE Head (no LoRA)")
        print("=" * 70)
        print(f"Device: {device}")
        if is_dist:
            print(f"Distributed: {dist_info['world_size']} nodes")
        print(f"Experts: {args.num_experts}, top-{args.top_k}, "
              f"hidden={args.expert_hidden}")
        print(f"Position: {args.position_type}")
        print(f"Epochs: {args.epochs}, LR: {args.lr}")

    # Frozen encoder — no LoRA
    if is_main_process():
        print(f"\nLoading CaLM encoder (frozen)...")
    encoder = CaLMEncoder(
        args.calm_dir, freeze=True, position_type=args.position_type,
    )
    total_enc = sum(p.numel() for p in encoder.parameters())

    # MoE head
    head = MoEHead(
        hidden_size=encoder.hidden_size,
        expert_hidden=args.expert_hidden,
        num_experts=args.num_experts,
        top_k=args.top_k,
        dropout=0.1,
        balance_coeff=args.balance_coeff,
    )
    n_head = sum(p.numel() for p in head.parameters())

    if is_main_process():
        print(f"  Encoder params: {total_enc:,} (frozen)")
        print(f"  MoE head params: {n_head:,} (trainable)")

    model = FrozenEncoderClassifier(encoder, head)

    if args.compile and device.type == "cuda":
        if is_main_process():
            print("\ntorch.compile (dynamic=None)...")
        if args.position_type == "alibi":
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

    # Load data
    if is_main_process():
        print(f"\nLoading baked data...")
    tr_samples = torch.load(args.baked_train, weights_only=False)
    val_samples = torch.load(args.baked_val, weights_only=False)
    if is_main_process():
        n_pos = sum(1 for s in tr_samples if s["label"] == 1)
        n_neg = len(tr_samples) - n_pos
        print(f"  Train: {len(tr_samples):,} ({n_pos:,} cod / {n_neg:,} nc)")
        print(f"  Val:   {len(val_samples):,}")
        ratio = n_head / len(tr_samples)
        print(f"  Params/sample: {ratio:.2f} "
              f"({'OK' if ratio < 5 else 'HIGH'})")

    barrier()

    t0 = time.time()
    log = train_moe_head(
        model, tr_samples, val_samples, device,
        token_budget=args.token_budget,
        max_batch=args.max_batch,
        epochs=args.epochs,
        lr=args.lr,
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
                "architecture": "moe_head",
                "position_type": args.position_type,
                "num_experts": args.num_experts,
                "expert_hidden": args.expert_hidden,
                "top_k": args.top_k,
                "n_head_params": n_head,
                "lora": False,
                "epochs_run": len(log),
                "train_time_sec": train_time,
                "world_size": get_world_size(),
                "token_budget": args.token_budget,
                "max_batch": args.max_batch,
                "lr": args.lr,
                "balance_coeff": args.balance_coeff,
                "compiled": args.compile,
                "log": log,
            }, f, indent=2)

        print(f"\nSaved: {args.output / 'moe_head.pt'}")
        print(f"Config: {args.output / 'training_log.json'}")

    barrier()
    if is_dist:
        cleanup_distributed()


if __name__ == "__main__":
    main()
