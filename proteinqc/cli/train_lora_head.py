"""LoRA fine-tune CaLM attention layers + train classification head jointly.

Supports single-GPU and multi-node distributed (DDP via torchrun).

Usage:
    # Single GPU
    python -m proteinqc.cli.train_lora_head \
        --benchmark-dir data/benchmark/ \
        --model-dir models/calm/ \
        --output models/heads/lora_mlp_v1/

    # Distributed (launched via bin/lora-train)
    torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 \
        --master_addr=IP --master_port=29500 \
        -m proteinqc.cli.train_lora_head [args...]
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

from proteinqc.cli.benchmark_multispecies import (
    codon_align_longest_orf,
    discover_datasets,
    read_fasta,
    read_fasta_multi,
)
from proteinqc.data.dataset import (
    MAX_SEQ_LEN,
    _bucket_pad,
    collate_binary,
    pre_tokenize,
    token_budget_batcher,
)
from proteinqc.data.tokenizer import CodonTokenizer
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
from proteinqc.models.classification_heads import GatedHead, LinearHead, MLPHead
from proteinqc.tools.codon_table import CodonTableManager
from proteinqc.tools.orf_scanner import ORFScanner

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ── Focal loss ────────────────────────────────────────────────────

class FocalBCEWithLogitsLoss(nn.Module):
    """Binary focal loss (Lin et al. 2017). Drop-in replacement for BCEWithLogitsLoss.

    Downweights easy samples so the model focuses on hard cases.
    loss = -alpha * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # BCE component (with pos_weight for class balance)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none",
        )
        # p_t = probability of correct class
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        # Focal modulator: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


# ── Combined module for DDP wrapping ──────────────────────────────

class LoRAClassifier(nn.Module):
    """Encoder + head in one module so DDP syncs all gradients."""

    def __init__(self, encoder: nn.Module, head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        cls_emb = self.encoder(input_ids, attention_mask)
        return self.head(cls_emb).squeeze(-1)


# ── Helpers ───────────────────────────────────────────────────────

def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_head(name: str, hidden_size: int = 768) -> nn.Module:
    heads = {
        "linear": lambda: LinearHead(hidden_size),
        "mlp": lambda: MLPHead(hidden_size, 256, 0.1),
        "gated": lambda: GatedHead(hidden_size, 256, 0.1),
    }
    if name not in heads:
        raise ValueError(f"Unknown head: {name}. Choose from {list(heads)}")
    return heads[name]()


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


def split_by_dataset(
    dataset_names: list[str], test_frac: float = 0.2, seed: int = 42,
) -> tuple[set[str], set[str]]:
    names = np.array(dataset_names)
    rng = np.random.default_rng(seed)
    rng.shuffle(names)
    n_test = max(12, int(test_frac * len(names)))
    return set(names[n_test:]), set(names[:n_test])


def apply_lora(encoder: CaLMEncoder, rank: int = 8, alpha: int = 16,
               dropout: float = 0.1) -> int:
    """Apply LoRA adapters to Q+K+V projections. Returns trainable param count."""
    from peft import LoraConfig, get_peft_model

    for param in encoder.parameters():
        param.requires_grad = True

    config = LoraConfig(
        r=rank, lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=dropout, bias="none",
    )
    peft_encoder = get_peft_model(encoder, config)
    if is_main_process():
        peft_encoder.print_trainable_parameters()
    return sum(p.numel() for p in peft_encoder.parameters() if p.requires_grad)


# ── Data loading ──────────────────────────────────────────────────

def load_all_sequences(
    datasets: list[dict],
    max_seqs_per_ds: int,
    align_fn,
    per_ds_cap: dict[str, int] | None = None,
) -> tuple[list[str], list[str], list[int], list[str]]:
    all_seqs: list[str] = []
    all_ids: list[str] = []
    all_labels: list[int] = []
    all_ds_names: list[str] = []

    for ds in datasets:
        ds_name = f"{ds['tool']}/{ds['species']}"
        # Per-dataset cap overrides global sample
        cap = max_seqs_per_ds
        if per_ds_cap and ds_name in per_ds_cap:
            cap = per_ds_cap[ds_name]
        half = cap // 2 if cap else 0

        if "mixed" in ds:
            if not ds["mixed"].exists():
                continue
        else:
            coding_paths = ds["coding"] if isinstance(ds["coding"], list) else [ds["coding"]]
            noncoding_paths = ds["noncoding"] if isinstance(ds["noncoding"], list) else [ds["noncoding"]]
            missing = [p for p in coding_paths + noncoding_paths if not Path(p).exists()]
            if missing:
                continue

        if "mixed" in ds:
            all_records = read_fasta(ds["mixed"], max_seqs=0)
            coding_records = [
                (h, s) for h, s in all_records if "NM_" in h or "mRNA" in h.lower()
            ]
            noncoding_records = [
                (h, s) for h, s in all_records
                if h not in {r[0] for r in coding_records}
            ]
            if half:
                coding_records = coding_records[:half]
                noncoding_records = noncoding_records[:half]
        else:
            coding_records = read_fasta_multi(ds["coding"], max_seqs=half)
            noncoding_records = read_fasta_multi(ds["noncoding"], max_seqs=half)

        count = 0
        for header, seq in coding_records:
            aligned = align_fn(seq)
            if len(aligned) >= 9:
                all_seqs.append(aligned)
                all_ids.append(header.split()[0])
                all_labels.append(1)
                all_ds_names.append(ds_name)
                count += 1

        for header, seq in noncoding_records:
            aligned = align_fn(seq)
            if len(aligned) >= 9:
                all_seqs.append(aligned)
                all_ids.append(header.split()[0])
                all_labels.append(0)
                all_ds_names.append(ds_name)
                count += 1

        if count > 0 and is_main_process():
            n_cod = sum(1 for l in all_labels[-count:] if l == 1)
            print(f"  {ds_name}: {count} seqs ({n_cod} coding)")

    return all_seqs, all_ids, all_labels, all_ds_names


# ── Forward pass ──────────────────────────────────────────────────

def batched_forward(
    samples: list[dict],
    model: nn.Module,
    device: torch.device,
    token_budget: int = 16_384,
    max_batch: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward pass with token-budget batching on pre-tokenized samples.

    Sorts by length for efficient padding, reassembles in original order.
    """
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


# ── Training loop ─────────────────────────────────────────────────

def train_lora(
    model: nn.Module,
    train_samples: list[dict],
    val_samples: list[dict],
    device: torch.device,
    token_budget: int = 16_384,
    max_batch: int = 512,
    epochs: int = 10,
    lora_lr: float = 2e-4,
    head_lr: float = 1e-3,
    grad_accum: int = 4,
    patience: int = 3,
    log_interval: int = 100,
    focal_gamma: float = 2.0,
    output_dir: Path | None = None,
) -> list[dict]:
    """Train LoRA adapters + head jointly. Supports DDP."""
    raw = unwrap(model)
    world_size = get_world_size()

    # Separate param groups: LoRA adapters get lower LR
    lora_params = [p for n, p in raw.encoder.named_parameters() if p.requires_grad]
    head_params = list(raw.head.parameters())

    all_trainable = lora_params + head_params  # cache for clip_grad_norm_

    fused = device.type == "cuda"
    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": lora_lr},
        {"params": head_params, "lr": head_lr},
    ], weight_decay=1e-4, fused=fused)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n_pos = sum(s["label"] for s in train_samples)
    n_neg = len(train_samples) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    criterion = FocalBCEWithLogitsLoss(gamma=focal_gamma, pos_weight=pos_weight)

    log: list[dict] = []
    best_val_loss = float("inf")
    best_encoder_state = None
    best_head_state = None
    wait = 0

    n_train = len(train_samples)

    # Determine this rank's shard of training data
    rank = 0
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()

    for epoch in range(epochs):
        model.train()
        rng = np.random.default_rng(42 + epoch)  # same perm on all ranks
        perm = rng.permutation(n_train)

        # Each rank takes every world_size-th sample (interleaved sharding)
        my_indices = perm[rank::world_size]
        my_samples = [train_samples[i] for i in my_indices]

        epoch_loss = 0.0
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
            labels = batch["labels"].to(device, dtype=torch.float32, non_blocking=True)

            logits = model(ids, mask)
            loss = criterion(logits, labels)
            if hasattr(raw.head, "get_balance_loss"):
                loss = loss + raw.head.get_balance_loss()
            loss = loss / grad_accum

            loss.backward()
            epoch_loss += loss.item() * grad_accum
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
                    bs = ids.shape[0]
                    print(
                        f"    [{epoch+1}/{epochs}] step {opt_step_count}  "
                        f"loss={avg:.4f}  bs={bs}  "
                        f"tok={n_tok:,}  {elapsed:.0f}s",
                        flush=True,
                    )

            if device.type == "mps":
                torch.mps.empty_cache()

        # Final gradient step
        if n_steps % grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(all_trainable, max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()

        # All-reduce avg loss across ranks
        avg_loss = epoch_loss / max(n_steps, 1)
        loss_tensor = torch.tensor([avg_loss], device=device)
        all_reduce_mean(loss_tensor)
        avg_loss = loss_tensor.item()

        # Validation (rank 0 only — no DDP forward needed)
        barrier()
        val_loss = 0.0
        val_acc = 0.0
        if is_main_process():
            raw.eval()
            val_logits, val_probs = batched_forward(
                val_samples, model, device, token_budget, max_batch,
            )
            val_labels_t = torch.tensor(
                [s["label"] for s in val_samples], dtype=torch.float32, device=device,
            )
            val_loss = criterion(val_logits, val_labels_t).item()
            val_preds = (val_probs.cpu().numpy() > 0.5).astype(int)
            val_true = np.array([s["label"] for s in val_samples])
            val_acc = float((val_preds == val_true).mean() * 100)

        # Broadcast val_loss to all ranks for early stopping
        vl_tensor = torch.tensor([val_loss], device=device)
        if torch.distributed.is_initialized():
            torch.distributed.broadcast(vl_tensor, src=0)
        val_loss = vl_tensor.item()

        opt_steps = max(n_steps // grad_accum, 1)
        entry = {
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "micro_batches": n_steps,
            "opt_steps": opt_steps,
            "tokens": n_tok,
        }
        log.append(entry)
        if is_main_process():
            print(
                f"  Epoch {epoch+1}/{epochs}: "
                f"train_loss={avg_loss:.4f}  val_loss={val_loss:.4f}  "
                f"val_acc={val_acc:.1f}%  "
                f"steps={opt_steps}  tok={n_tok:,}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_encoder_state = {
                k: v.cpu().clone() for k, v in raw.encoder.state_dict().items()
                if "lora" in k
            }
            best_head_state = {
                k: v.cpu().clone() for k, v in raw.head.state_dict().items()
            }
            # Save checkpoint immediately so progress survives crashes
            if output_dir is not None and is_main_process():
                output_dir.mkdir(parents=True, exist_ok=True)
                torch.save(best_encoder_state, output_dir / "adapter_model.pt")
                torch.save(best_head_state, output_dir / "head.pt")
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
    if best_encoder_state is not None:
        current = raw.encoder.state_dict()
        current.update({k: v.to(device) for k, v in best_encoder_state.items()})
        raw.encoder.load_state_dict(current)
    if best_head_state is not None:
        raw.head.load_state_dict({k: v.to(device) for k, v in best_head_state.items()})

    return log


# ── CLI ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LoRA fine-tune CaLM + classification head (single/multi-node)",
    )
    p.add_argument("--benchmark-dir", type=Path, default=PROJECT_ROOT / "data" / "benchmark")
    p.add_argument("--baked-train", type=Path, default=None,
                   help="Pre-baked train.pt (skips FASTA loading + tokenization)")
    p.add_argument("--baked-val", type=Path, default=None,
                   help="Pre-baked val.pt (skips FASTA loading + tokenization)")
    p.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models" / "calm")
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "models" / "heads" / "lora_mlp_v1")
    p.add_argument("--head", choices=["linear", "mlp", "gated"], default="mlp")
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lora-lr", type=float, default=2e-5)
    p.add_argument("--head-lr", type=float, default=1e-4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--sample", type=int, default=0, help="Max sequences per dataset (0=all)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--token-budget", type=int, default=16_384,
                   help="Max tokens per micro-batch (padded_len * batch_size)")
    p.add_argument("--max-batch", type=int, default=512,
                   help="Hard cap on batch size regardless of budget")
    p.add_argument("--focal-gamma", type=float, default=2.0,
                   help="Focal loss gamma (0=standard BCE, 2=default focal)")
    p.add_argument("--position-type", choices=["rotary", "alibi"], default="alibi",
                   help="Position encoding: 'rotary' (original RoPE, max 1026) or 'alibi' (no limit)")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True,
                   help="torch.compile encoder+head (CUDA only, on by default)")
    p.add_argument("--log-interval", type=int, default=100,
                   help="Print training progress every N optimizer steps (0=epoch only)")
    p.add_argument(
        "--benchmark-results", type=Path, default=None,
        help="Path to benchmark_multispecies_longest_orf.json for difficulty-weighted sampling",
    )
    p.add_argument(
        "--hard-threshold", type=float, default=88.0,
        help="Datasets below this ACC get all seqs; above get --easy-cap (default: 88%%)",
    )
    p.add_argument(
        "--easy-cap", type=int, default=500,
        help="Max seqs per dataset for datasets above --hard-threshold (default: 500)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Distributed setup ──
    dist_info = setup_distributed(backend="nccl")
    is_dist = dist_info is not None

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = select_device()

    # CUDA performance flags
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")  # TF32 for residual FP32 ops
        torch.backends.cudnn.benchmark = True        # autotuner for stable bucket shapes

    if is_main_process():
        print("=" * 70)
        print(f"  LoRA Fine-Tune CaLM + {args.head} Head")
        print("=" * 70)
        print(f"Device: {device}")
        if is_dist:
            print(f"Distributed: {dist_info['world_size']} nodes, rank {dist_info['rank']}")
        print(f"LoRA rank={args.lora_rank}, alpha={args.lora_alpha}")
        print(f"Epochs: {args.epochs}, LoRA LR: {args.lora_lr}, Head LR: {args.head_lr}")
        print(f"Token budget: {args.token_budget:,}, Max batch: {args.max_batch}")
        if args.sample:
            print(f"Sample per dataset: {args.sample}")

    # Init ORF scanner
    import proteinqc.cli.benchmark_multispecies as bm
    ctm = CodonTableManager()
    bm._orf_scanner = ORFScanner(ctm.get_genetic_code(1), min_codons=30)
    align_fn = codon_align_longest_orf

    # Load encoder (unfrozen for LoRA)
    if is_main_process():
        print(f"\nLoading CaLM encoder from {args.model_dir}...")
    encoder = CaLMEncoder(args.model_dir, freeze=False, position_type=args.position_type)

    # Apply LoRA (before moving to device / DDP)
    if is_main_process():
        print(f"\nApplying LoRA (r={args.lora_rank}, alpha={args.lora_alpha})...")
    n_lora = apply_lora(encoder, args.lora_rank, args.lora_alpha)

    tokenizer = CodonTokenizer(args.model_dir / "vocab.txt")

    # Build head
    head = build_head(args.head)
    n_head = sum(p.numel() for p in head.parameters())

    if is_main_process():
        print(f"  LoRA trainable params: {n_lora:,}")
        print(f"  Head trainable params: {n_head:,}")
        print(f"  Total trainable: {n_lora + n_head:,} / "
              f"{sum(p.numel() for p in encoder.parameters()):,} encoder")

    # Combine encoder + head
    model = LoRAClassifier(encoder, head)

    # torch.compile before DDP wrapping (CUDA only)
    if args.compile and device.type == "cuda":
        if is_main_process():
            print("\ntorch.compile (dynamic=None)...")
        if args.position_type == "rotary":
            encoder._ensure_rope(1026, device)
        else:
            encoder._ensure_alibi(MAX_SEQ_LEN, device)
        model = torch.compile(model, dynamic=None)

    # Wrap with DDP
    model = wrap_ddp(model, device, find_unused_params=True)

    # BF16 compress hook to halve allreduce volume
    if is_dist:
        from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
        model.register_comm_hook(state=None, hook=default_hooks.bf16_compress_hook)

    # ── Load data ──
    test_samples = []
    test_ds: list[str] = []
    test_datasets: set[str] = set()

    if args.baked_train and args.baked_val:
        # Fast path: load pre-baked .pt files (from bin/bake-eu7)
        if is_main_process():
            print(f"\nLoading pre-baked data...")
            print(f"  Train: {args.baked_train}")
            print(f"  Val:   {args.baked_val}")
        t_tok = time.time()
        tr_samples = torch.load(args.baked_train, weights_only=False)
        val_samples = torch.load(args.baked_val, weights_only=False)
        if is_main_process():
            n_pos = sum(1 for s in tr_samples if s["label"] == 1)
            n_neg = len(tr_samples) - n_pos
            print(f"  Train: {len(tr_samples):,} ({n_pos:,} cod / {n_neg:,} nc)")
            print(f"  Val:   {len(val_samples):,}")
            print(f"  Loaded in {time.time() - t_tok:.1f}s")
    else:
        # Legacy path: load from FASTA benchmark directories
        datasets = discover_datasets(args.benchmark_dir)

        ds_acc: dict[str, float] = {}
        bench_path = args.benchmark_results
        if bench_path is None:
            default_path = PROJECT_ROOT / "data" / "results" / "benchmark_multispecies_longest_orf.json"
            if default_path.exists():
                bench_path = default_path
        if bench_path and bench_path.exists():
            with open(bench_path) as f:
                bench_data = json.load(f)
            for entry in bench_data:
                key = f"{entry['tool']}/{entry['species']}"
                ds_acc[key] = entry["mlp_cls"]["ACC"]
            if is_main_process():
                n_hard = sum(1 for a in ds_acc.values() if a < args.hard_threshold)
                print(f"\nDifficulty-weighted sampling: threshold={args.hard_threshold}%")
                print(f"  Hard datasets (<{args.hard_threshold}%): {n_hard} (all seqs)")
                print(f"  Easy datasets (>={args.hard_threshold}%): {len(ds_acc) - n_hard} (cap {args.easy_cap})")

        per_ds_cap: dict[str, int] = {}
        for ds in datasets:
            ds_name = f"{ds['tool']}/{ds['species']}"
            if ds_name in ds_acc and ds_acc[ds_name] >= args.hard_threshold:
                per_ds_cap[ds_name] = args.easy_cap

        if is_main_process():
            print(f"\nDiscovered {len(datasets)} dataset pairs")
            print("Loading sequences...")

        all_seqs, all_ids, all_labels, all_ds_names = load_all_sequences(
            datasets, args.sample, align_fn, per_ds_cap=per_ds_cap,
        )
        if is_main_process():
            print(f"Total: {len(all_seqs):,} sequences")

        unique_ds = sorted(set(all_ds_names))
        train_datasets, test_datasets = split_by_dataset(unique_ds, seed=args.seed)
        if is_main_process():
            print(f"\nDataset split: {len(train_datasets)} train / {len(test_datasets)} test")

        train_seqs, train_labels = [], []
        test_seqs, test_labels = [], []
        for seq, label, ds_name in zip(all_seqs, all_labels, all_ds_names):
            if ds_name in train_datasets:
                train_seqs.append(seq)
                train_labels.append(label)
            else:
                test_seqs.append(seq)
                test_labels.append(label)
                test_ds.append(ds_name)

        rng = np.random.default_rng(args.seed)
        n = len(train_seqs)
        perm = rng.permutation(n)
        n_val = int(n * 0.1)

        val_seqs = [train_seqs[i] for i in perm[:n_val]]
        val_labels = [train_labels[i] for i in perm[:n_val]]
        tr_seqs = [train_seqs[i] for i in perm[n_val:]]
        tr_labels = [train_labels[i] for i in perm[n_val:]]

        n_pos = sum(tr_labels)
        n_neg = len(tr_labels) - n_pos
        if is_main_process():
            print(f"\nTraining: {len(tr_seqs):,} seqs ({n_pos:,} coding, {n_neg:,} nc)")
            print(f"Validation: {len(val_seqs):,} seqs")
            print(f"Test: {len(test_seqs):,} seqs across {len(test_datasets)} datasets")
            if is_dist:
                ws = get_world_size()
                print(f"  Per-rank: ~{len(tr_seqs) // ws:,} training seqs")

        if is_main_process():
            print("\nPre-tokenizing sequences...")
        t_tok = time.time()
        tr_samples = pre_tokenize(tr_seqs, tr_labels, tokenizer)
        val_samples = pre_tokenize(val_seqs, val_labels, tokenizer)
        test_samples = pre_tokenize(test_seqs, test_labels, tokenizer)
        if is_main_process():
            print(f"  Pre-tokenized {len(tr_samples) + len(val_samples) + len(test_samples):,} "
                  f"seqs in {time.time() - t_tok:.1f}s")

    barrier()

    # ── Train ──
    t0 = time.time()
    log = train_lora(
        model, tr_samples, val_samples, device,
        token_budget=args.token_budget,
        max_batch=args.max_batch,
        epochs=args.epochs,
        lora_lr=args.lora_lr,
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

    # ── Assessment (rank 0 only) ──
    if is_main_process():
        raw = unwrap(model)
        raw.encoder.eval()
        raw.head.eval()
        print("\nAssessing on held-out datasets...")
        results = []
        for ds_name in sorted(test_datasets):
            ds_samples = [s for s, d in zip(test_samples, test_ds) if d == ds_name]
            if len(ds_samples) < 2:
                continue

            _, probs = batched_forward(
                ds_samples, model, device, args.token_budget, args.max_batch,
            )
            ds_labels = np.array([s["label"] for s in ds_samples])
            preds = (probs.cpu().numpy() > 0.5).astype(int)
            m = compute_metrics(ds_labels, preds, probs.cpu().numpy())
            m["dataset"] = ds_name
            m["n"] = len(ds_samples)
            results.append(m)
            print(f"  {ds_name}: ACC={m['ACC']:.1f}%  MCC={m['MCC']:.1f}%  n={m['n']}")

        # Overall test
        overall: dict = {}
        if test_samples:
            _, all_probs = batched_forward(
                test_samples, model, device, args.token_budget, args.max_batch,
            )
            all_labels_np = np.array([s["label"] for s in test_samples])
            all_preds = (all_probs.cpu().numpy() > 0.5).astype(int)
            overall = compute_metrics(
                all_labels_np, all_preds, all_probs.cpu().numpy(),
            )
            print(
                f"\n[Overall test]  ACC={overall['ACC']:.2f}%  MCC={overall['MCC']:.2f}"
                f"  F1={overall['F1']:.2f}  AUC={overall['AUC']:.2f}"
            )

        # ── Save (rank 0 only) ──
        args.output.mkdir(parents=True, exist_ok=True)

        lora_state = {
            k: v.cpu() for k, v in raw.encoder.state_dict().items() if "lora" in k
        }
        torch.save(lora_state, args.output / "adapter_model.pt")

        lora_config = {
            "r": args.lora_rank, "lora_alpha": args.lora_alpha,
            "target_modules": ["q_proj", "k_proj", "v_proj"],
            "lora_dropout": 0.1, "bias": "none",
        }
        with open(args.output / "adapter_config.json", "w") as f:
            json.dump(lora_config, f, indent=2)

        torch.save(raw.head.state_dict(), args.output / "head.pt")

        with open(args.output / "training_log.json", "w") as f:
            json.dump({
                "head": args.head, "position_type": args.position_type,
                "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha,
                "epochs_run": len(log), "train_time_sec": train_time,
                "n_lora_params": n_lora, "n_head_params": n_head,
                "world_size": get_world_size(),
                "token_budget": args.token_budget,
                "max_batch": args.max_batch,
                "compiled": args.compile,
                "log": log,
            }, f, indent=2)

        with open(args.output / "test_results.json", "w") as f:
            json.dump({
                "head": args.head, "lora_rank": args.lora_rank,
                "overall": overall, "per_dataset": results,
            }, f, indent=2)

        print(f"\nSaved adapter  : {args.output / 'adapter_model.pt'}")
        print(f"Saved config   : {args.output / 'adapter_config.json'}")
        print(f"Saved head     : {args.output / 'head.pt'}")
        print(f"Saved results  : {args.output / 'test_results.json'}")

    barrier()
    if is_dist:
        cleanup_distributed()


if __name__ == "__main__":
    main()
