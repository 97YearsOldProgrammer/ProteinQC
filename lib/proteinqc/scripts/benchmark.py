#!/usr/bin/env python3
"""GeneT5-style benchmark: frozen CaLM encoder + classification heads.

Pipeline:
  1. Load RNA Challenge dataset (coding vs non-coding)
  2. Codon-align sequences (trim to multiple of 3)
  3. Extract [CLS] embeddings from frozen CaLM on MPS
  4. Train/test 3 heads: LinearHead, MLPHead, GatedHead
  5. Report ACC, PRE, REC, F1, MCC + gate statistics
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from proteinqc.metrics.metrics import compute_metrics_from_logits
from proteinqc.models.classification_heads import GatedHead, LinearHead, MLPHead

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BATCH_EMBED = 16     # batch size for embedding extraction (MPS memory)
BATCH_TRAIN = 64   # batch size for head training
LR = 1e-3
EPOCHS = 20
SEED = 42
TEST_RATIO = 0.2

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "rnachallenge"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
MODEL_ID = "multimolecule/calm"


def select_device() -> torch.device:
    """Pick best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def print_hardware(device: torch.device):
    """Print hardware summary."""
    print(f"PyTorch {torch.__version__}")
    print(f"Device : {device}")
    if device.type == "mps":
        print("Backend: Metal Performance Shaders (Apple GPU)")
        print("  ANE (Neural Engine): not accessible via PyTorch")
        print("  ANE requires CoreML conversion (deployment stage)")


def codon_align(seq: str) -> str:
    """Align sequence to reading frame. No length truncation.

    Finds the first ATG (start codon) and begins the reading frame there.
    If no ATG exists, falls back to frame 0. Full sequence is preserved.
    """
    seq = seq.upper().replace("U", "T")
    atg_pos = seq.find("ATG")
    if atg_pos >= 0:
        seq = seq[atg_pos:]
    trim = len(seq) - (len(seq) % 3)
    return seq[:trim]


def load_dataset(tsv_path: Path) -> tuple[list[str], list[int], list[str]]:
    """Load RNA Challenge TSV. Returns (sequences, labels, ids)."""
    sequences, labels, ids = [], [], []
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            seq = codon_align(row["sequence"])
            if len(seq) >= 9:  # minimum 3 codons
                sequences.append(seq)
                labels.append(int(row["label"]))
                ids.append(row["sequence_id"])
    return sequences, labels, ids


def stratified_split(
    labels: list[int],
    test_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Stratified train/test index split."""
    rng = np.random.RandomState(seed)
    pos = [i for i, y in enumerate(labels) if y == 1]
    neg = [i for i, y in enumerate(labels) if y == 0]

    rng.shuffle(pos)
    rng.shuffle(neg)

    n_pos_test = max(1, int(len(pos) * test_ratio))
    n_neg_test = max(1, int(len(neg) * test_ratio))

    test_idx = pos[:n_pos_test] + neg[:n_neg_test]
    train_idx = pos[n_pos_test:] + neg[n_neg_test:]

    rng.shuffle(test_idx)
    rng.shuffle(train_idx)

    return train_idx, test_idx


def extract_embeddings(
    sequences: list[str],
    device: torch.device,
    batch_size: int = BATCH_EMBED,
) -> torch.Tensor:
    """Extract [CLS] embeddings from frozen CaLM encoder.

    Full-length inference — no truncation, no chunking. CaLM uses RoPE
    positional embeddings which extrapolate to arbitrary sequence lengths.

    Adaptive batching: sequences are sorted by length and grouped so that
    total tokens per batch (max_len × batch_count) stays within a budget.
    Short sequences batch efficiently; long sequences run individually.
    Every sequence sees the full encoder in a single forward pass.

    Returns float32 tensor [N, 768] on CPU.
    """
    from multimolecule import AutoTokenizer, CaLmModel

    print(f"Loading CaLM tokenizer + model from '{MODEL_ID}' ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = CaLmModel.from_pretrained(MODEL_ID)
    model = model.to(device)
    model.eval()

    n = len(sequences)
    embeddings = torch.zeros(n, 768, dtype=torch.float32)

    # Sort by codon length for efficient batching (least padding waste)
    sorted_indices = sorted(range(n), key=lambda i: len(sequences[i]))

    # Token budget: max total tokens per batch to keep attention under ~4 GB
    # attention = batch × heads(12) × seq² × 4 bytes → budget ~16k tokens total
    TOKEN_BUDGET = 16_384

    start_time = time.time()
    done = 0
    i = 0

    while i < n:
        # Determine batch size based on longest sequence in this group
        max_codons = len(sequences[sorted_indices[i]]) // 3 + 2  # +2 special tokens
        adaptive_bs = max(1, TOKEN_BUDGET // max_codons)
        adaptive_bs = min(adaptive_bs, batch_size, n - i)

        batch_idx = sorted_indices[i : i + adaptive_bs]
        batch_seqs = [sequences[j] for j in batch_idx]

        encoded = tokenizer(
            batch_seqs,
            return_tensors="pt",
            padding=True,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            out = model(**encoded)
            cls = out.last_hidden_state[:, 0, :].cpu()

        for j, orig_idx in enumerate(batch_idx):
            embeddings[orig_idx] = cls[j]

        done += adaptive_bs
        i += adaptive_bs

        if done % (batch_size * 50) < adaptive_bs or done == n:
            elapsed = time.time() - start_time
            rate = done / elapsed if elapsed > 0 else 0
            max_len = encoded["input_ids"].shape[1]
            print(f"  [{done:>6}/{n}]  {rate:.0f} seq/s  "
                  f"(batch={adaptive_bs}, max_tokens={max_len})")

    elapsed = time.time() - start_time
    print(f"Done in {elapsed:.1f}s  ({n / elapsed:.0f} seq/s)")
    return embeddings


def train_head(
    head: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    device: torch.device,
    epochs: int = EPOCHS,
    lr: float = LR,
    batch_size: int = BATCH_TRAIN,
) -> list[float]:
    """Train a classification head. Returns per-epoch losses."""
    head = head.to(device)
    head.train()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)

    dataset = TensorDataset(train_x, train_y.float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)

            logits = head(x_batch)
            loss = criterion(logits, y_batch)

            # Add balance loss for GatedHead
            if hasattr(head, "get_balance_loss"):
                loss = loss + head.get_balance_loss().to(device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

    return losses


def test_head(
    head: nn.Module,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    device: torch.device,
) -> dict:
    """Test a classification head. Returns metrics dict."""
    head = head.to(device)
    head.eval()

    all_logits = []
    dataset = TensorDataset(test_x, test_y)
    loader = DataLoader(dataset, batch_size=BATCH_TRAIN, shuffle=False)

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            logits = head(x_batch)
            all_logits.append(logits.cpu())

    all_logits = torch.cat(all_logits)
    metrics = compute_metrics_from_logits(test_y, all_logits, threshold=0.5)

    # Gate stats
    if hasattr(head, "get_gate_stats"):
        metrics["gate"] = head.get_gate_stats()

    return metrics


def print_table(results: dict):
    """Print results as a formatted table."""
    print(f"\n{'Head':<12} {'ACC':>7} {'PRE':>7} {'REC':>7} {'F1':>7} {'MCC':>7}")
    print("-" * 55)
    for name, m in results.items():
        print(
            f"{name:<12} "
            f"{m['ACC']:>6.2f}% "
            f"{m['PRE']:>6.2f}% "
            f"{m['REC']:>6.2f}% "
            f"{m['F1']:>6.2f}% "
            f"{m['MCC']:>6.2f}%"
        )


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = select_device()
    print("=" * 60)
    print("  ProteinQC — RNA Classification Benchmark")
    print("  GeneT5-style: frozen encoder + trainable head")
    print("=" * 60)
    print()
    print_hardware(device)

    # --- Load data ---
    print(f"\nLoading RNA Challenge dataset ...")
    sequences, labels, ids = load_dataset(DATA_DIR / "rnachallenge.tsv")
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    print(f"  Total : {len(labels)}")
    print(f"  Coding: {n_pos}  Non-coding: {n_neg}")

    # --- Train/test split ---
    train_idx, test_idx = stratified_split(labels, TEST_RATIO, SEED)
    print(f"  Train : {len(train_idx)}  Test: {len(test_idx)}")

    # --- Extract embeddings ---
    print()
    all_embeddings = extract_embeddings(sequences, device)

    train_x = all_embeddings[train_idx]
    train_y = torch.tensor([labels[i] for i in train_idx], dtype=torch.long)
    test_x = all_embeddings[test_idx]
    test_y = torch.tensor([labels[i] for i in test_idx], dtype=torch.long)

    print(f"\nTrain embeddings: {train_x.shape}")
    print(f"Test  embeddings: {test_x.shape}")

    # --- Define heads ---
    heads = {
        "Linear": LinearHead(hidden_size=768),
        "MLP": MLPHead(hidden_size=768, mlp_hidden=256, dropout=0.1),
        "Gated": GatedHead(
            hidden_size=768, mlp_hidden=256, dropout=0.1, balance_loss_weight=0.01
        ),
    }

    for name, head in heads.items():
        n_params = sum(p.numel() for p in head.parameters())
        print(f"  {name:<8}: {n_params:>8,} params")

    # --- Train + test each head ---
    results = {}
    training_losses = {}

    for name, head in heads.items():
        print(f"\n--- Training {name} ({EPOCHS} epochs, lr={LR}) ---")
        t0 = time.time()
        losses = train_head(head, train_x, train_y, device)
        train_time = time.time() - t0
        print(f"  Loss: {losses[0]:.4f} → {losses[-1]:.4f}  ({train_time:.1f}s)")

        metrics = test_head(head, test_x, test_y, device)
        metrics["train_time_sec"] = train_time
        metrics["final_loss"] = losses[-1]
        results[name] = metrics
        training_losses[name] = losses

        if "gate" in metrics:
            g = metrics["gate"]
            print(f"  Gate: mean={g['gate_mean']:.3f}  "
                  f"shortcut={g['pct_shortcut']:.1f}%  mlp={g['pct_mlp']:.1f}%")

    # --- Print results ---
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print_table(results)

    # --- Save results ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # JSON report
    report = {
        "experiment": "RNA Challenge — CaLM frozen encoder + head comparison",
        "model": MODEL_ID,
        "device": str(device),
        "dataset": {
            "total": len(labels),
            "coding": n_pos,
            "non_coding": n_neg,
            "train": len(train_idx),
            "test": len(test_idx),
        },
        "hyperparams": {
            "epochs": EPOCHS,
            "lr": LR,
            "batch_size": BATCH_TRAIN,
            "seed": SEED,
        },
        "results": {},
    }
    for name, m in results.items():
        report["results"][name] = {
            k: v for k, v in m.items()
        }

    report_path = RESULTS_DIR / "benchmark_results.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nResults saved to {report_path}")

    # TSV summary
    tsv_path = RESULTS_DIR / "benchmark_summary.tsv"
    with open(tsv_path, "w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["head", "ACC", "PRE", "REC", "F1", "MCC",
                         "train_time_sec", "final_loss"])
        for name, m in results.items():
            writer.writerow([
                name,
                f"{m['ACC']:.2f}",
                f"{m['PRE']:.2f}",
                f"{m['REC']:.2f}",
                f"{m['F1']:.2f}",
                f"{m['MCC']:.2f}",
                f"{m['train_time_sec']:.2f}",
                f"{m['final_loss']:.4f}",
            ])
    print(f"Summary saved to {tsv_path}")

    # Training curves
    curves_path = RESULTS_DIR / "training_curves.json"
    with open(curves_path, "w") as f:
        json.dump(training_losses, f, indent=2)
    print(f"Training curves saved to {curves_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
