#!/usr/bin/env python3
"""GeneT5-style benchmark: frozen CaLM encoder + classification heads.

Pipeline:
  1. Load RNA Challenge dataset (coding vs non-coding)
  2. Codon-align sequences (trim to multiple of 3)
  3. Extract [CLS] embeddings from frozen CaLM on MPS
  4. Train/test 3 heads: LinearHead, MLPHead, GatedHead
  5. Report ACC, PRE, REC, F1, MCC + gate statistics

Uses pure PyTorch CaLM encoder — no multimolecule dependency.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from proteinqc.data.tokenizer import CodonTokenizer
from proteinqc.metrics.metrics import compute_metrics_from_logits
from proteinqc.models.calm_encoder import CaLMEncoder
from proteinqc.models.classification_heads import GatedHead, LinearHead, MLPHead

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BATCH_EMBED = 16     # batch size for embedding extraction (MPS memory)
BATCH_TRAIN = 64     # batch size for head training
LR = 1e-3
EPOCHS = 20
SEED = 42
TEST_RATIO = 0.2

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "rnachallenge"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
MODEL_DIR = PROJECT_ROOT / "models" / "calm"

# Token budget: max total tokens per batch.
# MPS lacks FlashAttention, so attention is O(seq^2) memory.
# Lower budget prevents OOM on long sequences.
TOKEN_BUDGET = 8_192


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RNA Classification Benchmark")
    parser.add_argument(
        "--verify", action="store_true",
        help="Compare embeddings against multimolecule (requires multimolecule)",
    )
    parser.add_argument(
        "--rmsnorm", action="store_true",
        help="Apply RMSNorm conversion before inference",
    )
    return parser.parse_args()


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def print_hardware(device: torch.device):
    print(f"PyTorch {torch.__version__}")
    print(f"Device : {device}")
    if device.type == "mps":
        print("Backend: Metal Performance Shaders (Apple GPU)")


def codon_align(seq: str) -> str:
    """Align sequence to reading frame, trim to codon boundary."""
    seq = seq.upper().replace("U", "T")
    atg_pos = seq.find("ATG")
    if atg_pos >= 0:
        seq = seq[atg_pos:]
    trim = len(seq) - (len(seq) % 3)
    return seq[:trim]


def load_dataset(tsv_path: Path) -> tuple[list[str], list[int], list[str]]:
    sequences, labels, ids = [], [], []
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            seq = codon_align(row["sequence"])
            if len(seq) >= 9:
                sequences.append(seq)
                labels.append(int(row["label"]))
                ids.append(row["sequence_id"])
    return sequences, labels, ids


def stratified_split(
    labels: list[int],
    test_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
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
    encoder: CaLMEncoder,
    tokenizer: CodonTokenizer,
    device: torch.device,
    batch_size: int = BATCH_EMBED,
) -> torch.Tensor:
    """Extract [CLS] embeddings using pure PyTorch encoder.

    Adaptive batching: sequences sorted by length, grouped by token budget.
    Returns float32 tensor [N, 768] on CPU.
    """
    encoder.to(device)
    encoder.eval()
    n = len(sequences)
    embeddings = torch.zeros(n, encoder.hidden_size, dtype=torch.float32)

    sorted_indices = sorted(range(n), key=lambda i: len(sequences[i]))

    start_time = time.time()
    done = 0
    i = 0

    while i < n:
        max_codons = len(sequences[sorted_indices[i]]) // 3 + 2
        adaptive_bs = max(1, TOKEN_BUDGET // max_codons)
        adaptive_bs = min(adaptive_bs, batch_size, n - i)

        batch_idx = sorted_indices[i : i + adaptive_bs]
        batch_seqs = [sequences[j] for j in batch_idx]

        encoded = tokenizer.batch_encode(batch_seqs, device=device)

        with torch.no_grad():
            cls = encoder(encoded["input_ids"], encoded["attention_mask"]).cpu()

        for j, orig_idx in enumerate(batch_idx):
            embeddings[orig_idx] = cls[j]

        # Free MPS memory pool (no FlashAttention → O(seq^2) allocations)
        if device.type == "mps" and max_codons > 500:
            torch.mps.empty_cache()

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
            if hasattr(head, "get_balance_loss"):
                loss = loss + head.get_balance_loss().to(device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        losses.append(epoch_loss / n_batches)
    return losses


def streaming_test(
    head: nn.Module,
    sequences: list[str],
    labels: list[int],
    test_idx: list[int],
    encoder: CaLMEncoder,
    tokenizer: CodonTokenizer,
    device: torch.device,
    batch_size: int = BATCH_EMBED,
) -> dict:
    """Streaming test: processes batch-by-batch, no stored test embeddings."""
    head = head.to(device)
    head.eval()
    encoder.eval()

    tp, tn, fp, fn = 0, 0, 0, 0
    sorted_test = sorted(test_idx, key=lambda i: len(sequences[i]))
    i = 0
    n = len(sorted_test)

    while i < n:
        max_codons = len(sequences[sorted_test[i]]) // 3 + 2
        adaptive_bs = max(1, TOKEN_BUDGET // max_codons)
        adaptive_bs = min(adaptive_bs, batch_size, n - i)

        batch_idx = sorted_test[i : i + adaptive_bs]
        batch_seqs = [sequences[j] for j in batch_idx]
        batch_labels = torch.tensor(
            [labels[j] for j in batch_idx], dtype=torch.long, device=device,
        )

        encoded = tokenizer.batch_encode(batch_seqs, device=device)
        with torch.no_grad():
            cls_emb = encoder(encoded["input_ids"], encoded["attention_mask"])
            logits = head(cls_emb).squeeze(-1)
            preds = (torch.sigmoid(logits) > 0.5).long()

        tp += ((preds == 1) & (batch_labels == 1)).sum().item()
        tn += ((preds == 0) & (batch_labels == 0)).sum().item()
        fp += ((preds == 1) & (batch_labels == 0)).sum().item()
        fn += ((preds == 0) & (batch_labels == 1)).sum().item()
        i += adaptive_bs

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total > 0 else 0.0
    pre = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * pre * rec / (pre + rec) if (pre + rec) > 0 else 0.0
    denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = ((tp * tn) - (fp * fn)) / denom if denom > 0 else 0.0

    metrics = {
        "ACC": acc * 100, "PRE": pre * 100, "REC": rec * 100,
        "F1": f1 * 100, "MCC": mcc * 100,
    }
    if hasattr(head, "get_gate_stats"):
        metrics["gate"] = head.get_gate_stats()
    return metrics


def test_head(
    head: nn.Module,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    device: torch.device,
) -> dict:
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
    if hasattr(head, "get_gate_stats"):
        metrics["gate"] = head.get_gate_stats()
    return metrics


def verify_against_multimolecule(
    sequences: list[str],
    encoder: CaLMEncoder,
    tokenizer: CodonTokenizer,
    device: torch.device,
    n_samples: int = 100,
):
    """Compare our [CLS] embeddings against multimolecule output."""
    print("\n--- Verification: Pure PyTorch vs multimolecule ---")
    try:
        from multimolecule import AutoTokenizer, CaLmModel
    except ImportError:
        print("  multimolecule not installed, skipping verification")
        return

    hf_tokenizer = AutoTokenizer.from_pretrained("multimolecule/calm")
    hf_model = CaLmModel.from_pretrained("multimolecule/calm").to(device)
    hf_model.eval()

    sample_seqs = sequences[:n_samples]
    max_diff = 0.0
    mean_diff = 0.0

    for seq in sample_seqs:
        encoded = tokenizer.batch_encode([seq], device=device)
        with torch.no_grad():
            our_cls = encoder(encoded["input_ids"], encoded["attention_mask"])

        hf_encoded = hf_tokenizer([seq], return_tensors="pt", padding=True)
        hf_encoded = {k: v.to(device) for k, v in hf_encoded.items()}
        with torch.no_grad():
            hf_out = hf_model(**hf_encoded)
            hf_cls = hf_out.last_hidden_state[:, 0, :]

        diff = (our_cls - hf_cls).abs()
        max_diff = max(max_diff, diff.max().item())
        mean_diff += diff.mean().item()

    mean_diff /= len(sample_seqs)
    print(f"  Samples compared: {len(sample_seqs)}")
    print(f"  Max absolute diff:  {max_diff:.2e}")
    print(f"  Mean absolute diff: {mean_diff:.2e}")
    match = "PASS" if max_diff < 1e-4 else "FAIL"
    print(f"  Result: {match} (threshold: 1e-4)")


def print_table(results: dict):
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
    args = parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = select_device()
    print("=" * 60)
    print("  ProteinQC — RNA Classification Benchmark")
    print("  Pure PyTorch CaLM encoder (no multimolecule)")
    print("=" * 60)
    print()
    print_hardware(device)

    # --- Load encoder + tokenizer ---
    print(f"\nLoading CaLM encoder from {MODEL_DIR} ...")
    encoder = CaLMEncoder(MODEL_DIR, freeze=True)
    encoder = encoder.to(device)
    encoder.eval()

    tokenizer = CodonTokenizer(MODEL_DIR / "vocab.txt")
    print(f"  Encoder: {sum(p.numel() for p in encoder.parameters()):,} params (frozen)")
    print(f"  Tokenizer: {tokenizer.vocab_size} tokens")

    # --- Optional RMSNorm conversion ---
    if args.rmsnorm:
        print("\nApplying RMSNorm conversion ...")
        from proteinqc.models.norm_convert import convert_layernorm_to_rmsnorm
        encoder = convert_layernorm_to_rmsnorm(encoder)
        print("  LayerNorm -> RMSNorm conversion complete")

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

    # --- Optional verification ---
    if args.verify:
        verify_against_multimolecule(sequences, encoder, tokenizer, device)

    # --- Extract train embeddings (fits in memory: ~64 MB) ---
    print(f"\nExtracting train embeddings ({len(train_idx)} sequences) ...")
    train_seqs = [sequences[i] for i in train_idx]
    train_embeddings = extract_embeddings(train_seqs, encoder, tokenizer, device)

    train_x = train_embeddings
    train_y = torch.tensor([labels[i] for i in train_idx], dtype=torch.long)
    print(f"Train embeddings: {train_x.shape}")

    # --- Extract test embeddings ---
    print(f"\nExtracting test embeddings ({len(test_idx)} sequences) ...")
    test_seqs = [sequences[i] for i in test_idx]
    test_embeddings = extract_embeddings(test_seqs, encoder, tokenizer, device)
    test_x = test_embeddings
    test_y = torch.tensor([labels[i] for i in test_idx], dtype=torch.long)
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
        print(f"  Loss: {losses[0]:.4f} -> {losses[-1]:.4f}  ({train_time:.1f}s)")

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

    report = {
        "experiment": "RNA Challenge — Pure PyTorch CaLM + head comparison",
        "model": str(MODEL_DIR),
        "device": str(device),
        "rmsnorm": args.rmsnorm,
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
        report["results"][name] = {k: v for k, v in m.items()}

    report_path = RESULTS_DIR / "benchmark_results.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nResults saved to {report_path}")

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

    curves_path = RESULTS_DIR / "training_curves.json"
    with open(curves_path, "w") as f:
        json.dump(training_losses, f, indent=2)
    print(f"Training curves saved to {curves_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
