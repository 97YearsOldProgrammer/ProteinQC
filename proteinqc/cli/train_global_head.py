"""Train ONE global classification head on frozen CLS embeddings from HDF5.

Compares LinearHead, MLPHead, GatedHead on dataset-level train/test split.
Evaluates per held-out dataset, compares to XGBoost baseline and old benchmark.

Usage:
    python -m proteinqc.cli.train_global_head \
        --embeddings data/embeddings/benchmark_embeddings.h5 \
        --output models/heads/global_mlp_v1.pt

    python -m proteinqc.cli.train_global_head \
        --embeddings data/embeddings/benchmark_embeddings.h5 \
        --output models/heads/global_mlp_v1.pt \
        --baseline-json data/results/benchmark_multispecies_longest_orf.json \
        --xgb-results models/combiner/test_results.json \
        --head mlp --epochs 50 --lr 1e-3
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from proteinqc.models.classification_heads import GatedHead, LinearHead, MLPHead

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_h5(path: Path) -> dict:
    """Load HDF5 embedding file into memory."""
    with h5py.File(path, "r") as f:
        data = {
            "embeddings": f["embeddings"][:],
            "labels": f["labels"][:],
            "dataset_idx": f["dataset_idx"][:],
            "dataset_names": [s.decode() if isinstance(s, bytes) else s
                              for s in f["dataset_names"][:]],
        }
    print(f"Loaded {len(data['labels']):,} embeddings, "
          f"{len(data['dataset_names'])} datasets from {path}")
    return data


def split_by_dataset(
    dataset_names: list[str],
    test_frac: float = 0.2,
    seed: int = 42,
) -> tuple[set[str], set[str]]:
    """Dataset-level train/test split matching train_combiner.py logic."""
    names = np.array(dataset_names)
    rng = np.random.default_rng(seed)
    rng.shuffle(names)
    n_test = max(12, int(test_frac * len(names)))
    test_set = set(names[:n_test])
    train_set = set(names[n_test:])
    return train_set, test_set


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    """ACC, MCC, F1, AUC — same interface as train_combiner.py."""
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


def build_head(name: str, hidden_size: int = 768) -> nn.Module:
    heads = {
        "linear": lambda: LinearHead(hidden_size),
        "mlp": lambda: MLPHead(hidden_size, 256, 0.1),
        "gated": lambda: GatedHead(hidden_size, 256, 0.1),
    }
    if name not in heads:
        raise ValueError(f"Unknown head: {name}. Choose from {list(heads)}")
    return heads[name]()


def train_head(
    head: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 2048,
    patience: int = 5,
) -> list[dict]:
    """Train head with early stopping on validation loss."""
    head = head.to(device)
    head.train()

    # Class imbalance weighting
    n_pos = float(train_y.sum())
    n_neg = float(len(train_y) - n_pos)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_ds = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    log: list[dict] = []
    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        head.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = head(batch_x).squeeze(-1)
            loss = criterion(logits, batch_y)
            if hasattr(head, "get_balance_loss"):
                loss = loss + head.get_balance_loss()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = epoch_loss / n_batches

        # Validation
        head.eval()
        with torch.no_grad():
            val_logits = head(val_x.to(device)).squeeze(-1)
            val_loss = criterion(val_logits, val_y.to(device)).item()
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_preds = (val_probs > 0.5).astype(int)
            val_acc = float((val_preds == val_y.numpy()).mean() * 100)

        entry = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": scheduler.get_last_lr()[0],
        }
        log.append(entry)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch+1:3d}/{epochs}: "
                f"train_loss={avg_train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.1f}%"
            )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch+1} (patience={patience})")
                break

    if best_state is not None:
        head.load_state_dict(best_state)
    head = head.to(device)
    return log


def evaluate_per_dataset(
    head: nn.Module,
    embeddings: np.ndarray,
    labels: np.ndarray,
    dataset_idx: np.ndarray,
    dataset_names: list[str],
    test_datasets: set[str],
    device: torch.device,
) -> list[dict]:
    """Evaluate head per held-out dataset."""
    head.eval()
    results = []

    for ds_i, ds_name in enumerate(dataset_names):
        if ds_name not in test_datasets:
            continue

        mask = dataset_idx == ds_i
        if mask.sum() < 2:
            continue

        x = torch.tensor(embeddings[mask], dtype=torch.float32)
        y = labels[mask].astype(int)

        with torch.no_grad():
            logits = head(x.to(device)).squeeze(-1).cpu()
            probs = torch.sigmoid(logits).numpy()
            preds = (probs > 0.5).astype(int)

        m = compute_metrics(y, preds, probs)
        m["dataset"] = ds_name
        m["n"] = int(mask.sum())
        m["n_coding"] = int((y == 1).sum())
        m["n_noncoding"] = int((y == 0).sum())
        results.append(m)

    return sorted(results, key=lambda r: r["ACC"])


def print_comparison(
    results: list[dict],
    baseline_json: Path | None,
    xgb_json: Path | None,
) -> None:
    """Print side-by-side comparison table."""
    baseline_map: dict[str, float] = {}
    if baseline_json and baseline_json.exists():
        with open(baseline_json) as f:
            for e in json.load(f):
                key = f"{e['tool']}/{e['species']}"
                baseline_map[key] = e.get("mlp_cls", {}).get("ACC", 0)

    xgb_map: dict[str, float] = {}
    if xgb_json and xgb_json.exists():
        with open(xgb_json) as f:
            for e in json.load(f):
                xgb_map[e["dataset"]] = e.get("ACC", 0)

    w = 110
    print("\n" + "=" * w)
    print(f"{'Dataset':<35} {'N':>6} {'Head ACC':>9} {'XGB ACC':>9} {'Old CaLM':>9} {'vs XGB':>8}")
    print("-" * w)

    head_accs = []
    for r in results:
        ds = r["dataset"]
        head_acc = r["ACC"]
        head_accs.append(head_acc)
        xgb_acc = xgb_map.get(ds, float("nan"))
        old_acc = baseline_map.get(ds, float("nan"))
        delta = head_acc - xgb_acc if not np.isnan(xgb_acc) else float("nan")
        delta_str = f"{delta:+.1f}%" if not np.isnan(delta) else "N/A"
        xgb_str = f"{xgb_acc:.1f}%" if not np.isnan(xgb_acc) else "N/A"
        old_str = f"{old_acc:.1f}%" if not np.isnan(old_acc) else "N/A"
        print(f"  {ds:<33} {r['n']:>6} {head_acc:>8.1f}% {xgb_str:>9} {old_str:>9} {delta_str:>8}")

    print("-" * w)
    mean_head = np.mean(head_accs)
    xgb_vals = [xgb_map.get(r["dataset"], float("nan")) for r in results]
    mean_xgb = np.nanmean(xgb_vals) if any(not np.isnan(v) for v in xgb_vals) else float("nan")
    old_vals = [baseline_map.get(r["dataset"], float("nan")) for r in results]
    mean_old = np.nanmean(old_vals) if any(not np.isnan(v) for v in old_vals) else float("nan")
    print(f"  {'MEAN':<33} {'':>6} {mean_head:>8.1f}%", end="")
    if not np.isnan(mean_xgb):
        print(f" {mean_xgb:>8.1f}%", end="")
    if not np.isnan(mean_old):
        print(f" {mean_old:>8.1f}%", end="")
    print()
    print("=" * w)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train global classification head on frozen CLS embeddings",
    )
    p.add_argument(
        "--embeddings", type=Path, required=True,
        help="HDF5 file from extract_embeddings.py",
    )
    p.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "models" / "heads" / "global_mlp_v1.pt",
        help="Output path for trained head state dict",
    )
    p.add_argument(
        "--baseline-json", type=Path, default=None,
        help="Old benchmark JSON for per-dataset CaLM comparison",
    )
    p.add_argument(
        "--xgb-results", type=Path, default=None,
        help="XGBoost test_results.json for comparison",
    )
    p.add_argument(
        "--head", choices=["linear", "mlp", "gated"], default="mlp",
        help="Head architecture (default: mlp)",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = select_device()
    print("=" * 70)
    print(f"  Global Head Training -- {args.head}")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Embeddings: {args.embeddings}")
    print(f"Head: {args.head}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}, Batch: {args.batch_size}")

    # Load data
    data = load_h5(args.embeddings)
    embeddings = data["embeddings"]
    labels = data["labels"]
    dataset_idx = data["dataset_idx"]
    dataset_names = data["dataset_names"]

    # Dataset-level split
    train_datasets, test_datasets = split_by_dataset(dataset_names, seed=args.seed)
    print(f"\nDataset split: {len(train_datasets)} train / {len(test_datasets)} test")
    print("Test datasets:")
    for ds in sorted(test_datasets):
        print(f"  - {ds}")

    # Build train/val arrays
    train_mask = np.array([
        dataset_names[di] in train_datasets for di in dataset_idx
    ])
    test_mask = ~train_mask

    train_emb = embeddings[train_mask]
    train_lab = labels[train_mask].astype(np.float32)

    # 10% of training as validation for early stopping
    rng = np.random.default_rng(args.seed)
    n_train = len(train_emb)
    idx = rng.permutation(n_train)
    n_val = int(n_train * 0.1)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    train_x = torch.tensor(train_emb[tr_idx], dtype=torch.float32)
    train_y = torch.tensor(train_lab[tr_idx], dtype=torch.float32)
    val_x = torch.tensor(train_emb[val_idx], dtype=torch.float32)
    val_y = torch.tensor(train_lab[val_idx], dtype=torch.float32)

    n_pos = int(train_y.sum())
    n_neg = len(train_y) - n_pos
    print(f"\nTraining: {len(train_y):,} seqs ({n_pos:,} coding, {n_neg:,} nc)")
    print(f"Validation: {len(val_y):,} seqs")
    print(f"Test: {int(test_mask.sum()):,} seqs across {len(test_datasets)} datasets")

    # Build and train head
    head = build_head(args.head)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"\n{args.head} head: {n_params:,} params")

    t0 = time.time()
    log = train_head(
        head, train_x, train_y, val_x, val_y,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
    )
    train_time = time.time() - t0
    print(f"\nTraining completed in {train_time:.1f}s ({len(log)} epochs)")

    # Evaluate per dataset
    results = evaluate_per_dataset(
        head, embeddings, labels, dataset_idx, dataset_names,
        test_datasets, device,
    )

    # Overall test metrics
    test_emb = embeddings[test_mask]
    test_lab = labels[test_mask].astype(int)
    head.eval()
    with torch.no_grad():
        test_logits = head(torch.tensor(test_emb, dtype=torch.float32).to(device))
        test_probs = torch.sigmoid(test_logits.squeeze(-1)).cpu().numpy()
        test_preds = (test_probs > 0.5).astype(int)
    overall = compute_metrics(test_lab, test_preds, test_probs)
    print(
        f"\n[Overall test]  ACC={overall['ACC']:.2f}%  MCC={overall['MCC']:.2f}"
        f"  F1={overall['F1']:.2f}  AUC={overall['AUC']:.2f}"
    )

    # Per-dataset comparison
    print_comparison(results, args.baseline_json, args.xgb_results)

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(head.cpu().state_dict(), args.output)

    log_path = args.output.parent / "training_log.json"
    with open(log_path, "w") as f:
        json.dump({"head": args.head, "epochs_run": len(log),
                    "train_time_sec": train_time, "log": log}, f, indent=2)

    results_path = args.output.parent / "test_results.json"
    with open(results_path, "w") as f:
        json.dump({"head": args.head, "overall": overall,
                    "per_dataset": results}, f, indent=2)

    print(f"\nSaved head     : {args.output}")
    print(f"Saved log      : {log_path}")
    print(f"Saved results  : {results_path}")


if __name__ == "__main__":
    main()
