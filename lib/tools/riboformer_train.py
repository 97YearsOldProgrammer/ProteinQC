"""
Riboformer training — pure PyTorch, zero TF dependency.

Replaces the original TF training.py. Same data format (xc.txt / yc.txt),
same normalization, same train/val/test split logic.

As a tool call:
    from proteinqc.tools.riboformer_train import train_riboformer
    train_riboformer(data_dir="datasets/ecoli", out="models/riboformer/ecoli.pt")

As CLI:
    python -m proteinqc.tools.riboformer_train \
        --data datasets/ecoli --out models/riboformer/ecoli.pt --epochs 15
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from proteinqc.tools.riboformer import RiboformerConfig, RiboformerPyTorch


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    epochs:     int   = 15
    batch_size: int   = 64
    lr:         float = 5e-4
    split:      float = 0.70     # fraction for training
    seed:       int   = 42
    device:     str   = "auto"   # "auto" | "mps" | "cpu" | "cuda"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(data_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load xc.txt (features) and yc.txt (targets) from data_dir.

    Returns:
        x: (N, 82) — first 40 cols = ribo-seq coverage, last 40 = codon indices
        y: (N,)    — target ribo-seq density
    """
    data_dir = Path(data_dir)
    xc_files = sorted(data_dir.glob("*xc.txt"))
    yc_files = sorted(data_dir.glob("*yc.txt"))

    if not xc_files or not yc_files:
        raise FileNotFoundError(
            f"xc.txt / yc.txt not found in {data_dir}. "
            "Run data_processing.py (or proteinqc process-ribo) first."
        )

    x = np.loadtxt(xc_files[0], delimiter="\t")
    y = np.loadtxt(yc_files[0], delimiter="\t")
    return x, y


def _normalize(x: np.ndarray, y: np.ndarray, wsize: int = 40):
    """Same normalization as original training.py."""
    x = x.copy()
    x[:, :wsize]  = x[:, :wsize]  / 100.0 - 5.0   # ribo-seq coverage
    x[:, wsize:]  = x[:, wsize:]  / 100.0 - 5.0   # codon indices (already scaled)
    y = y / 100.0 - 5.0
    return x, y


def _split(x: np.ndarray, y: np.ndarray, train_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    n   = len(x)
    idx = rng.permutation(n)
    n_train = int(train_frac * n)
    n_val   = (n - n_train) // 2
    tr_idx  = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    te_idx  = idx[n_train + n_val :]
    return (x[tr_idx], y[tr_idx],
            x[val_idx], y[val_idx],
            x[te_idx],  y[te_idx])


def _to_tensors(x: np.ndarray, y: np.ndarray,
                wsize: int, device: torch.device) -> TensorDataset:
    seq = torch.from_numpy(x[:, -wsize:].astype(np.int64)).to(device)    # codon indices
    exp = torch.from_numpy(x[:,  :wsize].astype(np.float32)).to(device)  # ribo coverage
    tgt = torch.from_numpy(y.astype(np.float32)).to(device)
    return TensorDataset(seq, exp, tgt)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_riboformer(
    data_dir: str | Path,
    out:      str | Path = "models/riboformer/model.pt",
    cfg:      TrainConfig | None = None,
    model_cfg: RiboformerConfig | None = None,
) -> dict:
    """Train Riboformer on a new species/condition and save weights.

    Args:
        data_dir:  Directory containing xc.txt and yc.txt.
        out:       Path to save the trained .pt weights.
        cfg:       Training hyperparameters.
        model_cfg: Model architecture config (default matches published model).

    Returns:
        dict with keys: best_val_loss, test_corr, train_corr, out_path
    """
    if cfg is None:
        cfg = TrainConfig()
    if model_cfg is None:
        model_cfg = RiboformerConfig()

    # device
    if cfg.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(cfg.device)
    print(f"Device: {device}")

    # data
    print("Loading data ...")
    x, y = load_data(data_dir)
    x, y = _normalize(x, y, model_cfg.wsize)
    x_tr, y_tr, x_val, y_val, x_te, y_te = _split(x, y, cfg.split, cfg.seed)
    print(f"  train={len(x_tr):,}  val={len(x_val):,}  test={len(x_te):,}")

    tr_ds  = _to_tensors(x_tr,  y_tr,  model_cfg.wsize, device)
    val_ds = _to_tensors(x_val, y_val, model_cfg.wsize, device)
    te_ds  = _to_tensors(x_te,  y_te,  model_cfg.wsize, device)

    tr_loader  = DataLoader(tr_ds,  batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size * 4)
    te_loader  = DataLoader(te_ds,  batch_size=cfg.batch_size * 4)

    # model
    model = RiboformerPyTorch(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    # optimizer + cosine LR decay (matches TF CosineDecay)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    total_steps = cfg.epochs * len(tr_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=0.0
    )
    loss_fn = nn.MSELoss()

    # training
    best_val_loss = float("inf")
    best_state = None

    print(f"\nTraining for {cfg.epochs} epochs ...")
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        model.train(True)
        train_loss = 0.0
        for seq, exp, tgt in tr_loader:
            optimizer.zero_grad()
            pred, _ = model(seq, exp)
            loss = loss_fn(pred.squeeze(), tgt)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item() * len(tgt)
        train_loss /= len(tr_ds)

        model.train(False)
        val_loss = 0.0
        with torch.no_grad():
            for seq, exp, tgt in val_loader:
                pred, _ = model(seq, exp)
                val_loss += loss_fn(pred.squeeze(), tgt).item() * len(tgt)
        val_loss /= len(val_ds)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"  epoch {epoch:3d}/{cfg.epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  "
              f"({time.time()-t0:.1f}s)")

    # evaluate best model
    model.load_state_dict(best_state)
    model.train(False)

    def _corr(loader):
        preds, targets = [], []
        with torch.no_grad():
            for seq, exp, tgt in loader:
                p, _ = model(seq, exp)
                preds.append(p.squeeze().cpu().numpy())
                targets.append(tgt.cpu().numpy())
        p = np.concatenate(preds)
        t = np.concatenate(targets)
        return float(np.corrcoef(p, t)[0, 1])

    test_corr  = _corr(te_loader)
    train_corr = _corr(tr_loader)
    print(f"\nTest  correlation: {test_corr:.4f}")
    print(f"Train correlation: {train_corr:.4f}")
    print(f"Best val MSE:      {best_val_loss:.4f}")

    # save
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out)
    print(f"Saved: {out}")

    return {
        "best_val_loss": best_val_loss,
        "test_corr":     test_corr,
        "train_corr":    train_corr,
        "out_path":      str(out),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Train Riboformer (PyTorch)")
    ap.add_argument("--data",    required=True, help="Directory with xc.txt and yc.txt")
    ap.add_argument("--out",     default="models/riboformer/model.pt")
    ap.add_argument("--epochs",  type=int,   default=15)
    ap.add_argument("--batch",   type=int,   default=64)
    ap.add_argument("--lr",      type=float, default=5e-4)
    ap.add_argument("--split",   type=float, default=0.70)
    ap.add_argument("--device",  default="auto")
    args = ap.parse_args()

    cfg = TrainConfig(
        epochs=args.epochs, batch_size=args.batch,
        lr=args.lr, split=args.split, device=args.device,
    )
    train_riboformer(args.data, args.out, cfg)


if __name__ == "__main__":
    main()
