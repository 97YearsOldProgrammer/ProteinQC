#!/usr/bin/env python3
"""Train MLPHead on RNA Challenge dataset and save weights to models/heads/.

Reproduces the same train/test split and hyperparameters as benchmark.py
(SEED=42, 80/20 split, 20 epochs, lr=1e-3). Saves only the MLP head state
dict (~1 MB) for use by CaLMScorer in the ORF pipeline.

Usage:
    python -m proteinqc.scripts.save_mlp_head
    # or
    save-mlp-head
"""

import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from proteinqc.data.tokenizer import CodonTokenizer
from proteinqc.models.calm_encoder import CaLMEncoder
from proteinqc.models.classification_heads import MLPHead

BATCH_EMBED = 16
BATCH_TRAIN = 64
LR = 1e-3
EPOCHS = 20
SEED = 42
TEST_RATIO = 0.2
TOKEN_BUDGET = 8_192

_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = _ROOT / "data" / "rnachallenge"
MODEL_DIR = _ROOT / "models" / "calm"
OUTPUT_DIR = _ROOT / "models" / "heads"


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _codon_align(seq: str) -> str:
    seq = seq.upper().replace("U", "T")
    atg_pos = seq.find("ATG")
    if atg_pos >= 0:
        seq = seq[atg_pos:]
    trim = len(seq) - (len(seq) % 3)
    return seq[:trim]


def _load_dataset(tsv_path: Path) -> tuple[list[str], list[int]]:
    sequences, labels = [], []
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            seq = _codon_align(row["sequence"])
            if len(seq) >= 9:
                sequences.append(seq)
                labels.append(int(row["label"]))
    return sequences, labels


def _extract_embeddings(
    sequences: list[str],
    encoder: CaLMEncoder,
    tokenizer: CodonTokenizer,
    device: torch.device,
) -> torch.Tensor:
    n = len(sequences)
    embeddings = torch.zeros(n, encoder.hidden_size, dtype=torch.float32)
    sorted_idx = sorted(range(n), key=lambda i: len(sequences[i]))

    i = 0
    while i < n:
        max_codons = len(sequences[sorted_idx[i]]) // 3 + 2
        bs = max(1, min(TOKEN_BUDGET // max_codons, BATCH_EMBED, n - i))
        batch_idx = sorted_idx[i : i + bs]
        encoded = tokenizer.batch_encode(
            [sequences[j] for j in batch_idx], device=device
        )
        with torch.no_grad():
            cls = encoder(encoded["input_ids"], encoded["attention_mask"]).cpu()
        for k, orig in enumerate(batch_idx):
            embeddings[orig] = cls[k]
        if device.type == "mps" and max_codons > 500:
            torch.mps.empty_cache()
        i += bs

    return embeddings


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = _select_device()
    print(f"Device: {device}")

    print(f"Loading CaLM encoder from {MODEL_DIR} ...")
    encoder = CaLMEncoder(MODEL_DIR, freeze=True).to(device)
    encoder.train(False)
    tokenizer = CodonTokenizer(MODEL_DIR / "vocab.txt")

    print("Loading RNA Challenge dataset ...")
    sequences, labels = _load_dataset(DATA_DIR / "rnachallenge.tsv")
    print(f"  {len(sequences)} sequences")

    # Identical split to benchmark.py
    rng = np.random.RandomState(SEED)
    pos = [i for i, y in enumerate(labels) if y == 1]
    neg = [i for i, y in enumerate(labels) if y == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    n_pos_test = max(1, int(len(pos) * TEST_RATIO))
    n_neg_test = max(1, int(len(neg) * TEST_RATIO))
    train_idx = pos[n_pos_test:] + neg[n_neg_test:]

    print(f"Extracting embeddings ({len(train_idx)} train sequences) ...")
    train_x = _extract_embeddings(
        [sequences[i] for i in train_idx], encoder, tokenizer, device
    )
    train_y = torch.tensor([labels[i] for i in train_idx], dtype=torch.float32)

    print(f"Training MLPHead ({EPOCHS} epochs, lr={LR}) ...")
    head = MLPHead(hidden_size=768, mlp_hidden=256, dropout=0.1).to(device)
    head.train()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR)
    loader = DataLoader(
        TensorDataset(train_x, train_y), batch_size=BATCH_TRAIN, shuffle=True
    )

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)
            loss = criterion(head(x_batch), y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1}/{EPOCHS}: loss={epoch_loss / len(loader):.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "mlp_head_v1.pt"
    torch.save(head.state_dict(), out_path)
    print(f"Saved MLPHead weights → {out_path}")


if __name__ == "__main__":
    main()
