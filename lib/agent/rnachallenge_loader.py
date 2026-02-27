"""Load RNA Challenge dataset for GRPO training.

Reads the pre-built TSV (27,283 sequences) and produces stratified
train/test splits matching the benchmark evaluation split.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path


def load_rnachallenge(
    tsv_path: Path | str,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Load RNA Challenge TSV into train/test splits.

    Args:
        tsv_path: Path to rnachallenge.tsv (columns: sequence_id, sequence, label).
        train_ratio: Fraction of data for training (rest is test).
        seed: Random seed for reproducible splits.

    Returns:
        (train, test) where each element is
        {"sequence_id": str, "sequence": str, "label": "coding"|"noncoding"}.
    """
    tsv_path = Path(tsv_path)
    if not tsv_path.exists():
        raise FileNotFoundError(f"RNA Challenge TSV not found: {tsv_path}")

    label_map = {"1": "coding", "0": "noncoding"}
    coding: list[dict] = []
    noncoding: list[dict] = []

    with open(tsv_path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            raw_label = row["label"].strip()
            label = label_map.get(raw_label)
            if label is None:
                raise ValueError(f"Unknown label '{raw_label}' for {row['sequence_id']}")
            entry = {
                "sequence_id": row["sequence_id"].strip(),
                "sequence": row["sequence"].strip(),
                "label": label,
            }
            if label == "coding":
                coding.append(entry)
            else:
                noncoding.append(entry)

    rng = random.Random(seed)
    rng.shuffle(coding)
    rng.shuffle(noncoding)

    def _split(items: list[dict]) -> tuple[list[dict], list[dict]]:
        n_train = int(len(items) * train_ratio)
        return items[:n_train], items[n_train:]

    train_c, test_c = _split(coding)
    train_nc, test_nc = _split(noncoding)

    train = train_c + train_nc
    test = test_c + test_nc
    rng.shuffle(train)
    rng.shuffle(test)

    return train, test
