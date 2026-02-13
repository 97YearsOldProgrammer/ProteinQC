"""Dataset utilities for RNA binary classification.

Handles loading test datasets in simple formats:
- FASTA with labels
- TSV/CSV with (sequence, label) columns
- JSON with sequence/label fields
"""

from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import Dataset


class RNABinaryDataset(Dataset):
    """Dataset for binary RNA classification (coding vs non-coding).

    Args:
        sequences: List of RNA sequences (strings)
        labels: List of binary labels (0=non-coding, 1=coding)
        tokenizer: Tokenizer for converting sequences to input_ids (optional)
    """

    def __init__(
        self,
        sequences: list[str],
        labels: list[int],
        tokenizer=None,
    ):
        if len(sequences) != len(labels):
            raise ValueError(
                f"Sequences ({len(sequences)}) and labels ({len(labels)}) "
                "must have same length"
            )

        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        """Get a single sample.

        Returns:
            sample: Dictionary with keys:
                - sequence: Raw RNA sequence (str)
                - label: Binary label (int tensor)
                - input_ids: Tokenized sequence (if tokenizer provided)
        """
        sample = {
            "sequence": self.sequences[idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }

        if self.tokenizer is not None:
            # Tokenize sequence (placeholder for now)
            # TODO: Implement codon-level tokenization
            sample["input_ids"] = torch.zeros(100, dtype=torch.long)

        return sample

    @classmethod
    def from_fasta(
        cls,
        fasta_path: Path | str,
        label_key: Literal["coding", "noncoding"] = "coding",
    ) -> "RNABinaryDataset":
        """Load dataset from FASTA file with labels in headers.

        Expected header format:
            >seq_id|label=coding
            >seq_id|label=noncoding

        Args:
            fasta_path: Path to FASTA file
            label_key: Key to extract from header (default: "coding")

        Returns:
            dataset: RNABinaryDataset instance
        """
        sequences = []
        labels = []

        with open(fasta_path) as f:
            current_seq = []
            current_label = None

            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    # Save previous sequence
                    if current_seq:
                        sequences.append("".join(current_seq))
                        labels.append(current_label)
                        current_seq = []

                    # Parse label from header
                    if f"label={label_key}" in line.lower():
                        current_label = 1
                    else:
                        current_label = 0
                else:
                    current_seq.append(line)

            # Save last sequence
            if current_seq:
                sequences.append("".join(current_seq))
                labels.append(current_label)

        return cls(sequences, labels)

    @classmethod
    def from_tsv(
        cls,
        tsv_path: Path | str,
        seq_col: str = "sequence",
        label_col: str = "label",
        sep: str = "\t",
    ) -> "RNABinaryDataset":
        """Load dataset from TSV/CSV file.

        Expected format:
            sequence    label
            ATGGCTA...  1
            GCTATCG...  0

        Args:
            tsv_path: Path to TSV/CSV file
            seq_col: Column name for sequences (default: "sequence")
            label_col: Column name for labels (default: "label")
            sep: Column separator (default: tab)

        Returns:
            dataset: RNABinaryDataset instance
        """
        import csv

        sequences = []
        labels = []

        with open(tsv_path) as f:
            reader = csv.DictReader(f, delimiter=sep)
            for row in reader:
                sequences.append(row[seq_col])
                labels.append(int(row[label_col]))

        return cls(sequences, labels)


def create_dummy_dataset(n_samples: int = 100) -> RNABinaryDataset:
    """Create dummy dataset for testing infrastructure.

    Args:
        n_samples: Number of samples to generate (default: 100)

    Returns:
        dataset: RNABinaryDataset with random sequences and labels
    """
    import random

    bases = ["A", "T", "G", "C"]
    sequences = []
    labels = []

    for _ in range(n_samples):
        # Generate random RNA sequence (300-600 bp)
        seq_len = random.randint(300, 600)
        seq = "".join(random.choice(bases) for _ in range(seq_len))
        sequences.append(seq)

        # Random binary label
        labels.append(random.randint(0, 1))

    return RNABinaryDataset(sequences, labels)
