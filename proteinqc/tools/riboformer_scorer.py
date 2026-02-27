"""Riboformer-based translation efficiency scorer for ORF classification.

Wraps the RiboformerPyTorch model with a sliding-window strategy so it can
score variable-length ORFs (the native model expects exactly 40 codons).

For ORF classification without ribo-seq data, uses uniform coverage (all 1s)
as a sequence-only baseline — the model still extracts useful codon-context
signals from the sequence branch alone.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import torch

from .codon_table import CODON_INDEX
from .riboformer import RiboformerConfig, RiboformerPyTorch

WINDOW_SIZE = 40  # Riboformer fixed input width (codons)
BATCH_MAX = 256   # max windows per GPU batch (40 codons each = small)
MAX_SEQ_LEN = 3_000 * 3  # 3000 codons max (9000 bases)

# Pad short ORFs with stop codon index. Coverage for padding is 0.0,
# so the ribo-seq branch contributes near-zero signal for padded positions.
_PAD_INDEX = CODON_INDEX["TAA"]

_VALID_DNA = re.compile(r"^[ACGTUacgtu]+$")


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _dna_to_codon_indices(dna_sequence: str) -> list[int]:
    """Convert a DNA sequence to a list of codon integer indices.

    Args:
        dna_sequence: DNA string, length should be a multiple of 3.

    Returns:
        List of codon indices (0..63) from CODON_INDEX.
        Unknown codons map to _PAD_INDEX.
    """
    seq = dna_sequence.upper().replace("U", "T")
    indices: list[int] = []
    for i in range(0, len(seq) - 2, 3):
        codon = seq[i : i + 3]
        indices.append(CODON_INDEX.get(codon, _PAD_INDEX))
    return indices


class RiboformerScorer:
    """Score ORF translation efficiency using Riboformer.

    Uses a 40-codon sliding window for variable-length ORFs:
      - ORFs >= 40 codons: slide with stride 1, batch all windows, average.
      - ORFs < 40 codons: right-pad with stop codon index to 40.

    Without ribo-seq data, coverage defaults to uniform (all 1.0),
    providing a sequence-only translation efficiency estimate.

    Args:
        weights_path: Path to converted .pt weights file.
        device: Compute device (auto-selects MPS/CUDA/CPU if None).
    """

    def __init__(
        self,
        weights_path: Path | str,
        device: Optional[torch.device] = None,
    ):
        self.device = device or _select_device()
        self.weights_path = Path(weights_path)

        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"Riboformer weights not found: {self.weights_path}\n"
                "Run: python -m proteinqc.tools.riboformer_convert --download-all"
            )

        cfg = RiboformerConfig()
        self.model = RiboformerPyTorch(cfg)
        state = torch.load(
            self.weights_path, map_location=self.device, weights_only=True
        )
        self.model.load_state_dict(state)
        self.model = self.model.to(self.device)
        self.model.train(False)

    def score_one(
        self,
        dna_sequence: str,
        coverage: Optional[list[float]] = None,
    ) -> float:
        """Score a single ORF for translation efficiency.

        Args:
            dna_sequence: DNA sequence (ATG...stop). Length should be
                          a multiple of 3 (codon-aligned).
            coverage: Per-codon ribo-seq coverage values. If None, uses
                      uniform coverage (sequence-only mode).

        Returns:
            Translation efficiency prediction (higher = more efficient).
        """
        cleaned = _validate_dna(dna_sequence)
        codon_ids = _dna_to_codon_indices(cleaned)
        n_codons = len(codon_ids)

        if n_codons == 0:
            return 0.0

        if coverage is not None and len(coverage) != n_codons:
            raise ValueError(
                f"Coverage length ({len(coverage)}) != codon count ({n_codons})"
            )

        cov = coverage if coverage is not None else [1.0] * n_codons

        if n_codons < WINDOW_SIZE:
            # Pad to 40 codons
            pad_len = WINDOW_SIZE - n_codons
            padded_ids = codon_ids + [_PAD_INDEX] * pad_len
            padded_cov = cov + [0.0] * pad_len
            return self._forward_batch(
                [padded_ids], [padded_cov]
            )[0]

        # Batch all sliding windows into a single forward pass
        n_windows = n_codons - WINDOW_SIZE + 1
        all_ids = [codon_ids[s : s + WINDOW_SIZE] for s in range(n_windows)]
        all_cov = [cov[s : s + WINDOW_SIZE] for s in range(n_windows)]

        preds = self._forward_batch(all_ids, all_cov)
        return sum(preds) / len(preds)

    def batch_score(
        self,
        sequences: list[str],
        coverages: Optional[list[list[float]]] = None,
    ) -> list[float]:
        """Score multiple ORFs for translation efficiency.

        Args:
            sequences: DNA sequences (ATG...stop), codon-aligned.
            coverages: Per-sequence coverage lists. If None, uses uniform.

        Returns:
            Translation efficiency scores, same order as input.
        """
        if not sequences:
            return []

        results: list[float] = []
        for i, seq in enumerate(sequences):
            cov = coverages[i] if coverages is not None else None
            results.append(self.score_one(seq, cov))
        return results

    def _forward_batch(
        self,
        all_ids: list[list[int]],
        all_cov: list[list[float]],
    ) -> list[float]:
        """Run multiple 40-codon windows through the model in batches.

        Uses BATCH_MAX to avoid OOM on large ORFs with many windows.

        Returns:
            Prediction for each window.
        """
        n = len(all_ids)
        predictions: list[float] = []
        i = 0

        while i < n:
            batch_end = min(i + BATCH_MAX, n)
            seq_tensor = torch.tensor(
                all_ids[i:batch_end], dtype=torch.long, device=self.device
            )
            exp_tensor = torch.tensor(
                all_cov[i:batch_end], dtype=torch.float32, device=self.device
            )

            with torch.no_grad():
                pred, _ = self.model(seq_tensor, exp_tensor)

            predictions.extend(pred.squeeze(-1).tolist())

            if self.device.type == "mps":
                torch.mps.empty_cache()

            i = batch_end

        return predictions


def _validate_dna(dna_sequence: str) -> str:
    """Validate and normalize a DNA sequence.

    Raises ValueError for non-DNA characters or oversized input.
    """
    if not dna_sequence:
        return ""
    cleaned = dna_sequence.strip()
    if not _VALID_DNA.match(cleaned):
        raise ValueError("Sequence contains non-DNA characters")
    if len(cleaned) > MAX_SEQ_LEN:
        raise ValueError(
            f"Sequence too long ({len(cleaned)} bases > {MAX_SEQ_LEN} limit)"
        )
    return cleaned
