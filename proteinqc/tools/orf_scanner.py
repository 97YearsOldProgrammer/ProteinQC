"""Exhaustive ORF scanner for transcript sequences.

Scans all 3 forward reading frames for open reading frames defined by ATG
start codons and species-specific stop codons (from GeneticCode). Deduplicates
sub-ORFs within the same frame + same stop codon, keeping the longest.

O(N) per reading frame where N is sequence length in nucleotides.
"""

from __future__ import annotations

from dataclasses import dataclass

from .codon_table import GeneticCode


@dataclass(frozen=True)
class ORFCandidate:
    """A candidate ORF identified by exhaustive scan."""

    seq: str            # DNA sequence, includes stop codon, len % 3 == 0
    start: int          # 0-based inclusive (nt position of start codon)
    stop: int           # 0-based exclusive (nt position after stop codon)
    frame: int          # reading frame: 0, 1, or 2
    length_codons: int  # total codons including stop codon
    start_codon: str    # e.g. "ATG"


class ORFScanner:
    """Exhaustive ORF scanner for 3 forward reading frames.

    Phase A: ATG-only start codons. Non-ATG starts deferred to Phase B.
    Phase A: Forward strand only. Reverse complement deferred to Phase B.

    Args:
        genetic_code: GeneticCode with species-specific stop/start codon sets
        min_codons: Minimum ORF length in codons (including stop), default 30
    """

    def __init__(self, genetic_code: GeneticCode, min_codons: int = 30):
        self.genetic_code = genetic_code
        self.min_codons = min_codons

    def scan(self, transcript: str) -> list[ORFCandidate]:
        """Scan transcript for ORF candidates in 3 forward reading frames.

        Deduplicates sub-ORFs: same frame + same stop → keep longest start.

        Args:
            transcript: DNA sequence (T not U), any length

        Returns:
            ORF candidates sorted by length_codons descending
        """
        seq = transcript.upper().replace("U", "T")
        candidates: list[ORFCandidate] = []

        for frame in range(3):
            candidates.extend(self._scan_frame(seq, frame))

        candidates.sort(key=lambda o: o.length_codons, reverse=True)
        return candidates

    def _scan_frame(self, seq: str, frame: int) -> list[ORFCandidate]:
        """Scan a single reading frame for ORFs."""
        stop_codons = self.genetic_code.stop_codons
        n = len(seq)
        n_codons = (n - frame) // 3

        # Map: stop_codon_index → earliest_atg_codon_index (for dedup)
        best_start: dict[int, int] = {}
        current_atg: int | None = None  # codon index of current open ATG

        for codon_idx in range(n_codons):
            pos = frame + codon_idx * 3
            codon = seq[pos : pos + 3]

            # ATG opens a new ORF (only if no open ORF already)
            if codon == "ATG" and current_atg is None:
                current_atg = codon_idx

            # Stop codon closes the current ORF
            if codon in stop_codons:
                if current_atg is not None:
                    stop_idx = codon_idx
                    if stop_idx not in best_start:
                        best_start[stop_idx] = current_atg
                    else:
                        best_start[stop_idx] = min(best_start[stop_idx], current_atg)
                current_atg = None

        candidates = []
        for stop_idx, atg_idx in best_start.items():
            length_codons = stop_idx - atg_idx + 1
            if length_codons < self.min_codons:
                continue

            orf_start_nt = frame + atg_idx * 3
            orf_stop_nt = frame + (stop_idx + 1) * 3
            orf_seq = seq[orf_start_nt:orf_stop_nt]

            candidates.append(
                ORFCandidate(
                    seq=orf_seq,
                    start=orf_start_nt,
                    stop=orf_stop_nt,
                    frame=frame,
                    length_codons=length_codons,
                    start_codon=orf_seq[:3],
                )
            )

        return candidates
