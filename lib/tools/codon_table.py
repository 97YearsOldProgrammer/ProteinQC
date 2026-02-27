"""Genetic code and codon usage frequency tables.

Loads stop/start codon rules via Biopython (by NCBI genetic code ID) and codon
usage frequencies via python-codon-tables (by species name). Both are loaded
once per session and cached.

Disk cache: ~/.cache/proteinqc/codon_tables/{taxid}.npz
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from Bio.Data import CodonTable as BioCodonTable

CACHE_DIR = Path.home() / ".cache" / "proteinqc" / "codon_tables"

# 64 standard DNA codons sorted lexicographically (used as vector indices)
STANDARD_CODONS: list[str] = sorted(
    a + b + c for a in "ACGT" for b in "ACGT" for c in "ACGT"
)
CODON_INDEX: dict[str, int] = {c: i for i, c in enumerate(STANDARD_CODONS)}


@dataclass(frozen=True)
class GeneticCode:
    """Codon classification rules for a NCBI genetic code."""

    stop_codons: frozenset  # DNA notation (T not U)
    start_codons: frozenset


@dataclass(frozen=True)
class CodonUsageTable:
    """Species-specific background codon usage frequencies."""

    taxid: int
    usage_vector: np.ndarray  # 64-dim, L1-normalized, indexed by STANDARD_CODONS


class CodonTableManager:
    """Session-level cache for genetic codes and codon usage tables.

    Usage::

        manager = CodonTableManager()
        code = manager.get_genetic_code(1)          # standard code
        usage = manager.get_usage("h_sapiens", 9606)
    """

    def __init__(self):
        self._code_cache: dict[int, GeneticCode] = {}
        self._usage_cache: dict[int, CodonUsageTable] = {}
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def get_genetic_code(self, code_id: int = 1) -> GeneticCode:
        """Get stop/start codons for an NCBI genetic code ID.

        Common IDs: 1=Standard, 2=Vertebrate Mitochondrial, 4=Mycoplasma.
        """
        if code_id in self._code_cache:
            return self._code_cache[code_id]

        bio_table = BioCodonTable.unambiguous_dna_by_id[code_id]
        code = GeneticCode(
            stop_codons=frozenset(bio_table.stop_codons),
            start_codons=frozenset(bio_table.start_codons),
        )
        self._code_cache[code_id] = code
        return code

    def get_usage(self, species_name: str, taxid: int) -> CodonUsageTable:
        """Get codon usage frequencies for a species (disk-cached).

        Args:
            species_name: python-codon-tables name (e.g. "h_sapiens")
            taxid: NCBI TaxID for cache key

        Returns:
            CodonUsageTable with 64-dim L1-normalized usage vector
        """
        if taxid in self._usage_cache:
            return self._usage_cache[taxid]

        cache_path = CACHE_DIR / f"{taxid}.npz"
        if cache_path.exists():
            data = np.load(cache_path)
            table = CodonUsageTable(taxid=taxid, usage_vector=data["usage_vector"])
            self._usage_cache[taxid] = table
            return table

        table = self._fetch_and_cache(species_name, taxid, cache_path)
        self._usage_cache[taxid] = table
        return table

    def _fetch_and_cache(
        self, species_name: str, taxid: int, cache_path: Path
    ) -> CodonUsageTable:
        from python_codon_tables import get_codons_table

        raw = get_codons_table(species_name)
        vec = _build_usage_vector(raw)
        np.savez(cache_path, usage_vector=vec)
        return CodonUsageTable(taxid=taxid, usage_vector=vec)


def _build_usage_vector(raw_table: dict) -> np.ndarray:
    """Convert a raw codon table dict to 64-dim L1-normalized vector."""
    vec = np.zeros(64, dtype=np.float32)
    for codon, freq in raw_table.items():
        dna_codon = codon.upper().replace("U", "T")
        if dna_codon in CODON_INDEX:
            vec[CODON_INDEX[dna_codon]] = float(freq)
    total = vec.sum()
    if total > 0:
        vec = vec / total
    return vec


def sequence_to_codon_vector(sequence: str) -> np.ndarray:
    """Compute 64-dim L1-normalized codon usage vector from a DNA sequence.

    Args:
        sequence: DNA sequence (T not U), length ideally a multiple of 3.

    Returns:
        64-dim float32 array, L1-normalized codon frequencies.
    """
    seq = sequence.upper().replace("U", "T")
    vec = np.zeros(64, dtype=np.float32)
    for i in range(0, len(seq) - 2, 3):
        codon = seq[i : i + 3]
        if len(codon) == 3 and codon in CODON_INDEX:
            vec[CODON_INDEX[codon]] += 1
    total = vec.sum()
    if total > 0:
        vec = vec / total
    return vec
