"""Species detection from codon usage signatures.

k-NN lookup in 64-dim codon usage space using a pre-built species reference
file (species_ref.npz). Pure numpy — no model training required.

species_ref.npz stores:
  taxids:        int64  [N]       — NCBI TaxIDs
  names:         bytes  [N, ...]  — python-codon-tables names (ASCII)
  display_names: bytes  [N, ...]  — human-readable names (ASCII)
  vectors:       f32    [N, 64]   — L1-normalized codon usage

String arrays stored as fixed-width ASCII bytes to avoid pickle dependency.

If the user provides a species hint, detection is skipped entirely.
Unknown species fall back to human (TaxID 9606).

Build the reference once with:
    python -m proteinqc.scripts.build_species_ref
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .codon_table import sequence_to_codon_vector

SPECIES_REF_PATH = Path(__file__).parent / "data" / "species_ref.npz"
FALLBACK_TAXID = 9606  # Homo sapiens

# Human-readable alias → (taxid, python-codon-tables name)
# Only species available in python-codon-tables package
SPECIES_ALIASES: dict[str, tuple[int, str]] = {
    "homo sapiens": (9606, "h_sapiens_9606"),
    "human": (9606, "h_sapiens_9606"),
    "mus musculus": (10090, "m_musculus_10090"),
    "mouse": (10090, "m_musculus_10090"),
    "drosophila melanogaster": (7227, "d_melanogaster_7227"),
    "fly": (7227, "d_melanogaster_7227"),
    "caenorhabditis elegans": (6239, "c_elegans_6239"),
    "worm": (6239, "c_elegans_6239"),
    "saccharomyces cerevisiae": (4932, "s_cerevisiae_4932"),
    "yeast": (4932, "s_cerevisiae_4932"),
    "escherichia coli": (316407, "e_coli_316407"),
    "e. coli": (316407, "e_coli_316407"),
    "gallus gallus": (9031, "g_gallus_9031"),
    "chicken": (9031, "g_gallus_9031"),
    "bacillus subtilis": (1423, "b_subtilis_1423"),
}


@dataclass(frozen=True)
class SpeciesMatch:
    taxid: int
    species_name: str   # python-codon-tables name (e.g. "h_sapiens")
    confidence: float   # 1.0 for explicit hints, (0,1] for kNN
    is_fallback: bool


class SpeciesDetector:
    """Detect species from codon usage using pre-built kNN reference.

    Loads species_ref.npz on first call and caches in memory.

    Args:
        ref_path: Path to species_ref.npz (default: bundled data file)
    """

    def __init__(self, ref_path: Path = SPECIES_REF_PATH):
        self.ref_path = ref_path
        self._ref: Optional[dict] = None

    def detect(
        self,
        sequence: str,
        hint: Optional[int | str] = None,
    ) -> SpeciesMatch:
        """Detect species from codon usage, or use hint if provided.

        Args:
            sequence: DNA sequence for codon usage computation
            hint: int TaxID, or str species name/alias. Skips kNN if provided.

        Returns:
            SpeciesMatch with taxid and species_name
        """
        if hint is not None:
            return self._resolve_hint(hint)

        if not self.ref_path.exists():
            return SpeciesMatch(
                taxid=FALLBACK_TAXID,
                species_name="h_sapiens",
                confidence=0.0,
                is_fallback=True,
            )

        query_vec = sequence_to_codon_vector(sequence)
        return self._knn_lookup(query_vec)

    def _resolve_hint(self, hint: int | str) -> SpeciesMatch:
        if isinstance(hint, int):
            try:
                ref = self._load_ref()
                matches = np.where(ref["taxids"] == hint)[0]
                if len(matches) > 0:
                    i = int(matches[0])
                    return SpeciesMatch(
                        taxid=int(ref["taxids"][i]),
                        species_name=ref["names"][i],
                        confidence=1.0,
                        is_fallback=False,
                    )
            except Exception:
                pass
            return SpeciesMatch(
                taxid=hint,
                species_name=str(hint),
                confidence=1.0,
                is_fallback=False,
            )

        key = str(hint).lower().strip()
        if key in SPECIES_ALIASES:
            taxid, pct_name = SPECIES_ALIASES[key]
            return SpeciesMatch(
                taxid=taxid,
                species_name=pct_name,
                confidence=1.0,
                is_fallback=False,
            )

        # Substring match on display names in ref
        try:
            ref = self._load_ref()
            for i, display in enumerate(ref["display_names"]):
                if key in display.lower():
                    return SpeciesMatch(
                        taxid=int(ref["taxids"][i]),
                        species_name=ref["names"][i],
                        confidence=1.0,
                        is_fallback=False,
                    )
        except Exception:
            pass

        return SpeciesMatch(
            taxid=FALLBACK_TAXID,
            species_name="h_sapiens",
            confidence=0.5,
            is_fallback=True,
        )

    def _load_ref(self) -> dict:
        if self._ref is not None:
            return self._ref
        # No allow_pickle needed: string arrays stored as fixed-width ASCII bytes
        data = np.load(self.ref_path)
        self._ref = {
            "taxids": data["taxids"],
            "names": [s.decode("ascii").strip() for s in data["names"]],
            "display_names": [s.decode("ascii").strip() for s in data["display_names"]],
            "vectors": data["vectors"],
        }
        return self._ref

    def _knn_lookup(self, query_vec: np.ndarray) -> SpeciesMatch:
        """Find nearest neighbor in codon usage space (L2 distance)."""
        ref = self._load_ref()
        vectors = ref["vectors"]  # [N, 64]

        diffs = vectors - query_vec[None, :]
        dists = np.sqrt((diffs**2).sum(axis=1))

        best_idx = int(np.argmin(dists))
        best_dist = float(dists[best_idx])
        confidence = 1.0 / (1.0 + best_dist)

        return SpeciesMatch(
            taxid=int(ref["taxids"][best_idx]),
            species_name=ref["names"][best_idx],
            confidence=confidence,
            is_fallback=False,
        )
