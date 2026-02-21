"""ORF detection pipeline: scan transcript → score with CaLM → rank.

Wires together:
  - SpeciesDetector  (codon usage kNN for species identification)
  - CodonTableManager (genetic code + codon usage tables)
  - ORFScanner        (exhaustive 3-frame ATG scan)
  - CaLMScorer        (frozen CaLM encoder + MLP head for coding probability)

Session state stored as instance attributes — components loaded once,
reused across all pipeline.run() calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .tools.calm_scorer import CaLMScorer
from .tools.codon_table import CodonTableManager
from .tools.orf_scanner import ORFCandidate, ORFScanner
from .tools.species_detect import SpeciesDetector, SpeciesMatch

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_MODEL_DIR = _PROJECT_ROOT / "models" / "calm"
_DEFAULT_HEAD_PATH = _PROJECT_ROOT / "models" / "heads" / "mlp_head_v1.pt"


@dataclass(frozen=True)
class ScoredORF:
    candidate: ORFCandidate
    score: float   # CaLM coding probability in [0, 1]
    rank: int      # 1-based, sorted by score descending


class ORFPipeline:
    """End-to-end ORF detection pipeline.

    Scans a transcript for all ORFs, scores each with CaLM, returns ranked list.
    All components are lazy-loaded on first call to run().

    Args:
        model_dir: CaLM model directory (default: models/calm/)
        head_weights_path: MLP head state dict (default: models/heads/mlp_head_v1.pt)
        min_codons: Minimum ORF length in codons including stop (default: 30)
        genetic_code_id: NCBI genetic code ID (default: 1 = standard)
    """

    def __init__(
        self,
        model_dir: Path | str = _DEFAULT_MODEL_DIR,
        head_weights_path: Path | str = _DEFAULT_HEAD_PATH,
        min_codons: int = 30,
        genetic_code_id: int = 1,
    ):
        self.model_dir = Path(model_dir)
        self.head_weights_path = Path(head_weights_path)
        self.min_codons = min_codons
        self.genetic_code_id = genetic_code_id

        self._table_manager: Optional[CodonTableManager] = None
        self._scorer: Optional[CaLMScorer] = None
        self._detector: Optional[SpeciesDetector] = None

    def _ensure_loaded(self):
        """Lazy-initialize pipeline components (called on first run)."""
        if self._table_manager is None:
            self._table_manager = CodonTableManager()
        if self._scorer is None:
            self._scorer = CaLMScorer(self.model_dir, self.head_weights_path)
        if self._detector is None:
            self._detector = SpeciesDetector()

    def run(
        self,
        transcript: str,
        species_hint: Optional[str | int] = None,
    ) -> list[ScoredORF]:
        """Scan transcript for ORFs, score with CaLM, return ranked results.

        Args:
            transcript: RNA or DNA transcript sequence (T or U accepted)
            species_hint: Species name ("Homo sapiens", "mouse") or NCBI TaxID.
                          If None, detected from codon usage. If no species_ref.npz,
                          falls back to human (TaxID 9606).

        Returns:
            ScoredORF list sorted by CaLM score descending.
            Empty list if no ORFs pass the min_codons threshold.
        """
        self._ensure_loaded()

        seq_dna = transcript.upper().replace("U", "T")

        # Species detection (for genetic code and codon context)
        species: SpeciesMatch = self._detector.detect(seq_dna, species_hint)

        # Genetic code determines stop codon set
        genetic_code = self._table_manager.get_genetic_code(self.genetic_code_id)

        # Exhaustive ORF scan
        scanner = ORFScanner(genetic_code, min_codons=self.min_codons)
        candidates = scanner.scan(seq_dna)

        if not candidates:
            return []

        # Score all candidates with CaLM
        seqs = [c.seq for c in candidates]
        raw_scores = self._scorer.batch_score(seqs)

        # Build and sort results
        paired = sorted(
            zip(candidates, raw_scores), key=lambda x: x[1], reverse=True
        )
        return [
            ScoredORF(candidate=cand, score=score, rank=i + 1)
            for i, (cand, score) in enumerate(paired)
        ]
