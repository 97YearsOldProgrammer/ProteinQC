"""ORF detection pipeline: scan transcript → score with CaLM → rank.

Components lazy-loaded on first run(), reused across calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .tools.calm_scorer import CaLMScorer
from .tools.codon_table import CodonTableManager
from .tools.orf_scanner import ORFCandidate, ORFScanner
from .tools.species_detect import SpeciesDetector, SpeciesMatch

if TYPE_CHECKING:
    from .tools.perplexity_scorer import PerplexityScorer
    from .tools.pfam_scanner import DomainHit, PfamScanner
    from .tools.riboformer_scorer import RiboformerScorer

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_MODEL_DIR = _PROJECT_ROOT / "models" / "calm"
_DEFAULT_HEAD_PATH = _PROJECT_ROOT / "models" / "heads" / "mlp_head_v1.pt"
_DEFAULT_PFAM_DIR = _PROJECT_ROOT / "models" / "pfam"


@dataclass(frozen=True)
class ScoredORF:
    candidate: ORFCandidate
    score: float   # CaLM coding probability in [0, 1]
    rank: int      # 1-based, sorted by score descending


@dataclass(frozen=True)
class EnrichedORF:
    """ORF with classification, perplexity, and domain annotations."""

    candidate: ORFCandidate
    calm_score: float                        # coding probability [0, 1]
    perplexity: Optional[float]              # lower = more natural; None if disabled
    translation_efficiency: Optional[float]  # Riboformer TE; None if disabled
    domain_hits: Optional[list[DomainHit]]   # None if disabled
    rank: int                                # 1-based, by calm_score desc


class ORFPipeline:
    """Scan transcript → score ORFs with CaLM → optionally enrich with perplexity/Pfam/Riboformer."""

    def __init__(
        self,
        model_dir: Path | str = _DEFAULT_MODEL_DIR,
        head_weights_path: Path | str = _DEFAULT_HEAD_PATH,
        min_codons: int = 30,
        genetic_code_id: int = 1,
        enable_perplexity: bool = False,
        enable_pfam: bool = False,
        enable_riboformer: bool = False,
        pfam_db_path: Optional[Path | str] = None,
        antifam_db_path: Optional[Path | str] = None,
        riboformer_weights: Optional[Path | str] = None,
    ):
        self.model_dir = Path(model_dir)
        self.head_weights_path = Path(head_weights_path)
        self.min_codons = min_codons
        self.genetic_code_id = genetic_code_id
        self.enable_perplexity = enable_perplexity
        self.enable_pfam = enable_pfam
        self.enable_riboformer = enable_riboformer
        self.pfam_db_path = (
            Path(pfam_db_path) if pfam_db_path else _DEFAULT_PFAM_DIR / "Pfam-A.hmm"
        )
        self.antifam_db_path = (
            Path(antifam_db_path) if antifam_db_path else _DEFAULT_PFAM_DIR / "AntiFam.hmm"
        )
        self.riboformer_weights = Path(riboformer_weights) if riboformer_weights else None

        self._table_manager: Optional[CodonTableManager] = None
        self._scorer: Optional[CaLMScorer] = None
        self._detector: Optional[SpeciesDetector] = None
        self._perplexity_scorer: Optional[PerplexityScorer] = None
        self._pfam_scanner: Optional[PfamScanner] = None
        self._riboformer_scorer: Optional[RiboformerScorer] = None

    def _ensure_loaded(self):
        """Lazy-initialize pipeline components (called on first run)."""
        if self._table_manager is None:
            self._table_manager = CodonTableManager()
        if self._scorer is None:
            self._scorer = CaLMScorer(self.model_dir, self.head_weights_path)
        if self._detector is None:
            self._detector = SpeciesDetector()
        if self.enable_perplexity and self._perplexity_scorer is None:
            from .tools.perplexity_scorer import PerplexityScorer
            self._perplexity_scorer = PerplexityScorer(self.model_dir)
        if self.enable_pfam and self._pfam_scanner is None:
            from .tools.pfam_scanner import PfamScanner
            antifam = self.antifam_db_path if self.antifam_db_path.exists() else None
            self._pfam_scanner = PfamScanner(self.pfam_db_path, antifam)
        if self.enable_riboformer and self._riboformer_scorer is None:
            from .tools.riboformer_scorer import RiboformerScorer
            if self.riboformer_weights is None:
                raise ValueError(
                    "enable_riboformer=True requires riboformer_weights path"
                )
            self._riboformer_scorer = RiboformerScorer(self.riboformer_weights)

    def _scan_and_score(
        self,
        transcript: str,
        species_hint: Optional[str | int],
    ) -> tuple[list[ORFCandidate], list[float]]:
        """Normalize → detect species → scan ORFs → CaLM score. Returns parallel lists."""
        self._ensure_loaded()

        seq_dna = transcript.upper().replace("U", "T")
        species: SpeciesMatch = self._detector.detect(seq_dna, species_hint)
        genetic_code = self._table_manager.get_genetic_code(self.genetic_code_id)

        scanner = ORFScanner(genetic_code, min_codons=self.min_codons)
        candidates = scanner.scan(seq_dna)

        if not candidates:
            return [], []

        seqs = [c.seq for c in candidates]
        raw_scores = self._scorer.batch_score(seqs)
        return candidates, raw_scores

    def run(
        self,
        transcript: str,
        species_hint: Optional[str | int] = None,
    ) -> list[ScoredORF]:
        """Scan → score → rank. Returns ScoredORF list sorted by score desc."""
        candidates, raw_scores = self._scan_and_score(transcript, species_hint)
        if not candidates:
            return []

        paired = sorted(
            zip(candidates, raw_scores), key=lambda x: x[1], reverse=True
        )
        return [
            ScoredORF(candidate=cand, score=score, rank=i + 1)
            for i, (cand, score) in enumerate(paired)
        ]

    def run_enriched(
        self,
        transcript: str,
        species_hint: Optional[str | int] = None,
    ) -> list[EnrichedORF]:
        """Scan → score → enrich with perplexity/Pfam/Riboformer. Sorted by calm_score desc."""
        candidates, raw_scores = self._scan_and_score(transcript, species_hint)
        if not candidates:
            return []

        seqs = [c.seq for c in candidates]

        perplexities: Optional[list[float]] = None
        if self._perplexity_scorer is not None:
            perplexities = self._perplexity_scorer.batch_score(seqs)

        te_scores: Optional[list[float]] = None
        if self._riboformer_scorer is not None:
            te_scores = self._riboformer_scorer.batch_score(seqs)

        domain_hits_list: Optional[list[list[DomainHit]]] = None
        if self._pfam_scanner is not None:
            from .tools.translate import translate
            proteins = [translate(s, self.genetic_code_id) for s in seqs]
            valid = [(i, p) for i, p in enumerate(proteins) if len(p) >= 10]
            if valid:
                valid_idx, valid_proteins = zip(*valid)
                hits = self._pfam_scanner.scan(list(valid_proteins))
                domain_hits_list = [[] for _ in seqs]
                for i, idx in enumerate(valid_idx):
                    domain_hits_list[idx] = hits[i]

        indices = sorted(
            range(len(candidates)), key=lambda i: raw_scores[i], reverse=True
        )
        results: list[EnrichedORF] = []
        for rank, idx in enumerate(indices, 1):
            results.append(
                EnrichedORF(
                    candidate=candidates[idx],
                    calm_score=raw_scores[idx],
                    perplexity=perplexities[idx] if perplexities else None,
                    translation_efficiency=te_scores[idx] if te_scores else None,
                    domain_hits=domain_hits_list[idx] if domain_hits_list else None,
                    rank=rank,
                )
            )
        return results
