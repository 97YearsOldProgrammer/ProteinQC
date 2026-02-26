"""ProteinQC tools: ORF scanning, scoring, species detection, and domain analysis."""

from .calm_scorer import CaLMScorer
from .codon_table import (
    CodonTableManager,
    CodonUsageTable,
    GeneticCode,
    sequence_to_codon_vector,
)
from .orf_scanner import ORFCandidate, ORFScanner
from .species_detect import SpeciesDetector, SpeciesMatch
from .translate import translate, translate_all_frames

__all__ = [
    "CaLMScorer",
    "CodonTableManager",
    "CodonUsageTable",
    "DomainHit",
    "GeneticCode",
    "ORFCandidate",
    "ORFScanner",
    "PerplexityScorer",
    "PfamScanner",
    "RiboformerScorer",
    "SpeciesDetector",
    "SpeciesMatch",
    "sequence_to_codon_vector",
    "translate",
    "translate_all_frames",
]


def __getattr__(name: str):
    """Lazy-load heavy tools that have external dependencies."""
    if name == "PerplexityScorer":
        from .perplexity_scorer import PerplexityScorer
        return PerplexityScorer
    if name == "PfamScanner":
        from .pfam_scanner import PfamScanner
        return PfamScanner
    if name == "DomainHit":
        from .pfam_scanner import DomainHit
        return DomainHit
    if name == "RiboformerScorer":
        from .riboformer_scorer import RiboformerScorer
        return RiboformerScorer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
