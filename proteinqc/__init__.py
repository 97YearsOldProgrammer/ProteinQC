"""ProteinQC: RNA coding potential prediction with CaLM.

Pure PyTorch implementation — no HuggingFace transformers dependency.
Gated LM head architecture for binary classification (coding vs non-coding).
"""

__version__ = "0.3.0"

from proteinqc.data.tokenizer import CodonTokenizer
from proteinqc.models.calm_encoder import CaLMEncoder
from proteinqc.models.classification_heads import (
    GatedHead,
    LinearHead,
    MLPHead,
)
from proteinqc.models.norm_convert import convert_layernorm_to_rmsnorm
from proteinqc.pipeline import EnrichedORF, ORFPipeline, ScoredORF

__all__ = [
    "CaLMEncoder",
    "CodonTokenizer",
    "EnrichedORF",
    "LinearHead",
    "MLPHead",
    "GatedHead",
    "convert_layernorm_to_rmsnorm",
    "ORFPipeline",
    "ScoredORF",
]
