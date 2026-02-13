"""ProteinQC: RNA coding potential prediction with CaLM.

Gated LM head architecture for binary classification (coding vs non-coding).
"""

__version__ = "0.1.0"

from lib.proteinqc.models.calm_encoder import CaLMEncoder
from lib.proteinqc.models.classification_heads import (
    GatedHead,
    LinearHead,
    MLPHead,
)

__all__ = [
    "CaLMEncoder",
    "LinearHead",
    "MLPHead",
    "GatedHead",
]
