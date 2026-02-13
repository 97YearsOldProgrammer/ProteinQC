"""CaLM encoder wrapper for extracting [CLS] embeddings.

Pure PyTorch + safetensors — no transformers, no multimolecule.
Loads pre-trained CaLM weights and provides forward pass for RNA sequences.
"""

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from safetensors.torch import load_file


class CaLMEncoder(nn.Module):
    """CaLM BERT encoder for RNA sequences (codon-level tokenization).

    Architecture:
        - 12 transformer layers
        - 768 hidden size
        - 12 attention heads
        - 3072 FFN intermediate size
        - RoPE positional embeddings
        - 131 codon-level vocab

    Args:
        model_dir: Path to CaLM model directory (contains config.json, model.safetensors)
        freeze: If True, freeze all encoder parameters (default: True)
    """

    def __init__(
        self,
        model_dir: Path | str,
        freeze: bool = True,
    ):
        super().__init__()
        self.model_dir = Path(model_dir)
        self.config = self._load_config()
        self.hidden_size = self.config["hidden_size"]

        # Load pre-trained weights
        weights = load_file(str(self.model_dir / "model.safetensors"))

        # Build minimal BERT-style encoder
        # For now, we'll store weights and implement forward pass
        # Full transformer implementation will be added when needed
        self.weights = weights

        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def _load_config(self) -> dict:
        """Load CaLM config.json."""
        config_path = self.model_dir / "config.json"
        with open(config_path) as f:
            return json.load(f)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through CaLM encoder.

        Args:
            input_ids: Tokenized codon sequences [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len] (optional)

        Returns:
            cls_embedding: [CLS] token embeddings [batch, hidden_size]

        Note:
            Full transformer forward pass will be implemented.
            For now, this is a placeholder that returns dummy embeddings.
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # TODO: Implement full BERT forward pass
        # For now, return dummy embeddings for testing infrastructure
        cls_embedding = torch.randn(
            batch_size,
            self.hidden_size,
            device=device,
            dtype=torch.float32,
        )

        return cls_embedding

    @property
    def device(self) -> torch.device:
        """Get device of model parameters."""
        # Return device of first parameter
        return next(self.parameters()).device
