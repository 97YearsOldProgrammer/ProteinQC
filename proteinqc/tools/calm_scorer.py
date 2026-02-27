"""CaLM-based ORF scorer using frozen encoder + trained MLP head.

Wraps CaLMEncoder (frozen) + MLPHead (pre-trained) for scoring ORF candidates.
Uses TOKEN_BUDGET=8192 adaptive batching (same pattern as benchmark.py).
Model loaded once on construction, reused across all calls.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

TOKEN_BUDGET = 8_192  # max total tokens per batch (MPS memory constraint)
BATCH_MAX = 16        # absolute max batch size


class CaLMScorer:
    """Score DNA sequences for coding potential using frozen CaLM + MLP head.

    Args:
        model_dir: Path to CaLM model directory (config.json + model.safetensors)
        head_weights_path: Path to saved MLPHead state dict (.pt file)
        device: Compute device. Auto-selects MPS/CUDA/CPU if None.
    """

    def __init__(
        self,
        model_dir: Path | str,
        head_weights_path: Path | str,
        device: Optional[torch.device] = None,
    ):
        from proteinqc.data.tokenizer import CodonTokenizer
        from proteinqc.models.calm_encoder import CaLMEncoder
        from proteinqc.models.classification_heads import MLPHead

        self.device = device or _select_device()
        self.model_dir = Path(model_dir)
        self.head_weights_path = Path(head_weights_path)

        self.tokenizer = CodonTokenizer(self.model_dir / "vocab.txt")

        self.encoder = CaLMEncoder(self.model_dir, freeze=True).to(self.device)
        self.encoder.train(False)  # inference mode

        self.head = MLPHead(hidden_size=768, mlp_hidden=256, dropout=0.0)
        state = torch.load(
            self.head_weights_path, map_location=self.device, weights_only=True
        )
        self.head.load_state_dict(state)
        self.head = self.head.to(self.device)
        self.head.train(False)  # inference mode

    def batch_score(self, sequences: list[str]) -> list[float]:
        """Score DNA sequences for coding potential.

        Uses adaptive token-budget batching: sequences sorted by length,
        grouped so total tokens per batch stays under TOKEN_BUDGET.
        Input order is preserved in output.

        Args:
            sequences: DNA sequences (T not U), ideally codon-aligned (len % 3 == 0).

        Returns:
            Coding probabilities in [0, 1], same order as input.
        """
        if not sequences:
            return []

        n = len(sequences)
        scores = [0.0] * n
        sorted_indices = sorted(range(n), key=lambda i: len(sequences[i]))

        i = 0
        while i < n:
            max_codons = len(sequences[sorted_indices[i]]) // 3 + 2
            adaptive_bs = max(1, TOKEN_BUDGET // max_codons)
            adaptive_bs = min(adaptive_bs, BATCH_MAX, n - i)

            batch_idx = sorted_indices[i : i + adaptive_bs]
            batch_seqs = [sequences[j] for j in batch_idx]

            encoded = self.tokenizer.batch_encode(batch_seqs, device=self.device)

            with torch.no_grad():
                cls_emb = self.encoder(
                    encoded["input_ids"], encoded["attention_mask"]
                )
                logits = self.head(cls_emb).squeeze(-1)
                probs = torch.sigmoid(logits).cpu().tolist()

            if isinstance(probs, float):
                probs = [probs]

            for j, orig_idx in enumerate(batch_idx):
                scores[orig_idx] = probs[j]

            if self.device.type == "mps" and max_codons > 500:
                torch.mps.empty_cache()

            i += adaptive_bs

        return scores


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
