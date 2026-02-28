"""CaLM pseudo-perplexity scorer for DNA coding sequences.

Computes pseudo-perplexity (PPPL) via masked-codon-prediction:
for each codon position, mask it, predict with CaLM's LM head,
and collect log P(true_codon | context). Lower PPPL = more natural
codon patterns (stronger coding signal).

Uses TOKEN_BUDGET adaptive batching matching CaLMScorer pattern.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch

from proteinqc.data.tokenizer import PAD_ID

# Conservative defaults for MPS/CPU; CUDA auto-scales in __init__
TOKEN_BUDGET = 8_192
BATCH_MAX = 16


class PerplexityScorer:
    """Compute pseudo-perplexity for DNA sequences using CaLM's LM head.

    Lower PPPL indicates more natural codon usage patterns (likely coding).
    Higher PPPL indicates unusual patterns (likely non-coding or spurious).

    Args:
        model_dir: Path to CaLM model directory (config.json + model.safetensors)
        device: Compute device. Auto-selects MPS/CUDA/CPU if None.
    """

    def __init__(
        self,
        model_dir: Path | str,
        device: Optional[torch.device] = None,
    ):
        from proteinqc.models.calm_encoder import CaLMEncoder

        from .calm_scorer import _select_device

        self.device = device or _select_device()
        self.model_dir = Path(model_dir)

        from proteinqc.data.tokenizer import CodonTokenizer

        self.tokenizer = CodonTokenizer(self.model_dir / "vocab.txt")
        self._mask_id = self.tokenizer.token_to_id["<mask>"]

        self.encoder = CaLMEncoder(
            self.model_dir, freeze=True, load_lm_head=True
        ).to(self.device)
        self.encoder.train(False)

        # CUDA with large VRAM: batch aggressively
        if self.device.type == "cuda":
            self._token_budget = 131_072  # 16x default
            self._batch_max = 512
        else:
            self._token_budget = TOKEN_BUDGET
            self._batch_max = BATCH_MAX

    def score_one(self, sequence: str) -> float:
        """Compute pseudo-perplexity for a single DNA sequence.

        Creates N masked copies (one per codon position), runs them through
        CaLM in adaptive batches, and collects log-likelihoods.

        Args:
            sequence: DNA sequence (T not U), should be codon-aligned.

        Returns:
            PPPL score (lower = more natural codon patterns).
        """
        token_ids = self.tokenizer.encode(sequence)

        # Codon positions are indices 1..N (excluding CLS=0 and EOS=last)
        num_codons = len(token_ids) - 2  # subtract CLS and EOS
        if num_codons <= 0:
            return float("inf")

        # Build masked variants: one per codon position
        base = torch.tensor(token_ids, dtype=torch.long)
        masked_ids = base.unsqueeze(0).expand(num_codons, -1).clone()
        true_tokens = torch.zeros(num_codons, dtype=torch.long)
        mask_positions = torch.zeros(num_codons, dtype=torch.long)

        for i in range(num_codons):
            pos = i + 1  # skip CLS at index 0
            mask_positions[i] = pos
            true_tokens[i] = masked_ids[i, pos].item()
            masked_ids[i, pos] = self._mask_id

        # Run through encoder in adaptive batches
        log_probs = self._batched_forward(masked_ids, true_tokens, mask_positions)

        # PPPL = exp(-mean(log_probs))
        avg_log_prob = log_probs.mean().item()
        return math.exp(-avg_log_prob)

    def batch_score(self, sequences: list[str]) -> list[float]:
        """Compute pseudo-perplexity for multiple DNA sequences.

        Args:
            sequences: DNA sequences (T not U), ideally codon-aligned.

        Returns:
            PPPL scores (lower = more natural), same order as input.
        """
        if not sequences:
            return []
        return [self.score_one(seq) for seq in sequences]

    def _batched_forward(
        self,
        masked_ids: torch.Tensor,
        true_tokens: torch.Tensor,
        mask_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Run masked variants through CaLM and collect log P(true | context).

        Args:
            masked_ids: [N, seq_len] — N masked copies of a single sequence
            true_tokens: [N] — the true token at each masked position
            mask_positions: [N] — the sequence index of the masked token in each row

        Returns:
            log_probs: [N] — log P(true_token) at each masked position
        """
        n = masked_ids.shape[0]
        seq_len = masked_ids.shape[1]
        log_probs = torch.zeros(n)

        # Adaptive batching based on sequence length
        adaptive_bs = max(1, self._token_budget // seq_len)
        adaptive_bs = min(adaptive_bs, self._batch_max, n)

        i = 0
        while i < n:
            end = min(i + adaptive_bs, n)
            batch_ids = masked_ids[i:end].to(self.device)
            batch_mask = batch_ids.ne(PAD_ID).long()

            with torch.no_grad():
                logits = self.encoder.forward_mlm(batch_ids, batch_mask)

            # Collect log prob at the masked position for each sample
            for j in range(end - i):
                pos = mask_positions[i + j].item()
                token_logits = logits[j, pos, :]
                token_log_probs = torch.log_softmax(token_logits, dim=-1)
                true_id = true_tokens[i + j].item()
                log_probs[i + j] = token_log_probs[true_id].cpu().item()

            if self.device.type == "mps" and seq_len > 500:
                torch.mps.empty_cache()

            i = end

        return log_probs
