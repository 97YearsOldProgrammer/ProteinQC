"""Simplified episode for single-generation GRPO (v2).

Eliminates multi-turn raw_responses and the "\n".join token concat bug.
Each episode stores a single completion_text with pre-computed tokens
so log-prob computation matches generation exactly.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EpisodeV2:
    """One classification episode: evidence prompt -> single LLM generation.

    Unlike v1 Episode, there is exactly ONE generation per episode.
    Prompt tokens and completion tokens are stored directly to avoid
    re-tokenization drift in the log-prob phase.

    Attributes:
        sequence_id: Unique sequence identifier.
        label: Ground truth — "coding" or "noncoding".
        prompt_tokens: Token IDs for the full chat prompt (cached).
        completion_text: Raw LLM output text.
        completion_tokens: Token IDs for the completion (cached).
        reasoning: Extracted <reasoning> text, or empty string.
        prediction: Extracted <classification> — "coding" or "noncoding".
        confidence: Extracted <confidence> float in [0, 1].
        reward: 1.0 if prediction == label, else 0.0.
    """

    sequence_id: str
    label: str
    prompt_tokens: tuple[int, ...]
    completion_text: str
    completion_tokens: tuple[int, ...]
    reasoning: str
    prediction: str
    confidence: float
    reward: float

    @property
    def is_correct(self) -> bool:
        return self.prediction == self.label

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict (omits token arrays)."""
        return {
            "sequence_id": self.sequence_id,
            "label": self.label,
            "completion_text": self.completion_text,
            "reasoning": self.reasoning,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "reward": self.reward,
        }
