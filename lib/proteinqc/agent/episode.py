"""Episode data structures for the ORF classification RL agent.

An episode represents one classification attempt: the agent receives an ORF
sequence, calls biological tools to gather evidence, then makes a coding vs
non-coding decision. Ground truth comes from GENCODE annotations.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolCall:
    """A single tool invocation within an episode."""

    name: str        # tool name from TOOL_SCHEMAS
    arguments: dict  # parsed arguments (shallow-frozen via dataclass)
    result: str      # stringified tool output


@dataclass(frozen=True)
class Episode:
    """One ORF classification episode: sequence -> tool calls -> decision.

    Immutable: build the full tool_calls tuple first, then construct once.
    Use Episode(...) or dataclasses.replace() to create modified copies.

    Attributes:
        transcript_id: GENCODE transcript ID (e.g. "ENST00000456328.2").
        orf_sequence: DNA sequence of the ORF (ATG...stop).
        ground_truth: "coding" or "noncoding" from GENCODE annotation.
        tool_calls: Ordered tuple of tool invocations the agent made.
        prediction: Agent's final classification, or None if episode incomplete.
        confidence: Agent's stated confidence in [0, 1], or None.
        reward: 1.0 if correct, 0.0 if wrong.
        raw_responses: Raw LLM response texts for log-prob computation.
    """

    transcript_id: str
    orf_sequence: str
    ground_truth: str
    tool_calls: tuple[ToolCall, ...] = ()
    prediction: str | None = None
    confidence: float | None = None
    reward: float = 0.0
    raw_responses: tuple[str, ...] = ()

    @property
    def is_correct(self) -> bool:
        return self.prediction == self.ground_truth

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "transcript_id": self.transcript_id,
            "orf_sequence": self.orf_sequence,
            "ground_truth": self.ground_truth,
            "tool_calls": [
                {"name": tc.name, "arguments": tc.arguments, "result": tc.result}
                for tc in self.tool_calls
            ],
            "prediction": self.prediction,
            "confidence": self.confidence,
            "reward": self.reward,
        }
