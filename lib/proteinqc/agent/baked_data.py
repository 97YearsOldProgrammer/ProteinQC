"""Pre-baked evidence data structures and loaders.

Evidence (CaLM score, perplexity, translation, Pfam domains) is computed
offline via `bake-evidence` and stored as JSONL. This module loads it
into typed dataclasses for the v2 GRPO trainer.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class BakedEvidence:
    """Pre-computed evidence for a single RNA sequence.

    All tool results are baked offline so the LLM sees them in one prompt
    (no multi-turn tool calling).

    Attributes:
        sequence_id: Unique identifier (e.g. RNA Challenge ID).
        sequence: Raw DNA sequence (T not U).
        label: Ground truth — "coding" or "noncoding".
        calm_score: CaLM coding probability in [0, 1], or None if skipped.
        perplexity: CaLM pseudo-perplexity, or None if skipped.
        translation: Amino acid sequence from translate(), or None.
        pfam_domains: List of domain hit strings, or empty list.
    """

    sequence_id: str
    sequence: str
    label: str
    species: str = "unknown"
    calm_score: float | None = None
    perplexity: float | None = None
    translation: str | None = None
    pfam_domains: tuple[str, ...] = field(default_factory=tuple)

    @property
    def n_bp(self) -> int:
        return len(self.sequence)

    @property
    def n_codons(self) -> int:
        return len(self.sequence) // 3

    @property
    def domain_summary(self) -> str:
        """One-line summary for prompt injection."""
        if not self.pfam_domains:
            return "no domain hits"
        return "; ".join(self.pfam_domains)

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "sequence_id": self.sequence_id,
            "sequence": self.sequence,
            "label": self.label,
            "species": self.species,
            "calm_score": self.calm_score,
            "perplexity": self.perplexity,
            "translation": self.translation,
            "pfam_domains": list(self.pfam_domains),
        }


def load_baked_evidence(path: Path | str) -> list[BakedEvidence]:
    """Load pre-baked evidence from a JSONL file.

    Each line is a JSON object matching BakedEvidence fields.

    Args:
        path: Path to evidence_baked.jsonl.

    Returns:
        List of BakedEvidence in file order.

    Raises:
        FileNotFoundError: If the JSONL file doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Baked evidence not found: {path}")

    items: list[BakedEvidence] = []
    with open(path) as fh:
        for line_num, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_num} of {path}: {exc}"
                ) from exc
            items.append(
                BakedEvidence(
                    sequence_id=obj["sequence_id"],
                    sequence=obj["sequence"],
                    label=obj["label"],
                    species=obj.get("species", "unknown"),
                    calm_score=obj.get("calm_score"),
                    perplexity=obj.get("perplexity"),
                    translation=obj.get("translation"),
                    pfam_domains=tuple(obj.get("pfam_domains", [])),
                )
            )
    return items


def split_baked(
    evidence: list[BakedEvidence],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[BakedEvidence], list[BakedEvidence]]:
    """Stratified train/test split on baked evidence.

    Splits coding and noncoding separately to preserve label balance.

    Args:
        evidence: Full list of baked evidence.
        train_ratio: Fraction for training (rest is test).
        seed: Random seed for reproducibility.

    Returns:
        (train, test) lists of BakedEvidence.
    """
    coding = [e for e in evidence if e.label == "coding"]
    noncoding = [e for e in evidence if e.label == "noncoding"]

    rng = random.Random(seed)
    rng.shuffle(coding)
    rng.shuffle(noncoding)

    def _split(items: list[BakedEvidence]) -> tuple[list[BakedEvidence], list[BakedEvidence]]:
        n = int(len(items) * train_ratio)
        return items[:n], items[n:]

    train_c, test_c = _split(coding)
    train_nc, test_nc = _split(noncoding)

    train = train_c + train_nc
    test = test_c + test_nc
    rng.shuffle(train)
    rng.shuffle(test)

    return train, test
