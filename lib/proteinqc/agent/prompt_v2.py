"""Constrained single-turn prompt for pre-baked evidence classification (v2).

The LLM receives ALL evidence upfront and produces ONE structured response.
No multi-turn tool calling — just reasoning + classification + confidence.
"""

from __future__ import annotations

import re

from .baked_data import BakedEvidence

SYSTEM_PROMPT_V2 = """\
You are an RNA bioinformatics expert classifying sequences as coding or non-coding.

You will receive pre-computed biological evidence for a sequence. Analyze ALL evidence, \
then respond EXACTLY in this format (no other text):

<reasoning>
Your analysis of the evidence (2-4 sentences). Mention which signals are most informative.
</reasoning>
<classification>coding</classification>
<confidence>0.85</confidence>

## Evidence interpretation guide:
- CaLM score: Coding probability from a codon-level BERT encoder (85M params, 131 codon vocab, \
trained on multi-species CDS). Range [0,1]. >0.5 suggests coding, <0.3 suggests noncoding.
- Perplexity: Pseudo-perplexity from masked codon prediction. Lower = more natural codon usage. \
<5 typical for real CDS, >8 for non-coding.
- Protein preview: First 50 amino acids from longest-ORF translation.
- Protein length: Full translated protein length in amino acids.
- Pfam domains: Known protein domain hits from Pfam-A HMM scan. Any hit = strong coding evidence.

Rules:
- confidence must be a number between 0.0 and 1.0
- classification must be exactly "coding" or "noncoding"
- Do NOT call any tools. All evidence is pre-computed."""

USER_TEMPLATE_V2 = """\
Classify this RNA sequence:

| Field | Value |
|-------|-------|
| Species | {species} |
| Length | {n_bp} bp ({n_codons} codons) |
| CaLM score | {calm_score} |
| Perplexity | {perplexity} |
| Protein (first 50aa) | {protein_preview} |
| Protein length | {protein_length} |
| Pfam domains | {pfam_domains} |"""

# Regex patterns for parsing structured output
_RE_REASONING = re.compile(
    r"<reasoning>\s*(.*?)\s*</reasoning>", re.DOTALL
)
_RE_CLASSIFICATION = re.compile(
    r"<classification>\s*(coding|noncoding)\s*</classification>"
)
_RE_CONFIDENCE = re.compile(
    r"<confidence>\s*([\d.]+)\s*</confidence>"
)


def build_evidence_prompt(evidence: BakedEvidence) -> list[dict]:
    """Build single-turn chat messages from pre-baked evidence.

    Args:
        evidence: Pre-computed evidence for one sequence.

    Returns:
        [system, user] message list for Llama 3.1 Instruct.
    """
    calm_str = f"{evidence.calm_score:.4f}" if evidence.calm_score is not None else "N/A"
    ppl_str = f"{evidence.perplexity:.2f}" if evidence.perplexity is not None else "N/A"

    if evidence.translation:
        preview = evidence.translation[:50]
        if len(evidence.translation) > 50:
            preview += "..."
        protein_str = preview
        protein_len = f"{len(evidence.translation)} aa"
    else:
        protein_str = "(no translation)"
        protein_len = "0 aa"

    user_msg = USER_TEMPLATE_V2.format(
        species=evidence.species,
        n_bp=evidence.n_bp,
        n_codons=evidence.n_codons,
        calm_score=calm_str,
        perplexity=ppl_str,
        protein_preview=protein_str,
        protein_length=protein_len,
        pfam_domains=evidence.domain_summary,
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT_V2},
        {"role": "user", "content": user_msg},
    ]


def parse_structured_output(text: str) -> tuple[str, str, float]:
    """Parse <reasoning>, <classification>, <confidence> from LLM output.

    Args:
        text: Raw LLM generation text.

    Returns:
        (reasoning, prediction, confidence) tuple.
        Falls back to ("", "noncoding", 0.1) for unparseable output.
    """
    reasoning_match = _RE_REASONING.search(text)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

    class_match = _RE_CLASSIFICATION.search(text)
    prediction = class_match.group(1) if class_match else "noncoding"

    conf_match = _RE_CONFIDENCE.search(text)
    if conf_match:
        try:
            confidence = float(conf_match.group(1))
            confidence = max(0.0, min(1.0, confidence))
        except ValueError:
            confidence = 0.1
    else:
        confidence = 0.1

    return reasoning, prediction, confidence
