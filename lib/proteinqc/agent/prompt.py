"""Prompt templates for the ORF classification agent.

Builds multi-turn prompts for tool-calling with Llama 3.1 8B Instruct.
The LLM selects which tool to call; the sequence is auto-injected.
"""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are an RNA bioinformatics expert classifying ORFs as coding or non-coding.

You have access to biological analysis tools. Use them to gather evidence, then classify.

## Tool interpretation guide:
- score_coding_potential: CaLM neural network score. >0.5 suggests coding.
  Very short ORFs (<100 codons) may score low even if coding.
- score_perplexity: Codon usage naturalness. Lower = more natural coding patterns.
  Coding genes typically show perplexity <5. Non-coding ORFs >8.
- scan_domains: Pfam protein domain hits. Known domains = strong coding evidence.
  AntiFam hits = spurious ORF signal. No hits is inconclusive.
- translate_orf: Translate to amino acid sequence. Use before domain scanning.

## Strategy:
1. Start with score_coding_potential -- it's fast and informative
2. If ambiguous (0.3-0.7), use score_perplexity for a second opinion
3. For borderline cases, translate and scan domains
4. When confident, call classify with your decision

Call ONE tool at a time. After each result, decide your next action.
Respond ONLY with a single <tool_call> block. No other text."""

USER_TEMPLATE = """\
Classify this ORF (length={n_codons} codons, {n_bp} bp):
Sequence (first 60bp): {seq_preview}...

Available tools: score_coding_potential, score_perplexity, scan_domains, translate_orf, classify

To call a tool, respond with:
<tool_call>
{{"name": "<tool_name>", "arguments": {{}}}}
</tool_call>

For classify, use:
<tool_call>
{{"name": "classify", "arguments": {{"label": "coding"|"noncoding", "confidence": 0.0-1.0}}}}
</tool_call>"""

TOOL_RESULT_TEMPLATE = """\
Tool result ({tool_name}): {result}

What tool do you want to call next? Respond with a single <tool_call> block."""


def build_initial_prompt(sequence: str) -> list[dict]:
    """Build the initial chat messages for a classification episode.

    Args:
        sequence: DNA ORF sequence.

    Returns:
        List of chat messages in Llama 3.1 Instruct format.
    """
    n_bp = len(sequence)
    n_codons = n_bp // 3
    seq_preview = sequence[:60]

    user_msg = USER_TEMPLATE.format(
        n_codons=n_codons,
        n_bp=n_bp,
        seq_preview=seq_preview,
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


def append_tool_result(
    messages: list[dict], tool_name: str, result: str
) -> list[dict]:
    """Append a tool result and prompt for next action.

    Args:
        messages: Existing conversation messages.
        tool_name: Name of the tool that was called.
        result: String result from ToolExecutor.

    Returns:
        Updated messages list (mutated in place and returned).
    """
    tool_msg = TOOL_RESULT_TEMPLATE.format(tool_name=tool_name, result=result)
    return [*messages, {"role": "user", "content": tool_msg}]


def parse_tool_call(text: str) -> dict | None:
    """Parse a <tool_call> block from LLM output.

    Returns:
        {"name": str, "arguments": dict} or None if parsing fails.
    """
    import json

    start = text.find("<tool_call>")
    end = text.find("</tool_call>")
    if start == -1 or end == -1:
        return None

    json_str = text[start + len("<tool_call>"):end].strip()
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        return None

    if "name" not in parsed:
        return None
    if "arguments" not in parsed:
        parsed["arguments"] = {}

    return parsed
