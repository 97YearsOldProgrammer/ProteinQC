"""smolagents-compatible tool wrappers for ProteinQC.

Lightweight tools use the @tool decorator (no state, fast).
Heavy tools subclass Tool for lazy model loading on first call.
"""

from __future__ import annotations

from smolagents import Tool, tool


# ---------------------------------------------------------------------------
# Lightweight tools (@tool decorator)
# ---------------------------------------------------------------------------


@tool
def translate_dna(sequence: str) -> str:
    """Translate a DNA sequence to an amino acid (protein) sequence.

    Uses the standard genetic code. Translation stops at the first
    in-frame stop codon. Incomplete trailing codons are ignored.

    Args:
        sequence: DNA sequence (A/T/G/C characters). U is accepted and
                  converted to T automatically.
    """
    from proteinqc.tools.translate import translate

    return translate(sequence)


@tool
def scan_orfs(sequence: str, min_codons: int = 30) -> str:
    """Scan a DNA transcript for open reading frames (ORFs) in all 3 forward frames.

    Finds ATG-initiated ORFs terminated by stop codons. Returns a formatted
    summary of candidates sorted by length (longest first). Each ORF entry
    shows frame, start/stop positions, and length in codons.

    Args:
        sequence: DNA transcript sequence (A/T/G/C).
        min_codons: Minimum ORF length in codons (including stop). Default 30.
    """
    from proteinqc.tools.codon_table import CodonTableManager
    from proteinqc.tools.orf_scanner import ORFScanner

    manager = CodonTableManager()
    code = manager.get_genetic_code(1)
    scanner = ORFScanner(code, min_codons=min_codons)
    candidates = scanner.scan(sequence)

    if not candidates:
        return "No ORFs found meeting the minimum length threshold."

    lines = [f"Found {len(candidates)} ORF(s):"]
    for i, orf in enumerate(candidates, 1):
        lines.append(
            f"  {i}. frame={orf.frame} start={orf.start} "
            f"stop={orf.stop} codons={orf.length_codons}"
        )
    return "\n".join(lines)


@tool
def gc_content(sequence: str) -> float:
    """Compute the GC content (fraction of G+C bases) of a DNA sequence.

    Returns a value between 0.0 and 1.0. Useful for distinguishing
    coding vs non-coding regions (coding tends toward ~0.4-0.6).

    Args:
        sequence: DNA sequence (A/T/G/C characters).
    """
    seq = sequence.upper().replace("U", "T")
    if not seq:
        return 0.0
    gc = sum(1 for c in seq if c in ("G", "C"))
    return gc / len(seq)


@tool
def kozak_score(sequence: str, atg_pos: int) -> float:
    """Score the Kozak consensus context around an ATG start codon.

    The Kozak sequence influences translation initiation efficiency.
    A strong Kozak context has R (A or G) at position -3 and G at +4
    relative to the A of ATG. Returns a score from 0.0 (weak) to 1.0
    (strong consensus).

    Args:
        sequence: Full DNA transcript sequence containing the ATG.
        atg_pos: 0-based position of the 'A' in ATG within the sequence.
    """
    seq = sequence.upper().replace("U", "T")
    score = 0.0
    total = 0.0

    # Position -3: R (A or G) is strongest signal
    pos_m3 = atg_pos - 3
    if 0 <= pos_m3 < len(seq):
        total += 0.5
        if seq[pos_m3] in ("A", "G"):
            score += 0.5

    # Position +4 (relative to A of ATG, so index atg_pos + 3): G preferred
    pos_p4 = atg_pos + 3
    if 0 <= pos_p4 < len(seq):
        total += 0.3
        if seq[pos_p4] == "G":
            score += 0.3

    # Position -1/-2: CC preferred
    for offset, weight in [(-1, 0.1), (-2, 0.1)]:
        pos = atg_pos + offset
        if 0 <= pos < len(seq):
            total += weight
            if seq[pos] == "C":
                score += weight

    return score / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Heavy tools (Tool subclass with lazy loading)
# ---------------------------------------------------------------------------


class CaLMScorerTool(Tool):
    """Score a DNA sequence for coding potential using CaLM encoder + MLP head.

    Returns a probability between 0 (non-coding) and 1 (coding).
    The CaLM model (~86M params) is loaded lazily on first call.
    """

    name = "calm_score"
    description = (
        "Score a DNA sequence for coding potential (protein-coding vs non-coding). "
        "Returns a float probability between 0.0 (non-coding) and 1.0 (coding). "
        "Uses the CaLM (Codon-Aware Language Model) encoder with a trained MLP head."
    )
    inputs = {
        "sequence": {
            "type": "string",
            "description": "DNA sequence (A/T/G/C characters) to score.",
        }
    }
    output_type = "number"

    def __init__(self, model_dir: str = "models/calm", head_path: str = "models/heads/mlp_head_v1.pt"):
        super().__init__()
        self._model_dir = model_dir
        self._head_path = head_path
        self._scorer = None

    def forward(self, sequence: str) -> float:
        if self._scorer is None:
            from proteinqc.tools.calm_scorer import CaLMScorer

            self._scorer = CaLMScorer(self._model_dir, self._head_path)
        scores = self._scorer.batch_score([sequence])
        return scores[0]


class PerplexityScorerTool(Tool):
    """Compute pseudo-perplexity of a DNA sequence using CaLM's LM head.

    Lower perplexity means more natural codon usage patterns (likely coding).
    The CaLM model is loaded lazily on first call.
    """

    name = "calm_perplexity"
    description = (
        "Compute pseudo-perplexity (PPPL) of a DNA sequence using CaLM's masked "
        "language model head. Lower values indicate more natural codon usage patterns "
        "(stronger coding signal). Higher values suggest non-coding or spurious ORFs."
    )
    inputs = {
        "sequence": {
            "type": "string",
            "description": "DNA sequence (A/T/G/C characters) to score.",
        }
    }
    output_type = "number"

    def __init__(self, model_dir: str = "models/calm"):
        super().__init__()
        self._model_dir = model_dir
        self._scorer = None

    def forward(self, sequence: str) -> float:
        if self._scorer is None:
            from proteinqc.tools.perplexity_scorer import PerplexityScorer

            self._scorer = PerplexityScorer(self._model_dir)
        return self._scorer.score_one(sequence)


class PfamScannerTool(Tool):
    """Scan a protein sequence against the Pfam-A HMM database for known domains.

    Returns a text summary of domain hits. Requires HMMER3 (hmmscan) installed.
    The Pfam database path must be valid and hmmpress'd.
    """

    name = "pfam_scan"
    description = (
        "Scan a protein (amino acid) sequence against the Pfam-A HMM database to "
        "identify known protein domains. Returns domain names, accessions, and "
        "E-values. Finding Pfam domains is strong evidence of a real protein."
    )
    inputs = {
        "protein_sequence": {
            "type": "string",
            "description": "Amino acid sequence (single-letter codes, e.g. MVLSPADKTN...).",
        }
    }
    output_type = "string"

    def __init__(self, pfam_db_path: str = "models/pfam/Pfam-A.hmm"):
        super().__init__()
        self._pfam_db_path = pfam_db_path
        self._scanner = None

    def forward(self, protein_sequence: str) -> str:
        if self._scanner is None:
            from proteinqc.tools.pfam_scanner import PfamScanner

            self._scanner = PfamScanner(self._pfam_db_path)
        results = self._scanner.scan([protein_sequence])
        hits = results[0]
        if not hits:
            return "No Pfam domains found."
        lines = [f"Found {len(hits)} domain(s):"]
        for h in hits:
            lines.append(
                f"  {h.domain_name} ({h.domain_id}) "
                f"E={h.e_value:.1e} score={h.score:.1f} "
                f"pos={h.ali_from}-{h.ali_to}"
            )
        return "\n".join(lines)
