"""Tool schemas and executor for the ORF classification agent.

TOOL_SCHEMAS defines the JSON function-calling interface the LLM sees.
ToolExecutor dispatches tool names to real Python function calls.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

TOOL_SCHEMAS: list[dict] = [
    {
        "name": "score_coding_potential",
        "description": (
            "Score DNA ORF for coding probability using CaLM classifier. "
            "Returns float in [0,1]. Higher = more likely coding."
        ),
        "parameters": {
            "sequence": {
                "type": "string",
                "description": "DNA sequence (ATG...stop)",
            }
        },
    },
    {
        "name": "score_perplexity",
        "description": (
            "Compute codon pattern naturalness (pseudo-perplexity). "
            "Lower = more natural coding patterns."
        ),
        "parameters": {
            "sequence": {
                "type": "string",
                "description": "DNA sequence (ATG...stop)",
            }
        },
    },
    {
        "name": "scan_domains",
        "description": (
            "Scan translated protein for known Pfam domains and AntiFam "
            "spurious signals. Returns domain hits with E-values."
        ),
        "parameters": {
            "sequence": {
                "type": "string",
                "description": "DNA sequence (ATG...stop) — will be translated internally",
            }
        },
    },
    {
        "name": "score_translation_efficiency",
        "description": (
            "Predict translation efficiency using Riboformer. "
            "Higher = more efficiently translated."
        ),
        "parameters": {
            "sequence": {
                "type": "string",
                "description": "DNA sequence (ATG...stop)",
            }
        },
    },
    {
        "name": "translate_orf",
        "description": (
            "Translate DNA ORF to amino acid sequence. "
            "Useful before domain scanning."
        ),
        "parameters": {
            "sequence": {
                "type": "string",
                "description": "DNA sequence (ATG...stop)",
            },
            "genetic_code_id": {
                "type": "integer",
                "default": 1,
                "description": "NCBI genetic code ID (1=standard)",
            },
        },
    },
    {
        "name": "classify",
        "description": (
            "Submit final classification: coding or non-coding. "
            "Terminates the episode."
        ),
        "parameters": {
            "label": {
                "type": "string",
                "enum": ["coding", "noncoding"],
                "description": "Final classification",
            },
            "confidence": {
                "type": "number",
                "description": "Confidence in [0, 1]",
            },
        },
    },
]

# Tool names that gather evidence (everything except classify)
EVIDENCE_TOOLS = frozenset(
    s["name"] for s in TOOL_SCHEMAS if s["name"] != "classify"
)


class ToolExecutor:
    """Dispatch tool names to real Python function calls.

    Lazy-loads heavy components (CaLM, Riboformer, Pfam) on first use.

    Args:
        calm_model_dir: Path to CaLM model directory.
        calm_head_path: Path to MLP head weights.
        riboformer_weights: Path to Riboformer .pt weights (optional).
        pfam_db_path: Path to Pfam-A.hmm (optional).
        antifam_db_path: Path to AntiFam.hmm (optional).
    """

    def __init__(
        self,
        calm_model_dir: Optional[Path | str] = None,
        calm_head_path: Optional[Path | str] = None,
        riboformer_weights: Optional[Path | str] = None,
        pfam_db_path: Optional[Path | str] = None,
        antifam_db_path: Optional[Path | str] = None,
    ):
        self._calm_model_dir = Path(calm_model_dir) if calm_model_dir else None
        self._calm_head_path = Path(calm_head_path) if calm_head_path else None
        self._riboformer_weights = Path(riboformer_weights) if riboformer_weights else None
        self._pfam_db_path = Path(pfam_db_path) if pfam_db_path else None
        self._antifam_db_path = Path(antifam_db_path) if antifam_db_path else None

        self._calm_scorer = None
        self._perplexity_scorer = None
        self._riboformer_scorer = None
        self._pfam_scanner = None

    _KNOWN_TOOLS = frozenset(s["name"] for s in TOOL_SCHEMAS)

    def execute(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool and return stringified result.

        All tool errors are caught and returned as "error: ..." strings
        so the RL agent sees consistent string outputs (never exceptions).

        Args:
            tool_name: One of the names in TOOL_SCHEMAS.
            arguments: Parsed arguments matching the schema.

        Returns:
            String representation of the tool output, or error string.

        Raises:
            ValueError: If tool_name is unknown (not a valid tool).
        """
        if tool_name not in self._KNOWN_TOOLS:
            raise ValueError(f"Unknown tool: {tool_name}")
        try:
            return self._dispatch(tool_name, arguments)
        except Exception as exc:
            return f"error: {type(exc).__name__}: {exc}"

    def _dispatch(self, tool_name: str, arguments: dict) -> str:
        """Route tool_name to the appropriate executor method."""
        if tool_name == "score_coding_potential":
            return self._exec_calm(arguments["sequence"])
        if tool_name == "score_perplexity":
            return self._exec_perplexity(arguments["sequence"])
        if tool_name == "scan_domains":
            return self._exec_pfam(arguments["sequence"])
        if tool_name == "score_translation_efficiency":
            return self._exec_riboformer(arguments["sequence"])
        if tool_name == "translate_orf":
            return self._exec_translate(
                arguments["sequence"],
                arguments.get("genetic_code_id", 1),
            )
        # classify — always succeeds
        return f"classification={arguments['label']} confidence={arguments.get('confidence', 'N/A')}"

    def _exec_calm(self, sequence: str) -> str:
        if self._calm_scorer is None:
            from proteinqc.tools.calm_scorer import CaLMScorer
            if self._calm_model_dir is None or self._calm_head_path is None:
                return "error: CaLM model paths not configured"
            self._calm_scorer = CaLMScorer(self._calm_model_dir, self._calm_head_path)
        scores = self._calm_scorer.batch_score([sequence])
        return f"{scores[0]:.6f}"

    def _exec_perplexity(self, sequence: str) -> str:
        if self._perplexity_scorer is None:
            from proteinqc.tools.perplexity_scorer import PerplexityScorer
            if self._calm_model_dir is None:
                return "error: CaLM model path not configured"
            self._perplexity_scorer = PerplexityScorer(self._calm_model_dir)
        scores = self._perplexity_scorer.batch_score([sequence])
        return f"{scores[0]:.4f}"

    def _exec_riboformer(self, sequence: str) -> str:
        if self._riboformer_scorer is None:
            from proteinqc.tools.riboformer_scorer import RiboformerScorer
            if self._riboformer_weights is None:
                return "error: Riboformer weights not configured"
            self._riboformer_scorer = RiboformerScorer(self._riboformer_weights)
        scores = self._riboformer_scorer.batch_score([sequence])
        return f"{scores[0]:.6f}"

    def _exec_pfam(self, sequence: str) -> str:
        from proteinqc.tools.translate import translate

        protein = translate(sequence)
        if len(protein) < 10:
            return "protein too short for domain scan (<10 aa)"

        if self._pfam_scanner is None:
            from proteinqc.tools.pfam_scanner import PfamScanner
            if self._pfam_db_path is None:
                return "error: Pfam database path not configured"
            antifam = self._antifam_db_path if (
                self._antifam_db_path and self._antifam_db_path.exists()
            ) else None
            self._pfam_scanner = PfamScanner(self._pfam_db_path, antifam)

        hits_list = self._pfam_scanner.scan([protein])
        hits = hits_list[0] if hits_list else []
        if not hits:
            return "no domain hits"
        parts = [
            f"{h.domain_name}(E={h.e_value:.1e},antifam={h.is_antifam})"
            for h in hits
        ]
        return "; ".join(parts)

    def _exec_translate(self, sequence: str, genetic_code_id: int) -> str:
        from proteinqc.tools.translate import translate
        protein = translate(sequence, genetic_code_id)
        return protein if protein else "(empty translation)"
