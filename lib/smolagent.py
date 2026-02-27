"""smolagents CodeAgent factory for ProteinQC.

Provides create_agent() which wires up all ProteinQC tools with a
CodeAgent backed by MLX (Apple Silicon), Transformers (CUDA), or
an inference API (HuggingFace / OpenAI / Anthropic via LiteLLM).
"""

from __future__ import annotations

from smolagents import CodeAgent

from proteinqc.smolagents_tools import (
    CaLMScorerTool,
    PerplexityScorerTool,
    PfamScannerTool,
    gc_content,
    kozak_score,
    scan_orfs,
    translate_dna,
)

SYSTEM_PROMPT_SUFFIX = """\

You are a bioinformatics assistant specializing in RNA and protein analysis.
You have access to tools for analyzing DNA/RNA sequences:

- translate_dna: Convert DNA to protein sequence
- scan_orfs: Find open reading frames in a transcript
- gc_content: Measure GC content of a sequence
- kozak_score: Evaluate translation initiation context
- calm_score: Score coding potential with CaLM neural network
- calm_perplexity: Measure codon usage naturalness
- pfam_scan: Search for known protein domains

When analyzing a sequence, consider:
1. First scan for ORFs to find candidate coding regions
2. Score promising ORFs with calm_score for coding probability
3. Translate high-scoring ORFs and check for Pfam domains
4. Use GC content and Kozak context as supporting evidence

Always show your reasoning and intermediate results.
"""


def _build_model(backend: str, model_id: str):
    """Instantiate the LLM backend."""
    if backend == "mlx":
        from smolagents import MLXModel

        return MLXModel(model_id)

    if backend == "transformers":
        from smolagents import TransformersModel

        return TransformersModel(model_id=model_id)

    if backend == "api":
        from smolagents import InferenceClientModel

        return InferenceClientModel(model_id=model_id)

    if backend == "litellm":
        from smolagents import LiteLLMModel

        return LiteLLMModel(model_id=model_id)

    raise ValueError(
        f"Unknown backend '{backend}'. Choose from: mlx, transformers, api, litellm"
    )


def create_agent(
    backend: str = "mlx",
    model_id: str = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
    model_dir: str = "models/calm",
    head_path: str = "models/heads/mlp_head_v1.pt",
    pfam_db: str = "models/pfam/Pfam-A.hmm",
    verbose: bool = False,
) -> CodeAgent:
    """Create a smolagents CodeAgent with all ProteinQC tools loaded.

    Args:
        backend: LLM backend - "mlx" (Apple Silicon local), "transformers"
                 (CUDA local), "api" (HF Inference API), or "litellm".
        model_id: Model identifier for the chosen backend.
        model_dir: Path to CaLM model directory.
        head_path: Path to trained MLP head weights.
        pfam_db: Path to Pfam-A HMM database.
        verbose: If True, print generated code before execution.

    Returns:
        Configured CodeAgent ready for .run() calls.
    """
    model = _build_model(backend, model_id)

    tools = [
        translate_dna,
        scan_orfs,
        gc_content,
        kozak_score,
        CaLMScorerTool(model_dir=model_dir, head_path=head_path),
        PerplexityScorerTool(model_dir=model_dir),
        PfamScannerTool(pfam_db_path=pfam_db),
    ]

    return CodeAgent(
        tools=tools,
        model=model,
        additional_authorized_imports=["re", "math"],
        verbosity_level=2 if verbose else 1,
    )
