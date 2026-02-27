"""CLI entry point for the ProteinQC smolagents CodeAgent.

Usage:
    # Single-shot query
    python -m proteinqc.scripts.run_agent --query "Is ATGCGA... coding?"

    # Interactive REPL
    python -m proteinqc.scripts.run_agent

    # With specific backend
    python -m proteinqc.scripts.run_agent --backend transformers --model meta-llama/Llama-3.1-8B-Instruct
"""

from __future__ import annotations

import argparse
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_agent",
        description="ProteinQC sequence analysis agent powered by smolagents",
    )
    p.add_argument(
        "--backend",
        choices=["mlx", "transformers", "api", "litellm"],
        default="mlx",
        help="LLM backend (default: mlx for Apple Silicon)",
    )
    p.add_argument(
        "--model",
        default="mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
        help="Model ID for the chosen backend",
    )
    p.add_argument(
        "--model-dir",
        default="models/calm",
        help="Path to CaLM model directory",
    )
    p.add_argument(
        "--head-path",
        default="models/heads/mlp_head_v1.pt",
        help="Path to MLP head weights",
    )
    p.add_argument(
        "--pfam-db",
        default="models/pfam/Pfam-A.hmm",
        help="Path to Pfam-A HMM database",
    )
    p.add_argument(
        "--query",
        default=None,
        help="Single-shot query (omit for interactive REPL)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Show generated code before execution",
    )
    return p.parse_args()


def main():
    args = parse_args()

    from proteinqc.smolagent import create_agent

    agent = create_agent(
        backend=args.backend,
        model_id=args.model,
        model_dir=args.model_dir,
        head_path=args.head_path,
        pfam_db=args.pfam_db,
        verbose=args.verbose,
    )

    if args.query:
        result = agent.run(args.query)
        print(result)
        return

    # Interactive REPL
    print("ProteinQC Agent (type 'quit' or Ctrl+C to exit)")
    print("-" * 50)
    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not query or query.lower() in ("quit", "exit", "q"):
            break
        result = agent.run(query)
        print(result)


if __name__ == "__main__":
    main()
