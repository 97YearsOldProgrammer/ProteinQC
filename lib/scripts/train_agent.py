"""GRPO agent training on RNA Challenge dataset.

Entry point that wires together MLX Llama 3.1 8B, biological tools,
GRPO trainer, and GeneT5-style logging. Handles all prerequisites
(model downloads, dependency checks) before training starts.

Usage:
    train-agent --dry-run 5          # Quick smoke test (5 sequences)
    train-agent                       # Full training (3 epochs)
    train-agent --epochs 1 --lr 1e-5  # Custom hyperparameters
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _check_prerequisites(args: argparse.Namespace) -> bool:
    """Verify all prerequisites before training. Returns True if OK."""
    ok = True

    # 1. Check MLX dependencies
    print("Checking prerequisites...", flush=True)
    try:
        import mlx.core  # noqa: F401
        import mlx_lm  # noqa: F401
        print("  [OK] mlx + mlx-lm importable", flush=True)
    except ImportError:
        print(
            '  [FAIL] mlx or mlx-lm not installed. Run: pip install -e ".[agent]"',
            file=sys.stderr,
        )
        ok = False

    # 2. Llama model (auto-downloads on first load, just warn about size)
    print(f"  [INFO] Model: {args.model} (auto-downloads ~4.7 GB on first run)", flush=True)

    # 3. CaLM model
    calm_dir = Path(args.calm_dir)
    calm_weights = calm_dir / "model.safetensors"
    if calm_weights.exists():
        print(f"  [OK] CaLM model: {calm_dir}", flush=True)
    else:
        print(f"  [FAIL] CaLM model not found: {calm_weights}", file=sys.stderr)
        print("    Run: download-calm", file=sys.stderr)
        ok = False

    # 4. MLP head
    head_path = Path(args.head_path)
    if head_path.exists():
        print(f"  [OK] MLP head: {head_path}", flush=True)
    else:
        print(f"  [FAIL] MLP head not found: {head_path}", file=sys.stderr)
        print("    Run: save-mlp-head", file=sys.stderr)
        ok = False

    # 5. RNA Challenge data
    data_path = Path(args.data)
    if data_path.exists():
        print(f"  [OK] RNA Challenge data: {data_path}", flush=True)
    else:
        print(f"  [FAIL] Data not found: {data_path}", file=sys.stderr)
        ok = False

    # 6. Pfam DB (optional)
    pfam_path = Path(args.pfam_db) if args.pfam_db else None
    if pfam_path and pfam_path.exists():
        print(f"  [OK] Pfam DB: {pfam_path}", flush=True)
    elif pfam_path:
        print(f"  [WARN] Pfam DB not found: {pfam_path} — scan_domains disabled", flush=True)
    else:
        print("  [INFO] Pfam DB not configured — scan_domains disabled", flush=True)

    return ok


def _save_config(args: argparse.Namespace, output_dir: Path, train_size: int, test_size: int) -> None:
    """Save full hyperparameter config as JSON for reproducibility."""
    config = {
        "model": args.model,
        "lora_rank": args.lora_rank,
        "lora_layers": args.lora_layers,
        "dataset": str(args.data),
        "train_size": train_size,
        "test_size": test_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "beta": args.beta,
        "group_size": args.group_size,
        "grad_accum": args.grad_accum,
        "max_tools_per_episode": args.max_tools,
        "warmup_ratio": args.warmup_ratio,
        "seed": args.seed,
        "tools_enabled": ["score_coding_potential", "score_perplexity", "translate_orf"]
        + (["scan_domains"] if args.pfam_db and Path(args.pfam_db).exists() else []),
        "riboformer_enabled": False,
        "baseline_acc": 0.9562,
        "baseline_f1": 0.9630,
        "baseline_mcc": 0.9092,
        "dry_run": args.dry_run,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    config_path = output_dir / "grpo_config.json"
    with open(config_path, "w") as fh:
        json.dump(config, fh, indent=2)
    print(f"  Config saved: {config_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GRPO agent training on RNA Challenge"
    )

    # Data
    parser.add_argument("--data", default="data/rnachallenge/rnachallenge.tsv",
                        help="Path to RNA Challenge TSV")
    parser.add_argument("--output-dir", default="data/ProteinQC/RL",
                        help="Base output directory")

    # Model
    parser.add_argument("--model", default="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
                        help="HuggingFace model ID for mlx-lm")
    parser.add_argument("--lora-rank", type=int, default=16,
                        help="LoRA adapter rank")
    parser.add_argument("--lora-layers", type=int, default=8,
                        help="Number of transformer layers for LoRA")

    # Tool backends
    parser.add_argument("--calm-dir", default="models/calm",
                        help="CaLM model directory")
    parser.add_argument("--head-path", default="models/heads/mlp_head_v1.pt",
                        help="MLP head weights path")
    parser.add_argument("--pfam-db", default=None,
                        help="Pfam-A.hmm path (optional)")

    # Training
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--group-size", type=int, default=4,
                        help="Episodes per sequence for GRPO")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--max-tools", type=int, default=3,
                        help="Max tool calls per episode")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--beta", type=float, default=0.05,
                        help="KL penalty coefficient")
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)

    # Logging
    parser.add_argument("--log-every", type=int, default=10,
                        help="Log training metrics every N steps")
    parser.add_argument("--save-every", type=int, default=1000,
                        help="Save checkpoint every N steps")
    parser.add_argument("--assess-every", type=int, default=500,
                        help="Run test assessment every N steps")

    # v2: Pre-baked evidence mode
    parser.add_argument("--baked-evidence", default=None,
                        help="Path to evidence_baked.jsonl for v2 trainer")
    parser.add_argument("--max-tokens", type=int, default=300,
                        help="Max tokens per LLM generation (v2 only)")

    # Debug
    parser.add_argument("--dry-run", type=int, default=None,
                        help="Run only N sequences for debugging")

    args = parser.parse_args()

    # Prerequisite checks
    if not _check_prerequisites(args):
        print("\nPrerequisite check failed. Fix the issues above and retry.",
              file=sys.stderr)
        sys.exit(1)

    # Branch: v2 (pre-baked evidence) or v1 (multi-turn tool calling)
    if args.baked_evidence:
        _run_v2(args)
    else:
        _run_v1(args)


def _run_v2(args: argparse.Namespace) -> None:
    """V2 training path: pre-baked evidence, single generation, fixed grad accum."""
    from proteinqc.agent.baked_data import load_baked_evidence, split_baked

    baked_path = Path(args.baked_evidence)
    if not baked_path.exists():
        print(f"Baked evidence not found: {baked_path}", file=sys.stderr)
        sys.exit(1)

    print(f"\nLoading baked evidence: {baked_path}", flush=True)
    evidence = load_baked_evidence(baked_path)
    train_data, test_data = split_baked(evidence, seed=args.seed)

    if args.dry_run:
        train_data = train_data[:args.dry_run]
        test_data = test_data[:min(args.dry_run, len(test_data))]
        print(f"  Dry run: {len(train_data)} train, {len(test_data)} test", flush=True)
    else:
        print(f"  Loaded: {len(train_data)} train, {len(test_data)} test", flush=True)

    # Create timestamped output directory
    ts = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"run_v2_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    _save_config(args, output_dir, len(train_data), len(test_data))

    # Initialize MLX backend
    from proteinqc.agent.mlx_backend import MLXBackend

    print(f"\nInitializing MLX backend...", flush=True)
    backend = MLXBackend(
        model_name=args.model,
        lora_rank=args.lora_rank,
        lora_layers=args.lora_layers,
    )

    # Initialize v2 trainer
    from proteinqc.agent.train_grpo_v2 import GRPOTrainerV2

    trainer = GRPOTrainerV2(
        backend=backend,
        train_data=train_data,
        test_data=test_data,
        output_dir=output_dir,
        group_size=args.group_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        beta=args.beta,
        warmup_ratio=args.warmup_ratio,
        max_tokens=args.max_tokens,
        seed=args.seed,
        log_every=args.log_every,
        save_every=args.save_every,
        assess_every=args.assess_every,
    )

    # Run training
    print(f"\nStarting GRPO v2 training...", flush=True)
    start = time.time()
    results = trainer.train(epochs=args.epochs)
    elapsed = time.time() - start

    print(f"\n{'='*60}", flush=True)
    print(f"  GRPO v2 training complete in {elapsed / 3600:.1f} hours", flush=True)
    print(f"  Total steps: {results['total_steps']}", flush=True)
    print(f"  Best accuracy: {results['best_accuracy']:.4f}", flush=True)
    print(f"  Output: {output_dir}", flush=True)
    print(f"  Baseline: ACC=95.62% F1=96.30% MCC=90.92%", flush=True)
    print(f"{'='*60}", flush=True)


def _run_v1(args: argparse.Namespace) -> None:
    """V1 training path: multi-turn tool calling (original)."""
    # Create timestamped output directory
    ts = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"run_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    from proteinqc.agent.rnachallenge_loader import load_rnachallenge

    print(f"\nLoading data: {args.data}", flush=True)
    train_data, test_data = load_rnachallenge(args.data, seed=args.seed)

    if args.dry_run:
        train_data = train_data[:args.dry_run]
        test_data = test_data[:min(args.dry_run, len(test_data))]
        print(f"  Dry run: {len(train_data)} train, {len(test_data)} test", flush=True)
    else:
        print(f"  Loaded: {len(train_data)} train, {len(test_data)} test", flush=True)

    # Save config
    _save_config(args, output_dir, len(train_data), len(test_data))

    # Initialize tool executor
    from proteinqc.agent.tool_schema import ToolExecutor

    pfam_path = args.pfam_db if args.pfam_db and Path(args.pfam_db).exists() else None
    tool_executor = ToolExecutor(
        calm_model_dir=args.calm_dir,
        calm_head_path=args.head_path,
        pfam_db_path=pfam_path,
    )

    # Initialize MLX backend
    from proteinqc.agent.mlx_backend import MLXBackend

    print(f"\nInitializing MLX backend...", flush=True)
    backend = MLXBackend(
        model_name=args.model,
        lora_rank=args.lora_rank,
        lora_layers=args.lora_layers,
    )

    # Initialize trainer
    from proteinqc.agent.train_grpo import GRPOTrainer

    trainer = GRPOTrainer(
        backend=backend,
        tool_executor=tool_executor,
        train_data=train_data,
        test_data=test_data,
        output_dir=output_dir,
        group_size=args.group_size,
        grad_accum=args.grad_accum,
        max_tools_per_episode=args.max_tools,
        lr=args.lr,
        beta=args.beta,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
        log_every=args.log_every,
        save_every=args.save_every,
        assess_every=args.assess_every,
    )

    # Run training
    print(f"\nStarting GRPO training...", flush=True)
    start = time.time()
    results = trainer.train(epochs=args.epochs)
    elapsed = time.time() - start

    # Summary
    print(f"\n{'='*60}", flush=True)
    print(f"  Training complete in {elapsed / 3600:.1f} hours", flush=True)
    print(f"  Total steps: {results['total_steps']}", flush=True)
    print(f"  Best accuracy: {results['best_accuracy']:.4f}", flush=True)
    print(f"  Output: {output_dir}", flush=True)
    print(f"  Baseline: ACC=95.62% F1=96.30% MCC=90.92%", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
