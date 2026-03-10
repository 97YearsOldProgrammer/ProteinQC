"""Upcycle CaLM encoder: replace dense FFN with MoE FFN in top layers.

Creates a new model checkpoint where layers [moe_start..11] have their
dense FFN replaced with N-expert MoE FFN. Each expert is initialized
from the original pretrained FFN weights (upcycling), so at init the
model output is identical regardless of routing.

Usage:
    python -m proteinqc.cli.upcycle_moe
    python -m proteinqc.cli.upcycle_moe --moe-start 8 --num-experts 4 --top-k 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def upcycle(
    model_dir: Path,
    output_dir: Path,
    moe_start: int = 8,
    num_experts: int = 4,
    top_k: int = 1,
    position_type: str = "alibi",
    expert_intermediate: int | None = None,
) -> dict:
    """Create upcycled MoE model from pretrained CaLM weights.

    Layers 0..moe_start-1: dense FFN (frozen during training).
    Layers moe_start..11: MoE FFN with N experts, each initialized
    from the original FFN weights.

    Args:
        expert_intermediate: FFN intermediate size per expert. If None,
            uses the dense intermediate_size (3072). Smaller values reduce
            capacity and overfitting risk. When smaller than dense, weights
            are truncated from pretrained FFN (preserving top dimensions).

    Returns manifest dict with architecture info.
    """
    config_path = model_dir / "config.json"
    weights_path = model_dir / "model.safetensors"

    with open(config_path) as f:
        config = json.load(f)

    raw = load_file(str(weights_path))
    num_layers = config["num_hidden_layers"]
    hidden_size = config["hidden_size"]
    intermediate_size = config["intermediate_size"]
    expert_inter = expert_intermediate or intermediate_size

    print(f"CaLM config: {num_layers} layers, hidden={hidden_size}, "
          f"intermediate={intermediate_size}")
    print(f"MoE layers: {moe_start}-{num_layers - 1} "
          f"({num_layers - moe_start} layers)")
    print(f"Experts: {num_experts}, top-k: {top_k}")
    if expert_inter != intermediate_size:
        ratio = (num_experts * expert_inter) / intermediate_size
        print(f"Expert intermediate: {expert_inter} "
              f"(total capacity: {ratio:.1f}x dense)")

    # Build the upcycled state dict
    # Dense layers (0..moe_start-1): copy weights as-is
    # MoE layers (moe_start..11): replicate FFN into N experts
    moe_state: dict[str, torch.Tensor] = {}

    # Copy embeddings + final layernorm
    for key, val in raw.items():
        if key.startswith("model.embeddings."):
            new_key = key.replace("model.embeddings.", "")
            moe_state[new_key] = val
        elif key.startswith("model.encoder.emb_layer_norm_after."):
            new_key = key.replace("model.encoder.", "")
            moe_state[new_key] = val

    # Copy each layer
    for i in range(num_layers):
        src = f"model.encoder.layer.{i}"
        dst = f"layers.{i}"

        # Attention (same for both dense and MoE layers)
        moe_state[f"{dst}.attn_ln.weight"] = raw[f"{src}.attention.layer_norm.weight"]
        moe_state[f"{dst}.attn_ln.bias"] = raw[f"{src}.attention.layer_norm.bias"]
        moe_state[f"{dst}.q_proj.weight"] = raw[f"{src}.attention.self.query.weight"]
        moe_state[f"{dst}.q_proj.bias"] = raw[f"{src}.attention.self.query.bias"]
        moe_state[f"{dst}.k_proj.weight"] = raw[f"{src}.attention.self.key.weight"]
        moe_state[f"{dst}.k_proj.bias"] = raw[f"{src}.attention.self.key.bias"]
        moe_state[f"{dst}.v_proj.weight"] = raw[f"{src}.attention.self.value.weight"]
        moe_state[f"{dst}.v_proj.bias"] = raw[f"{src}.attention.self.value.bias"]
        moe_state[f"{dst}.out_proj.weight"] = raw[f"{src}.attention.output.dense.weight"]
        moe_state[f"{dst}.out_proj.bias"] = raw[f"{src}.attention.output.dense.bias"]

        # FFN LayerNorm
        moe_state[f"{dst}.ffn_ln.weight"] = raw[f"{src}.layer_norm.weight"]
        moe_state[f"{dst}.ffn_ln.bias"] = raw[f"{src}.layer_norm.bias"]

        if i < moe_start:
            # Dense FFN: copy as-is
            moe_state[f"{dst}.ffn_up.weight"] = raw[f"{src}.intermediate.dense.weight"]
            moe_state[f"{dst}.ffn_up.bias"] = raw[f"{src}.intermediate.dense.bias"]
            moe_state[f"{dst}.ffn_down.weight"] = raw[f"{src}.output.dense.weight"]
            moe_state[f"{dst}.ffn_down.bias"] = raw[f"{src}.output.dense.bias"]
        else:
            # MoE FFN: fused weight format [E, K, N]
            # CaLM intermediate.dense.weight is [3072, 768] (PyTorch convention)
            up_w = raw[f"{src}.intermediate.dense.weight"]   # [3072, 768]
            down_w = raw[f"{src}.output.dense.weight"]       # [768, 3072]

            # Truncate to expert_inter if smaller than dense intermediate
            # up_w.T: [768, 3072] → [:, :expert_inter] = [768, expert_inter]
            # down_w.T: [3072, 768] → [:expert_inter, :] = [expert_inter, 768]
            up_slice = up_w.T[:, :expert_inter].clone()     # [768, expert_inter]
            down_slice = down_w.T[:expert_inter, :].clone() # [expert_inter, 768]

            fused_up = torch.stack(
                [up_slice.clone() for _ in range(num_experts)],
            )  # [E, 768, expert_inter]
            fused_down = torch.stack(
                [down_slice.clone() for _ in range(num_experts)],
            )  # [E, expert_inter, 768]

            moe_state[f"{dst}.moe_ffn.expert_weights.up_weights"] = fused_up
            moe_state[f"{dst}.moe_ffn.expert_weights.down_weights"] = fused_down

            # Router: small random init (uniform routing at start)
            router_w = torch.empty(num_experts, hidden_size)
            torch.nn.init.kaiming_uniform_(router_w, a=1.0)
            router_w *= 0.01
            moe_state[f"{dst}.moe_ffn.router.weight"] = router_w

    # Compute stats
    total_params = sum(v.numel() for v in moe_state.values())
    dense_ffn_per_layer = hidden_size * intermediate_size * 2
    expert_ffn_per_layer = hidden_size * expert_inter * 2 * num_experts
    n_moe_layers = num_layers - moe_start

    # Trainable during training: MoE FFN + routers + ffn_ln in MoE layers + head
    trainable_moe = n_moe_layers * (
        expert_ffn_per_layer         # expert weights (no biases)
        + num_experts * hidden_size  # router
        + hidden_size * 2            # ffn_ln weight + bias
    )

    print(f"\nParam counts:")
    print(f"  Total model:     {total_params:,}")
    print(f"  Dense FFN/layer: {dense_ffn_per_layer:,}")
    print(f"  MoE FFN/layer:   {expert_ffn_per_layer:,} "
          f"({num_experts}×{expert_inter})")
    print(f"  Trainable MoE:   {trainable_moe:,}")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "model_moe.pt"
    torch.save(moe_state, out_path)

    manifest = {
        "base_model": str(model_dir),
        "position_type": position_type,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "expert_intermediate_size": expert_inter,
        "moe_start": moe_start,
        "num_experts": num_experts,
        "top_k": top_k,
        "total_params": total_params,
        "trainable_moe_params": trainable_moe,
        "dense_layers": list(range(moe_start)),
        "moe_layers": list(range(moe_start, num_layers)),
    }
    manifest_path = output_dir / "moe_config.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    sz_mb = out_path.stat().st_size / 1e6
    print(f"\nSaved: {out_path} ({sz_mb:.1f} MB)")
    print(f"Config: {manifest_path}")

    return manifest


def main() -> None:
    p = argparse.ArgumentParser(
        description="Upcycle CaLM encoder with MoE FFN layers",
    )
    p.add_argument("--model-dir", type=Path,
                   default=PROJECT_ROOT / "models" / "calm")
    p.add_argument("--output", type=Path,
                   default=PROJECT_ROOT / "models" / "calm_moe")
    p.add_argument("--moe-start", type=int, default=8,
                   help="First layer to replace with MoE (0-indexed, default: 8)")
    p.add_argument("--num-experts", type=int, default=4)
    p.add_argument("--top-k", type=int, default=1)
    p.add_argument("--expert-intermediate", type=int, default=None,
                   help="FFN intermediate size per expert (default: same as "
                        "dense=3072). Smaller = less capacity, less overfitting")
    p.add_argument("--position-type", choices=["rotary", "alibi"],
                   default="alibi")
    args = p.parse_args()

    print("=" * 60)
    print("  CaLM -> CaLM-MoE Upcycling")
    print("=" * 60)

    upcycle(
        args.model_dir, args.output,
        moe_start=args.moe_start,
        num_experts=args.num_experts,
        top_k=args.top_k,
        position_type=args.position_type,
        expert_intermediate=args.expert_intermediate,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
