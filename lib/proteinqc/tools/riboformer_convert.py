"""
Convert Riboformer TF SavedModel weights → PyTorch .pt

One-time dev tool. Requires: pip install tensorflow-cpu
End users receive pre-converted .pt files and never need TF.

Usage:
    python -m proteinqc.tools.riboformer_convert \
        --model_dir path/to/bacteria_cm_mg \
        --out      models/riboformer/bacteria_cm_mg.pt

    # or download + convert all bundled species:
    python -m proteinqc.tools.riboformer_convert --download-all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

BUNDLED_MODELS = [
    "bacteria_cm_mg",
    "covid_model",
    "worm_aging",
    "yeast_aging",
    "yeast_disome",
]

GH_BASE = (
    "https://github.com/lingxusb/Riboformer/raw/main/models"
)


def _require_tf():
    try:
        import tensorflow as tf  # noqa: F401
        return tf
    except ImportError:
        sys.exit(
            "TensorFlow required for weight conversion.\n"
            "Install once: pip install tensorflow-cpu\n"
            "After converting all bundled models, TF is no longer needed."
        )


def _download_saved_model(model_name: str, dest: Path) -> Path:
    """Download a TF SavedModel directory from GitHub into dest/model_name/."""
    import urllib.request

    model_dir = dest / model_name
    subdirs = ["", "/variables"]
    files = {
        "": ["saved_model.pb", "keras_metadata.pb"],
        "/variables": ["variables.index", "variables.data-00000-of-00001"],
    }

    for sub, fnames in files.items():
        (model_dir / sub.lstrip("/")).mkdir(parents=True, exist_ok=True)
        for fname in fnames:
            url = f"{GH_BASE}/{model_name}{sub}/{fname}"
            dst = model_dir / sub.lstrip("/") / fname
            if dst.exists():
                print(f"  skip (exists): {dst}")
                continue
            print(f"  downloading: {url}")
            urllib.request.urlretrieve(url, dst)

    return model_dir


def _get_weights(tf_model) -> dict[str, np.ndarray]:
    """Extract all weights from a loaded Keras model as {name: ndarray}."""
    weights = {}
    for layer in tf_model.layers:
        for var in layer.weights:
            weights[var.name] = var.numpy()
    return weights


def _find(weights: dict, *fragments: str) -> np.ndarray:
    """Find a weight tensor whose name contains all fragments."""
    for name, w in weights.items():
        if all(f in name for f in fragments):
            return w
    available = "\n  ".join(weights.keys())
    raise KeyError(f"No weight matching {fragments!r} in:\n  {available}")


def convert(model_dir: str | Path, out_path: str | Path) -> None:
    """Load TF SavedModel and write a PyTorch state_dict .pt file."""
    tf = _require_tf()
    from proteinqc.tools.riboformer import RiboformerConfig, RiboformerPyTorch

    print(f"Loading TF model from {model_dir} ...")
    tf_model = tf.keras.models.load_model(str(model_dir))
    w = _get_weights(tf_model)

    print("Weight keys found:")
    for k in sorted(w):
        print(f"  {k}  {w[k].shape}")

    cfg   = RiboformerConfig()
    model = RiboformerPyTorch(cfg)
    sd    = model.state_dict()

    def _set(pt_key: str, arr: np.ndarray):
        t = torch.from_numpy(arr.copy())
        if sd[pt_key].shape != t.shape:
            raise ValueError(
                f"{pt_key}: expected {sd[pt_key].shape}, got {t.shape}"
            )
        sd[pt_key] = t

    # ---- Token + Position Embedding ----
    _set("embedding.token_emb.weight",
         _find(w, "token_and_position_embedding", "token_emb", "embeddings:0"))
    _set("embedding.pos_emb.weight",
         _find(w, "token_and_position_embedding", "pos_emb", "embeddings:0"))

    # ---- Conv2D Tower (Branch 1) ----
    # TF Conv2D kernel: (kH, kW, C_in, C_out) → PyTorch: (C_out, C_in, kH, kW)
    for i in range(cfg.conv2d_layers):
        k = _find(w, f"conv_tower", "conv2d", f"_{i}" if i > 0 else "/conv2d/", "kernel:0")
        _set(f"conv2d_tower.{i}.conv.weight", np.transpose(k, (3, 2, 0, 1)))
        bn_g = _find(w, f"conv_tower", f"_{i}" if i > 0 else "/batch", "gamma:0")
        bn_b = _find(w, f"conv_tower", f"_{i}" if i > 0 else "/batch", "beta:0")
        bn_m = _find(w, f"conv_tower", f"_{i}" if i > 0 else "/batch", "moving_mean:0")
        bn_v = _find(w, f"conv_tower", f"_{i}" if i > 0 else "/batch", "moving_variance:0")
        _set(f"conv2d_tower.{i}.bn.weight",       bn_g)
        _set(f"conv2d_tower.{i}.bn.bias",         bn_b)
        _set(f"conv2d_tower.{i}.bn.running_mean", bn_m)
        _set(f"conv2d_tower.{i}.bn.running_var",  bn_v)

    # ---- Conv1D Tower (Branch 2) ----
    # TF Conv1D kernel: (L, C_in, C_out) → PyTorch: (C_out, C_in, L)
    for i in range(len(cfg.conv1d_filters)):
        k = _find(w, "conv_tower_1", f"_{i}" if i > 0 else "/conv1d/", "kernel:0")
        _set(f"conv1d_tower.{i}.conv.weight", np.transpose(k, (2, 1, 0)))
        bn_g = _find(w, "conv_tower_1", f"_{i}" if i > 0 else "/batch", "gamma:0")
        bn_b = _find(w, "conv_tower_1", f"_{i}" if i > 0 else "/batch", "beta:0")
        bn_m = _find(w, "conv_tower_1", f"_{i}" if i > 0 else "/batch", "moving_mean:0")
        bn_v = _find(w, "conv_tower_1", f"_{i}" if i > 0 else "/batch", "moving_variance:0")
        _set(f"conv1d_tower.{i}.bn.weight",       bn_g)
        _set(f"conv1d_tower.{i}.bn.bias",         bn_b)
        _set(f"conv1d_tower.{i}.bn.running_mean", bn_m)
        _set(f"conv1d_tower.{i}.bn.running_var",  bn_v)

    # ---- Transformer Block 1 (sequence branch) ----
    # TF MHA: separate query/key/value/output_0 kernels (in_dim, num_heads*key_dim)
    # Our _MHA: q_proj, k_proj, v_proj, o_proj — same layout, direct copy + transpose
    def _load_mha(tf_prefix: str, pt_prefix: str):
        for proj in ("query", "key", "value"):
            k = _find(w, tf_prefix, proj, "kernel:0")          # (in, heads*key_dim)
            _set(f"{pt_prefix}.{proj[0]}_proj.weight", k.T)    # PyTorch Linear: (out, in)
        ok = _find(w, tf_prefix, "output_0", "kernel:0")       # (heads*key_dim, in)
        _set(f"{pt_prefix}.o_proj.weight", ok.T)

    def _load_transformer(tf_prefix: str, pt_prefix: str):
        _load_mha(tf_prefix, f"{pt_prefix}.mha")
        ff1_k = _find(w, tf_prefix, "dense", "kernel:0")
        ff1_b = _find(w, tf_prefix, "dense", "bias:0")
        ff2_k = _find(w, tf_prefix, "dense_1", "kernel:0")
        ff2_b = _find(w, tf_prefix, "dense_1", "bias:0")
        _set(f"{pt_prefix}.mlp.0.weight", ff1_k.T)
        _set(f"{pt_prefix}.mlp.0.bias",   ff1_b)
        _set(f"{pt_prefix}.mlp.2.weight", ff2_k.T)
        _set(f"{pt_prefix}.mlp.2.bias",   ff2_b)
        ln_g = _find(w, tf_prefix, "layer_normalization", "gamma:0")
        ln_b = _find(w, tf_prefix, "layer_normalization", "beta:0")
        _set(f"{pt_prefix}.layernorm.weight", ln_g)
        _set(f"{pt_prefix}.layernorm.bias",   ln_b)

    _load_transformer("transformer_block",   "seq_transformer")
    _load_transformer("transformer_block_1", "exp_transformer")

    # ---- Heads ----
    def _load_dense(tf_name: str, pt_prefix: str):
        k = _find(w, tf_name, "kernel:0")
        b = _find(w, tf_name, "bias:0")
        _set(f"{pt_prefix}.weight", k.T)
        _set(f"{pt_prefix}.bias",   b)

    _load_dense("head1/dense",  "seq_head.1")   # Linear after Flatten+Dropout
    _load_dense("head2/dense",  "exp_head.1")
    _load_dense("read_depth",   "final_dense.0")

    # ---- Save ----
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, out_path)
    print(f"Saved PyTorch weights: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Convert Riboformer TF → PyTorch")
    ap.add_argument("--model_dir", help="Path to TF SavedModel directory")
    ap.add_argument("--out",       help="Output .pt path")
    ap.add_argument("--download-all", action="store_true",
                    help="Download all bundled models from GitHub and convert")
    ap.add_argument("--tmp-dir", default="/tmp/riboformer_models",
                    help="Temp dir for downloaded models")
    args = ap.parse_args()

    if args.download_all:
        tmp = Path(args.tmp_dir)
        out_dir = Path("models/riboformer")
        for name in BUNDLED_MODELS:
            print(f"\n=== {name} ===")
            model_dir = _download_saved_model(name, tmp)
            convert(model_dir, out_dir / f"{name}.pt")
        print("\nAll models converted.")
    elif args.model_dir and args.out:
        convert(args.model_dir, args.out)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
