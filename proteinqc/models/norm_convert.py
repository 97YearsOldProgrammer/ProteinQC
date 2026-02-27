"""LayerNorm → RMSNorm conversion for Pre-LN transformers.

Based on Jiang & Gu (NeurIPS 2023): for Pre-LN architectures with
approximately zero-mean residual streams, LayerNorm reduces to RMSNorm.

Conversion absorbs the LayerNorm bias into the next linear layer,
then replaces LayerNorm with RMSNorm. One-time operation, no retraining.
~10% inference speedup from skipping mean computation.
"""

import torch
import torch.nn as nn


def _replace_ln_with_rmsnorm(
    ln: nn.LayerNorm,
    next_linears: list[nn.Linear],
) -> nn.RMSNorm:
    """Replace a LayerNorm with RMSNorm, absorbing bias into next layers.

    For Pre-LN: output = W @ (LN(x)) + b = W @ (gamma * norm(x) + beta) + b
    When norm(x) ≈ rms_norm(x) (zero-mean residual stream):
        new_bias = old_bias + W @ beta
        RMSNorm uses same gamma (weight), no bias.
    """
    gamma = ln.weight.data.clone()
    beta = ln.bias.data.clone()

    # Absorb beta into each following linear layer's bias
    for linear in next_linears:
        linear.bias.data += linear.weight.data @ beta

    # Create RMSNorm with same gamma
    rmsnorm = nn.RMSNorm(gamma.shape[0], eps=ln.eps, device=gamma.device)
    rmsnorm.weight.data.copy_(gamma)
    return rmsnorm


def convert_layernorm_to_rmsnorm(model: nn.Module) -> nn.Module:
    """Convert all Pre-LN LayerNorms to RMSNorm in a CaLM encoder.

    Handles three types of LayerNorm in the CaLM architecture:
    1. attn_ln  → bias absorbed into q_proj, k_proj, v_proj
    2. ffn_ln   → bias absorbed into ffn_up
    3. emb_layer_norm_after → bias stored as additive buffer

    Args:
        model: CaLMEncoder instance (modified in-place)

    Returns:
        The same model with RMSNorm replacing LayerNorm
    """
    for layer in model.layers:
        # Attention Pre-LN: absorb bias into Q, K, V projections
        layer.attn_ln = _replace_ln_with_rmsnorm(
            layer.attn_ln, [layer.q_proj, layer.k_proj, layer.v_proj],
        )

        # FFN Pre-LN: absorb bias into ffn_up
        layer.ffn_ln = _replace_ln_with_rmsnorm(
            layer.ffn_ln, [layer.ffn_up],
        )

    # Final LayerNorm: no following linear in encoder, store bias as buffer
    final_gamma = model.emb_layer_norm_after.weight.data.clone()
    final_beta = model.emb_layer_norm_after.bias.data.clone()
    final_eps = model.emb_layer_norm_after.eps

    model.emb_layer_norm_after = nn.RMSNorm(
        final_gamma.shape[0], eps=final_eps, device=final_gamma.device,
    )
    model.emb_layer_norm_after.weight.data.copy_(final_gamma)
    model.register_buffer("_final_ln_bias", final_beta)

    # Patch forward to add final bias after RMSNorm
    original_forward = model.forward

    def patched_forward(input_ids, attention_mask=None):
        cls_emb = original_forward(input_ids, attention_mask)
        return cls_emb + model._final_ln_bias.to(cls_emb.device)

    model.forward = patched_forward
    return model
