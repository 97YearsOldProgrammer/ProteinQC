"""Mixture-of-Experts FFN layer for CaLM encoder upcycling.

Replaces the dense FFN (768->3072->768) with N parallel expert FFNs
and a learned router. Standard top-k gating with load-balancing loss.

Expert weights stored in fused format [E, K, N] for efficient dispatch:
  - Triton grouped GEMM (GB10 tuned) when available
  - Batched matmul fallback on CUDA
  - Loop fallback on CPU/MPS

Upcycling: all experts initialized from the same pretrained FFN weights,
so at init the model produces identical outputs regardless of routing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusedExpertWeights(nn.Module):
    """Expert weights in fused [E, K, N] format for grouped GEMM dispatch."""

    def __init__(self, num_experts: int, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.up_weights = nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size),
        )
        self.down_weights = nn.Parameter(
            torch.empty(num_experts, intermediate_size, hidden_size),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.up_weights, a=1.0)
        nn.init.kaiming_uniform_(self.down_weights, a=1.0)


class MoEFFN(nn.Module):
    """Mixture-of-Experts FFN with top-k routing and Triton dispatch.

    Dispatch priority: triton > bmm > loop.
    CaLM FFN: Linear(hidden->intermediate) -> GELU -> Linear(intermediate->hidden).

    Args:
        hidden_size: Model hidden dimension (768 for CaLM).
        intermediate_size: FFN intermediate dimension (3072 for CaLM).
        num_experts: Number of expert FFNs.
        top_k: Number of experts activated per token.
        balance_coeff: Weight for load-balancing auxiliary loss.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_experts: int = 4,
        top_k: int = 1,
        balance_coeff: float = 0.01,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.balance_coeff = balance_coeff
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        nn.init.kaiming_uniform_(self.router.weight, a=1.0)

        self.expert_weights = FusedExpertWeights(
            num_experts, hidden_size, intermediate_size,
        )

        self._last_balance_loss = torch.tensor(0.0)
        self._last_load: torch.Tensor | None = None

        self._triton_available = False
        try:
            from proteinqc.models.triton_moe import (
                triton_grouped_gemm,
                triton_grouped_gemm_scatter,
            )
            self._triton_gemm = triton_grouped_gemm
            self._triton_scatter = triton_grouped_gemm_scatter
            self._triton_available = True
        except ImportError:
            pass

    def _prepare_expert_batches(self, top_k_indices, top_k_probs, num_tokens, device):
        """Sort tokens by expert assignment for grouped dispatch."""
        flat_indices = top_k_indices.reshape(-1)
        flat_probs = top_k_probs.reshape(-1)
        flat_tokens = (
            torch.arange(num_tokens, device=device)
            .unsqueeze(1).expand(-1, self.top_k).reshape(-1)
        )

        sorted_idx = torch.argsort(flat_indices, stable=True)
        sorted_experts = flat_indices[sorted_idx]
        sorted_tokens = sorted_idx_tokens = flat_tokens[sorted_idx]
        sorted_weights = flat_probs[sorted_idx]

        ones = torch.ones(sorted_experts.shape[0], dtype=torch.long, device=device)
        expert_counts = torch.zeros(self.num_experts, dtype=torch.long, device=device)
        expert_counts.scatter_add_(0, sorted_experts.long(), ones)
        expert_offsets = torch.zeros(
            self.num_experts + 1, dtype=torch.long, device=device,
        )
        expert_offsets[1:] = expert_counts.cumsum(0)

        return sorted_tokens, sorted_weights, expert_offsets, expert_counts

    def _forward_triton(self, x_flat, top_k_indices, top_k_probs):
        """Triton grouped GEMM: 2 kernel launches for up + down."""
        num_tokens = x_flat.shape[0]
        device = x_flat.device

        sorted_tokens, sorted_weights, expert_offsets, _ = \
            self._prepare_expert_batches(
                top_k_indices, top_k_probs, num_tokens, device,
            )

        if sorted_tokens.shape[0] == 0:
            return torch.zeros_like(x_flat)

        sorted_input = x_flat[sorted_tokens]
        offs = expert_offsets.to(torch.int32)

        up_w = self.expert_weights.up_weights
        down_w = self.expert_weights.down_weights

        hidden = self._triton_gemm(sorted_input, up_w, offs)
        hidden = F.gelu(hidden)
        output = self._triton_scatter(
            hidden, down_w, offs, sorted_weights, sorted_tokens, num_tokens,
        )
        return output

    @torch.compiler.disable
    def _forward_bmm(self, x_flat, top_k_indices, top_k_probs):
        """Batched matmul: 2 bmm calls for up + down."""
        num_tokens = x_flat.shape[0]
        device = x_flat.device

        sorted_tokens, sorted_weights, expert_offsets, expert_counts = \
            self._prepare_expert_batches(
                top_k_indices, top_k_probs, num_tokens, device,
            )

        output = torch.zeros_like(x_flat)
        if sorted_tokens.shape[0] == 0:
            return output

        max_count = expert_counts.max().item()
        positions = torch.arange(max_count, device=device)
        mask = positions.unsqueeze(0) < expert_counts.unsqueeze(1)

        flat_idx = expert_offsets[:-1].unsqueeze(1) + positions.unsqueeze(0)
        flat_idx = flat_idx.clamp(max=sorted_tokens.shape[0] - 1)

        token_ids = sorted_tokens[flat_idx.reshape(-1)].reshape(
            self.num_experts, max_count,
        )
        padded_input = (
            x_flat[token_ids] * mask.unsqueeze(-1).to(x_flat.dtype)
        )

        up_out = torch.bmm(padded_input, self.expert_weights.up_weights)
        hidden = F.gelu(up_out)
        expert_out = torch.bmm(hidden, self.expert_weights.down_weights)

        padded_w = sorted_weights[flat_idx.reshape(-1)].reshape(
            self.num_experts, max_count, 1,
        )
        weighted = (
            expert_out * padded_w * mask.unsqueeze(-1).to(expert_out.dtype)
        ).to(output.dtype)

        flat_out = weighted.reshape(-1, self.hidden_size)
        flat_ids = token_ids.reshape(-1)
        flat_mask = mask.reshape(-1)

        output.scatter_add_(
            0,
            flat_ids[flat_mask].unsqueeze(-1).expand(-1, self.hidden_size),
            flat_out[flat_mask],
        )
        return output

    def _forward_loop(self, x_flat, top_k_indices, top_k_probs):
        """Loop fallback for CPU/MPS."""
        num_tokens = x_flat.shape[0]
        device = x_flat.device

        sorted_tokens, sorted_weights, expert_offsets, _ = \
            self._prepare_expert_batches(
                top_k_indices, top_k_probs, num_tokens, device,
            )

        output = torch.zeros_like(x_flat)
        up_w = self.expert_weights.up_weights
        down_w = self.expert_weights.down_weights

        for e in range(self.num_experts):
            start = expert_offsets[e].item()
            end = expert_offsets[e + 1].item()
            if start == end:
                continue

            token_ids = sorted_tokens[start:end]
            weights = sorted_weights[start:end].unsqueeze(-1)

            expert_input = x_flat[token_ids]
            hidden = F.gelu(expert_input @ up_w[e])
            out = hidden @ down_w[e]

            output.index_add_(0, token_ids, (out * weights).to(output.dtype))

        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, H = x.shape
        num_tokens = B * S
        x_flat = x.view(num_tokens, H)

        router_logits = self.router(x_flat)
        router_probs = torch.softmax(router_logits, dim=-1)

        top_weights, top_indices = torch.topk(
            router_probs, self.top_k, dim=-1,
        )
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)

        if self.training:
            self._compute_balance_loss(router_probs, top_indices)

        if self._triton_available and x_flat.is_cuda:
            output = self._forward_triton(x_flat, top_indices, top_weights)
        elif x_flat.is_cuda:
            output = self._forward_bmm(x_flat, top_indices, top_weights)
        else:
            output = self._forward_loop(x_flat, top_indices, top_weights)

        return output.view(B, S, H)

    def _compute_balance_loss(
        self,
        router_probs: torch.Tensor,
        top_indices: torch.Tensor,
    ) -> None:
        """Switch Transformer load-balancing loss (all top-k assignments)."""
        N = self.num_experts
        flat_experts = top_indices.reshape(-1)
        one_hot = F.one_hot(flat_experts, num_classes=N).float()
        f = one_hot.mean(dim=0)
        P = router_probs.reshape(-1, N).mean(dim=0)

        self._last_balance_loss = self.balance_coeff * N * (f * P).sum()
        self._last_load = f.detach()

    def get_balance_loss(self) -> torch.Tensor:
        return self._last_balance_loss

    def get_load_stats(self) -> dict[str, float]:
        if self._last_load is None:
            return {}
        load = self._last_load
        return {
            f"expert_{i}_frac": load[i].item()
            for i in range(self.num_experts)
        }


class MoETransformerLayer(nn.Module):
    """Transformer layer with MoE FFN replacing dense FFN.

    Attention is frozen (weights from pretrained CaLM). Only the MoE FFN
    and its router are trainable. Supports RoPE and ALiBi position encoding.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        layer_norm_eps: float,
        attn_dropout: float,
        hidden_dropout: float,
        num_experts: int = 4,
        top_k: int = 1,
        balance_coeff: float = 0.01,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.attn_ln = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout_p = attn_dropout
        self.hidden_dropout1 = nn.Dropout(hidden_dropout)

        self.ffn_ln = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.moe_ffn = MoEFFN(
            hidden_size, intermediate_size,
            num_experts=num_experts, top_k=top_k,
            balance_coeff=balance_coeff,
        )
        self.hidden_dropout2 = nn.Dropout(hidden_dropout)

    def _reshape_for_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        return x.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask,
        cos=None,
        sin=None,
        alibi_bias=None,
    ) -> torch.Tensor:
        from proteinqc.models.calm_encoder import _apply_rope

        try:
            from flash_attn import flash_attn_func
            _has_fa = True
        except ImportError:
            _has_fa = False

        # --- Pre-LN Attention (frozen path) ---
        h = self.attn_ln(x)
        q = self._reshape_for_heads(self.q_proj(h))
        k = self._reshape_for_heads(self.k_proj(h))
        v = self._reshape_for_heads(self.v_proj(h))

        if cos is not None and sin is not None:
            q, k = _apply_rope(q, k, cos, sin)

        if _has_fa and q.is_cuda:
            from proteinqc.models.calm_encoder import _alibi_slopes, _FA_ALIBI_SLOPES
            fa_dtype = torch.bfloat16
            q_fa = q.transpose(1, 2).to(fa_dtype)
            k_fa = k.transpose(1, 2).to(fa_dtype)
            v_fa = v.transpose(1, 2).to(fa_dtype)
            fa_kwargs: dict = {"causal": False}
            if alibi_bias is not None:
                nh = self.num_heads
                if nh not in _FA_ALIBI_SLOPES:
                    _FA_ALIBI_SLOPES[nh] = _alibi_slopes(nh)
                fa_kwargs["alibi_slopes"] = _FA_ALIBI_SLOPES[nh].to(
                    device=q.device, dtype=torch.float32,
                )
            attn_out = flash_attn_func(q_fa, k_fa, v_fa, **fa_kwargs)
            attn_out = attn_out.to(x.dtype).reshape(x.shape)
        else:
            if alibi_bias is not None:
                pad_bias = torch.zeros_like(attn_mask, dtype=q.dtype)
                pad_bias = pad_bias.masked_fill(~attn_mask, float("-inf"))
                sdpa_mask = pad_bias + alibi_bias[:, :, :q.shape[2], :q.shape[2]]
            else:
                sdpa_mask = attn_mask
            attn_out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=sdpa_mask,
                dropout_p=self.attn_dropout_p if self.training else 0.0,
            )
            attn_out = attn_out.transpose(1, 2).contiguous().view(x.shape)

        attn_out = self.hidden_dropout1(self.out_proj(attn_out))
        x = x + attn_out

        # --- Pre-LN MoE FFN (trainable path) ---
        h = self.ffn_ln(x)
        h = self.moe_ffn(h)
        h = self.hidden_dropout2(h)
        x = x + h

        return x
