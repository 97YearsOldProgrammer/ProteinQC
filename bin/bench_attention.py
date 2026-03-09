#!/usr/bin/env python3
"""Benchmark attention backends on current GPU.

Tests flash_attn, SDPA (cuDNN), and FlexAttention with ALiBi
using CaLM-shaped tensors (12 heads, 64 head_dim, BF16).
"""

import time
import torch
import torch.nn.functional as F

device = "cuda"
num_heads = 12
head_dim = 64
batch = 8
n_iter = 50

slopes = torch.tensor(
    [2 ** (-8 * (i + 1) / num_heads) for i in range(num_heads)],
    device=device, dtype=torch.bfloat16,
)

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"PyTorch: {torch.__version__}")
print(f"Config: B={batch}, H={num_heads}, D={head_dim}, BF16, {n_iter} iters\n")

header = f"{'SeqLen':>8} {'flash_attn':>12} {'SDPA+ALiBi':>12} {'SDPA plain':>12} {'FlexAttn':>12}"
print(header)
print("-" * len(header))

for seq_len in [128, 256, 512, 1024]:
    q = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)

    # ALiBi bias matrix for SDPA
    pos = torch.arange(seq_len, device=device)
    bias = (pos.unsqueeze(0) - pos.unsqueeze(1)).float()
    alibi_bias = (slopes.view(num_heads, 1, 1) * bias.unsqueeze(0)).to(torch.bfloat16)
    alibi_bias = alibi_bias.unsqueeze(0).expand(batch, -1, -1, -1)

    # 1. flash_attn
    fa_ms = "N/A"
    try:
        from flash_attn import flash_attn_func
        q_fa = q.permute(0, 2, 1, 3).contiguous()
        k_fa = k.permute(0, 2, 1, 3).contiguous()
        v_fa = v.permute(0, 2, 1, 3).contiguous()
        for _ in range(3):
            flash_attn_func(q_fa, k_fa, v_fa, causal=False)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_iter):
            flash_attn_func(q_fa, k_fa, v_fa, causal=False)
        torch.cuda.synchronize()
        fa_ms = f"{(time.time() - t0) / n_iter * 1000:.2f}ms"
    except Exception as e:
        fa_ms = f"ERR"

    # 2. SDPA with ALiBi bias
    for _ in range(3):
        F.scaled_dot_product_attention(q, k, v, attn_mask=alibi_bias)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iter):
        F.scaled_dot_product_attention(q, k, v, attn_mask=alibi_bias)
    torch.cuda.synchronize()
    sdpa_alibi_ms = f"{(time.time() - t0) / n_iter * 1000:.2f}ms"

    # 3. SDPA plain (no mask)
    for _ in range(3):
        F.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iter):
        F.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()
    sdpa_plain_ms = f"{(time.time() - t0) / n_iter * 1000:.2f}ms"

    # 4. FlexAttention with compiled ALiBi score_mod
    flex_ms = "N/A"
    try:
        from torch.nn.attention.flex_attention import flex_attention
        slopes_flex = slopes.clone()

        def alibi_mod(score, b, h, q_idx, kv_idx):
            return score + (q_idx - kv_idx) * slopes_flex[h]

        compiled_flex = torch.compile(flex_attention)
        for _ in range(3):
            compiled_flex(q, k, v, score_mod=alibi_mod)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_iter):
            compiled_flex(q, k, v, score_mod=alibi_mod)
        torch.cuda.synchronize()
        flex_ms = f"{(time.time() - t0) / n_iter * 1000:.2f}ms"
    except Exception as e:
        flex_ms = f"ERR({e})"

    print(f"{seq_len:>8} {fa_ms:>12} {sdpa_alibi_ms:>12} {sdpa_plain_ms:>12} {flex_ms:>12}")

print("\nDone.")
