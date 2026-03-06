#!/usr/bin/env python3
"""CUDA profiling for LoRA training — synthetic data, no data pipeline dependency.

Generates fake CaLM-shaped batches to isolate GPU compute profile.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from proteinqc.models.calm_encoder import CaLMEncoder
from proteinqc.models.classification_heads import GatedHead


class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        return ((1 - p_t) ** self.gamma * bce).mean()


class LoRAClassifier(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, input_ids, attention_mask):
        return self.head(self.encoder(input_ids, attention_mask)).squeeze(-1)


def make_batch(bs, seq_len, vocab=131, device="cuda"):
    ids = torch.randint(5, vocab, (bs, seq_len), device=device)
    mask = torch.ones(bs, seq_len, dtype=torch.long, device=device)
    labels = torch.randint(0, 2, (bs,), device=device).float()
    return ids, mask, labels


def timed_step(model, ids, mask, labels, criterion, optimizer, all_trainable):
    """Run one training step, return per-component wall-clock times (ms)."""
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    logits = model(ids, mask)
    torch.cuda.synchronize()
    t_fwd = time.perf_counter()

    loss = criterion(logits, labels)
    torch.cuda.synchronize()
    t_loss = time.perf_counter()

    loss.backward()
    torch.cuda.synchronize()
    t_bwd = time.perf_counter()

    torch.nn.utils.clip_grad_norm_(all_trainable, 1.0)
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.synchronize()
    t_opt = time.perf_counter()

    return {
        "forward": (t_fwd - t0) * 1000,
        "loss": (t_loss - t_fwd) * 1000,
        "backward": (t_bwd - t_loss) * 1000,
        "optim": (t_opt - t_bwd) * 1000,
        "total": (t_opt - t0) * 1000,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, default=Path("models/calm"))
    p.add_argument("--n-warmup", type=int, default=5)
    p.add_argument("--n-profile", type=int, default=30)
    p.add_argument("--compile", action="store_true", default=True)
    p.add_argument("--no-compile", dest="compile", action="store_false")
    args = p.parse_args()

    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA: {torch.version.cuda}, PyTorch: {torch.__version__}")

    # Load model
    print("\nLoading CaLM encoder...")
    t0 = time.time()
    encoder = CaLMEncoder(args.model_dir, freeze=False)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    print("Applying LoRA (r=8, alpha=16)...")
    from peft import LoraConfig, get_peft_model
    for param in encoder.parameters():
        param.requires_grad = True
    encoder = get_peft_model(encoder, LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=0.1, bias="none",
    ))
    encoder.print_trainable_parameters()

    head = GatedHead(768, 256, 0.1)
    model = LoRAClassifier(encoder, head)

    if args.compile:
        print("torch.compile (dynamic=None)...")
        encoder._ensure_rope(1026, device)
        model = torch.compile(model, dynamic=None)

    model.to(device)

    lora_params = [p for n, p in encoder.named_parameters() if p.requires_grad]
    head_params = list(head.parameters())
    all_trainable = lora_params + head_params
    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": 2e-5},
        {"params": head_params, "lr": 1e-4},
    ], weight_decay=1e-4, fused=True)
    criterion = FocalBCEWithLogitsLoss(gamma=2.0)

    # (batch_size, seq_len) — all ~16384 tokens per step
    batch_configs = [
        (256, 64),
        (64, 256),
        (32, 512),
        (16, 1024),
    ]

    # Warmup
    print(f"\nWarmup ({args.n_warmup} steps/shape, triggers compile)...")
    model.train()
    t_warmup = time.time()
    for bs, sl in batch_configs:
        for _ in range(args.n_warmup):
            ids, mask, labels = make_batch(bs, sl, device=device)
            logits = model(ids, mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        torch.cuda.synchronize()
        print(f"  ({bs}x{sl}) compiled")
    print(f"  Total warmup: {time.time() - t_warmup:.1f}s")

    # Profile each shape
    for bs, sl in batch_configs:
        print(f"\n{'='*70}")
        print(f"  batch={bs}, seq_len={sl}, tokens/step={bs*sl}")
        print(f"{'='*70}")

        model.train()
        timings = {"forward": [], "loss": [], "backward": [], "optim": [], "total": []}

        for _ in range(args.n_profile):
            ids, mask, labels = make_batch(bs, sl, device=device)
            t = timed_step(model, ids, mask, labels, criterion, optimizer, all_trainable)
            for k in timings:
                timings[k].append(t[k])

        print(f"\n  {'Component':<12} {'Mean ms':>10} {'Std ms':>10} {'% Total':>10}")
        print(f"  {'-'*44}")
        mean_total = sum(timings["total"]) / len(timings["total"])
        for name in ["forward", "backward", "loss", "optim"]:
            vals = timings[name]
            mean = sum(vals) / len(vals)
            std = (sum((v - mean)**2 for v in vals) / len(vals)) ** 0.5
            pct = mean / mean_total * 100
            print(f"  {name:<12} {mean:>10.2f} {std:>10.2f} {pct:>9.1f}%")
        std_total = (sum((v - mean_total)**2 for v in timings["total"]) / len(timings["total"])) ** 0.5
        print(f"  {'TOTAL':<12} {mean_total:>10.2f} {std_total:>10.2f}")
        print(f"  Throughput: {bs*sl/mean_total*1000:,.0f} tok/s, {1000/mean_total:.1f} steps/s")

    # Mixed-shape throughput (simulating real training)
    print(f"\n{'='*70}")
    print(f"  MIXED SHAPE THROUGHPUT (simulating real epoch)")
    print(f"{'='*70}")

    import random
    random.seed(42)
    # Distribution: 40% short, 30% medium, 20% long, 10% max
    shape_weights = [(256, 64, 0.4), (64, 256, 0.3), (32, 512, 0.2), (16, 1024, 0.1)]

    model.train()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    n_steps = 100
    t_start = time.time()
    total_tok = 0

    for _ in range(n_steps):
        r = random.random()
        cum = 0.0
        for bs, sl, w in shape_weights:
            cum += w
            if r < cum:
                break
        ids, mask, labels = make_batch(bs, sl, device=device)
        logits = model(ids, mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_tok += bs * sl

    torch.cuda.synchronize()
    elapsed = time.time() - t_start
    sps = n_steps / elapsed
    tps = total_tok / elapsed

    peak_mb = torch.cuda.max_memory_allocated() / 1e6

    print(f"\n  {n_steps} mixed steps in {elapsed:.1f}s")
    print(f"  {sps:.1f} steps/sec, {tps:,.0f} tokens/sec")
    print(f"  Peak VRAM: {peak_mb:.0f} MB / 24576 MB ({peak_mb/24576*100:.0f}%)")

    # Training time estimate
    print(f"\n{'='*70}")
    print(f"  TRAINING TIME ESTIMATE (eu7 dataset)")
    print(f"{'='*70}")

    n_samples = 461_679
    avg_tok = 662  # from baked data stats
    total_tok_epoch = n_samples * avg_tok
    est_steps = total_tok_epoch / 16_384
    est_sec = est_steps / sps

    print(f"  Dataset: {n_samples:,} samples, ~{avg_tok} avg tokens")
    print(f"  Total tokens/epoch: {total_tok_epoch:,.0f}")
    print(f"  Est. steps/epoch: ~{est_steps:,.0f}")
    print(f"  Measured: {sps:.1f} steps/s, {tps:,.0f} tok/s")
    print(f"  Est. time/epoch: {est_sec:.0f}s ({est_sec/60:.1f}min)")
    print(f"  Est. 5 epochs: {est_sec*5:.0f}s ({est_sec*5/60:.1f}min)")
    print(f"  + val/eval ~15%: ~{est_sec*5*1.15/60:.0f}min total")
    print(f"  Peak VRAM: {peak_mb:.0f} MB — {'OK' if peak_mb < 22000 else 'TIGHT'}")


if __name__ == "__main__":
    main()
