# ProteinQC

## Project Overview

RL/MCTS-based system for mRNA ORF exploration and protein quality prediction.
CaLM (Codon-Aware Language Model) is the backbone encoder — 12-layer BERT, 768 hidden, 131 codon-level vocab, 85.75M params, RoPE positional embeddings.

## Hardware

- Mac mini M4 Pro — 14 cores (10P + 4E), 64GB unified memory
- 16-core Apple Neural Engine (ANE)
- PyTorch MPS backend for GPU acceleration

## Architecture: LoRA ALiBi GatedHead

CaLM encoder (frozen, RoPE) with LoRA adapters + ALiBi position encoding in the classification head:

```
CaLM encoder (frozen, 85.75M params)
       ↓
  LoRA adapters (r=8, alpha=16, q/k/v_proj — 442K params)
       ↓
  ALiBi attention (no sequence length limit)
       ↓
  [CLS] embedding (768-dim)
       ↓
  GatedHead (264K params)
       ↓
  Gate (Linear → Sigmoid) → g ∈ [0,1]
       ↓
  ┌────┴────┐
  │         │
(1-g)×Lin  g×MLP
  │         │
  └────┬────┘
       ↓
  logits (coding vs non-coding)
```

- **LoRA**: rank=8, alpha=16, targets q/k/v projections (442K trainable params)
- **ALiBi**: replaces RoPE for the head — no 1026-token sequence length cap
- **GatedHead**: shortcut (linear) + MLP paths, learned per-sample routing
- **Current best**: `models/heads/lora_alibi_gated_v1/` (epoch 2, val_acc=90.4%)

## Project Roadmap

### Phase 1: Inference & Benchmarking (COMPLETE)
- Load CaLM weights from safetensors (no transformers dependency)
- Evaluate raw CaLM embeddings on RNA classification (coding vs non-coding)
- Benchmark against 24 published tools, 30+ species, 80 dataset pairs
- Implement gated LM head with LoRA + ALiBi in pure `torch.nn`
- Result: frozen CaLM + LoRA ALiBi GatedHead achieves 83.8% avg ACC, 92.5% AUC zero-shot across 2.1M sequences

### Phase 2: Multi-Signal Scoring System (current)
- **Feature extraction complete**: 17 features per sequence across 80 datasets (2.1M seqs)
  - CaLM score, ORF length, codon_tai, Pfam hits, GC content, kozak, entropy
- **XGBoost combiner**: v2 trained on tool-call features (`models/combiner/xgb_v2.json`)
  - SHAP analysis identifies which tools matter most per failure mode
- **Key insight**: CaLM encoder stays frozen forever — intelligence is in the routing
- **Weak spots**: short sequences (<300nt), some plant species — XGBoost compensates

### Phase 3: Tool-Use Agent (ReAct)
- **Tool-calling policy**: agent learns WHEN to call which tools
- **RL on tool-use policy**: GRPO with Qwen2.5-7B-Instruct
- **Framework**: `pip install proteinqc`

### Design Principles
- **CaLM is never fine-tuned** — 85.75M params frozen, pretrained representations are strong enough
- **Scoring system is lightweight ML** — XGBoost on ~12 scalar features, not deep learning
- **Agent learns routing, not biology** — which tools to call, not how to interpret sequences
- **Short ORFs are the key failure mode** — additional signals (Pfam, codon usage) compensate

## Dependencies

### Core
```
torch>=2.0          # MPS backend for Apple Silicon
safetensors         # CaLM weight loading
huggingface_hub     # Model downloads
biopython           # Sequence I/O
pyhmmer             # Pfam domain scanning
```

### Phase 2 (scoring system)
```
scikit-learn        # ML combiner, logistic regression baseline
xgboost             # Gradient boosted trees for tabular features
shap                # Feature importance analysis
pandas / pyarrow    # Feature table management
```

### Phase 3 (agent)
```
mlx / mlx-lm        # Apple Silicon inference + LoRA
smolagents           # Tool-calling agent framework
```

## Code Conventions

- Pure PyTorch for model code — no unnecessary framework abstractions
- Feature-organized: `proteinqc/models/`, `proteinqc/data/`, `proteinqc/tools/`, `proteinqc/cli/`
- Immutable data flow — new objects, no mutation
- Files: 200-400 lines typical, 800 max
- Functions: <50 lines, max 4 levels nesting
- Type hints on all public interfaces
- Device-agnostic: `device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")`

## Key Files

```
models/calm/
  config.json           # CaLM architecture config (768 hidden, 12 layers, 131 vocab)
  model.safetensors     # Pre-trained weights (328 MB)
  vocab.txt             # 131 codon-level tokens

models/heads/
  lora_alibi_gated_v1/  # Current best: LoRA + ALiBi + GatedHead

models/combiner/
  xgb_v2.json           # XGBoost multi-signal combiner

proteinqc/
  cli/                  # CLI entry points (extract_features, benchmark_zeroshot, etc.)
  models/               # CaLM encoder, classification heads
  tools/                # calm_scorer, orf_scanner, pfam_scanner, riboformer, etc.
  agent/                # RL agent (MLX backend, GRPO, episodes)
  pipeline.py           # ORFPipeline orchestrator

doc/
  research_landscape.md          # Full research survey, top 10 papers, tool catalog
  encoder_tokenizer_reference.md # Technical reference for ESM-2, CodonBERT, CaLM
```

## Model Reference

| Parameter | Value |
|-----------|-------|
| Architecture | BERT (CaLmForPreTraining) |
| Hidden size | 768 |
| Attention heads | 12 |
| Layers | 12 |
| FFN intermediate | 3072 |
| Vocab | 131 (64 standard codons + ambiguous + special) |
| Position encoding | RoPE (encoder), ALiBi (head) |
| Max seq length | 1026 tokens (encoder), unlimited (ALiBi head) |
| Encoder params | ~85.75M (frozen) |
| LoRA params | ~442K (trainable) |
| Head params | ~264K (trainable) |
| Weights format | safetensors (float32) |
