# ProteinQC

## Project Overview

RL/MCTS-based system for mRNA ORF exploration and protein quality prediction.
CaLM (Codon-Aware Language Model) is the backbone encoder — 12-layer BERT, 768 hidden, 131 codon-level vocab, 85.75M params, RoPE positional embeddings.

## Hardware

- Mac mini M4 Pro — 14 cores (10P + 4E), 64GB unified memory
- 16-core Apple Neural Engine (ANE)
- PyTorch MPS backend for GPU acceleration
- CoreML/ANE path available for deployment-stage inference

## Architecture: Gated LM Head

Replace the default melting point classification head with a **gated dual-path head** for binary classification (coding vs non-coding protein):

```
CaLM encoder (frozen) → [CLS] embedding (768-dim)
                              ↓
                         Gate (Linear → Sigmoid) → g ∈ [0,1]
                              ↓
              ┌───────────────┴───────────────┐
              │                               │
         (1 - g) × Shortcut              g × MLP
         (Linear: 768→1,               (768→256→256→1,
          single bias,                   GELU, Dropout,
          logistic regression)           deep nonlinear)
              │                               │
              └───────────────┬───────────────┘
                              ↓
                     logits (binary classification)
```

- **Shortcut path**: Single linear layer — handles easy-to-classify sequences (clear ORF signals)
- **MLP path**: Deep nonlinear — handles ambiguous cases (short ORFs, lncRNAs with coding-like features)
- **Gate**: Learned sigmoid routing — model decides per-sample which path dominates
- **Balance loss**: `|mean(g) - 0.5|` regularization to prevent gate collapse

## Project Roadmap

### Phase 1: Inference & Benchmarking (current)
- Load CaLM weights from safetensors (no transformers dependency)
- Evaluate raw CaLM embeddings on RNA classification (coding vs non-coding)
- Benchmark against existing tools from `doc/benchmark_report.tsv`
- Implement gated LM head in pure `torch.nn`
- Test on MPS backend, verify ANE compatibility path

### Phase 2: SFT on Small Dataset
- Curate small labeled dataset (coding/non-coding RNA sequences)
- Supervised fine-tuning of the gated head (encoder frozen initially)
- Optional: unfreeze top-N encoder layers for domain adaptation
- Establish baseline metrics (ACC, PRE, REC, F1, MCC)

### Phase 3: Reinforcement Learning
- Define reward function using existing QC tool signals:
  - mRNA stability (RNAdegformer)
  - Translation efficiency (Riboformer)
  - Protein solubility (NetSolP)
  - Protein half-life (PLTNUM)
  - Degron detection (deepDegron)
- MCTS for ORF search space exploration (different AUG starts, stop codons, reading frames)
- Self-play or adversarial scoring dynamics

### Phase 4: Chain-of-Thought SFT
- Distill RL-learned reasoning into explicit chain-of-thought traces
- SFT on (sequence, reasoning chain, prediction) triples
- Goal: interpretable model that explains WHY a sequence is coding/non-coding
- Portable encoder that internalizes multi-signal QC scoring

## Dependencies

### Core (required now)
```
torch>=2.0          # MPS backend for Apple Silicon
safetensors         # CaLM weight loading
huggingface_hub     # Model downloads
```

### Phase 2+ (SFT and beyond)
```
multimolecule       # HuggingFace-compatible CaLM wrapper
datasets            # HuggingFace datasets for training data
accelerate          # Training utilities
```

### Optional (deployment)
```
coremltools         # Convert to CoreML for ANE inference
ane_transformers    # ANE-optimized transformer blocks
mlx                 # Apple-native ML framework alternative
```

## Code Conventions

- Pure PyTorch for model code — no unnecessary framework abstractions
- Feature-organized: `src/model/`, `src/data/`, `src/train/`, `src/eval/`
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

src/
  download_calm.py      # HuggingFace model download utility
  inspect_calm.py       # Weight inspection and config verification

doc/
  research_landscape.md          # Full research survey, top 10 papers, tool catalog
  encoder_tokenizer_reference.md # Technical reference for ESM-2, CodonBERT, CaLM
  benchmark_report.tsv           # ORF detection benchmark (60 tools/variants)
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
| Position encoding | Rotary (RoPE) |
| Max seq length | 1026 tokens |
| Total params | ~85.75M |
| Weights format | safetensors (float32) |
