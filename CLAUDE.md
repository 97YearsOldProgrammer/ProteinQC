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

### Phase 1: Inference & Benchmarking (COMPLETE)
- Load CaLM weights from safetensors (no transformers dependency)
- Evaluate raw CaLM embeddings on RNA classification (coding vs non-coding)
- Benchmark against 24 published tools, 30+ species, ~70 dataset pairs
- Implement gated LM head in pure `torch.nn`
- Result: frozen CaLM + fresh MLP achieves ~87% avg accuracy zero-shot

### Phase 2: Multi-Signal Scoring System (current)
- **Confusion matrix analysis**: classify all benchmark errors as TP/FP/TN/FN
- **Feature engineering**: run all tool calls on benchmark sequences
  - CaLM score, CaLM perplexity, ORF length, codon_tai, Pfam hits, GC content
- **ML combiner**: XGBoost/scikit-learn on tool call features
  - Goal: patch CaLM's weak spots (short seqs, certain species) with additional signals
  - SHAP analysis to identify which tools matter most per failure mode
- **Key insight**: CaLM encoder stays frozen forever — intelligence is in the routing

### Phase 3: Tool-Use Agent (ReAct)
- **Tool-calling policy**: agent learns WHEN to call which tools
  - Short ambiguous ORF → call Pfam + codon_tai
  - Long clear coding seq → CaLM score alone suffices, skip expensive tools
- **RL on tool-use policy**: GRPO/PPO to optimize tool selection efficiency
  - Reward = classification accuracy + tool call cost penalty
  - DeepSeek-R1 / web-agent playbook applied to bioinformatics
- **Framework**: HuggingFace-compatible pipeline
  - `transformers` API for CaLM inference
  - Tool-calling agent as separate module
  - Ship as `pip install proteinqc`

### Design Principles
- **CaLM is never fine-tuned** — 85.75M params frozen, pretrained representations are strong enough
- **Scoring system is lightweight ML** — XGBoost on ~8-12 scalar features, not deep learning
- **Agent learns routing, not biology** — which tools to call, not how to interpret sequences
- **Short ORFs are the key failure mode** — additional signals (Pfam, codon usage) compensate

## Dependencies

### Core
```
torch>=2.0          # MPS backend for Apple Silicon
safetensors         # CaLM weight loading
huggingface_hub     # Model downloads
```

### Phase 2 (scoring system)
```
scikit-learn        # ML combiner, logistic regression baseline
xgboost             # Gradient boosted trees for tabular features
shap                # Feature importance analysis
pandas              # Feature table management
```

### Phase 3 (agent)
```
trl                 # GRPO training for tool-use policy
peft                # LoRA adapters for agent LLM
transformers        # HuggingFace pipeline integration
```

### Optional (deployment)
```
coremltools         # Convert to CoreML for ANE inference
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
