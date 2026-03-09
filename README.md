# ProteinQC

Coding vs non-coding RNA classification using CaLM (Codon-Aware Language Model) with multi-signal scoring.

CaLM is a 12-layer BERT encoder (85.75M params) pretrained on codon-level sequences. ProteinQC wraps it with a LoRA ALiBi GatedHead and a bioinformatics tool suite to classify mRNA transcripts as protein-coding or non-coding.

## Benchmark

**Zero-shot across 80 datasets (2.1M sequences, 19 benchmark suites, 57 species):**
- **83.8% mean accuracy, 92.5% mean AUC**

| Range | Examples |
|-------|---------|
| 95-99% | LGC-RefSeq Vertebrates/Invertebrates/Plants, Tomato, C. elegans |
| 85-95% | CPAT Human, Mouse, Zebrafish (long), Arabidopsis |
| 73-85% | Short sequences (<300nt), Amborella, Potato |

CaLM encoder is frozen — all intelligence is in the LoRA adapters, gated head, and XGBoost combiner.

## Install

```bash
pip install -e .
```

Requires Python 3.9+ and PyTorch 2.0+. Model weights (~328 MB) download automatically on first run:

```bash
download-calm          # or: python bin/download-calm
```

### Optional dependencies

```bash
pip install -e ".[features]"     # pandas, pyarrow — feature extraction
pip install -e ".[scoring]"      # xgboost, shap — ML combiner
pip install -e ".[agent]"        # mlx — RL agent (macOS only)
pip install -e ".[dev]"          # pytest, ruff — development
```

## Usage

### Quick classification

```python
from proteinqc.tools.calm_scorer import CaLMScorer

scorer = CaLMScorer(model_dir="models/calm", head_weights="models/heads/lora_alibi_gated_v1")
scores = scorer.batch_score(["ATGAAAGCTTGA..."])
print(scores[0])  # 0.0 - 1.0
```

### Feature extraction (Phase 2)

```bash
extract-features --coding data/pc.fa --noncoding data/nc.fa --output features.parquet
```

Extracts 17 features per sequence: CaLM score, ORF metrics, codon usage, GC content, and more.

### Benchmark reproduction

```bash
benchmark-zeroshot --data-dir data/benchmark/ --output results.json
```

## Project structure

```
proteinqc/              # installable package
  models/               # CaLM encoder, classification heads
  data/                 # tokenizer, dataset loaders
  tools/                # bioinformatics tools (ORF scanner, codon table, Pfam, etc.)
  agent/                # RL agent infrastructure (GRPO, episodes)
  cli/                  # CLI entry points
  pipeline.py           # ORFPipeline orchestrator
bin/                    # executable scripts
models/calm/            # CaLM weights and config (gitignored, downloaded on demand)
models/heads/           # classification head weights
models/combiner/        # XGBoost combiner weights
data/features/          # extracted feature parquets
data/results/           # benchmark results (tracked)
doc/                    # research docs and methods
tests/                  # pytest
```

## Architecture

```
Input mRNA sequence
       |
  Codon tokenizer (131 vocab)
       |
  CaLM encoder (12-layer BERT, frozen)
       |
  LoRA adapters (r=8, alpha=16, q/k/v)
       |
  ALiBi attention (no length limit)
       |
  [CLS] embedding (768-dim)
       |
  GatedHead (shortcut + MLP, learned routing)  -->  coding probability
```

The gated head routes easy sequences through a single linear layer and hard sequences through a deeper MLP, learned per-sample.

## Roadmap

- [x] Phase 1 — CaLM inference and multi-species benchmarking (83.8% avg ACC, 92.5% AUC)
- [x] Phase 2 — Multi-signal scoring (XGBoost combiner on 17 features + SHAP analysis)
- [ ] Phase 3 — Tool-use RL agent (learns when to call expensive tools)

## License

MIT
