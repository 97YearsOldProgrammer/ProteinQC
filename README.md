# ProteinQC

Coding vs non-coding RNA classification using CaLM (Codon-Aware Language Model) with multi-signal scoring.

CaLM is a 12-layer BERT encoder (85.75M params) pretrained on codon-level sequences. ProteinQC wraps it with lightweight classification heads and a bioinformatics tool suite to classify mRNA transcripts as protein-coding or non-coding.

## Benchmark

**90.8% mean accuracy** across 80 dataset/species combinations (19 published benchmark suites, 57 species).

| Range | Examples |
|-------|---------|
| 95-99% | Vertebrates, Plants, S. cerevisiae, Tomato, Fruitfly (long) |
| 85-95% | Human, Mouse, Zebrafish (long), Arabidopsis, Rice |
| 77-85% | Short sequences (<300nt), Amborella, Potato |

CaLM encoder is frozen — all intelligence is in the routing and scoring layers.

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
pip install -e ".[agent]"        # mlx — RL agent (macOS only)
pip install -e ".[dev]"          # pytest, ruff — development
```

## Usage

### Quick classification

```python
from proteinqc.tools.calm_scorer import CaLMScorer

scorer = CaLMScorer(model_dir="models/calm", head_weights="models/heads/mlp_head.pt")
result = scorer.score("ATGAAAGCTTGA...")
print(result.coding_probability)  # 0.0 - 1.0
```

### Feature extraction (Phase 2)

```bash
extract-features --coding data/pc.fa --noncoding data/nc.fa --output features.parquet
```

Extracts 17 features per sequence: CaLM score, ORF metrics, codon usage, GC content, and more.

### Benchmark reproduction

```bash
benchmark-multispecies --data-dir data/benchmark/ --output results.json
```

## Project structure

```
proteinqc/              # installable package
  models/               # CaLM encoder, classification heads
  data/                 # tokenizer, dataset loaders
  tools/                # bioinformatics tools (ORF scanner, codon table, Pfam, etc.)
  agent/                # RL agent infrastructure (GRPO, episodes)
  metrics/              # evaluation metrics
  cli/                  # CLI entry points
  pipeline.py           # ORFPipeline orchestrator
bin/                    # executable scripts
models/calm/            # CaLM weights and config (gitignored, downloaded on demand)
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
  [CLS] embedding (768-dim)
       |
  Classification head (MLP or Gated)  -->  coding probability
```

The gated head routes easy sequences through a single linear layer and hard sequences through a deeper MLP, learned per-sample.

## Roadmap

- [x] Phase 1 — CaLM inference and multi-species benchmarking (90.8% avg)
- [ ] Phase 2 — Multi-signal scoring (XGBoost on tool-call features + SHAP analysis)
- [ ] Phase 3 — Tool-use RL agent (learns when to call expensive tools)

## License

MIT
