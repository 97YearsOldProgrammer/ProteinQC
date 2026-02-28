# Multi-Species CaLM Benchmark: Supplementary Methods

## 1. Overview

We evaluated the CaLM (Codon-Aware Language Model) encoder on independent benchmark
datasets from 9 published RNA classification tools, spanning 16 species and covering
both animal and plant kingdoms. This evaluation tests whether CaLM's codon-level
representations, learned from pre-training on codon sequences, generalize to coding
vs. non-coding RNA classification across diverse taxa without tool-specific retraining.

All datasets are entirely independent from the RNA Challenge training set used to
train the MLP classification head. This separation enables a clean assessment of
both zero-shot transfer (MLP head) and representation quality (linear probe).

## 2. CaLM Encoder

| Parameter           | Value                                           |
|---------------------|-------------------------------------------------|
| Architecture        | BERT (CaLmForPreTraining)                       |
| Hidden size         | 768                                             |
| Attention heads     | 12                                               |
| Layers              | 12                                               |
| FFN intermediate    | 3,072                                            |
| Vocab size          | 131 (64 standard codons + ambiguous + special)  |
| Position encoding   | Rotary (RoPE)                                    |
| Max sequence length | 1,026 tokens (3,078 bp)                         |
| Total parameters    | 85,750,000 (all frozen during evaluation)       |
| Weights format      | safetensors (float32, 328 MB)                   |
| Source              | multimolecule/calm on HuggingFace Hub           |

The encoder is loaded from local safetensors weights with no dependency on the
`transformers` library. All encoder parameters are frozen; no gradient computation
occurs during embedding extraction.

## 3. Codon Alignment Preprocessing

CaLM operates at the codon level (3-nucleotide tokens), requiring that input
sequences be aligned to the correct reading frame. The following preprocessing
is applied uniformly to all sequences:

1. **Case normalization**: Convert to uppercase; replace U with T.
2. **Character filtering**: Remove all characters except A, C, G, T.
3. **ATG search**: Find the first occurrence of ATG (start codon) to establish
   the reading frame. If found, trim the 5' prefix before the ATG.
4. **Codon boundary trim**: Truncate the 3' end so the sequence length is a
   multiple of 3.
5. **Minimum length filter**: Sequences shorter than 9 bp (3 codons) after
   alignment are excluded.

This procedure is necessary because most benchmark datasets provide full-length
mRNA transcripts including 5'UTR, CDS, and 3'UTR, rather than CDS-only sequences.
The ATG search heuristic establishes the reading frame that produces meaningful
codon tokens from CaLM's vocabulary.

## 4. Data Provenance

### 4.1 CPAT

- **Citation**: Wang L, Park HJ, Dasari S, Wang S, Kocher J-P, Li W. CPAT:
  Coding-Potential Assessment Tool using an alignment-free logistic regression
  model. *Nucleic Acids Research*. 2013;41(6):e74. doi:10.1093/nar/gkt006
- **Data source**: CPAT SourceForge repository.
  https://sourceforge.net/projects/rna-cpat/files/
- **Species**: *Homo sapiens*
- **Data format**: Separate FASTA files for coding (mRNA) and non-coding (RNA).
  Full-length mRNA transcripts with UTRs, not CDS-only.
- **Files and sequence counts**:

| File | Sequences |
|------|-----------|
| `Human_test_coding_mRNA.fa` | 4,000 |
| `Human_test_noncoding_RNA.fa` | 4,000 |
| `Human_coding_transcripts_mRNA.fa` | 17,984 |
| `Human_noncoding_transcripts_RNA.fa` | 11,519 |

- **Total**: 37,503 sequences (21,984 coding + 15,519 non-coding)
- **Disk size**: 83 MB
- **Notes**: Two dataset splits (train and test) are evaluated separately. Headers
  use UCSC Genome Browser format with genomic coordinates (`hg19_ct_UserTrack`).

### 4.2 NCResNet

- **Citation**: Zhang Y, Jia C, Fullwood MJ, Kwoh CK. DeepCPP: a deep neural
  network based on nucleotide bias information and minimum distribution similarity
  feature selection for RNA coding potential prediction. *Briefings in Bioinformatics*.
  2021;22(2):2073-2084. Note: The NCResNet datasets were compiled and distributed
  by Wen J, Liu Y, Shi Y, Huang H, Deng B, Xiao X. A classification model for
  lncRNA and mRNA based on k-mers and a convolutional neural network. *BMC Bioinformatics*.
  2019;20:469. The extended species datasets (cow, rat, C. elegans) come from
  Singh U, Wurtele ES. orfipy: a fast and flexible tool for extracting ORFs.
  *Bioinformatics*. 2021;37(18):3019-3020, and the NCResNet benchmark collection.
- **Data source**: NCResNet GitHub repository supplementary data.
  https://github.com/Wuyang15/NCResNet
- **Species**: *Homo sapiens*, *Mus musculus*, *Saccharomyces cerevisiae*,
  *Danio rerio*, *Drosophila melanogaster* (core 5 species); *Bos taurus*,
  *Rattus norvegicus*, *Caenorhabditis elegans* (extended test species)
- **Data format**: Separate FASTA files per species, split into "long" (>200 bp)
  and "short" (<200 bp or mixed) subsets. Balanced class sizes per species-length
  combination. Full-length transcripts, not CDS-only.
- **Files and sequence counts** (core 5 species used in benchmark):

| Species | Subset | Coding | Non-coding |
|---------|--------|--------|------------|
| Human | long | 8,241 | 8,241 |
| Human | short | 641 | 641 |
| Mouse | long | 19,930 | 19,930 |
| Mouse | short | 846 | 846 |
| S. cerevisiae | long | 413 | 413 |
| S. cerevisiae | short | 413 | 413 |
| Zebrafish | long | 10,662 | 10,662 |
| Zebrafish | short | 387 | 387 |
| Fruitfly | long | 4,098 | 4,098 |
| Fruitfly | short | 381 | 381 |

- **Extended test species** (available but only 5 core species used by default):

| Species | Subset | Coding | Non-coding |
|---------|--------|--------|------------|
| Cow | long | 284 | 284 |
| Cow | short | 744 | 744 |
| Rat | long | 4,589 | 4,589 |
| Rat | short | 1,080 | 1,080 |
| C. elegans | long | 582 | 582 |
| C. elegans | short | 1,493 | 1,493 |

- **Total (all species)**: 115,320 sequences (core 5: 91,674; extended 3: 23,646)
- **Disk size**: 265 MB (includes train/ directory with human training data:
  26,650 coding + 23,983 non-coding)
- **Notes**: The "long" subset contains sequences with median ~2,779 bp (coding)
  and ~586 bp (non-coding) for human. The "short" subset has median ~958 bp (coding)
  and ~625 bp (non-coding) for human. Coding sequences use RefSeq accessions
  (`NM_` prefix); non-coding use Ensembl transcript IDs (`ENST` prefix).

### 4.3 LncFinder

- **Citation**: Han S, Liang Y, Ma Q, Xu Y, Zhang Y, Du W, Wang C, Li Y.
  LncFinder: an integrated platform for long non-coding RNA identification
  utilizing sequence intrinsic composition, structural information and
  physicochemical property. *Briefings in Bioinformatics*. 2019;20(6):2009-2027.
  doi:10.1093/bib/bby065
- **Data source**: LncFinder CRAN package supplementary data.
  https://CRAN.R-project.org/package=LncFinder
- **Species**: *Homo sapiens* (Human B), *Mus musculus*, *Gallus gallus* (Chicken),
  *Danio rerio* (Zebrafish), *Caenorhabditis elegans*, *Triticum aestivum* (Wheat)
- **Data format**: Nested directory structure `Species/Species/*.fa`. Files
  prefixed `pct.*` = protein-coding transcripts; `lnc.*` = lncRNA. Some species
  have train/test splits. Full-length transcripts.
- **Files and sequence counts**:

| Species | Files | Coding | Non-coding | Total |
|---------|-------|--------|------------|-------|
| C. elegans | 2 | 1,645 | 1,645 | 3,290 |
| Chicken | 2 | 8,000 | 8,000 | 16,000 |
| Human B | 4 (train+test) | 10,500 | 10,500 | 21,000 |
| Mouse | 4 (train+test) | 6,000 | 6,000 | 12,000 |
| Wheat | 4 (train+test) | 6,000 | 6,000 | 12,000 |
| Zebrafish | 2 | 4,000 | 4,000 | 8,000 |

- **Total**: 72,290 sequences (36,145 coding + 36,145 non-coding)
- **Disk size**: 121 MB
- **Notes**: All FASTA files within a species directory are concatenated during
  loading (train + test combined). The benchmark script uses `rglob("*.fa")` to
  discover all files per species. Coding files identified by `pct` prefix,
  non-coding by `lnc` prefix.

### 4.4 longdist

- **Citation**: Schneider HW, Raiol T, Brigido MM, Walter MEMT, Stadler PF.
  A support vector machine based method to distinguish long non-coding RNAs from
  protein coding transcripts. *BMC Genomics*. 2017;18:804.
  doi:10.1186/s12864-017-4178-4
- **Data source**: longdist supplementary material.
  https://github.com/SchneiderCompBio/longdist (or supplementary data)
- **Species**: *Mus musculus* (GRCm38 assembly)
- **Data format**: Separate FASTA files. Coding file contains Ensembl CDS
  sequences with `cds` biotype annotation; non-coding file contains `lincRNA`
  and `ncrna` biotype transcripts.
- **Files and sequence counts**:

| File | Sequences |
|------|-----------|
| `GRCm38.pct.fa` | 61,427 |
| `GRCm38.lncRNA.fa` | 12,646 |

- **Total**: 74,073 sequences (61,427 coding + 12,646 non-coding)
- **Disk size**: 113 MB
- **Notes**: Heavily imbalanced (4.86:1 coding to non-coding ratio). Coding
  sequences appear to be CDS-only (Ensembl `cds` biotype), while non-coding
  sequences are full-length lncRNA transcripts. Headers follow Ensembl format
  with assembly coordinates.

### 4.5 LncRNA-Mdeep

- **Citation**: Yang C, Yang L, Zhou M, Xie H, Zhang C, Wang MD, Zhu H.
  LncRNA-Mdeep: An Alignment-Free Predictor for Distinguishing Long Non-Coding
  RNAs from Protein-Coding Transcripts by Multimodal Deep Learning.
  *International Journal of Molecular Sciences*. 2020;21(18):5222.
- **Data source**: LncRNA-Mdeep supplementary data.
  https://github.com/yangchaogit/LncRNA-Mdeep
- **Species**: *Homo sapiens*
- **Data format**: Separate test FASTA files. Full-length transcripts.
- **Files and sequence counts**:

| File | Sequences |
|------|-----------|
| `human_PCT_test.fa` | 6,000 |
| `human_lncRNA_test.fa` | 6,000 |

- **Total**: 12,000 sequences (6,000 coding + 6,000 non-coding, balanced)
- **Disk size**: 15 MB
- **Notes**: Test set only. Headers use Ensembl transcript IDs.

### 4.6 DeepCPP

- **Citation**: Zhang S, Hu H, Jiang T, Zhang L, Zeng J. TITER: predicting
  translation initiation sites by deep learning. *Bioinformatics*. 2017;33(14):
  i234-i242. Note: The DeepCPP sORF datasets were compiled by Zhang Y, Jia C,
  Fullwood MJ, Kwoh CK. DeepCPP: a deep neural network based on nucleotide bias
  information and minimum distribution similarity feature selection for RNA coding
  potential prediction. *Briefings in Bioinformatics*. 2021;22(2):2073-2084.
  doi:10.1093/bib/bbaa039
- **Data source**: DeepCPP supplementary data.
  https://github.com/JoeHsiao/DeepCPP (or Singh & Roy 2022 benchmark collection)
- **Species**: *Homo sapiens* (short ORFs / sORFs)
- **Data format**: Separate FASTA files for coding sORFs (mRNA containing short
  ORFs) and non-coding sORFs (lncRNA-derived sORFs). Also includes a combined
  file (`humansorf.fa`).
- **Files and sequence counts**:

| File | Sequences |
|------|-----------|
| `human_mrnasorf.fa` | 232 |
| `human_lncsorf.fa` | 232 |
| `humansorf.fa` | 464 (combined) |

- **Total**: 464 sequences (232 coding + 232 non-coding, balanced)
- **Disk size**: 1.6 MB
- **Notes**: This is a specialized small ORF dataset. Coding sequences are
  RefSeq mRNAs (`NM_` accessions); non-coding sequences are Ensembl lncRNA
  transcripts. The sORF focus makes this a particularly challenging test case,
  as short coding sequences may lack the statistical signatures of full-length ORFs.

### 4.7 LncADeep

- **Citation**: Yang C, Yang L, Zhou M, Xie H, Zhang C, Wang MD, Zhu H.
  LncADeep: an ab initio lncRNA identification and functional annotation tool
  based on a deep learning framework. *Bioinformatics*. 2018;34(22):3825-3834.
  doi:10.1093/bioinformatics/bty428
- **Data source**: LncADeep GitHub repository.
  https://github.com/cyang235/LncADeep
- **Species**: *Homo sapiens*
- **Data format**: Single mixed FASTA file containing both coding and non-coding
  sequences. Coding/non-coding labels inferred from headers: `NM_` prefix or
  `mRNA` in description indicates coding; all others classified as non-coding.
- **Files and sequence counts**:

| File | Sequences |
|------|-----------|
| `lncRNA_mRNA_test.fa` | 100 (50 coding + 50 non-coding) |

- **Total**: 100 sequences (50 coding + 50 non-coding, balanced)
- **Disk size**: 200 KB
- **Notes**: Smallest dataset in the benchmark. Coding sequences are RefSeq
  human mRNAs with full gene descriptions. Non-coding sequences are Ensembl
  and GENCODE lncRNA transcripts (`ENST`/`OTTHUMT` accessions). Full-length
  transcripts with UTRs.

### 4.8 RNAplonc

- **Citation**: Negri TDC, Alves WAL, Bugatti PH, Saito PTM, Domingues DS,
  Paschoal AR. Pattern recognition analysis on long noncoding RNAs: a tool for
  prediction in plants. *Briefings in Bioinformatics*. 2019;20(2):682-689.
  doi:10.1093/bib/bby034
- **Data source**: RNAplonc GitHub repository.
  https://github.com/TatianneNegworworski/RNAplonc
- **Species**: *Amborella trichopoda*, *Brachypodium distachyon*,
  *Citrus sinensis*, *Manihot esculenta* (Cassava), *Ricinus communis*,
  *Sorghum bicolor*, *Solanum tuberosum* (Potato), *Zea mays* (Maize)
- **Data format**: Separate directories for `coding/` and `noncoding/` FASTA files.
  Each species has one file per class. Balanced class sizes per species. Coding
  sequences appear to be CDS or full-length mRNA; non-coding sequences are lncRNAs.
  Species naming differs between coding (Latin abbreviation, e.g., `atrichopoda`)
  and non-coding (common name, e.g., `amborella`) directories.
- **Files and sequence counts**:

| Species | Common name | Coding | Non-coding |
|---------|-------------|--------|------------|
| A. trichopoda | Amborella | 3,823 | 3,823 |
| B. distachyon | Brachypodium | 4,868 | 4,868 |
| C. sinensis | Citrus | 2,292 | 2,292 |
| M. esculenta | Cassava | 3,017 | 3,017 |
| R. communis | Ricinus | 4,080 | 4,080 |
| S. bicolor | Sorghum | 4,541 | 4,541 |
| S. tuberosum | Solanum | 5,607 | 5,607 |
| Z. mays | Maize | 12,071 | 12,071 |

- **Total**: 80,598 sequences (40,299 coding + 40,299 non-coding)
- **Disk size**: 108 MB
- **Notes**: Plant-specific benchmark. All species have perfectly balanced coding
  and non-coding counts. CaLM was pre-trained primarily on vertebrate/model organism
  codon usage, so plant species test cross-kingdom generalization. Coding sequence
  headers suggest CDS annotations from gene models (e.g., `evm_27.model` for
  Amborella).

### 4.9 lncRNAnet

- **Citation**: Baek J, Lee B, Kwon S, Yoon S. LncRNAnet: long non-coding RNA
  identification using deep learning. *Bioinformatics*. 2018;34(22):3889-3897.
  doi:10.1093/bioinformatics/bty418
- **Data source**: lncRNAnet supplementary data.
  https://github.com/JooBok/lncRNAnet
- **Species**: *Homo sapiens*, *Mus musculus*
- **Data format**: Separate FASTA files for human (`HT`) and mouse (`MT`)
  lncRNA transcripts. Additional `_100` files contain 100-sequence subsets.
  These files contain ONLY non-coding sequences (lncRNAs from GENCODE/Ensembl);
  no coding counterpart files are present.
- **Files and sequence counts**:

| File | Sequences | Description |
|------|-----------|-------------|
| `HT.fasta` | 7,000 | Human lncRNAs (full set) |
| `HT_100.fasta` | 100 | Human lncRNAs (subset) |
| `MT.fasta` | 7,000 | Mouse lncRNAs (full set) |
| `MT_100.fasta` | 100 | Mouse lncRNAs (subset) |

- **Total**: 14,200 sequences (all non-coding)
- **Disk size**: 17 MB
- **Notes**: This dataset is present in `data/benchmark/lncRNAnet/` but is NOT
  used in the current benchmark script because it lacks paired coding sequences.
  It contains only lncRNA sequences from GENCODE/HAVANA annotations (Ensembl
  transcript IDs with OTTHUMT/OTTMUST cross-references). Headers follow the
  pipe-delimited Ensembl format: `ENST|ENSG|OTTHUMG|OTTHUMT|name|gene|length`.

## 5. Evaluation Methodology

### 5.1 Two Evaluation Modes

Each dataset is evaluated in two independent modes:

**Mode 1: MLP Head Zero-Shot Transfer (from RNA Challenge)**

The pre-trained MLP classification head, trained on the RNA Challenge dataset
(27,283 sequences: 16,243 coding + 11,040 non-coding, CDS-only sequences from
multi-species RefSeq/Ensembl sources), is applied directly to each benchmark
dataset without any retraining. This tests whether the decision boundary learned
on one data distribution transfers to another.

The MLP head architecture is:
```
Linear(768, 256) -> GELU -> Dropout(0.1) ->
Linear(256, 256) -> GELU -> Dropout(0.1) ->
Linear(256, 1)
```
Weights loaded from `models/heads/mlp_head_v1.pt`.

**Mode 2: Fresh Linear Probe (per-dataset)**

A single-layer linear probe (`nn.Linear(768, 1)`) is trained from scratch on
each dataset independently, using an 80/20 random split (seed=42). This
evaluates the intrinsic quality of CaLM's [CLS] representations for separating
coding from non-coding sequences, independent of any prior training distribution.

Linear probe training:
- Optimizer: Adam, lr=1e-3
- Loss: BCEWithLogitsLoss
- Epochs: 50 (on the 80% training split)
- Weight init: N(0, 0.02), bias=0
- No regularization beyond early stopping at 50 epochs

### 5.2 Embedding Extraction

All sequences are processed through the frozen CaLM encoder to extract 768-dim
[CLS] token embeddings. Adaptive batching is used to handle variable-length
sequences efficiently:

- **TOKEN_BUDGET**: 8,192 tokens per batch
- **Maximum batch size**: 16 sequences
- **Adaptive sizing**: `batch_size = min(16, max(1, 8192 / max_codons))`
- **Sort-by-length**: Sequences sorted by length before batching to minimize
  padding waste
- **MPS cache management**: `torch.mps.empty_cache()` called after batches
  with sequences exceeding 500 codons (1,500 bp)

### 5.3 Metrics

All metrics are computed at the binary classification level (coding=1, non-coding=0):

| Metric | Formula |
|--------|---------|
| **ACC** (Accuracy) | (TP + TN) / (TP + TN + FP + FN) |
| **PRE** (Precision) | TP / (TP + FP) |
| **REC** (Recall) | TP / (TP + FN) |
| **F1** (F1-score) | 2 * PRE * REC / (PRE + REC) |
| **MCC** (Matthews Correlation Coefficient) | (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)) |

All metrics are reported as percentages (multiplied by 100). MCC ranges from
-100 (perfect inverse prediction) to +100 (perfect prediction), with 0
indicating random performance.

### 5.4 Compute Environment

| Component | Specification |
|-----------|---------------|
| Hardware | Apple Mac mini M4 Pro |
| CPU | 14 cores (10 Performance + 4 Efficiency) |
| Memory | 64 GB unified |
| GPU backend | PyTorch MPS (Metal Performance Shaders) |
| PyTorch | >= 2.0 |
| Random seed | 42 (torch + numpy) |

## 6. Key Finding: Domain Mismatch

The most notable result is the systematic failure of the MLP head (Mode 1) on
these independent datasets, contrasted with reasonable performance of the
linear probe (Mode 2).

**MLP head mean performance**: ACC ~21%, MCC ~ -50% (inverted predictions)

**Linear probe mean performance**: ACC ~77%, MCC ~ +55% (meaningful separation)

The MLP head was trained on the RNA Challenge dataset, which contains primarily
**CDS-only** sequences (coding sequences trimmed to the open reading frame,
without UTRs). The independent benchmark datasets contain **full-length mRNA
transcripts** including 5'UTR + CDS + 3'UTR. This distribution mismatch causes
the MLP head to systematically misclassify coding transcripts as non-coding:
when a full-length mRNA is fed to CaLM, the 5'UTR and 3'UTR regions produce
codon tokens that resemble non-coding patterns, overwhelming the CDS signal.

The linear probe, trained fresh on each dataset's own distribution, learns the
correct decision boundary for full-length transcripts and achieves substantially
better classification. This demonstrates that CaLM's [CLS] embeddings DO contain
discriminative information for coding vs. non-coding classification across diverse
species, but the decision boundary learned on CDS-only data does not transfer to
full-length transcript data.

**Implication**: For fair cross-dataset evaluation, the linear probe (Mode 2) is
the appropriate metric. The MLP head results quantify the domain shift between
CDS-only and full-length transcript distributions.

## 7. Dataset Summary

| Tool | Species evaluated | Sequences (coding + non-coding) | Format |
|------|-------------------|------|--------|
| CPAT | Human | 37,503 (21,984 + 15,519) | Full-length mRNA + UTRs |
| NCResNet | Human, Mouse, S. cerevisiae, Zebrafish, Fruitfly | 91,674 (balanced per species) | Full-length transcripts, long/short split |
| LncFinder | Human, Mouse, Chicken, Zebrafish, C. elegans, Wheat | 72,290 (balanced) | Full-length transcripts |
| longdist | Mouse (GRCm38) | 74,073 (61,427 + 12,646) | Mixed: CDS (coding) + full-length (non-coding) |
| LncRNA-Mdeep | Human | 12,000 (balanced) | Full-length transcripts |
| DeepCPP | Human (sORFs) | 464 (balanced) | Short ORF sequences |
| LncADeep | Human | 100 (balanced) | Mixed file, full-length |
| RNAplonc | 8 plant species | 80,598 (balanced per species) | CDS/mRNA (coding) + lncRNA |
| lncRNAnet* | Human, Mouse | 14,200 (non-coding only) | lncRNA only, not used |

*lncRNAnet is present in the data directory but excluded from the benchmark because
it lacks paired coding sequences.

**Grand total (used in benchmark)**: ~368,702 sequences across 28 dataset pairs
from 8 tools and 16 species.

## 8. Reference

This multi-species benchmark was motivated by:

Singh U, Roy SW. A large-scale benchmark study of tools for the classification
of protein-coding and non-coding RNAs. *Nucleic Acids Research*. 2022;50(22):
e131. doi:10.1093/nar/gkac1092. PMCID: PMC9757047.

Singh and Roy assembled 135 benchmark datasets from 24 RNA classification tools,
evaluating each tool's predictions across multiple species and training configurations.
Their study revealed that most tools perform poorly when applied to species or data
distributions different from their training data. Our evaluation uses a subset of 9
tools from their collection, testing whether a single pre-trained codon-level language
model (CaLM) can provide universal representations that generalize across these
diverse datasets.

The `doc/benchmark_report.tsv` file in this repository contains performance results
for 60 tool/model combinations evaluated on the RNA Challenge dataset, which serves
as the baseline for comparison.
