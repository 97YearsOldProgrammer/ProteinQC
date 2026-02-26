# mRNA to Protein Quality Control: Research Landscape

> Research survey for the ProteinQC project — an RL/MCTS-based system for mRNA ORF exploration
> and protein quality prediction.
>
> Last updated: 2025-02-10

---

## Table of Contents

- [Top 10 Papers to Read](#top-10-papers-to-read)
- [1. mRNA Foundation Models](#1-mrna-foundation-models)
- [2. Protein Foundation Models](#2-protein-foundation-models)
- [3. mRNA Degradation and Stability](#3-mrna-degradation-and-stability)
- [4. Translation Efficiency and Ribosome Dynamics](#4-translation-efficiency-and-ribosome-dynamics)
- [5. Codon Optimization and mRNA Design](#5-codon-optimization-and-mrna-design)
- [6. Protein Degradation and Degron Prediction](#6-protein-degradation-and-degron-prediction)
- [7. Protein Aggregation Prediction](#7-protein-aggregation-prediction)
- [8. Protein Solubility Prediction](#8-protein-solubility-prediction)
- [9. Protein Disorder and Stability](#9-protein-disorder-and-stability)
- [10. Protein Structure Quality Assessment](#10-protein-structure-quality-assessment)
- [11. RNA Secondary Structure Prediction](#11-rna-secondary-structure-prediction)
- [The Gap: What Doesn't Exist Yet](#the-gap-what-doesnt-exist-yet)

---

## Top 10 Papers to Read

These are ordered by priority — read them in this order to build up intuition for the field
before designing the RL system.

### 1. ESM-2: Language models of protein sequences at the scale of evolution

- **Why read this**: This is THE protein encoder you'd use as your pre-trained backbone. Understanding how ESM-2 learns protein representations (up to 15B params, single-sequence, no MSA needed) is essential. Every downstream protein QC tool in the field builds on top of this.
- **Authors**: Lin Z, Akin H, Rao R, et al. (Meta AI)
- **Journal**: Science, 2023
- **Citations**: ~4,000+
- **Link**: https://www.science.org/doi/10.1126/science.ade2574
- **GitHub**: https://github.com/facebookresearch/esm
- **Key insight**: A single protein language model can replace MSA-based methods for structure prediction. The learned embeddings capture evolutionary, structural, and functional information — all useful as QC features.

### 2. LinearDesign: Algorithm for optimized mRNA design improves stability and immunogenicity

- **Why read this**: The landmark paper showing that mRNA sequence optimization dramatically impacts protein output. Uses lattice parsing from computational linguistics to jointly optimize stability + codon usage. 128x antibody boost in vivo. This is the baseline your system needs to beat or integrate.
- **Authors**: Zhang H, Zhang L, Lin A, et al. (Baidu Research)
- **Journal**: Nature, 2023
- **Citations**: ~400+
- **Link**: https://www.nature.com/articles/s41586-023-06127-z
- **GitHub**: https://github.com/LinearDesignSoftware/LinearDesign
- **Key insight**: The mRNA sequence space for a given protein is astronomically large (exponential in synonymous codons), but structured search (not brute force) can find optimal solutions in O(n^3). Your MCTS is an alternative search strategy.

### 3. Codon language embeddings provide strong signals for protein engineering

- **Why read this**: This is the paper that proves your core thesis — codon choice encodes information about protein stability and expression BEYOND amino acid identity. Different synonymous codons produce the same protein but with different folding outcomes. This is why mRNA-level QC matters.
- **Authors**: (Nature Machine Intelligence, 2024)
- **Journal**: Nature Machine Intelligence, 2024
- **Citations**: ~50
- **Link**: https://www.nature.com/articles/s42256-024-00791-0
- **Key insight**: A language model trained on codons (not amino acids) captures translation-rate-dependent features that predict protein stability. Synonymous mutations are NOT silent at the protein quality level.

### 4. CodonBERT: Large language models for mRNA design and optimization

- **Why read this**: The mRNA-side foundation model from Sanofi. Pretrained on 10M+ mRNA coding sequences with codon-level tokenization. You'd potentially use this (or mRNABERT) as your mRNA encoder alongside ESM-2 as your protein encoder.
- **Authors**: Li S, Moayedpour S, et al. (Sanofi)
- **Journal**: Genome Research, 2024
- **Citations**: ~50
- **Link**: https://pubmed.ncbi.nlm.nih.gov/38951026/
- **GitHub**: https://github.com/Sanofi-Public/CodonBERT
- **Key insight**: Codon-level tokenization outperforms nucleotide-level for mRNA property prediction. The masked language model objective naturally learns codon usage patterns across organisms.

### 5. Riboformer: Predicting context-dependent translation dynamics

- **Why read this**: This gives you the translation dynamics scoring signal. Predicts ribosome density at codon resolution — where ribosomes stall, speed up, or fall off. Ribosome stalling directly affects co-translational folding (and thus protein quality). This would be a key component of your game scoring system.
- **Authors**: (Nature Communications, 2024)
- **Journal**: Nature Communications, 2024
- **Citations**: ~50
- **Link**: https://www.nature.com/articles/s41467-024-46241-8
- **GitHub**: https://github.com/lingxusb/Riboformer
- **Key insight**: Translation speed is NOT uniform across codons. Context-dependent pausing affects protein folding. A transformer can predict these dynamics from sequence alone.

### 6. RNAdegformer: Accurate prediction of mRNA degradation at nucleotide resolution

- **Why read this**: Predicts WHERE an mRNA will degrade at single-nucleotide resolution. If the mRNA breaks down before translation completes, you get truncated/garbage protein. This is a direct scoring signal for your system — mRNA stability = protein yield.
- **Authors**: He S, Gao B, Sabnis R, Sun Q
- **Journal**: Briefings in Bioinformatics, 2023
- **Citations**: ~80
- **Link**: https://academic.oup.com/bib/article/24/1/bbac581/6986359
- **GitHub**: https://github.com/Shujun-He/RNAdegformer
- **Key insight**: Convolution + self-attention captures both local chemical vulnerability and global structural context for degradation prediction. Generalizes to sequences much longer than training data.

### 7. deepDegron: Systematic characterization of mutations altering protein degradation

- **Why read this**: Predicts degron sequences — the signals that tell the cell to destroy a protein via the ubiquitin-proteasome pathway. If the protein your mRNA produces has strong degrons, it gets tagged for destruction. Critical for understanding protein half-life from sequence.
- **Authors**: Tokheim C, et al.
- **Journal**: Molecular Cell, 2021
- **Citations**: ~200
- **GitHub**: https://github.com/ctokheim/deepDegron
- **Key insight**: N-terminal and C-terminal degrons can be predicted from sequence. Cancer mutations often disrupt degrons to stabilize oncoproteins. Your system could learn to predict whether an ORF's protein product will be rapidly degraded.

### 8. RiboDecode: Deep generative optimization of mRNA codon sequences

- **Why read this**: Current state-of-the-art. Beats LinearDesign. Uses deep generative framework learning from ribosome profiling data. 10x stronger antibody response for influenza, equivalent neuroprotection at 1/5th dose. This is what your RL system competes against or could extend.
- **Authors**: (Nature Communications, 2025)
- **Journal**: Nature Communications, 2025
- **Citations**: New
- **Link**: https://www.nature.com/articles/s41467-025-64894-x
- **Key insight**: Context-aware codon optimization (considering m1Psi modification, circular mRNAs) matters more than just optimizing CAI or MFE. Generative exploration of codon space finds solutions that rule-based optimizers miss.

### 9. PLTNUM: Prediction of protein half-lives from amino acid sequences by protein language models

- **Why read this**: Directly predicts the key QC output — protein half-life — using SaProt/ESM2 fine-tuned on SILAC mass spec data. This is your ground truth signal. If an mRNA produces a protein with a 2-hour half-life vs 48-hour half-life, that's a QC classification right there.
- **Authors**: Sagawa T, et al.
- **Journal**: bioRxiv, 2024
- **Citations**: New (preprint)
- **GitHub**: https://github.com/sagawatatsuya/PLTNUM
- **HuggingFace**: https://huggingface.co/sagawa/PLTNUM-ESM2-NIH3T3
- **Key insight**: Protein half-life can be predicted from sequence alone (71% accuracy for short/long classification). Cross-species transfer works (mouse→human). SHAP analysis reveals degron-like motifs driving short half-life.

### 10. mRNABERT: Advancing mRNA sequence design with a universal language model

- **Why read this**: The most comprehensive mRNA foundation model as of 2025. Dual tokenization (nucleotide + codon) with cross-modality contrastive learning that integrates protein semantics. Covers full-length mRNA (5'UTR + CDS + 3'UTR). If you need one mRNA encoder to rule them all, this is the current best.
- **Authors**: (Nature Communications, 2025)
- **Journal**: Nature Communications, 2025
- **Citations**: New
- **Link**: https://www.nature.com/articles/s41467-025-65340-8
- **Key insight**: Dual tokenization captures both fine-grained nucleotide patterns and codon-level semantics. Cross-modality contrastive learning bridges mRNA and protein representations — exactly the bridge your QC system needs.

---

## Full Tool Catalog

### 1. mRNA Foundation Models

| Tool | Year | Citations | Problem Type | GitHub | Verdict |
|------|------|-----------|-------------|--------|---------|
| **CodonBERT** | 2024 | ~50 | Foundation + downstream regression | [Sanofi-Public/CodonBERT](https://github.com/Sanofi-Public/CodonBERT) | Strong candidate for mRNA encoder |
| **mRNABERT** | 2025 | New | Foundation + generative design | TBD | Most comprehensive, covers all mRNA regions |
| **mRNA-LM** | 2025 | New | Foundation (CLIP-style 5'UTR+CDS+3'UTR) | TBD | First full-length integrated model |
| **mRNA2vec** | 2025 | ~10 | Foundation (data2vec framework) | TBD | Competitive with UTR-LM and CodonBERT |
| **UTR-LM** | 2024 | ~80 | Regression (translation efficiency) | [a96123155/UTR-LM](https://github.com/a96123155/UTR-LM) | Best for 5'UTR-specific tasks |
| **RNA-FM** | 2024 | ~100 | Foundation (23M RNA seqs) | [ml4bio/RNA-FM](https://github.com/ml4bio/RNA-FM) | More general RNA, not mRNA-specific |
| **RESM** | 2025 | New | Foundation (RNA via ESM-2 transfer) | TBD | Clever trick — maps RNA to pseudo-protein |
| **LucaOne** | 2025 | New | Foundation (unified DNA/RNA/protein) | TBD | Ambitious, 170K species |

**Key papers:**
- CodonBERT: [Genome Research 2024](https://pubmed.ncbi.nlm.nih.gov/38951026/)
- mRNABERT: [Nature Communications 2025](https://www.nature.com/articles/s41467-025-65340-8)
- mRNA-LM: [Nucleic Acids Research 2025](https://academic.oup.com/nar/article/53/3/gkaf044/7997216)
- mRNA2vec: [AAAI 2025 / arXiv](https://arxiv.org/abs/2408.09048)
- UTR-LM: [Nature Machine Intelligence 2024](https://www.nature.com/articles/s42256-024-00823-9)
- RNA-FM / RhoFold+: [Nature Methods 2024](https://www.nature.com/articles/s41592-024-02487-0)
- LucaOne: [Nature Machine Intelligence 2025](https://www.nature.com/articles/s42256-025-01044-4)

---

### 2. Protein Foundation Models

| Tool | Year | Citations | Problem Type | GitHub | Verdict |
|------|------|-----------|-------------|--------|---------|
| **ESM-2 / ESM3** | 2023/2024 | ~4,000+ | Foundation (up to 15B params) | [facebookresearch/esm](https://github.com/facebookresearch/esm) | THE protein encoder |
| **ProtTrans (ProtT5)** | 2022 | ~1,500+ | Foundation (BERT/T5 on proteins) | [rostlab/ProtTrans](https://github.com/agemagician/ProtTrans) | Solid alternative to ESM |

**Key papers:**
- ESM-2: [Science 2023](https://www.science.org/doi/10.1126/science.ade2574)
- ProtTrans: [IEEE TPAMI 2022](https://ieeexplore.ieee.org/document/9477085)

---

### 3. mRNA Degradation and Stability

| Tool | Year | Citations | Problem Type | GitHub | Verdict |
|------|------|-----------|-------------|--------|---------|
| **RNAdegformer** | 2023 | ~80 | Regression (nucleotide-level degradation) | [Shujun-He/RNAdegformer](https://github.com/Shujun-He/RNAdegformer) | Scoring signal for mRNA stability |
| **OpenVaccine** | 2022 | ~200 | Regression (nucleotide-level degradation) | [Kaggle solutions](https://www.kaggle.com/c/stanford-covid-vaccine) | Foundational dataset (6K RNAs) |
| **Saluki** | 2022 | ~100 | Regression (mRNA half-life, r=0.77) | TBD | mRNA half-life from sequence |
| **iCodon** | 2022 | ~100 | Regression + optimization | Web tool | Codon composition to stability |
| **Massively Parallel Decay** | 2024 | ~15 | Regression (50K+ mRNAs in bacteria) | TBD | Large-scale dataset |

**Key papers:**
- RNAdegformer: [Briefings in Bioinformatics 2023](https://academic.oup.com/bib/article/24/1/bbac581/6986359)
- OpenVaccine: [Nature Machine Intelligence 2022](https://www.nature.com/articles/s42256-022-00571-8)
- Saluki: [Genome Biology 2022](https://link.springer.com/article/10.1186/s13059-022-02811-x)
- Massively Parallel Decay: [Nature Communications 2024](https://www.nature.com/articles/s41467-024-54059-7)

---

### 4. Translation Efficiency and Ribosome Dynamics

| Tool | Year | Citations | Problem Type | GitHub | Verdict |
|------|------|-----------|-------------|--------|---------|
| **Riboformer** | 2024 | ~50 | Regression (codon-level ribosome density) | [lingxusb/Riboformer](https://github.com/lingxusb/Riboformer) | Translation bottleneck detection |
| **Translatomer** | 2024 | ~30 | Regression (cell-type-specific translation) | TBD | Context-dependent prediction |
| **RiboTIE** | 2025 | New | Classification (which ORFs are translated) | TBD | Directly relevant to ORF exploration |
| **RIBO-former** | 2023 | ~20 | Classification (translated ORF detection) | TBD | Complementary to RiboTIE |
| **Optimus** | 2019 | ~400 | Regression (mean ribosome load from 5'UTR) | TBD | Pioneering; 300K UTR dataset |

**Key papers:**
- Riboformer: [Nature Communications 2024](https://www.nature.com/articles/s41467-024-46241-8)
- Translatomer: [Nature Machine Intelligence 2024](https://www.nature.com/articles/s42256-024-00915-6)
- RiboTIE: [Nature Communications 2025](https://www.nature.com/articles/s41467-025-56543-0)
- Optimus: [Nature Biotechnology 2019](https://www.nature.com/articles/s41587-019-0164-5)

---

### 5. Codon Optimization and mRNA Design

| Tool | Year | Citations | Problem Type | GitHub | Verdict |
|------|------|-----------|-------------|--------|---------|
| **LinearDesign** | 2023 | ~400 | Optimization (lattice parsing) | [LinearDesignSoftware/LinearDesign](https://github.com/LinearDesignSoftware/LinearDesign) | Landmark — baseline to beat |
| **RiboDecode** | 2025 | New | Generative + regression | TBD | Beats LinearDesign in vivo |
| **iDRO** | 2023 | ~60 | Seq-to-seq (BiLSTM-CRF + RNA-BART) | TBD | Closest to end-to-end mRNA optimization |
| **CodonTransformer** | 2025 | New | Generative (164 organisms) | TBD | Multispecies codon optimizer |
| **ICOR** | 2023 | ~30 | Generative (BiLSTM for E. coli) | TBD | Narrow scope |
| **UTRGAN** | 2025 | ~20 | Generative (5'UTR design) | TBD | GAN-based UTR generation |
| **Codon Language Embeddings** | 2024 | ~50 | Embedding model | TBD | Proves codons encode protein stability |

**Key papers:**
- LinearDesign: [Nature 2023](https://www.nature.com/articles/s41586-023-06127-z)
- RiboDecode: [Nature Communications 2025](https://www.nature.com/articles/s41467-025-64894-x)
- iDRO: [Briefings in Bioinformatics 2023](https://academic.oup.com/bib/article/24/1/bbad001/6987657)
- Codon Language Embeddings: [Nature Machine Intelligence 2024](https://www.nature.com/articles/s42256-024-00791-0)
- CodonTransformer: [Nature Communications 2025](https://www.nature.com/articles/s41467-025-58588-7)

---

### 6. Protein Degradation and Degron Prediction

| Tool | Year | Citations | Problem Type | GitHub | Verdict |
|------|------|-----------|-------------|--------|---------|
| **deepDegron** | 2021 | ~200 | Classification (degron vs non-degron) | [ctokheim/deepDegron](https://github.com/ctokheim/deepDegron) | Degradation signals from sequence |
| **Degpred** | 2022 | ~80 | Classification (BERT-based) | [CHAOHOU-97/degpred](https://github.com/CHAOHOU-97/degpred) | Maps degrons to E3 ligases |
| **MetaDegron** | 2024 | New | Classification (PLM-based, 21 E3 ligases) | [Web](http://modinfor.com/MetaDegron/) | Most comprehensive degron predictor |
| **PLTNUM** | 2024 | New | Classification (protein half-life) | [sagawatatsuya/PLTNUM](https://github.com/sagawatatsuya/PLTNUM) | Direct QC output metric |
| **DeepUbi** | 2019 | ~150 | Classification (ubiquitination sites) | [Sunmile/DeepUbi](https://github.com/Sunmile/DeepUbi) | Ubiquitination = degradation tag |
| **N-end rule** | 1986 | ~15,000 | Rule-based (N-terminal → half-life) | N/A | Classic biology, free feature |

**Key papers:**
- deepDegron: [Molecular Cell 2021](https://www.cell.com/molecular-cell/fulltext/S1097-2765(21)00757-8)
- Degpred: [BMC Biology 2022](https://link.springer.com/article/10.1186/s12915-022-01364-6)
- MetaDegron: [Briefings in Bioinformatics 2024](https://academic.oup.com/bib/article/25/6/bbae519/7828723)
- PLTNUM: [bioRxiv 2024](https://github.com/sagawatatsuya/PLTNUM)

---

### 7. Protein Aggregation Prediction

| Tool | Year | Citations | Problem Type | GitHub | Verdict |
|------|------|-----------|-------------|--------|---------|
| **TANGO** | 2004 | ~2,500 | Regression (aggregation propensity) | [Web](https://tango.crg.es/) | Classic — statistical mechanics based |
| **AGGRESCAN** | 2007 | ~800 | Regression (in-vivo derived scale) | [Web](http://bioinf.uab.es/aggrescan/) | In-vivo validated |
| **Zyggregator** | 2008 | ~600 | Regression (aggregation rates) | [Web](https://www-vendruscolo.ch.cam.ac.uk/camsolmethod.html) | Predicts rates, not just propensity |
| **PASTA 2.0** | 2014 | ~350 | Regression (amyloid aggregation energy) | [Web](http://old.protein.bio.unipd.it/pasta2/) | Amyloid-specific |
| **AggreProt** | 2024 | New | Classification (DNN ensemble) | [Web](https://loschmidt.chemi.muni.cz/aggreprot/) | Engineering suggestions included |
| **AggrescanAI** | 2025 | New | Regression (PLM-based, no structure) | [Web](https://biocomp.chem.uw.edu.pl/aggrescanai/) | Modern PLM version |
| **AggNet** | 2025 | New | Classification/Regression (PLM-based) | TBD | Cutting edge |

**Key papers:**
- TANGO: [Nature Biotechnology 2004](https://www.nature.com/articles/nbt1012)
- AGGRESCAN: [BMC Bioinformatics 2007](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-8-65)
- AggreProt: [NAR 2024](https://academic.oup.com/nar/article/52/W1/W170/7670898)
- AggNet: [Protein Science 2025](https://onlinelibrary.wiley.com/doi/10.1002/pro.70031)

---

### 8. Protein Solubility Prediction

| Tool | Year | Citations | Problem Type | GitHub | Verdict |
|------|------|-----------|-------------|--------|---------|
| **NetSolP** | 2022 | ~200 | Classification + Regression (ESM-1b based) | [tvinet/NetSolP-1.0](https://github.com/tvinet/NetSolP-1.0) | State-of-the-art, PLM-based |
| **GATSol** | 2024 | New | Regression (Graph Attention + ESM + AF) | [binbinbinv/GATSol](https://github.com/binbinbinv/GATSol) | Structure-aware, latest |
| **CamSol** | 2015 | ~500 | Regression (pH-dependent, designs mutations) | [Web](https://www-vendruscolo.ch.cam.ac.uk/camsolmethod.html) | Also does rational design |
| **DeepSol** | 2018 | ~400 | Classification (first DL solubility predictor) | [GitHub](https://github.com/sameerkhurana10/DSOL_rv0.2) | Foundational |
| **Protein-Sol** | 2017 | ~350 | Regression (E. coli cell-free system) | [Web](https://protein-sol.manchester.ac.uk/) | Practical for E. coli |
| **SoluProt** | 2021 | ~150 | Classification (gradient boosting) | [Web](https://loschmidt.chemi.muni.cz/soluprot/) | E. coli focused |
| **SOLpro** | 2009 | ~400 | Classification (SVM, earliest ML) | N/A | Superseded |

**Key papers:**
- NetSolP: [Bioinformatics 2022](https://academic.oup.com/bioinformatics/article/38/4/941/6444984)
- GATSol: [BMC Bioinformatics 2024](https://github.com/binbinbinv/GATSol)
- CamSol: [Journal of Molecular Biology 2015](https://www.sciencedirect.com/science/article/pii/S0022283614006542)
- DeepSol: [Bioinformatics 2018](https://academic.oup.com/bioinformatics/article/34/15/2605/4938490)

---

### 9. Protein Disorder and Stability

| Tool | Year | Citations | Problem Type | GitHub | Verdict |
|------|------|-----------|-------------|--------|---------|
| **IUPred3** | 2021 | ~400 | Regression (disorder probability 0-1) | [Web](https://iupred3.elte.hu/) | IDRs = chaperone targets, aggregation prone |
| **flDPnn** | 2021 | ~200 | Classification (CAID winner + disorder functions) | [Web](https://www.nature.com/articles/s41467-021-24773-7) | Functionally annotates disorder |
| **PONDR** | 2001-2010 | ~2,000+ | Regression (disorder score) | [Web](https://pondr.com/) | Well-established family |
| **MobiDB** | 2025 | ~500 | Consensus (8 predictors + AlphaFold) | [Database](https://mobidb.org/) | Pre-computed for all UniProt |
| **AlphaFold pLDDT** | 2021 | ~30,000+ | Regression (confidence = foldability) | [Pre-computed](https://alphafold.ebi.ac.uk/) | Free, pre-computed, pLDDT<50 = disordered |
| **ThermoMPNN** | 2024 | ~100 | Regression (ddG from ProteinMPNN) | [Kuhlman-Lab/ThermoMPNN](https://github.com/Kuhlman-Lab/ThermoMPNN) | Fast stability prediction |
| **SPURS** | 2025 | New | Regression (rewires ESM+ProteinMPNN) | [luo-group/SPURS](https://github.com/luo-group/SPURS) | Cutting edge stability |
| **DDMut** | 2023 | ~70 | Regression (ddG, Siamese network) | [Web+API](https://biosig.lab.uq.edu.au/ddmut) | Multi-mutation support |
| **DeepDDG** | 2019 | ~300 | Regression (ddG) | [Web](http://protein.org.cn/ddg.html) | Well-established |
| **RaSP** | 2023 | ~150 | Regression (ddG, fast) | [eLife](https://elifesciences.org/articles/82593) | On-par with biophysics methods |

**Key papers:**
- AlphaFold: [Nature 2021](https://www.nature.com/articles/s41586-021-03819-2)
- IUPred3: [NAR 2021](https://academic.oup.com/nar/article/49/W1/W297/6275862)
- flDPnn: [Nature Communications 2021](https://www.nature.com/articles/s41467-021-24773-7)
- ThermoMPNN: [PNAS 2024](https://www.pnas.org/doi/10.1073/pnas.2314853121)
- SPURS: [Nature Communications 2025](https://www.nature.com/articles/s41467-025-67609-4)

---

### 10. Protein Structure Quality Assessment

| Tool | Year | Citations | Problem Type | Verdict |
|------|------|-----------|-------------|---------|
| **QMEANDisCo** | 2020 | ~730 | Regression (lDDT + global quality) | Gold standard for structure QA |
| **DeepAccNet** | 2021 | ~300 | Regression (per-residue lDDT) | Baker lab, guides refinement |
| **DeepUMQA-X** | 2025 | New | Regression (CASP16 top performer) | Cutting edge |
| **GraphCPLMQA** | 2023 | ~15 | Regression (PLM + graph networks) | PLM embeddings improve QA |
| **VoroMQA** | 2017 | ~180 | Regression (Voronoi tessellation) | CASP/CAPRI strong |
| **ProQ3D** | 2017 | ~230 | Regression (DL-based MQA) | Superseded by newer methods |
| **ModFOLD9** | 2024 | ~5 | Regression (6 DL methods integrated) | Latest version |

**Key papers:**
- QMEANDisCo: [Bioinformatics 2020](https://academic.oup.com/bioinformatics/article/36/6/1765/5614424)
- DeepAccNet: [Nature Communications 2021](https://www.nature.com/articles/s41467-021-21511-x)

---

### 11. RNA Secondary Structure Prediction

| Tool | Year | Citations | Problem Type | Verdict |
|------|------|-----------|-------------|---------|
| **BPfold** | 2025 | New | Regression/Classification (base pair energy) | Best generalization to unseen RNA families |
| **UFold** | 2022 | ~300 | Classification (U-Net, ~160ms/seq) | Fast, practical for screening |
| **SPOT-RNA** | 2019 | ~500 | Classification (2D DNN + transfer learning) | Foundational transfer learning for RNA |
| **RNADiffFold** | 2024 | ~15 | Generative (diffusion model) | Uncertainty-aware structure sampling |

**Key papers:**
- BPfold: [Nature Communications 2025](https://www.nature.com/articles/s41467-025-60048-1)
- UFold: [NAR 2022](https://academic.oup.com/nar/article/50/3/e14/6430845)
- SPOT-RNA: [Nature Communications 2019](https://www.nature.com/articles/s41467-019-13395-9)

---

### Expression Level Prediction

| Tool | Year | Citations | Problem Type | Verdict |
|------|------|-----------|-------------|---------|
| **Xpresso** | 2020 | ~300 | Regression (promoter+stability→expression) | Explains 59% human, 71% mouse variation |
| **MP-TRANS** | 2025 | New | Regression + generative (88 species models) | Enables expression of previously unexpressible proteins |
| **Cross-Context TL** | 2024 | ~20 | Regression (transfer learning for generalization) | Addresses the generalization problem |
| **Mechanistic Features** | 2025 | New | Regression (mechanistic + DL hybrid) | Shows mechanistic features improve generalization |

**Key papers:**
- Xpresso: [Cell Reports 2020](https://www.cell.com/cell-reports/fulltext/S2211-1247(20)30616-1)
- Cross-Context TL: [NAR 2024](https://academic.oup.com/nar/article/52/13/e58/7691520)
- Mechanistic Features: [NAR 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11773361/)

---

## The Gap: What Doesn't Exist Yet

### No end-to-end mRNA → protein QC pipeline

Every tool above solves ONE piece. Nobody has connected them. The closest attempts:

1. **iDRO** (2023) — optimizes full-length mRNA but doesn't assess downstream protein quality
2. **LinearDesign** (2023) — optimizes mRNA stability + codon usage but no protein-level QC
3. **SONAR** (2025) — predicts cell-type-specific protein abundance from mRNA features, but not protein *quality*

### Nobody uses RL or game-theoretic approaches

The entire field is supervised learning (regression/classification) or rule-based optimization.
No MCTS, no self-play, no learned scoring functions that bootstrap through adversarial dynamics.

### Nobody explores the ORF search space

Existing codon optimizers take a FIXED amino acid sequence and optimize synonymous codons.
Nobody is exploring different possible ORFs from a given genomic region — different AUG start
codons, different stop codons, overlapping reading frames. This is the combinatorial game tree
that MCTS is designed to search.

### The generalization problem is unsolved

Multiple 2024-2025 papers explicitly show that supervised models fail to generalize outside
their training distribution. This validates the RL approach — instead of learning from
potentially unreliable labeled data, learn from the game dynamics of the scoring system.

### Integration of existing QC tools as scoring signals

The building blocks exist:
- **mRNA stability**: RNAdegformer
- **Translation efficiency**: Riboformer
- **Protein solubility**: NetSolP
- **Protein aggregation**: AggrescanAI
- **Protein half-life**: PLTNUM
- **Degron detection**: deepDegron
- **Structural foldability**: AlphaFold pLDDT

These can all serve as reward components in an RL scoring system, with the learned model
eventually internalizing these signals into a portable, pre-trained encoder.
