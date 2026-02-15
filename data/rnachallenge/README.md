# RNA Challenge Dataset

Binary classification benchmark: coding RNA vs non-coding RNA.

## Source

RNA Challenge benchmark dataset, used across 60+ ORF detection tools
(see `doc/benchmark_report.tsv` for full tool comparison).

## Files

| File | Description |
|------|-------------|
| `rnachallenge.tsv` | Main dataset: sequence_id, sequence, label (TSV) |
| `mRNAs.fa` | Coding RNA sequences (FASTA format) |
| `ncRNAs.fa` | Non-coding RNA sequences (FASTA format) |

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total sequences | 27,283 |
| Coding (label=1) | 16,243 (59.5%) |
| Non-coding (label=0) | 11,040 (40.5%) |

## Sequence Length Distribution (bp)

| Subset | Min | Median | Mean | Max |
|--------|-----|--------|------|-----|
| All | 21 | 787 | 1,635 | 26,831 |
| Coding | 21 | 532 | 664 | 10,809 |
| Non-coding | 369 | 2,072 | 3,065 | 26,831 |

Non-coding sequences are substantially longer on average (3x coding).
This reflects biological reality: lncRNAs tend to be longer transcripts.

## Source Organisms

| Source | Count | Fraction |
|--------|-------|----------|
| RefSeq (multi-species) | 5,152 | 18.9% |
| Mouse | 3,289 | 12.1% |
| Human | 3,259 | 11.9% |
| Ensembl (multi-species) | 2,873 | 10.5% |
| CPC2/lncRNA (CNT IDs) | 2,724 | 10.0% |
| Drosophila | 649 | 2.4% |
| C. elegans | 399 | 1.5% |
| RefSeq ncRNA | 139 | 0.5% |
| Soybean | 18 | 0.1% |

Multi-species coverage: vertebrates (human, mouse), invertebrates
(fly, worm), plants (soybean). Good taxonomic diversity.

## Codon Alignment Notes

Only 7.2% of coding sequences begin with ATG (start codon).
Most have 5'UTR upstream of the CDS. For codon-level tokenization
(e.g., CaLM), we find the first ATG to establish the correct
reading frame before chunking into triplets.

Interestingly, 13.7% of non-coding sequences start with ATG
(higher than coding) — spurious ATGs occur by chance in longer
non-coding transcripts.

## CaLM Tokenization Constraints

- CaLM uses codon-level (3-mer) tokenization
- Input must be a multiple of 3 nucleotides
- Max sequence length: 1,026 tokens = 3,078 bp
- Sequences longer than 3,078 bp are truncated after frame alignment
- Minimum: 3 codons (9 bp) required

## Label Convention

- `1` = coding RNA (mRNA with protein-coding ORF)
- `0` = non-coding RNA (lncRNA, miscRNA, other non-coding transcripts)
