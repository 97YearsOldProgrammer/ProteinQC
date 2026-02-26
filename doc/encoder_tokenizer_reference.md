# Encoder Tokenizer & Output Reference

Technical reference for the tokenizers and model outputs of three encoder-style language models relevant to the ProteinQC project. All three return **probability distributions** (via logits → softmax) over their respective vocabularies at each sequence position.

---

## Quick Comparison

| Property | ESM-2 | CodonBERT | CaLM |
|----------|-------|-----------|------|
| **Input level** | Amino acid (protein) | Codon (mRNA) | Codon (mRNA/cDNA) |
| **Vocab size** | 33 | 130 (pretrain) / 69 (benchmark) | 69 |
| **Tokenizer type** | Character-level | WordLevel | WordLevel |
| **Hidden dim** | 1280 | 768 | 768 |
| **Layers** | 33 | 12 | 12 |
| **Heads** | 20 | 12 | 12 |
| **Position encoding** | Rotary (RoPE) | Learned | Rotary (RoPE) |
| **Max length** | 1026 | 1024 | 1024 |
| **Pre-training** | MLM | MLM + NSP | MLM |
| **Origin** | Meta (Facebook AI) | Sanofi | Oxford (OPIG) |
| **Paper** | Rives et al. PNAS 2021 | Li et al. bioRxiv 2023 | Outeiral & Deane, Nat Mach Intell 2024 |
| **Repo** | [facebookresearch/esm](https://github.com/facebookresearch/esm) | [Sanofi-Public/CodonBERT](https://github.com/Sanofi-Public/CodonBERT) | [oxpig/CaLM](https://github.com/oxpig/CaLM) |

For context: **DNABERT-2** uses BPE with 4,096 learned variable-length DNA tokens — a fundamentally different philosophy from the fixed-vocabulary approaches above.

---

## 1. ESM-2 (Protein Encoder)

### Vocabulary (33 tokens)

```
Index  Token       Description
─────  ─────       ───────────
0      <cls>       Classification / BOS
1      <pad>       Padding
2      <eos>       End of sequence
3      <unk>       Unknown
4-23   L A G V S   20 standard amino acids
       E R T I D
       P K Q N F
       Y M H W C
24     X           Any / unknown amino acid
25     B           Asparagine or Aspartic acid (N/D)
26     U           Selenocysteine
27     Z           Glutamine or Glutamic acid (Q/E)
28     O           Pyrrolysine
29     .           Gap (alignment)
30     -           Gap (alignment)
31     <null_1>    Padding to align vocab to multiple of 8
32     <mask>      Mask token for MLM
```

Construction: 4 prepend + 27 standard + 1 null padding + 1 append = **33**.

No `tokenizer.json` file — just `vocab.txt`, `special_tokens_map.json`, and `tokenizer_config.json` on HuggingFace.

### Model Config (esm2_t33_650M_UR50D)

```json
{
  "hidden_size": 1280,
  "num_hidden_layers": 33,
  "num_attention_heads": 20,
  "intermediate_size": 5120,
  "position_embedding_type": "rotary",
  "token_dropout": true,
  "vocab_size": 33,
  "max_position_embeddings": 1026
}
```

### Forward Output

```python
result = model(tokens, repr_layers=[33], return_contacts=True)
```

| Key | Shape | Description | When |
|-----|-------|-------------|------|
| `logits` | `(B, T, 33)` | Raw logits over vocab (→ softmax for probs) | Always |
| `representations` | `{layer: (B, T, 1280)}` | Hidden states at requested layers | Always (empty if `repr_layers=[]`) |
| `attentions` | `(B, 33, 20, T, T)` | Attention weights all layers/heads | `need_head_weights=True` |
| `contacts` | `(B, T_seq, T_seq)` | Contact prediction (sigmoid output) | `return_contacts=True` |

### HuggingFace Model Heads

| Class | Output |
|-------|--------|
| `EsmModel` | `last_hidden_state: (B, T, 1280)`, `pooler_output: (B, 1280)` |
| `EsmForMaskedLM` | `logits: (B, T, 33)` |
| `EsmForSequenceClassification` | `logits: (B, num_labels)` |
| `EsmForTokenClassification` | `logits: (B, T, num_labels)` |
| `EsmForProteinFolding` | `positions`, `plddt`, `ptm`, `distogram_logits`, etc. |

### Sample Usage

**Native ESM library:**

```python
import torch, esm

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()

data = [("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG")]
batch_labels, batch_strs, batch_tokens = batch_converter(data)

with torch.no_grad():
    results = model(batch_tokens, repr_layers=[33], return_contacts=True)

logits = results["logits"]                    # (B, T, 33) — raw logits
probs = torch.softmax(logits, dim=-1)         # (B, T, 33) — probabilities
reps = results["representations"][33]         # (B, T, 1280)
contacts = results["contacts"]               # (B, T_seq, T_seq)
```

**HuggingFace:**

```python
from transformers import AutoTokenizer, EsmForMaskedLM
import torch

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
model = EsmForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D")

inputs = tokenizer("MKTVRQERLK", return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits  # (1, T, 33)
```

---

## 2. CodonBERT (mRNA Encoder)

### Vocabulary

**Two versions exist in the repo — this is a known bug.**

#### Pretrain vocabulary (130 tokens) — used by `pretrain.py` and `finetune.py`

Alphabet: `AUGCN` (5 letters, N = ambiguous base)

```
5 special tokens: [PAD]=0, [UNK]=1, [CLS]=2, [SEP]=3, [MASK]=4
125 codons: all 5^3 trinucleotide combos from {A, U, G, C, N}
Total: 130 tokens
```

This is what the pretrained model was actually trained with.

#### Benchmark vocabulary (69 tokens) — used by `utils/tokenizer.py` and `extract_embed.py`

Alphabet: `AUGC` (4 letters, no N)

```
5 special tokens: [PAD]=0, [UNK]=1, [CLS]=2, [SEP]=3, [MASK]=4
64 codons: all 4^3 standard RNA codons
```

Index mapping (69 tokens):

```
 0: [PAD]    5: AAA     9: AUA    13: AGA    17: ACA
 1: [UNK]    6: AAU    10: AUU    14: AGU    18: ACU
 2: [CLS]    7: AAG    11: AUG    15: AGG    19: ACG
 3: [SEP]    8: AAC    12: AUC    16: AGC    20: ACC
 4: [MASK]

21: UAA    25: UUA    29: UGA    33: UCA
22: UAU    26: UUU    30: UGU    34: UCU
23: UAG    27: UUG    31: UGG    35: UCG
24: UAC    28: UUC    32: UGC    36: UCC

37: GAA    41: GUA    45: GGA    49: GCA
38: GAU    42: GUU    46: GGU    50: GCU
39: GAG    43: GUG    47: GGG    51: GCG
40: GAC    44: GUC    48: GGC    52: GCC

53: CAA    57: CUA    61: CGA    65: CCA
54: CAU    58: CUU    62: CGU    66: CCU
55: CAG    59: CUG    63: CGG    67: CCG
56: CAC    60: CUC    64: CGC    68: CCC
```

### Vocabulary Mismatch Bug

The pretrained model uses 130 tokens (AUGCN), but the benchmark tokenizer builds 69 tokens (AUGC). In the AUGCN generation, N-containing codons are **interleaved** (e.g., after `AAC` at index 8 comes `AAN` at index 9, shifting all subsequent indices). Using the 69-token tokenizer with the 130-token checkpoint produces **misaligned embeddings**.

### Model Config

```
Architecture: HuggingFace BertForPreTraining
Hidden size: 768
Intermediate size: 3072
Attention heads: 12
Layers: 12
Max positions: 1024
Vocab size: 130 (pretrain)
```

### Forward Output

**Pre-training (`BertForPreTraining`):**

| Key | Shape | Description |
|-----|-------|-------------|
| `prediction_logits` | `(B, 1024, 130)` | MLM logits over vocab |
| `seq_relationship_logits` | `(B, 2)` | Next Sentence Prediction logits |
| `hidden_states` | tuple of `(B, T, 768)` | Per-layer hidden states (opt) |

**Fine-tuning (`BertForSequenceClassification`):**

| Key | Shape | Description |
|-----|-------|-------------|
| `logits` | `(B, num_labels)` | Classification logits |

**Embedding extraction:**

```python
model = BertForPreTraining.from_pretrained(model_dir)
outputs = model(input_ids, labels=input_ids, output_hidden_states=True)
hidden_states = outputs[2]
embeddings = hidden_states[-1].squeeze()[1:-1]  # strip [CLS]/[SEP], shape (num_codons, 768)
```

### Tokenization Code

```python
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import BertProcessing

def mytok(seq, kmer_len, s):
    seq = seq.upper().replace("T", "U")
    kmer_list = []
    for j in range(0, (len(seq) - kmer_len) + 1, s):
        kmer_list.append(seq[j : j + kmer_len])
    return kmer_list

# Build vocab (AUGCN for pretrain, AUGC for benchmarks)
lst_ele = list('AUGCN')  # or 'AUGC'
lst_voc = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
for a1 in lst_ele:
    for a2 in lst_ele:
        for a3 in lst_ele:
            lst_voc.extend([f'{a1}{a2}{a3}'])
dic_voc = dict(zip(lst_voc, range(len(lst_voc))))

tokenizer = Tokenizer(WordLevel(vocab=dic_voc, unk_token="[UNK]"))
tokenizer.add_special_tokens(['[PAD]', '[CLS]', '[UNK]', '[SEP]', '[MASK]'])
tokenizer.pre_tokenizer = Whitespace()
tokenizer.post_processor = BertProcessing(("[SEP]", dic_voc['[SEP]']), ("[CLS]", dic_voc['[CLS]']))

# Usage: DNA -> codon list -> space-separated string -> tokenize
codons = mytok("ATGGCTAGCTTAAGC", 3, 3)   # ['AUG', 'GCU', 'AGC', 'UUA', 'AGC']
input_str = " ".join(codons)                # "AUG GCU AGC UUA AGC"
```

### Key Files

| File | Purpose |
|------|---------|
| `benchmarks/utils/tokenizer.py` | Shared tokenizer (AUGC, 69 tokens) |
| `benchmarks/CodonBERT/pretrain.py` | Pre-training (AUGCN, 130 tokens) |
| `benchmarks/CodonBERT/finetune.py` | Fine-tuning with LoRA support |
| `benchmarks/CodonBERT/extract_embed.py` | Embedding extraction |

No HuggingFace model page. Weights via [Sanofi CDN zip](https://cdn.prod.accelerator.sanofi/llm/CodonBERT.zip).

---

## 3. CaLM (Codon Adaptation Language Model)

### Vocabulary (69 tokens)

Defined in `calm/alphabet.py`. Uses `CodonModel` architecture config.

```
Index  Token     Index  Token     Index  Token     Index  Token
─────  ─────     ─────  ─────     ─────  ─────     ─────  ─────
 0     <cls>      4     AAA        8     AUA       12     ACA
 1     <pad>      5     AAU        9     AUU       13     ACU
 2     <eos>      6     AAC       10     AUC       14     ACC
 3     <unk>      7     AAG       11     AUG       15     ACG

16     AGA       20     UAA*      24     UUA       28     UCA
17     AGU       21     UAU       25     UUU       29     UCU
18     AGC       22     UAC       26     UUC       30     UCC
19     AGG       23     UAG*      27     UUG       31     UCG

32     UGA*      36     CAA       40     CUA       44     CCA
33     UGU       37     CAU       41     CUU       45     CCU
34     UGC       38     CAC       42     CUC       46     CCC
35     UGG       39     CAG       43     CUG       47     CCG

48     CGA       52     GAA       56     GUA       60     GCA
49     CGU       53     GAU       57     GUU       61     GCU
50     CGC       54     GAC       58     GUC       62     GCC
51     CGG       55     GAG       59     GUG       63     GCG

64     GGA       68     <mask>
65     GGU
66     GGC
67     GGG
```

`*` = stop codons (UAA, UAG, UGA)

Construction: 4 prepend (`<cls>`, `<pad>`, `<eos>`, `<unk>`) + 64 codons + 1 append (`<mask>`) = **69**.

No `tokenizer.json` files — all in Python code.

### Model Config

```python
{
    'num_layers': 12,
    'embed_dim': 768,
    'ffn_embed_dim': 3072,
    'attention_heads': 12,
    'max_positions': 1024,
    'rope_embedding': True,      # RoPE, not learned positions
    'mask_proportion': 0.25,     # 25% masking (vs 15% for BERT/ESM)
}
```

### Forward Output

```python
result = model(tokens, repr_layers=[12], need_head_weights=True)
```

| Key | Shape | Description | When |
|-----|-------|-------------|------|
| `logits` | `(B, T, 69)` | Raw logits over 69-token codon vocab | Always |
| `representations` | `{layer: (B, T, 768)}` | Hidden states at requested layers | Always (empty if `repr_layers=[]`) |
| `attentions` | `(B, 12, 12, T, T)` | Attention weights | `need_head_weights=True` |

### User-Facing API

```python
from calm import CaLM

model = CaLM()

# Averaged embedding — one vector per sequence
vec = model.embed_sequence("ATGGCGCTAAAGCGGATC", average=True)   # (1, 768)

# Per-codon embeddings
vecs = model.embed_sequence("ATGGCGCTAAAGCGGATC", average=False) # (1, T, 768)
# T = num_codons + 2 (includes <cls> and <eos>)

# Raw model forward
from calm.sequence import CodonSequence
seq = CodonSequence("ATGGCGCTAAAGCGGATC")
tokens = model.tokenize(seq)
output = model(tokens)
logits = output["logits"]                        # (1, T, 69)
probs = torch.softmax(logits, dim=-1)            # (1, T, 69) — codon probabilities
```

DNA input is auto-converted to RNA (T->U) by `CodonSequence`.

### Training Details

- **MLM**: 25% masking (80% mask, 10% unchanged, 10% random)
- **Loss**: CrossEntropyLoss over 69 tokens
- **Optimizer**: AdamW, lr=4e-4, weight_decay=0.1
- **Schedule**: warmup_cosine (1000 warmup steps, 121000 total)
- **LM Head**: RobertaLMHead with **tied weights** (shares embedding matrix)

### Key Files

| File | Purpose |
|------|---------|
| `calm/alphabet.py` | 69-token vocabulary, tokenizer, batch converter |
| `calm/sequence.py` | `CodonSequence` — DNA->RNA codon splitting |
| `calm/model.py` | `ProteinBertModel` — 12-layer transformer (ESM-1b fork) |
| `calm/pretrained.py` | `CaLM` class — user API, weight download |
| `calm/pipeline.py` | Training data pipeline (masking) |
| `training.py` | Training loop |

---

## Tokenization Philosophy

Why these models tokenize differently:

| Model | Unit | Why |
|-------|------|-----|
| **ESM-2** | Single amino acid (20+) | High info per token — each AA is a distinct chemical entity. Character-level is natural. |
| **CodonBERT** | Single codon (64+) | The genetic code defines codons as the biological "word". Codon usage bias is the signal. |
| **CaLM** | Single codon (64) | Same as CodonBERT — codons are the natural tokenization unit for coding sequences. |
| **DNABERT-2** | BPE k-mers (4096) | Only 4 bases -> very low info per character. BPE learns variable-length patterns to compress. |

The codon-level models (CodonBERT, CaLM) and protein-level model (ESM-2) don't need BPE because their base alphabets already carry enough information per token. DNA models need BPE because `{A,T,G,C}` alone is too low-entropy per character.

---

## Relevance to ProteinQC

For the RL/MCTS approach:

1. **ESM-2 logits** -> per-position amino acid probability as a protein quality signal
2. **CaLM/CodonBERT logits** -> per-codon probability as codon optimality signal
3. **ESM-2 embeddings** -> protein-level feature vectors for scoring
4. **CaLM embeddings** -> codon-level feature vectors capturing codon usage bias
5. **ESM-2 contacts** -> structural plausibility signal
6. All three return **logits** (not probabilities) — apply `softmax(logits, dim=-1)` for probabilities
