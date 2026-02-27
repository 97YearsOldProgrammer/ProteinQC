#!/usr/bin/env python3
"""Multi-species CaLM benchmark on independent datasets.

Tests CaLM on FASTA datasets from 24 published RNA classification tools
covering 30+ species — completely independent from the RNA Challenge training set.

Tests two modes (one forward pass, two pooling strategies):
  1. Fresh MLP on [CLS] embedding (768→256→256→1)
  2. Fresh MLP on mean-pooled embedding (768→256→256→1)

Usage:
    python benchmark_multispecies.py
    python benchmark_multispecies.py --max-seqs 2000  # cap per dataset
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from proteinqc.data.tokenizer import CodonTokenizer
from proteinqc.models.calm_encoder import CaLMEncoder
from proteinqc.models.classification_heads import MLPHead
from proteinqc.tools.codon_table import CodonTableManager
from proteinqc.tools.orf_scanner import ORFScanner

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "calm"
BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
TOKEN_BUDGET = 8_192
BATCH_EMBED = 16
MAX_CODONS = 1024  # CaLM max: 1026 tokens = 1024 codons + [CLS] + [SEP]


def parse_args():
    p = argparse.ArgumentParser(description="Multi-species CaLM benchmark")
    p.add_argument("--max-seqs", type=int, default=0,
                   help="Max sequences per dataset (0=all)")
    p.add_argument("--align", choices=["first-atg", "longest-orf"],
                   default="first-atg",
                   help="Alignment strategy (default: first-atg)")
    return p.parse_args()


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def codon_align(seq: str) -> str:
    seq = seq.upper().replace("U", "T")
    seq = re.sub(r"[^ACGT]", "", seq)
    atg_pos = seq.find("ATG")
    if atg_pos >= 0:
        seq = seq[atg_pos:]
    trim = len(seq) - (len(seq) % 3)
    seq = seq[:trim]
    if len(seq) > MAX_CODONS * 3:
        seq = seq[:MAX_CODONS * 3]
    return seq


# Module-level scanner, initialized lazily in main()
_orf_scanner: ORFScanner | None = None


def codon_align_longest_orf(seq: str) -> str:
    """Align to longest ORF, fallback to first-ATG."""
    seq = seq.upper().replace("U", "T")
    seq = re.sub(r"[^ACGT]", "", seq)
    if _orf_scanner is not None:
        candidates = _orf_scanner.scan(seq)
        if candidates:
            orf_seq = candidates[0].seq  # longest ORF
            if len(orf_seq) > MAX_CODONS * 3:
                orf_seq = orf_seq[:MAX_CODONS * 3]
            return orf_seq
    # Fallback: first-ATG
    atg_pos = seq.find("ATG")
    if atg_pos >= 0:
        seq = seq[atg_pos:]
    trim = len(seq) - (len(seq) % 3)
    seq = seq[:trim]
    if len(seq) > MAX_CODONS * 3:
        seq = seq[:MAX_CODONS * 3]
    return seq


def read_fasta(path: Path, max_seqs: int = 0) -> list[tuple[str, str]]:
    """Read FASTA file, return list of (header, sequence)."""
    records = []
    header = ""
    seq_parts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header and seq_parts:
                    records.append((header, "".join(seq_parts)))
                    if max_seqs and len(records) >= max_seqs:
                        return records
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line)
    if header and seq_parts:
        records.append((header, "".join(seq_parts)))
    return records


def _add_pair(datasets: list, tool: str, species: str, d: Path,
              coding: str, noncoding: str):
    """Helper: add a coding/noncoding pair if both files exist."""
    c, nc = d / coding, d / noncoding
    if c.exists() and nc.exists():
        datasets.append({"tool": tool, "species": species,
                         "coding": c, "noncoding": nc})


def discover_datasets(benchmark_dir: Path) -> list[dict]:
    """Auto-discover coding/noncoding FASTA pairs from benchmark directory."""
    datasets = []

    # --- CPAT ---
    cpat = benchmark_dir / "cpat"
    if cpat.exists():
        _add_pair(datasets, "CPAT", "Human (test)", cpat,
                  "Human_test_coding_mRNA.fa", "Human_test_noncoding_RNA.fa")
        _add_pair(datasets, "CPAT", "Human (train)", cpat,
                  "Human_coding_transcripts_mRNA.fa",
                  "Human_noncoding_transcripts_RNA.fa")

    # --- NCResNet (8 species, long+short) ---
    ncresnet = benchmark_dir / "ncresnet"
    if ncresnet.exists():
        species_map = {
            "0human": "Human", "1mouse": "Mouse",
            "2s.cerevisiae": "S. cerevisiae", "3zebrafish": "Zebrafish",
            "4fruitfly": "Fruitfly", "5cow": "Cow",
            "6rat": "Rat", "7c.elegans": "C. elegans",
        }
        for prefix, species in species_map.items():
            for length in ["long", "short"]:
                # Original 5 use .fa, extras use .fasta and _test suffix
                for ext, suffix in [(".fa", ""), (".fasta", "_test")]:
                    pc = ncresnet / f"{prefix}_{length}_pc{suffix}{ext}"
                    nc = ncresnet / f"{prefix}_{length}_nc{suffix}{ext}"
                    if pc.exists() and nc.exists():
                        datasets.append({
                            "tool": "NCResNet", "species": f"{species} ({length})",
                            "coding": pc, "noncoding": nc,
                        })
                        break

    # --- LncFinder (6 species, nested: Species/Species/*.fa) ---
    lncfinder = benchmark_dir / "lncfinder"
    if lncfinder.exists():
        for species_dir in sorted(lncfinder.iterdir()):
            if not species_dir.is_dir():
                continue
            all_files = list(species_dir.rglob("*.fa")) + list(
                species_dir.rglob("*.fasta"))
            coding_files = [f for f in all_files if f.stem.lower().startswith("pct")]
            noncoding_files = [f for f in all_files if f.stem.lower().startswith("lnc")]
            if coding_files and noncoding_files:
                datasets.append({
                    "tool": "LncFinder", "species": species_dir.name,
                    "coding": coding_files,
                    "noncoding": noncoding_files,
                })

    # --- longdist (Mouse + Human) ---
    longdist = benchmark_dir / "longdist"
    if longdist.exists():
        _add_pair(datasets, "longdist", "Mouse", longdist,
                  "GRCm38.pct.fa", "GRCm38.lncRNA.fa")
        _add_pair(datasets, "longdist", "Human", longdist,
                  "Homo_sapiens.GRCh38.cdna.all.fa",
                  "Homo_sapiens.GRCh38.ncrna.fa")

    # --- LncRNA-Mdeep (human) ---
    _add_pair(datasets, "LncRNA-Mdeep", "Human", benchmark_dir / "lncrna_mdeep",
              "human_PCT_test.fa", "human_lncRNA_test.fa")

    # --- DeepCPP (human sORF) ---
    _add_pair(datasets, "DeepCPP", "Human (sORF)", benchmark_dir / "deepcpp",
              "human_mrnasorf.fa", "human_lncsorf.fa")

    # --- LncADeep (human, mixed file — split by header) ---
    lncadeep = benchmark_dir / "lncadeep"
    if lncadeep.exists():
        mixed_file = lncadeep / "lncRNA_mRNA_test.fa"
        if mixed_file.exists():
            datasets.append({"tool": "LncADeep", "species": "Human",
                             "mixed": mixed_file})

    # --- RNAplonc (plants) ---
    rnaplonc = benchmark_dir / "rnaplonc"
    if rnaplonc.exists():
        coding_dir = rnaplonc / "coding"
        noncoding_dir = rnaplonc / "noncoding"
        if coding_dir.exists() and noncoding_dir.exists():
            species_names = {
                "atrichopoda": "Amborella", "bdistachyon": "Brachypodium",
                "csinensis": "Citrus", "mesculenta": "Cassava",
                "rcommunis": "Ricinus", "sbicolor": "Sorghum",
                "suberosum": "Solanum", "zmays": "Maize",
            }
            noncoding_names = {
                "amborella": "atrichopoda", "brachypodium": "bdistachyon",
                "citrus": "csinensis", "manihot": "mesculenta",
                "ricinus": "rcommunis", "sorghum": "sbicolor",
                "solanum": "suberosum", "zea": "zmays",
            }
            for nc_file in sorted(noncoding_dir.glob("*.fasta")):
                nc_name = nc_file.stem.lower()
                if nc_name in noncoding_names:
                    coding_key = noncoding_names[nc_name]
                    c_file = coding_dir / f"{coding_key}.fasta"
                    if c_file.exists():
                        display = species_names.get(coding_key, coding_key)
                        datasets.append({
                            "tool": "RNAplonc", "species": display,
                            "coding": c_file, "noncoding": nc_file,
                        })

    # --- LncRNAnet ---
    lncnet = benchmark_dir / "lncRNAnet"
    if lncnet.exists():
        _add_pair(datasets, "LncRNAnet", "Human", lncnet, "HT.fasta", "MT.fasta")

    # --- MRNN ---
    mrnn = benchmark_dir / "mrnn"
    if mrnn.exists():
        _add_pair(datasets, "mRNN", "Human (train)", mrnn,
                  "mRNAs.TRAIN.fa", "lncRNAs.TRAIN.fa")
        _add_pair(datasets, "mRNN", "Human (test)", mrnn,
                  "mRNAs.TEST.fa", "lncRNAs.TEST.fa")
        _add_pair(datasets, "mRNN", "Mouse (test)", mrnn,
                  "mRNAs.MOUSETEST.fa", "lncRNAs.MOUSETEST.fa")

    # --- RNAC ---
    rnac = benchmark_dir / "rnac"
    if rnac.exists():
        for sp, label in [("Human", "Human"), ("Mouse", "Mouse"),
                          ("Arabidopsis_thaliana", "Arabidopsis"),
                          ("Celegans", "C. elegans")]:
            _add_pair(datasets, "RNAC", label, rnac,
                      f"{sp}_mRNA.fa", f"{sp}_ncRNA.fa")

    # --- PLEK ---
    _add_pair(datasets, "PLEK", "Human", benchmark_dir / "plek",
              "plek2_mrna.fa", "plek2_lncrna.fa")

    # --- CPPred (6 species + integrated) ---
    cppred = benchmark_dir / "cppred"
    if cppred.exists():
        _add_pair(datasets, "CPPred", "Human (train)", cppred,
                  "Human.coding_RNA_training.fa", "Homo38.ncrna_training.fa")
        _add_pair(datasets, "CPPred", "Human (test)", cppred,
                  "Human_coding_RNA_test.fa", "Homo38_ncrna_test.fa")
        _add_pair(datasets, "CPPred", "Mouse", cppred,
                  "Mouse_coding_RNA.fa", "Mouse_ncrna.fa")
        _add_pair(datasets, "CPPred", "Zebrafish", cppred,
                  "Zebrafish_coding_RNA.fa", "Zebrafish_ncrna.fa")
        _add_pair(datasets, "CPPred", "Fruit fly", cppred,
                  "Fruit_fly_coding_RNA.fa", "Fruit_fly_ncrna.fa")
        _add_pair(datasets, "CPPred", "S. cerevisiae", cppred,
                  "S.cerevisiae_coding_RNA.fa", "S.cerevisiae_ncrna.fa")
        _add_pair(datasets, "CPPred", "Integrated (train)", cppred,
                  "Integrated.coding_RNA_training.fa",
                  "Integrated.ncrna_training.fa")
        _add_pair(datasets, "CPPred", "Integrated (test)", cppred,
                  "Integrated_coding_RNA_test.fa", "Integrated_ncrna_test.fa")

    # --- PLncPRO (Human, Mouse, 10+ plant species) ---
    plncpro = benchmark_dir / "plncpro" / "plncpro_data"
    if plncpro.exists():
        # Human (hg24)
        _add_pair(datasets, "PLncPRO", "Human", plncpro / "hg24" / "train",
                  "hg24_pct_train_5000.fa", "hg24_lnct_train_5000.fa")
        # Mouse (mm8)
        _add_pair(datasets, "PLncPRO", "Mouse", plncpro / "mm8" / "train",
                  "m8_pct_train_2500.fa", "m8_lnct_train_2500.fa")
        # Plant species
        plant_dir = plncpro / "plant_new_fasta"
        plant_names = {
            "amt": "Amborella", "at": "Arabidopsis", "cr": "Chlamydomonas",
            "gm": "Soybean", "os": "Rice", "pp": "Physcomitrella",
            "sm": "Selaginella", "st": "Potato", "vv": "Grape", "zm": "Maize",
        }
        for abbr, name in plant_names.items():
            train = plant_dir / abbr / "train"
            _add_pair(datasets, "PLncPRO", name, train,
                      f"{abbr}_pct_train.fa", f"{abbr}_lnct_train.fa")
        # Multi-species groups
        _add_pair(datasets, "PLncPRO", "Dicot (combined)",
                  plant_dir / "dicot_no_st" / "train",
                  "pct_train.fa", "lnct_train.fa")
        _add_pair(datasets, "PLncPRO", "Monocot (combined)",
                  plant_dir / "monocot" / "train",
                  "monocot_pct_train.fa", "monocot_lnct_train.fa")

    # --- LGC (6 species + RefSeq taxonomic groups) ---
    lgc = benchmark_dir / "lgc"
    if lgc.exists():
        lgc6 = lgc / "sequence-6species"
        if lgc6.exists():
            for sp_file, label in [
                ("Homo-sapiens", "Human"), ("Mus-musculus", "Mouse"),
                ("Danio-rerio", "Zebrafish"),
                ("Caenorhaditis elegans", "C. elegans"),
                ("Oryza-sativa", "Rice"),
                ("Solanum-lycopersicum", "Tomato"),
            ]:
                _add_pair(datasets, "LGC", label, lgc6,
                          f"{sp_file}_cd.fa", f"{sp_file}_lnc.fa")
        # RefSeq taxonomic groups (NM=coding, NR=noncoding)
        for abbr, label in [("ma", "Mammals"), ("ve", "Vertebrates"),
                            ("inve", "Invertebrates"), ("pl", "Plants")]:
            _add_pair(datasets, "LGC-RefSeq", label, lgc,
                      f"{abbr}.NM.fasta", f"{abbr}.NR.fasta")

    # --- CREMA (multi-species) ---
    crema = benchmark_dir / "crema"
    if crema.exists():
        lnc = crema / "all_lncRNA_nodup.fa"
        if lnc.exists():
            for sp_file, label in [
                ("h_sapiens_random3000.fa", "Human"),
                ("a_thaliana_random3000.fa", "Arabidopsis"),
                ("o_sativa_random3000.fa", "Rice"),
            ]:
                c = crema / sp_file
                if c.exists():
                    datasets.append({"tool": "CREMA", "species": label,
                                     "coding": c, "noncoding": lnc})

    # --- FEELnc (Human + Mouse, GENCODE) ---
    feelnc = benchmark_dir / "feelnc"
    if feelnc.exists():
        _add_pair(datasets, "FEELnc", "Human (GENCODEv24)", feelnc,
                  "gencode.v24.pc_transcripts.fa",
                  "gencode.v24.lncRNA_transcripts.fa")
        _add_pair(datasets, "FEELnc", "Mouse (GENCODEvM4)", feelnc,
                  "gencode.vM4.pc_transcripts.fa",
                  "gencode.vM4.lncRNA_transcripts.fa")

    # --- RNAsamba (Human, GENCODE v32) ---
    _add_pair(datasets, "RNAsamba", "Human (GENCODEv32)",
              benchmark_dir / "rnasamba",
              "gencode.v32.pc_transcripts.fa",
              "gencode.v32.lncRNA_transcripts.fa")

    # --- PreLnc (Human, GENCODE v29) ---
    _add_pair(datasets, "PreLnc", "Human (GENCODEv29)",
              benchmark_dir / "prelnc",
              "gencode.v29.pc_transcripts.fa",
              "gencode.v29.lncRNA_transcripts.fa")

    return datasets


def extract_embeddings(
    sequences: list[str],
    encoder: CaLMEncoder,
    tokenizer: CodonTokenizer,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract [CLS] and mean-pooled embeddings with adaptive batching.

    Returns:
        (cls_embeddings, mean_embeddings) each [N, hidden_size]
    """
    n = len(sequences)
    cls_emb = torch.zeros(n, encoder.hidden_size, dtype=torch.float32)
    mean_emb = torch.zeros(n, encoder.hidden_size, dtype=torch.float32)
    sorted_indices = sorted(range(n), key=lambda i: len(sequences[i]))

    i = 0
    while i < n:
        max_codons = len(sequences[sorted_indices[i]]) // 3 + 2
        adaptive_bs = max(1, TOKEN_BUDGET // max_codons)
        adaptive_bs = min(adaptive_bs, BATCH_EMBED, n - i)

        batch_idx = sorted_indices[i: i + adaptive_bs]
        batch_seqs = [sequences[j] for j in batch_idx]
        encoded = tokenizer.batch_encode(batch_seqs, device=device)

        with torch.no_grad():
            # Single _encode call, derive both [CLS] and mean-pool
            full = encoder._encode(encoded["input_ids"], encoded["attention_mask"])
            cls_batch = full[:, 0, :].cpu()
            mask = encoded["attention_mask"].unsqueeze(-1).to(full.dtype)
            mean_batch = ((full * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)).cpu()

        for j, orig_idx in enumerate(batch_idx):
            cls_emb[orig_idx] = cls_batch[j]
            mean_emb[orig_idx] = mean_batch[j]

        if device.type == "mps":
            torch.mps.empty_cache()

        i += adaptive_bs

    return cls_emb, mean_emb


def compute_metrics(tp: int, tn: int, fp: int, fn: int) -> dict:
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total > 0 else 0.0
    pre = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * pre * rec / (pre + rec) if (pre + rec) > 0 else 0.0
    denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = ((tp * tn) - (fp * fn)) / denom if denom > 0 else 0.0
    return {
        "ACC": acc * 100, "PRE": pre * 100, "REC": rec * 100,
        "F1": f1 * 100, "MCC": mcc * 100,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "total": total,
    }


def _confusion_metrics(preds: torch.Tensor, labels: torch.Tensor) -> dict:
    """Compute metrics from predictions and labels tensors."""
    tp = ((preds == 1) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    return compute_metrics(tp, tn, fp, fn)


def _train_and_eval_head(
    head: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    epochs: int = 100,
    lr: float = 1e-3,
) -> dict:
    """Train a fresh classification head on train split, evaluate on test."""
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    head.train()
    for _ in range(epochs):
        logits = head(train_x).squeeze(-1)
        loss = criterion(logits, train_y)
        if hasattr(head, "get_balance_loss"):
            loss = loss + head.get_balance_loss()
        opt.zero_grad()
        loss.backward()
        opt.step()

    head.eval()
    with torch.no_grad():
        test_logits = head(test_x).squeeze(-1)
        test_preds = (torch.sigmoid(test_logits) > 0.5).long()
    return _confusion_metrics(test_preds, test_y.long())


def read_fasta_multi(paths, max_seqs: int = 0) -> list[tuple[str, str]]:
    """Read from a single Path or a list of Paths."""
    if isinstance(paths, (str, Path)):
        return read_fasta(Path(paths), max_seqs)
    records = []
    for p in paths:
        records.extend(read_fasta(Path(p), max_seqs=0))
        if max_seqs and len(records) >= max_seqs:
            return records[:max_seqs]
    return records


def run_on_dataset(
    ds: dict,
    encoder: CaLMEncoder,
    tokenizer: CodonTokenizer,
    device: torch.device,
    max_seqs: int = 0,
    align_fn=codon_align,
) -> dict:
    """Run CaLM on one dataset: fresh MLP on [CLS] and mean-pooled embeddings."""
    half = max_seqs // 2 if max_seqs else 0

    if "mixed" in ds:
        all_records = read_fasta(ds["mixed"], max_seqs=0)
        coding_records = [(h, s) for h, s in all_records
                          if "NM_" in h or "mRNA" in h.lower()]
        noncoding_records = [(h, s) for h, s in all_records
                             if h not in {r[0] for r in coding_records}]
        if half:
            coding_records = coding_records[:half]
            noncoding_records = noncoding_records[:half]
    else:
        coding_records = read_fasta_multi(ds["coding"], max_seqs=half)
        noncoding_records = read_fasta_multi(ds["noncoding"], max_seqs=half)

    sequences, labels = [], []
    for _, seq in coding_records:
        aligned = align_fn(seq)
        if len(aligned) >= 9:
            sequences.append(aligned)
            labels.append(1)
    for _, seq in noncoding_records:
        aligned = align_fn(seq)
        if len(aligned) >= 9:
            sequences.append(aligned)
            labels.append(0)

    if len(sequences) < 10:
        return {"error": f"Too few sequences ({len(sequences)})"}

    n_coding = sum(labels)
    n_noncoding = len(labels) - n_coding

    # One forward pass → both [CLS] and mean-pooled embeddings
    cls_emb, mean_emb = extract_embeddings(sequences, encoder, tokenizer, device)
    labels_t = torch.tensor(labels, dtype=torch.long)

    # 80/20 split
    n = len(sequences)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(42))
    n_train = int(n * 0.8)
    train_idx, test_idx = perm[:n_train], perm[n_train:]
    train_y = labels_t[train_idx].float()
    test_y = labels_t[test_idx]

    cls_train, cls_test = cls_emb[train_idx], cls_emb[test_idx]
    mean_train, mean_test = mean_emb[train_idx], mean_emb[test_idx]

    # Mode 1: Fresh MLP on [CLS]
    mlp_cls = _train_and_eval_head(
        MLPHead(hidden_size=768, mlp_hidden=256, dropout=0.1),
        cls_train, train_y, cls_test, test_y,
    )

    # Mode 2: Fresh MLP on mean-pooled
    mlp_mean = _train_and_eval_head(
        MLPHead(hidden_size=768, mlp_hidden=256, dropout=0.1),
        mean_train, train_y, mean_test, test_y,
    )

    return {
        "tool": ds["tool"],
        "species": ds["species"],
        "n_coding": n_coding,
        "n_noncoding": n_noncoding,
        "n_total": len(sequences),
        "mlp_cls": mlp_cls,
        "mlp_mean": mlp_mean,
    }


def main():
    args = parse_args()
    torch.manual_seed(42)
    np.random.seed(42)

    device = select_device()
    align_mode = args.align
    align_fn = codon_align
    if align_mode == "longest-orf":
        global _orf_scanner
        ctm = CodonTableManager()
        _orf_scanner = ORFScanner(ctm.get_genetic_code(1), min_codons=30)
        align_fn = codon_align_longest_orf

    print("=" * 70)
    print("  Multi-Species CaLM Benchmark -- align=" + align_mode)
    print("=" * 70)
    print("Device: " + str(device))
    print("Alignment: " + align_mode)
    if args.max_seqs:
        print("Max seqs per dataset: " + str(args.max_seqs))

    print("\nLoading CaLM encoder from " + str(MODEL_DIR) + " ...")
    encoder = CaLMEncoder(MODEL_DIR, freeze=True).to(device)
    encoder.eval()
    tokenizer = CodonTokenizer(MODEL_DIR / "vocab.txt")
    n_params = sum(p.numel() for p in encoder.parameters())
    print("  " + f"{n_params:,}" + " params (frozen)")

    datasets = discover_datasets(BENCHMARK_DIR)
    print(f"\nDiscovered {len(datasets)} dataset pairs:")
    for ds in datasets:
        print(f"  {ds['tool']:15s} | {ds['species']}")

    all_results = []
    for i, ds in enumerate(datasets):
        if "mixed" in ds:
            if not ds["mixed"].exists():
                print(f"\n[{i+1}/{len(datasets)}] SKIP {ds['tool']} {ds['species']} -- file missing")
                continue
        else:
            coding_paths = ds["coding"] if isinstance(ds["coding"], list) else [ds["coding"]]
            noncoding_paths = ds["noncoding"] if isinstance(ds["noncoding"], list) else [ds["noncoding"]]
            missing = [p for p in coding_paths + noncoding_paths if not Path(p).exists()]
            if missing:
                print(f"\n[{i+1}/{len(datasets)}] SKIP {ds['tool']} {ds['species']} -- {len(missing)} files missing")
                continue

        print(f"\n[{i+1}/{len(datasets)}] {ds['tool']} | {ds['species']}")
        t0 = time.time()
        result = run_on_dataset(ds, encoder, tokenizer, device, args.max_seqs, align_fn)
        elapsed = time.time() - t0

        if "error" in result:
            print(f"  ERROR: {result['error']}")
            continue

        mc = result["mlp_cls"]
        mm = result["mlp_mean"]
        print(f"  {result['n_total']:>6} seqs ({result['n_coding']} coding, {result['n_noncoding']} nc)")
        print(f"  MLP [CLS]:      ACC={mc['ACC']:.1f}%  F1={mc['F1']:.1f}%  MCC={mc['MCC']:.1f}%")
        print(f"  MLP [MeanPool]: ACC={mm['ACC']:.1f}%  F1={mm['F1']:.1f}%  MCC={mm['MCC']:.1f}%")
        print(f"  ({elapsed:.1f}s)")

        result["time_sec"] = elapsed
        all_results.append(result)

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        p_suffix = "_longest_orf" if align_mode == "longest-orf" else ""
        partial_path = RESULTS_DIR / f"benchmark_multispecies{p_suffix}_partial.json"
        with open(partial_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # Summary table
    w = 120
    print("\n" + "=" * w)
    hdr = (f"{'Tool':<15} {'Species':<22} {'N':>7}"
           f"  {'CLS ACC':>8} {'CLS MCC':>8}"
           f"  {'TP':>5} {'TN':>5} {'FP':>5} {'FN':>5}"
           f"  {'Mean ACC':>8} {'Mean MCC':>9}")
    print(hdr)
    print("-" * w)
    for r in all_results:
        mc = r["mlp_cls"]
        row = (f"{r['tool']:<15} {r['species']:<22} {r['n_total']:>7}"
               f"  {mc['ACC']:>7.1f}% {mc['MCC']:>7.1f}%"
               f"  {mc['TP']:>5} {mc['TN']:>5} {mc['FP']:>5} {mc['FN']:>5}"
               f"  {r['mlp_mean']['ACC']:>7.1f}% {r['mlp_mean']['MCC']:>8.1f}%")
        print(row)

    if all_results:
        print("-" * w)
        cls_acc = np.mean([r["mlp_cls"]["ACC"] for r in all_results])
        cls_mcc = np.mean([r["mlp_cls"]["MCC"] for r in all_results])
        mean_acc = np.mean([r["mlp_mean"]["ACC"] for r in all_results])
        mean_mcc = np.mean([r["mlp_mean"]["MCC"] for r in all_results])
        print(f"{'MEAN':<15} {'':<22} {'':<7}"
              f"  {cls_acc:>7.1f}% {cls_mcc:>7.1f}%"
              f"  {mean_acc:>7.1f}% {mean_mcc:>8.1f}%")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_longest_orf" if align_mode == "longest-orf" else ""
    out_path = RESULTS_DIR / f"benchmark_multispecies{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\nResults saved to " + str(out_path))
    total_time = sum(r.get("time_sec", 0) for r in all_results)
    print(f"Total time: {total_time:.0f}s")


if __name__ == "__main__":
    main()
