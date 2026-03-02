#!/usr/bin/env python3
"""Add Pfam annotations to existing feature parquets.

Reads raw sequences from FASTAs/JSONL, ORF-scans for proteins,
runs pyhmmer hmmsearch against Pfam-A, and updates parquets in-place.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from proteinqc.tools.orf_scanner import ORFScanner
from proteinqc.tools.pfam_scanner import PfamScanner

PFAM_DB = _ROOT / "models" / "pfam" / "Pfam-A.hmm"
FEATURES_DIR = _ROOT / "data" / "features"
BENCHMARK_DIR = _ROOT / "data" / "benchmark" / "ncresnet"
JSONL_PATH = _ROOT / "data" / "rnachallenge" / "evidence_baked.jsonl"

# Dataset name -> (pc_fasta, nc_fasta) mapping
NCRESNET_DATASETS = {
    "ncresnet/0human_short": ("0human_short_pc.fa", "0human_short_nc.fa"),
    "ncresnet/1mouse_short": ("1mouse_short_pc.fa", "1mouse_short_nc.fa"),
    "ncresnet/2s.cerevisiae_short": ("2s.cerevisiae_short_pc.fa", "2s.cerevisiae_short_nc.fa"),
    "ncresnet/3zebrafish_short": ("3zebrafish_short_pc.fa", "3zebrafish_short_nc.fa"),
    "ncresnet/4fruitfly_short": ("4fruitfly_short_pc.fasta", "4fruitfly_short_nc.fasta"),
    "ncresnet/5cow_short": ("5cow_short_pc_test.fasta", "5cow_short_nc_test.fasta"),
    "ncresnet/6rat_short": ("6rat_short_pc_test.fasta", "6rat_short_nc_test.fasta"),
    "ncresnet/7c.elegans_short": ("7c.elegans_short_pc_test.fasta", "7c.elegans_short_nc_test.fasta"),
}

PARQUET_FILES = {
    "ncresnet/0human_short": "0human_short.parquet",
    "ncresnet/1mouse_short": "1mouse_short.parquet",
    "ncresnet/2s.cerevisiae_short": "2s.cerevisiae_short.parquet",
    "ncresnet/3zebrafish_short": "3zebrafish_short.parquet",
    "ncresnet/4fruitfly_short": "4fruitfly_short.parquet",
    "ncresnet/5cow_short": "5cow_short.parquet",
    "ncresnet/6rat_short": "6rat_short.parquet",
    "ncresnet/7c.elegans_short": "7c.elegans_short.parquet",
    "RNAChallenge": "rna_challenge.parquet",
}


def parse_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header = ""
    chunks: list[str] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header:
                    records.append((header, "".join(chunks)))
                header = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line.upper().replace("U", "T"))
        if header:
            records.append((header, "".join(chunks)))
    return records


def get_longest_protein(seq: str) -> str:
    """ORF scan and return longest protein sequence."""
    scanner = ORFScanner(min_codons=10)
    orfs = scanner.scan(seq)
    if not orfs:
        return ""
    longest = max(orfs, key=lambda o: o.length_codons)
    return longest.protein


def collect_proteins() -> dict[str, str]:
    """Collect sequence_id -> protein for all datasets."""
    proteins: dict[str, str] = {}

    # NCResNet datasets from FASTAs
    for ds_name, (pc_file, nc_file) in NCRESNET_DATASETS.items():
        pc_path = BENCHMARK_DIR / pc_file
        nc_path = BENCHMARK_DIR / nc_file
        if not pc_path.exists() or not nc_path.exists():
            print(f"  SKIP {ds_name}: FASTAs missing", file=sys.stderr)
            continue
        for fasta_path in [pc_path, nc_path]:
            for seq_id, seq in parse_fasta(fasta_path):
                prot = get_longest_protein(seq)
                if prot:
                    proteins[seq_id] = prot
        print(f"  {ds_name}: {len(proteins)} proteins so far", flush=True)

    # RNA Challenge from JSONL
    if JSONL_PATH.exists():
        n_before = len(proteins)
        with open(JSONL_PATH) as fh:
            for line in fh:
                rec = json.loads(line)
                seq_id = rec["sequence_id"]
                seq = rec["sequence"]
                prot = get_longest_protein(seq)
                if prot:
                    proteins[seq_id] = prot
        print(
            f"  RNAChallenge: +{len(proteins) - n_before} proteins "
            f"({len(proteins)} total)",
            flush=True,
        )

    return proteins


def main():
    print("=" * 60)
    print("Pfam Annotation Pipeline")
    print(f"Database: {PFAM_DB}")
    print("=" * 60)

    if not PFAM_DB.exists():
        print(f"ERROR: {PFAM_DB} not found", file=sys.stderr)
        sys.exit(1)

    # Step 1: Collect all proteins
    print("\n[1/3] Collecting proteins from raw sequences...", flush=True)
    t0 = time.perf_counter()
    id_to_protein = collect_proteins()
    dt = time.perf_counter() - t0
    print(f"  {len(id_to_protein):,} proteins in {dt:.1f}s", flush=True)

    if not id_to_protein:
        print("No proteins found, exiting.", file=sys.stderr)
        sys.exit(1)

    # Step 2: Run Pfam on all proteins at once
    print("\n[2/3] Running Pfam hmmsearch...", flush=True)
    seq_ids = list(id_to_protein.keys())
    protein_seqs = [id_to_protein[sid] for sid in seq_ids]

    # Filter out very short proteins (< 10 AA)
    valid = [(sid, prot) for sid, prot in zip(seq_ids, protein_seqs) if len(prot) >= 10]
    if not valid:
        print("No proteins >= 10 AA, exiting.", file=sys.stderr)
        sys.exit(1)

    valid_ids, valid_prots = zip(*valid)
    valid_ids = list(valid_ids)
    valid_prots = list(valid_prots)

    print(f"  {len(valid_prots):,} proteins >= 10 AA", flush=True)

    scanner = PfamScanner(PFAM_DB)
    t0 = time.perf_counter()
    all_hits = scanner.scan(valid_prots)
    dt = time.perf_counter() - t0
    print(f"  Pfam scan: {dt:.1f}s ({len(valid_prots)/dt:.0f} seq/s)", flush=True)

    # Build lookup: seq_id -> (num_domains, best_evalue)
    pfam_results: dict[str, tuple[int, float | None]] = {}
    for sid, hits in zip(valid_ids, all_hits):
        n = len(hits)
        best_e = min(h.e_value for h in hits) if hits else None
        pfam_results[sid] = (n, best_e)

    n_with_hits = sum(1 for n, _ in pfam_results.values() if n > 0)
    print(f"  {n_with_hits:,} / {len(pfam_results):,} proteins have Pfam hits", flush=True)

    # Step 3: Update parquets
    print("\n[3/3] Updating parquets...", flush=True)
    for ds_name, pq_file in PARQUET_FILES.items():
        pq_path = FEATURES_DIR / pq_file
        if not pq_path.exists():
            print(f"  SKIP {pq_file}: not found", file=sys.stderr)
            continue

        df = pd.read_parquet(pq_path)
        n_updated = 0
        n_domains_col = []
        best_e_col = []

        for _, row in df.iterrows():
            sid = row["sequence_id"]
            if sid in pfam_results:
                n_dom, best_e = pfam_results[sid]
                n_domains_col.append(n_dom)
                best_e_col.append(best_e)
                n_updated += 1
            else:
                n_domains_col.append(0)
                best_e_col.append(None)

        df["num_pfam_domains"] = n_domains_col
        df["best_pfam_evalue"] = best_e_col
        df.to_parquet(pq_path, index=False)
        with_hits = sum(1 for n in n_domains_col if n > 0)
        print(f"  {pq_file}: {n_updated} annotated, {with_hits} with domains", flush=True)

    # Rebuild all_datasets.parquet
    all_dfs = []
    for pq_file in PARQUET_FILES.values():
        pq_path = FEATURES_DIR / pq_file
        if pq_path.exists():
            all_dfs.append(pd.read_parquet(pq_path))
    if all_dfs:
        merged = pd.concat(all_dfs, ignore_index=True)
        merged.to_parquet(FEATURES_DIR / "all_datasets.parquet", index=False)
        print(f"\n  all_datasets.parquet: {len(merged):,} rows", flush=True)

    print("\nDone!", flush=True)


if __name__ == "__main__":
    main()
