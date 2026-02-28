"""Extract scalar features from RNA sequences for Phase 2 XGBoost combiner.

Three input modes:
    # Single FASTA pair
    python -m proteinqc.cli.extract_features \
        --coding data/benchmark/ncresnet/0human_short_pc.fa \
        --noncoding data/benchmark/ncresnet/0human_short_nc.fa \
        --dataset "NCResNet/Human_short" \
        --output data/features/ncresnet_human_short.parquet

    # Bulk directory: finds *_pc*/*_nc* pairs, loads CaLM once for all
    python -m proteinqc.cli.extract_features \
        --fasta-dir data/benchmark/ncresnet/ --filter "*short*" \
        --output-dir data/features/

    # JSONL (reuses pre-baked CaLM/perplexity)
    python -m proteinqc.cli.extract_features \
        --jsonl data/rnachallenge/evidence_baked.jsonl \
        --dataset "RNAChallenge" --skip-perplexity \
        --output data/features/rna_challenge.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _parse_fasta(path: Path) -> list[tuple[str, str]]:
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


def _load_fasta_pair(
    coding_path: Path, noncoding_path: Path, dataset: str,
) -> list[dict]:
    items: list[dict] = []
    for hdr, seq in _parse_fasta(coding_path):
        items.append({"sequence_id": hdr, "sequence": seq, "label": 1, "dataset": dataset})
    for hdr, seq in _parse_fasta(noncoding_path):
        items.append({"sequence_id": hdr, "sequence": seq, "label": 0, "dataset": dataset})
    return items


def _load_jsonl(jsonl_path: Path, dataset: str) -> list[dict]:
    items: list[dict] = []
    with open(jsonl_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            raw = obj.get("label", "")
            label = raw if isinstance(raw, int) else (1 if raw == "coding" else 0)
            items.append({
                "sequence_id": obj["sequence_id"],
                "sequence": obj["sequence"],
                "label": label,
                "dataset": dataset,
                "species": obj.get("species", "unknown"),
                "_calm_score": obj.get("calm_score"),
                "_perplexity": obj.get("perplexity"),
            })
    return items


def _discover_pairs(fasta_dir: Path, pattern: str) -> list[tuple[str, Path, Path]]:
    """Find *_pc*/*_nc* FASTA pairs in a directory."""
    pairs = []
    for pc in sorted(fasta_dir.glob(pattern)):
        if "_pc" not in pc.name:
            continue
        nc = pc.parent / pc.name.replace("_pc", "_nc")
        if nc.exists():
            dataset = pc.stem.split("_pc")[0]
            pairs.append((dataset, pc, nc))
    return pairs


# ---------------------------------------------------------------------------
# Tier 1: CPU features (pure function, process-safe)
# ---------------------------------------------------------------------------


def _cpu_features_one(sequence: str) -> dict:
    """All Tier 1 scalar features for one DNA sequence."""
    from proteinqc.tools.codon_table import CodonTableManager, sequence_to_codon_vector
    from proteinqc.tools.orf_scanner import ORFScanner
    from proteinqc.tools.translate import translate

    seq = sequence.upper().replace("U", "T")
    seq_len = len(seq)

    gc_count = seq.count("G") + seq.count("C")
    gc_content = gc_count / seq_len if seq_len > 0 else 0.0

    manager = CodonTableManager()
    code = manager.get_genetic_code(1)
    scanner = ORFScanner(code, min_codons=10)
    orfs = scanner.scan(seq)

    num_orfs = len(orfs)
    protein = ""
    if orfs:
        best = orfs[0]
        longest_orf_codons = best.length_codons
        orf_fraction = (best.stop - best.start) / seq_len if seq_len > 0 else 0.0
        protein = translate(best.seq)
        protein_length = len(protein)
        kozak = 0.0
        pos_m3, pos_p4 = best.start - 3, best.start + 3
        if 0 <= pos_m3 < seq_len and seq[pos_m3] in ("A", "G"):
            kozak += 0.5
        if 0 <= pos_p4 < seq_len and seq[pos_p4] == "G":
            kozak += 0.5
    else:
        longest_orf_codons = 0
        orf_fraction = 0.0
        protein_length = 0
        kozak = 0.0

    vec = sequence_to_codon_vector(seq)
    nz = vec[vec > 0]
    entropy = float(-np.sum(nz * np.log2(nz))) if len(nz) > 0 else 0.0
    sorted_v = np.sort(vec)
    total = sorted_v.sum()
    if total > 0:
        idx = np.arange(1, len(sorted_v) + 1)
        gini = max(0.0, float(
            (2 * np.sum(idx * sorted_v) / (len(sorted_v) * total))
            - (len(sorted_v) + 1) / len(sorted_v)
        ))
    else:
        gini = 0.0

    return {
        "seq_length_bp": seq_len,
        "gc_content": round(gc_content, 6),
        "longest_orf_codons": longest_orf_codons,
        "num_orfs": num_orfs,
        "orf_fraction": round(orf_fraction, 6),
        "kozak_score": kozak,
        "protein_length": protein_length,
        "codon_entropy": round(entropy, 6),
        "codon_gini": round(gini, 6),
        "_protein": protein,
    }


def _cpu_features_parallel(sequences: list[str], n_workers: int) -> list[dict]:
    if n_workers <= 1 or len(sequences) < 200:
        return [_cpu_features_one(s) for s in sequences]
    chunksize = max(1, len(sequences) // (n_workers * 4))
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        return list(pool.map(_cpu_features_one, sequences, chunksize=chunksize))


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "seq_length_bp", "gc_content", "longest_orf_codons", "num_orfs",
    "orf_fraction", "kozak_score", "protein_length", "codon_entropy",
    "codon_gini", "calm_score", "perplexity", "num_pfam_domains",
    "best_pfam_evalue",
]


def _run(items: list[dict], args: argparse.Namespace,
         output_map: dict[str, Path]) -> None:
    """Process items: Tier 1 (CPU parallel), Tier 2 (GPU batched), write parquets."""
    import pandas as pd

    # Incremental resume across all outputs
    done_ids: set[str] = set()
    for path in output_map.values():
        if path.exists():
            done_ids.update(pd.read_parquet(path)["sequence_id"])
    if done_ids:
        items = [it for it in items if it["sequence_id"] not in done_ids]
        print(f"  Resuming: {len(done_ids)} done, {len(items)} remaining", flush=True)
    if not items:
        print("  All done.", flush=True)
        return

    sequences = [it["sequence"] for it in items]
    n = len(items)
    timings: dict[str, float] = {}

    # --- Tier 1: CPU ---
    n_workers = args.workers or min(os.cpu_count() or 1, 8)
    mode = f"{n_workers} workers" if n >= 200 and n_workers > 1 else "serial"
    print(f"\nTier 1  CPU features  ({n} seqs, {mode})", flush=True)
    t0 = time.perf_counter()
    cpu_feats = _cpu_features_parallel(sequences, n_workers)
    dt = time.perf_counter() - t0
    timings["tier1_cpu"] = dt
    print(f"  {dt:.2f}s  ({n / dt:.0f} seq/s)", flush=True)

    # --- Tier 2: GPU ---
    calm_scores: list[float | None] = [None] * n
    ppl_scores: list[float | None] = [None] * n

    if not args.skip_gpu:
        prebaked_calm = [it.get("_calm_score") for it in items]
        prebaked_ppl = [it.get("_perplexity") for it in items]
        need_calm = [i for i, v in enumerate(prebaked_calm) if v is None]
        need_ppl = [i for i, v in enumerate(prebaked_ppl) if v is None]

        calm_dir = Path(args.calm_dir)
        head_path = Path(args.head_path)

        if need_calm and (calm_dir / "model.safetensors").exists() and head_path.exists():
            print(f"\nTier 2a CaLM scoring  ({len(need_calm)} seqs)", flush=True)
            from proteinqc.tools.calm_scorer import CaLMScorer

            scorer = CaLMScorer(calm_dir, head_path)
            CALM_MAX_BP = 3072
            t0 = time.perf_counter()
            scores = scorer.batch_score([sequences[i][:CALM_MAX_BP] for i in need_calm])
            for idx, score in zip(need_calm, scores):
                calm_scores[idx] = score
            dt = time.perf_counter() - t0
            timings["tier2_calm"] = dt
            print(f"  {dt:.2f}s  ({len(need_calm) / dt:.0f} seq/s)", flush=True)
            del scorer
        elif need_calm:
            print("  WARN: CaLM model/head not found, skipping", file=sys.stderr)

        for i, v in enumerate(prebaked_calm):
            if v is not None:
                calm_scores[i] = v

        if not args.skip_perplexity and need_ppl:
            if (calm_dir / "model.safetensors").exists():
                print(f"\nTier 2b Perplexity    ({len(need_ppl)} seqs)", flush=True)
                from proteinqc.tools.perplexity_scorer import PerplexityScorer

                ppl = PerplexityScorer(calm_dir)
                t0 = time.perf_counter()
                results = ppl.batch_score([sequences[i] for i in need_ppl])
                for idx, score in zip(need_ppl, results):
                    ppl_scores[idx] = score
                dt = time.perf_counter() - t0
                timings["tier2_perplexity"] = dt
                print(f"  {dt:.2f}s  ({len(need_ppl) / dt:.1f} seq/s)", flush=True)
                del ppl

        for i, v in enumerate(prebaked_ppl):
            if v is not None:
                ppl_scores[i] = v

    # --- Tier 3: Pfam ---
    pfam_n_domains: list[int | None] = [None] * n
    pfam_best_evalue: list[float | None] = [None] * n
    if args.pfam and args.pfam_db:
        pfam_path = Path(args.pfam_db)
        if pfam_path.exists():
            from proteinqc.tools.pfam_scanner import PfamScanner

            print(f"\nTier 3  Pfam scanning", flush=True)
            pfam = PfamScanner(pfam_path)
            proteins, pidx = [], []
            for i, f in enumerate(cpu_feats):
                if len(f["_protein"]) >= 10:
                    proteins.append(f["_protein"])
                    pidx.append(i)
            t0 = time.perf_counter()
            if proteins:
                hits_list = pfam.scan(proteins)
                for idx, hits in zip(pidx, hits_list):
                    pfam_n_domains[idx] = len(hits)
                    pfam_best_evalue[idx] = min(h.e_value for h in hits) if hits else None
            dt = time.perf_counter() - t0
            timings["tier3_pfam"] = dt
            print(f"  {dt:.2f}s  ({len(proteins)} proteins)", flush=True)

    # --- Assemble & write per-dataset ---
    rows = []
    for i, item in enumerate(items):
        f = cpu_feats[i]
        rows.append({
            "sequence_id": item["sequence_id"],
            "label": item["label"],
            "dataset": item["dataset"],
            "species": item.get("species", "unknown"),
            "seq_length_bp": f["seq_length_bp"],
            "gc_content": f["gc_content"],
            "longest_orf_codons": f["longest_orf_codons"],
            "num_orfs": f["num_orfs"],
            "orf_fraction": f["orf_fraction"],
            "kozak_score": f["kozak_score"],
            "protein_length": f["protein_length"],
            "codon_entropy": f["codon_entropy"],
            "codon_gini": f["codon_gini"],
            "calm_score": calm_scores[i],
            "perplexity": ppl_scores[i],
            "num_pfam_domains": pfam_n_domains[i],
            "best_pfam_evalue": pfam_best_evalue[i],
        })
    df = pd.DataFrame(rows)

    for dataset, path in output_map.items():
        ds_df = df[df["dataset"] == dataset]
        if path.exists():
            existing = pd.read_parquet(path)
            ds_df = pd.concat([existing, ds_df], ignore_index=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        ds_df.to_parquet(path, index=False)

    # --- Profile ---
    total_t = sum(timings.values())
    print(f"\n{'='*55}")
    print(f"Done: {n} new rows across {len(output_map)} datasets")
    for tier, dt in sorted(timings.items()):
        pct = 100 * dt / total_t if total_t > 0 else 0
        print(f"  {tier:25s} {dt:8.2f}s  ({pct:5.1f}%)")
    print(f"  {'TOTAL':25s} {total_t:8.2f}s")

    if df["label"].nunique() == 2:
        print(f"\nCoding vs Noncoding (mean):")
        for col in ["longest_orf_codons", "orf_fraction", "gc_content", "calm_score"]:
            if col in df.columns and df[col].notna().any():
                c = df.loc[df["label"] == 1, col].mean()
                nc = df.loc[df["label"] == 0, col].mean()
                print(f"  {col:25s}  coding={c:.4f}  nc={nc:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract scalar features for Phase 2 XGBoost combiner",
    )
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--coding", help="Coding FASTA (use with --noncoding)")
    grp.add_argument("--jsonl", help="Pre-baked JSONL path")
    grp.add_argument("--fasta-dir", help="Directory with *_pc*/*_nc* FASTA pairs")

    ap.add_argument("--noncoding", help="Noncoding FASTA path")
    ap.add_argument("--dataset", help="Dataset tag (auto-derived in --fasta-dir mode)")
    ap.add_argument("--output", help="Output .parquet (single-dataset mode)")
    ap.add_argument("--output-dir", default="data/features", help="Output dir (bulk mode)")
    ap.add_argument("--filter", default="*", help="Glob filter for --fasta-dir")

    ap.add_argument("--skip-gpu", action="store_true", help="CPU features only")
    ap.add_argument("--skip-perplexity", action="store_true", help="Skip perplexity")
    ap.add_argument("--pfam", action="store_true", help="Enable Pfam scanning")
    ap.add_argument("--pfam-db", default=None, help="Pfam-A.hmm path")
    ap.add_argument("--calm-dir", default="models/calm")
    ap.add_argument("--head-path", default="models/heads/mlp_head_v1.pt")
    ap.add_argument("--workers", type=int, default=0, help="CPU workers (0=auto)")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    if args.fasta_dir:
        pairs = _discover_pairs(Path(args.fasta_dir), args.filter)
        if not pairs:
            print(f"No *_pc*/*_nc* pairs found in {args.fasta_dir}", file=sys.stderr)
            sys.exit(1)
        out_dir = Path(args.output_dir)
        dir_name = Path(args.fasta_dir).name
        items: list[dict] = []
        output_map: dict[str, Path] = {}
        for name, pc, nc in pairs:
            ds = f"{dir_name}/{name}"
            items.extend(_load_fasta_pair(pc, nc, ds))
            output_map[ds] = out_dir / f"{name}.parquet"
        print(f"Loaded {len(items)} sequences across {len(pairs)} datasets", flush=True)
    elif args.coding:
        if not args.noncoding:
            ap.error("--coding requires --noncoding")
        if not args.dataset or not args.output:
            ap.error("--coding mode requires --dataset and --output")
        items = _load_fasta_pair(Path(args.coding), Path(args.noncoding), args.dataset)
        output_map = {args.dataset: Path(args.output)}
        print(f"Loaded {len(items)} sequences [{args.dataset}]", flush=True)
    else:
        if not args.dataset or not args.output:
            ap.error("--jsonl mode requires --dataset and --output")
        items = _load_jsonl(Path(args.jsonl), args.dataset)
        output_map = {args.dataset: Path(args.output)}
        print(f"Loaded {len(items)} sequences [{args.dataset}]", flush=True)

    _run(items, args, output_map)


if __name__ == "__main__":
    main()
