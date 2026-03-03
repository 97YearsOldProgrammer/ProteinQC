"""Extract 768-dim CLS embeddings from frozen CaLM for all benchmark sequences.

Saves to HDF5 with schema:
    embeddings   float32  [N, 768]    CLS embeddings
    labels       int8     [N]         1=coding, 0=noncoding
    dataset_idx  int32    [N]         index into dataset_names
    dataset_names str[]   [D]         "NCResNet/Human (short)", etc.
    seq_ids      str[]    [N]         sequence IDs from FASTA headers

Usage:
    # Full extraction (~2M seqs, ~2-4 hours on M4 Pro)
    python -m proteinqc.cli.extract_embeddings \
        --benchmark-dir data/benchmark/ \
        --model-dir models/calm/ \
        --output data/embeddings/benchmark_embeddings.h5

    # Quick test (~20 min)
    python -m proteinqc.cli.extract_embeddings \
        --benchmark-dir data/benchmark/ \
        --model-dir models/calm/ \
        --output data/embeddings/benchmark_embeddings.h5 \
        --sample 1000

    # Shard across DGX nodes
    python -m proteinqc.cli.extract_embeddings \
        --benchmark-dir data/benchmark/ \
        --model-dir models/calm/ \
        --output data/embeddings/benchmark_embeddings.h5 \
        --nnodes 2 --node-rank 0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from proteinqc.cli.benchmark_multispecies import (
    codon_align_longest_orf,
    discover_datasets,
    extract_embeddings,
    read_fasta,
    read_fasta_multi,
)
from proteinqc.data.tokenizer import CodonTokenizer
from proteinqc.models.calm_encoder import CaLMEncoder
from proteinqc.tools.codon_table import CodonTableManager
from proteinqc.tools.orf_scanner import ORFScanner

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_dataset_sequences(
    ds: dict,
    max_seqs: int,
    align_fn,
) -> tuple[list[str], list[str], list[int]]:
    """Load and align sequences from one dataset.

    Returns:
        (seq_ids, aligned_sequences, labels)  where label=1 coding, 0 noncoding
    """
    half = max_seqs // 2 if max_seqs else 0

    if "mixed" in ds:
        all_records = read_fasta(ds["mixed"], max_seqs=0)
        coding_records = [
            (h, s) for h, s in all_records if "NM_" in h or "mRNA" in h.lower()
        ]
        noncoding_records = [
            (h, s) for h, s in all_records
            if h not in {r[0] for r in coding_records}
        ]
        if half:
            coding_records = coding_records[:half]
            noncoding_records = noncoding_records[:half]
    else:
        coding_records = read_fasta_multi(ds["coding"], max_seqs=half)
        noncoding_records = read_fasta_multi(ds["noncoding"], max_seqs=half)

    seq_ids: list[str] = []
    sequences: list[str] = []
    labels: list[int] = []

    for header, seq in coding_records:
        aligned = align_fn(seq)
        if len(aligned) >= 9:
            seq_ids.append(header.split()[0])
            sequences.append(aligned)
            labels.append(1)

    for header, seq in noncoding_records:
        aligned = align_fn(seq)
        if len(aligned) >= 9:
            seq_ids.append(header.split()[0])
            sequences.append(aligned)
            labels.append(0)

    return seq_ids, sequences, labels


def write_h5(
    output_path: Path,
    all_embeddings: list[np.ndarray],
    all_labels: list[np.ndarray],
    all_dataset_idx: list[np.ndarray],
    all_seq_ids: list[list[str]],
    dataset_names: list[str],
) -> None:
    """Write accumulated results to HDF5."""
    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    dataset_idx = np.concatenate(all_dataset_idx, axis=0)
    seq_ids: list[str] = []
    for batch in all_seq_ids:
        seq_ids.extend(batch)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.create_dataset("embeddings", data=embeddings, dtype="float32")
        f.create_dataset("labels", data=labels, dtype="int8")
        f.create_dataset("dataset_idx", data=dataset_idx, dtype="int32")

        dt_str = h5py.string_dtype()
        ds_names = f.create_dataset(
            "dataset_names", shape=(len(dataset_names),), dtype=dt_str,
        )
        for i, name in enumerate(dataset_names):
            ds_names[i] = name

        ds_ids = f.create_dataset(
            "seq_ids", shape=(len(seq_ids),), dtype=dt_str,
        )
        for i, sid in enumerate(seq_ids):
            ds_ids[i] = sid

    n = len(labels)
    gb = embeddings.nbytes / 1e9
    print(f"\nSaved {n:,} embeddings ({gb:.2f} GB) to {output_path}")
    print(f"  Datasets: {len(dataset_names)}")
    print(f"  Coding: {int((labels == 1).sum()):,}  Noncoding: {int((labels == 0).sum()):,}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract CLS embeddings from frozen CaLM for benchmark sequences",
    )
    p.add_argument(
        "--benchmark-dir", type=Path, default=PROJECT_ROOT / "data" / "benchmark",
        help="Directory containing cpat/, ncresnet/, lncfinder/, etc.",
    )
    p.add_argument(
        "--model-dir", type=Path, default=PROJECT_ROOT / "models" / "calm",
        help="CaLM weights directory (config.json + model.safetensors + vocab.txt)",
    )
    p.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "data" / "embeddings" / "benchmark_embeddings.h5",
        help="HDF5 output path",
    )
    p.add_argument(
        "--sample", type=int, default=0,
        help="Random subsample per dataset (0=all sequences)",
    )
    p.add_argument(
        "--nnodes", type=int, default=1,
        help="Total number of nodes (for sharding across machines)",
    )
    p.add_argument(
        "--node-rank", type=int, default=0,
        help="This node's rank (0-indexed)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(42)
    np.random.seed(42)

    device = select_device()
    print("=" * 70)
    print("  CLS Embedding Extraction")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Benchmark dir: {args.benchmark_dir}")
    print(f"Model dir: {args.model_dir}")
    print(f"Output: {args.output}")
    if args.sample:
        print(f"Sample per dataset: {args.sample}")
    if args.nnodes > 1:
        print(f"Node: {args.node_rank + 1}/{args.nnodes}")

    # Init ORF scanner for longest-orf alignment
    import proteinqc.cli.benchmark_multispecies as bm
    ctm = CodonTableManager()
    bm._orf_scanner = ORFScanner(ctm.get_genetic_code(1), min_codons=30)
    align_fn = codon_align_longest_orf

    # Load encoder
    print(f"\nLoading CaLM encoder from {args.model_dir}...")
    encoder = CaLMEncoder(args.model_dir, freeze=True).to(device)
    encoder.eval()
    tokenizer = CodonTokenizer(args.model_dir / "vocab.txt")
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"  {n_params:,} params (frozen)")

    # Discover datasets
    datasets = discover_datasets(args.benchmark_dir)
    print(f"\nDiscovered {len(datasets)} dataset pairs")

    # Shard across nodes
    if args.nnodes > 1:
        datasets = [
            ds for i, ds in enumerate(datasets) if i % args.nnodes == args.node_rank
        ]
        print(f"This node processes {len(datasets)} datasets")

    # Accumulate results
    all_embeddings: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_dataset_idx: list[np.ndarray] = []
    all_seq_ids: list[list[str]] = []
    dataset_names: list[str] = []
    total_seqs = 0
    t_start = time.time()

    for i, ds in enumerate(datasets):
        ds_name = f"{ds['tool']}/{ds['species']}"

        # Check files exist
        if "mixed" in ds:
            if not ds["mixed"].exists():
                print(f"  [{i+1}/{len(datasets)}] SKIP {ds_name} -- file missing")
                continue
        else:
            coding_paths = ds["coding"] if isinstance(ds["coding"], list) else [ds["coding"]]
            noncoding_paths = ds["noncoding"] if isinstance(ds["noncoding"], list) else [ds["noncoding"]]
            missing = [p for p in coding_paths + noncoding_paths if not Path(p).exists()]
            if missing:
                print(f"  [{i+1}/{len(datasets)}] SKIP {ds_name} -- {len(missing)} files missing")
                continue

        t0 = time.time()
        seq_ids, sequences, labels = load_dataset_sequences(ds, args.sample, align_fn)

        if len(sequences) < 2:
            print(f"  [{i+1}/{len(datasets)}] SKIP {ds_name} -- too few sequences ({len(sequences)})")
            continue

        n_coding = sum(labels)
        n_nc = len(labels) - n_coding

        # Extract CLS embeddings (reuse adaptive batching from benchmark)
        cls_emb, _ = extract_embeddings(sequences, encoder, tokenizer, device)

        ds_idx = len(dataset_names)
        dataset_names.append(ds_name)
        all_embeddings.append(cls_emb.numpy())
        all_labels.append(np.array(labels, dtype=np.int8))
        all_dataset_idx.append(np.full(len(labels), ds_idx, dtype=np.int32))
        all_seq_ids.append(seq_ids)

        total_seqs += len(sequences)
        elapsed = time.time() - t0
        print(
            f"  [{i+1}/{len(datasets)}] {ds_name}: "
            f"{len(sequences)} seqs ({n_coding} coding, {n_nc} nc) "
            f"[{elapsed:.1f}s]"
        )

        # Incremental save every 10 datasets
        if (i + 1) % 10 == 0:
            write_h5(
                args.output,
                all_embeddings, all_labels,
                all_dataset_idx, all_seq_ids,
                dataset_names,
            )

    # Final save
    if dataset_names:
        write_h5(
            args.output,
            all_embeddings, all_labels,
            all_dataset_idx, all_seq_ids,
            dataset_names,
        )

    total_time = time.time() - t_start
    print(f"\nTotal: {total_seqs:,} sequences, {len(dataset_names)} datasets, {total_time:.0f}s")


if __name__ == "__main__":
    main()
