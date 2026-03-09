"""Zero-shot benchmark: evaluate pre-trained LoRA+GatedHead on all datasets.

No training, no 80/20 split. Pure inference with the merged LoRA encoder
and gated classification head. Reports per-dataset and overall metrics.

Usage:
    python -m proteinqc.cli.benchmark_zeroshot
    python -m proteinqc.cli.benchmark_zeroshot \
        --head-dir models/heads/lora_alibi_gated_v1 \
        --token-budget 16384 --max-batch 256
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from proteinqc.cli.benchmark_multispecies import (
    codon_align_longest_orf,
    discover_datasets,
    read_fasta,
    read_fasta_multi,
)
from proteinqc.data.dataset import (
    _bucket_pad,
    collate_binary,
    pre_tokenize,
)
from proteinqc.data.tokenizer import CodonTokenizer
from proteinqc.tools.calm_scorer import _load_lora_checkpoint
from proteinqc.tools.codon_table import CodonTableManager
from proteinqc.tools.orf_scanner import ORFScanner

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray,
) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score, f1_score, matthews_corrcoef, roc_auc_score,
    )
    m: dict[str, float] = {}
    m["ACC"] = float(accuracy_score(y_true, y_pred) * 100)
    m["MCC"] = float(matthews_corrcoef(y_true, y_pred) * 100)
    m["F1"] = float(f1_score(y_true, y_pred, average="macro") * 100)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    m["TP"] = tp
    m["TN"] = tn
    m["FP"] = fp
    m["FN"] = fn
    try:
        m["AUC"] = float(roc_auc_score(y_true, y_prob) * 100)
    except ValueError:
        m["AUC"] = float("nan")
    return m


def batched_inference(
    samples: list[dict],
    encoder: torch.nn.Module,
    head: torch.nn.Module,
    device: torch.device,
    token_budget: int = 16_384,
    max_batch: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Zero-shot inference with token-budget dynamic batching.

    Returns (predictions, probabilities) as numpy arrays in original order.
    """
    n = len(samples)
    all_probs = np.zeros(n, dtype=np.float32)
    sorted_indices = sorted(range(n), key=lambda i: len(samples[i]["input_ids"]))

    i = 0
    while i < n:
        max_pad = _bucket_pad(len(samples[sorted_indices[i]]["input_ids"]))
        bs = 1
        while i + bs < n and bs < max_batch:
            next_pad = _bucket_pad(len(samples[sorted_indices[i + bs]]["input_ids"]))
            new_max = max(max_pad, next_pad)
            if new_max * (bs + 1) > token_budget:
                break
            max_pad = new_max
            bs += 1

        batch_samples = [samples[sorted_indices[j]] for j in range(i, i + bs)]
        batch = collate_binary(batch_samples)
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            cls_emb = encoder(ids, mask)
            logits = head(cls_emb).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()

        for j in range(bs):
            all_probs[sorted_indices[i + j]] = probs[j]

        i += bs

    all_preds = (all_probs > 0.5).astype(int)
    return all_preds, all_probs


def load_dataset_sequences(
    ds: dict, align_fn, max_seqs: int = 0,
) -> tuple[list[str], list[int], list[str]]:
    """Load and align sequences from a dataset pair.

    Returns (sequences, labels, sequence_ids).
    """
    half = max_seqs // 2 if max_seqs else 0

    if "mixed" in ds:
        if not ds["mixed"].exists():
            return [], [], []
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
        coding_paths = ds["coding"] if isinstance(ds["coding"], list) else [ds["coding"]]
        noncoding_paths = ds["noncoding"] if isinstance(ds["noncoding"], list) else [ds["noncoding"]]
        missing = [p for p in coding_paths + noncoding_paths if not Path(p).exists()]
        if missing:
            return [], [], []
        coding_records = read_fasta_multi(ds["coding"], max_seqs=half)
        noncoding_records = read_fasta_multi(ds["noncoding"], max_seqs=half)

    sequences, labels, seq_ids = [], [], []
    for header, seq in coding_records:
        aligned = align_fn(seq)
        if len(aligned) >= 9:
            sequences.append(aligned)
            labels.append(1)
            seq_ids.append(header.split()[0])
    for header, seq in noncoding_records:
        aligned = align_fn(seq)
        if len(aligned) >= 9:
            sequences.append(aligned)
            labels.append(0)
            seq_ids.append(header.split()[0])

    return sequences, labels, seq_ids


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Zero-shot benchmark: LoRA+GatedHead on all datasets",
    )
    p.add_argument("--model-dir", type=Path,
                   default=PROJECT_ROOT / "models" / "calm")
    p.add_argument("--head-dir", type=Path,
                   default=PROJECT_ROOT / "models" / "heads" / "lora_alibi_gated_v1")
    p.add_argument("--benchmark-dir", type=Path,
                   default=PROJECT_ROOT / "data" / "benchmark")
    p.add_argument("--output", type=Path,
                   default=PROJECT_ROOT / "data" / "results" / "benchmark_zeroshot.json")
    p.add_argument("--token-budget", type=int, default=16_384,
                   help="Max tokens per batch (default: 16384 = 2048*8)")
    p.add_argument("--max-batch", type=int, default=256,
                   help="Hard cap on batch size")
    p.add_argument("--max-seqs", type=int, default=0,
                   help="Max sequences per dataset (0=all)")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True,
                   help="torch.compile encoder (CUDA only)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(42)

    device = select_device()

    print("=" * 80)
    print("  Zero-Shot Benchmark: LoRA ALiBi + GatedHead")
    print("=" * 80)
    print(f"Device:       {device}")
    print(f"Model:        {args.model_dir}")
    print(f"Head:         {args.head_dir}")
    print(f"Token budget: {args.token_budget:,}")
    print(f"Max batch:    {args.max_batch}")

    # Init ORF scanner for longest-orf alignment
    import proteinqc.cli.benchmark_multispecies as bm
    ctm = CodonTableManager()
    bm._orf_scanner = ORFScanner(ctm.get_genetic_code(1), min_codons=30)
    align_fn = codon_align_longest_orf

    # Auto-detect position_type from training log
    training_log = args.head_dir / "training_log.json"
    position_type = "rotary"
    if training_log.exists():
        with open(training_log) as f:
            position_type = json.load(f).get("position_type", "rotary")

    # Load model (LoRA merged + head)
    print(f"\nLoading LoRA checkpoint from {args.head_dir}...")
    print(f"  Position encoding: {position_type}")
    encoder, head = _load_lora_checkpoint(
        args.model_dir, args.head_dir, device,
        position_type=position_type,
    )
    encoder.eval()
    head.eval()

    # torch.compile on CUDA
    if args.compile and device.type == "cuda":
        print("torch.compile (dynamic=None)...")
        from proteinqc.data.dataset import MAX_SEQ_LEN
        encoder._ensure_alibi(MAX_SEQ_LEN, device)
        encoder = torch.compile(encoder, dynamic=None)

    n_params = sum(p.numel() for p in encoder.parameters())
    n_head = sum(p.numel() for p in head.parameters())
    print(f"  Encoder: {n_params:,} params (frozen, LoRA merged)")
    print(f"  Head:    {n_head:,} params")

    tokenizer = CodonTokenizer(args.model_dir / "vocab.txt")

    # Discover datasets
    datasets = discover_datasets(args.benchmark_dir)
    print(f"\nDiscovered {len(datasets)} dataset pairs")

    all_results = []
    all_scores: list[dict] = []  # per-sequence scores for XGBoost
    total_seqs = 0
    total_time = 0.0

    for i, ds in enumerate(datasets):
        ds_name = f"{ds['tool']}/{ds['species']}"
        print(f"\n[{i+1}/{len(datasets)}] {ds_name}", end="", flush=True)

        sequences, labels, seq_ids = load_dataset_sequences(ds, align_fn, args.max_seqs)
        if len(sequences) < 10:
            print(f"  SKIP ({len(sequences)} seqs)")
            continue

        n_coding = sum(labels)
        n_noncoding = len(labels) - n_coding

        samples = pre_tokenize(sequences, labels, tokenizer)

        t0 = time.time()
        preds, probs = batched_inference(
            samples, encoder, head, device,
            args.token_budget, args.max_batch,
        )
        elapsed = time.time() - t0
        total_time += elapsed
        total_seqs += len(sequences)

        y_true = np.array(labels)
        m = compute_metrics(y_true, preds, probs)
        m["dataset"] = ds_name
        m["tool"] = ds["tool"]
        m["species"] = ds["species"]
        m["n_coding"] = n_coding
        m["n_noncoding"] = n_noncoding
        m["n_total"] = len(sequences)
        m["time_sec"] = elapsed
        all_results.append(m)

        # Collect per-sequence scores
        for j in range(len(sequences)):
            all_scores.append({
                "dataset": ds_name,
                "sequence_id": seq_ids[j],
                "label": labels[j],
                "calm_score": float(probs[j]),
                "seq_length": len(sequences[j]),
            })

        print(
            f"  {len(sequences):>6} seqs  "
            f"ACC={m['ACC']:.1f}%  MCC={m['MCC']:.1f}%  "
            f"AUC={m['AUC']:.1f}%  "
            f"({elapsed:.1f}s)"
        )

        # Save partial results
        args.output.parent.mkdir(parents=True, exist_ok=True)
        partial = args.output.with_suffix(".partial.json")
        with open(partial, "w") as f:
            json.dump(all_results, f, indent=2)

    # Summary table
    w = 120
    print("\n" + "=" * w)
    hdr = (
        f"{'Tool':<15} {'Species':<25} {'N':>7}"
        f"  {'ACC':>7} {'MCC':>7} {'AUC':>7} {'F1':>7}"
        f"  {'TP':>5} {'TN':>5} {'FP':>5} {'FN':>5}"
    )
    print(hdr)
    print("-" * w)
    for r in all_results:
        row = (
            f"{r['tool']:<15} {r['species']:<25} {r['n_total']:>7}"
            f"  {r['ACC']:>6.1f}% {r['MCC']:>6.1f}% {r['AUC']:>6.1f}% {r['F1']:>6.1f}%"
            f"  {r['TP']:>5} {r['TN']:>5} {r['FP']:>5} {r['FN']:>5}"
        )
        print(row)

    if all_results:
        print("-" * w)
        avg_acc = np.mean([r["ACC"] for r in all_results])
        avg_mcc = np.mean([r["MCC"] for r in all_results])
        avg_auc = np.nanmean([r["AUC"] for r in all_results])
        avg_f1 = np.mean([r["F1"] for r in all_results])
        total_n = sum(r["n_total"] for r in all_results)
        print(
            f"{'MEAN':<15} {'(' + str(len(all_results)) + ' datasets)':<25} {total_n:>7}"
            f"  {avg_acc:>6.1f}% {avg_mcc:>6.1f}% {avg_auc:>6.1f}% {avg_f1:>6.1f}%"
        )
        print(f"\nTotal: {total_seqs:,} sequences in {total_time:.0f}s")
        print(f"Throughput: {total_seqs / max(total_time, 1):.0f} seq/s")

    # Save final results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "model": str(args.head_dir),
            "token_budget": args.token_budget,
            "max_batch": args.max_batch,
            "device": str(device),
            "total_seqs": total_seqs,
            "total_time_sec": total_time,
            "avg_acc": float(avg_acc) if all_results else 0,
            "avg_mcc": float(avg_mcc) if all_results else 0,
            "avg_auc": float(np.nanmean([r["AUC"] for r in all_results])) if all_results else 0,
            "per_dataset": all_results,
        }, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Save per-sequence scores and merge into existing feature parquets
    if all_scores:
        import pandas as pd
        scores_df = pd.DataFrame(all_scores)
        scores_path = args.output.with_name("benchmark_zeroshot_scores.parquet")
        scores_df.to_parquet(scores_path, index=False)
        print(f"Scores saved to {scores_path} ({len(scores_df):,} rows)")

        # Merge into existing feature parquets
        features_dir = PROJECT_ROOT / "data" / "features"
        if features_dir.exists():
            n_updated = 0
            for pq_path in sorted(features_dir.glob("*.parquet")):
                if pq_path.name in ("all_datasets.parquet",
                                    "benchmark_zeroshot_scores.parquet"):
                    continue
                feat_df = pd.read_parquet(pq_path)
                if "dataset" not in feat_df.columns or "sequence_id" not in feat_df.columns:
                    continue
                ds_name = feat_df["dataset"].iloc[0]
                ds_scores = scores_df[scores_df["dataset"] == ds_name]
                if ds_scores.empty:
                    continue
                score_map = dict(zip(ds_scores["sequence_id"], ds_scores["calm_score"]))
                matched = feat_df["sequence_id"].isin(score_map)
                if matched.sum() == 0:
                    continue
                feat_df.loc[matched, "calm_score"] = feat_df.loc[matched, "sequence_id"].map(score_map)
                feat_df.to_parquet(pq_path, index=False)
                n_updated += 1
            print(f"Updated calm_score in {n_updated} feature parquets")

    # Clean up partial
    partial = args.output.with_suffix(".partial.json")
    if partial.exists():
        partial.unlink()


if __name__ == "__main__":
    main()
