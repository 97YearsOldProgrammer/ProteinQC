#!/usr/bin/env python3
"""RNA Challenge re-test: compare 3 CaLM input preprocessing strategies.

Strategy A (baseline): first-ATG alignment (current approach)
Strategy B (longest ORF): ORFScanner finds longest ORF, use that frame
Strategy C (3-frame max): score all 3 frames, take max

For each strategy: train fresh GatedHead on 80/20 split, report metrics.
Also score with pretrained MLPHead (zero-shot) for comparison.
"""

import csv
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from proteinqc.data.tokenizer import CodonTokenizer
from proteinqc.models.calm_encoder import CaLMEncoder
from proteinqc.models.classification_heads import GatedHead, MLPHead
from proteinqc.tools.codon_table import CodonTableManager
from proteinqc.tools.orf_scanner import ORFScanner

# ---------------------------------------------------------------------------
BATCH_EMBED = 16
BATCH_TRAIN = 64
TOKEN_BUDGET = 8_192
LR = 1e-3
EPOCHS = 20
SEED = 42
TEST_RATIO = 0.2

_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = _ROOT / "data" / "rnachallenge"
RESULTS_DIR = _ROOT / "data" / "results"
MODEL_DIR = _ROOT / "models" / "calm"
HEAD_PATH = _ROOT / "models" / "heads" / "mlp_head_v1.pt"


# ---------------------------------------------------------------------------
# Alignment strategies
# ---------------------------------------------------------------------------

def _clean(seq: str) -> str:
    seq = seq.upper().replace("U", "T")
    return re.sub(r"[^ACGT]", "", seq)


def align_first_atg(seq: str) -> str:
    """Strategy A: find first ATG, trim to codon boundary."""
    seq = _clean(seq)
    atg_pos = seq.find("ATG")
    if atg_pos >= 0:
        seq = seq[atg_pos:]
    trim = len(seq) - (len(seq) % 3)
    return seq[:trim]


def align_longest_orf(seq: str, scanner: ORFScanner) -> tuple[str, int]:
    """Strategy B: longest ORF from ORFScanner, fallback to first-ATG.

    Returns (aligned_seq, frame_used). frame=-1 means fallback.
    """
    seq = _clean(seq)
    candidates = scanner.scan(seq)
    if candidates:
        best = candidates[0]  # sorted by length_codons desc
        return best.seq, best.frame
    # Fallback: first-ATG
    return align_first_atg(seq), -1


def align_3frame(seq: str) -> list[str]:
    """Strategy C: return all 3 reading frames, each codon-aligned."""
    seq = _clean(seq)
    frames = []
    for offset in range(3):
        trimmed = seq[offset:]
        trim = len(trimmed) - (len(trimmed) % 3)
        frames.append(trimmed[:trim])
    return frames


# ---------------------------------------------------------------------------
# Device + embedding extraction
# ---------------------------------------------------------------------------

def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def extract_embeddings(
    sequences: list[str],
    encoder: CaLMEncoder,
    tokenizer: CodonTokenizer,
    device: torch.device,
    label: str = "",
) -> torch.Tensor:
    """Extract [CLS] embeddings with adaptive batching. Returns [N, 768] on CPU."""
    n = len(sequences)
    embeddings = torch.zeros(n, encoder.hidden_size, dtype=torch.float32)
    sorted_idx = sorted(range(n), key=lambda i: len(sequences[i]))

    t0 = time.time()
    i = 0
    done = 0

    while i < n:
        max_codons = len(sequences[sorted_idx[i]]) // 3 + 2
        bs = max(1, min(TOKEN_BUDGET // max_codons, BATCH_EMBED, n - i))
        batch_idx = sorted_idx[i : i + bs]

        encoded = tokenizer.batch_encode(
            [sequences[j] for j in batch_idx], device=device
        )
        with torch.no_grad():
            cls = encoder(encoded["input_ids"], encoded["attention_mask"]).cpu()

        for k, orig in enumerate(batch_idx):
            embeddings[orig] = cls[k]

        if device.type == "mps" and max_codons > 500:
            torch.mps.empty_cache()

        done += bs
        i += bs
        if done % (BATCH_EMBED * 100) < bs or done == n:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            print(f"  {label} [{done:>6}/{n}]  {rate:.0f} seq/s")

    elapsed = time.time() - t0
    print(f"  {label} done: {elapsed:.1f}s ({n / elapsed:.0f} seq/s)")
    return embeddings


# ---------------------------------------------------------------------------
# Train / evaluate helpers
# ---------------------------------------------------------------------------

def stratified_split(labels: list[int]) -> tuple[list[int], list[int]]:
    rng = np.random.RandomState(SEED)
    pos = [i for i, y in enumerate(labels) if y == 1]
    neg = [i for i, y in enumerate(labels) if y == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    n_pos_test = max(1, int(len(pos) * TEST_RATIO))
    n_neg_test = max(1, int(len(neg) * TEST_RATIO))
    test_idx = pos[:n_pos_test] + neg[:n_neg_test]
    train_idx = pos[n_pos_test:] + neg[n_neg_test:]
    rng.shuffle(test_idx)
    rng.shuffle(train_idx)
    return train_idx, test_idx


def train_head(
    head: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    device: torch.device,
) -> list[float]:
    head = head.to(device).train()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR)
    loader = DataLoader(
        TensorDataset(train_x, train_y.float()),
        batch_size=BATCH_TRAIN, shuffle=True,
    )
    losses = []
    for _ in range(EPOCHS):
        epoch_loss = 0.0
        for x_b, y_b in loader:
            x_b, y_b = x_b.to(device), y_b.to(device).unsqueeze(1)
            logits = head(x_b)
            loss = criterion(logits, y_b)
            if hasattr(head, "get_balance_loss"):
                loss = loss + head.get_balance_loss().to(device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(loader))
    return losses


def compute_metrics(preds: torch.Tensor, labels: torch.Tensor) -> dict:
    tp = ((preds == 1) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0
    pre = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * pre * rec / (pre + rec) if (pre + rec) else 0
    denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = ((tp * tn) - (fp * fn)) / denom if denom else 0
    return {
        "ACC": acc * 100, "F1": f1 * 100, "MCC": mcc * 100,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
    }


def evaluate_head(
    head: nn.Module,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    device: torch.device,
) -> dict:
    head = head.to(device).eval()
    all_logits = []
    loader = DataLoader(
        TensorDataset(test_x, test_y), batch_size=BATCH_TRAIN, shuffle=False,
    )
    with torch.no_grad():
        for x_b, _ in loader:
            all_logits.append(head(x_b.to(device)).cpu())
    logits = torch.cat(all_logits).squeeze(-1)
    preds = (torch.sigmoid(logits) > 0.5).long()
    return compute_metrics(preds, test_y.long())


def score_all(
    head: nn.Module,
    embeddings: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Return sigmoid scores [N] for all embeddings."""
    head = head.to(device).eval()
    scores = []
    loader = DataLoader(embeddings, batch_size=BATCH_TRAIN, shuffle=False)
    with torch.no_grad():
        for (x_b,) in DataLoader(TensorDataset(embeddings), batch_size=BATCH_TRAIN):
            logits = head(x_b.to(device)).cpu().squeeze(-1)
            scores.append(torch.sigmoid(logits))
    return torch.cat(scores)


def score_3frame_max(
    head: nn.Module,
    emb_3f: torch.Tensor,
    device: torch.device,
) -> tuple[list[float], list[int]]:
    """Score N seqs with 3 frames each, return (max_scores, best_frames)."""
    head = head.to(device).eval()
    n = emb_3f.shape[0]
    scores_out, frames_out = [], []
    with torch.no_grad():
        for i in range(n):
            frame_embs = emb_3f[i].to(device)  # [3, 768]
            logits = head(frame_embs).squeeze(-1)  # [3]
            s = torch.sigmoid(logits).cpu()
            best = s.argmax().item()
            scores_out.append(s[best].item())
            frames_out.append(best)
    return scores_out, frames_out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    t_start = time.time()

    device = select_device()
    print("=" * 65)
    print("  RNA Challenge Re-test: 3 Preprocessing Strategies")
    print("=" * 65)
    print(f"Device: {device}")

    # --- Load encoder + tokenizer ---
    print(f"\nLoading CaLM encoder from {MODEL_DIR} ...")
    encoder = CaLMEncoder(MODEL_DIR, freeze=True).to(device)
    encoder.eval()
    tokenizer = CodonTokenizer(MODEL_DIR / "vocab.txt")

    # --- Load ORF scanner ---
    ctm = CodonTableManager()
    genetic_code = ctm.get_genetic_code(1)
    scanner = ORFScanner(genetic_code, min_codons=30)

    # --- Load raw data ---
    print("Loading RNA Challenge TSV ...")
    raw_seqs, labels, seq_ids = [], [], []
    with open(DATA_DIR / "rnachallenge.tsv") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            raw_seqs.append(row["sequence"])
            labels.append(int(row["label"]))
            seq_ids.append(row["sequence_id"])

    n = len(raw_seqs)
    n_pos = sum(labels)
    print(f"  Total: {n}  Coding: {n_pos}  Non-coding: {n - n_pos}")

    # --- Align all sequences under each strategy ---
    print("\nAligning sequences ...")
    seqs_a: list[str] = []
    seqs_b: list[str] = []
    frames_b: list[int] = []
    seqs_c_flat: list[str] = []  # 3*N entries

    for raw in raw_seqs:
        seqs_a.append(align_first_atg(raw))
        aligned_b, frame_b = align_longest_orf(raw, scanner)
        seqs_b.append(aligned_b)
        frames_b.append(frame_b)
        three = align_3frame(raw)
        seqs_c_flat.extend(three)

    # Filter: need >= 9 nt in all strategies
    MIN_LEN = 9
    valid = []
    for i in range(n):
        if len(seqs_a[i]) >= MIN_LEN and len(seqs_b[i]) >= MIN_LEN:
            c_ok = all(len(seqs_c_flat[i * 3 + f]) >= MIN_LEN for f in range(3))
            if c_ok:
                valid.append(i)

    print(f"  Valid sequences (>= {MIN_LEN} nt in all strategies): {len(valid)}/{n}")

    f_ids = [seq_ids[i] for i in valid]
    f_labels = [labels[i] for i in valid]
    f_seqs_a = [seqs_a[i] for i in valid]
    f_seqs_b = [seqs_b[i] for i in valid]
    f_frames_b = [frames_b[i] for i in valid]
    f_seqs_c = []
    for i in valid:
        f_seqs_c.append([seqs_c_flat[i * 3 + f] for f in range(3)])

    nv = len(valid)
    train_idx, test_idx = stratified_split(f_labels)
    print(f"  Train: {len(train_idx)}  Test: {len(test_idx)}")

    # --- Extract embeddings ---
    print("\n--- Strategy A: first-ATG ---")
    emb_a = extract_embeddings(f_seqs_a, encoder, tokenizer, device, label="A")

    print("\n--- Strategy B: longest-ORF ---")
    emb_b = extract_embeddings(f_seqs_b, encoder, tokenizer, device, label="B")

    print("\n--- Strategy C: 3-frame (all frames) ---")
    flat_c = []
    for frames in f_seqs_c:
        flat_c.extend(frames)
    emb_c_flat = extract_embeddings(flat_c, encoder, tokenizer, device, label="C")
    emb_c_frames = emb_c_flat.view(nv, 3, -1)  # [N, 3, 768]

    # --- Train/test split ---
    labels_t = torch.tensor(f_labels, dtype=torch.long)
    train_y = labels_t[train_idx]
    test_y = labels_t[test_idx]

    # --- Strategy A: train + test ---
    print(f"\n{'=' * 50}")
    print(f"  Training GatedHead — Strategy A (first-ATG)")
    print(f"{'=' * 50}")
    head_a = GatedHead(hidden_size=768, mlp_hidden=256, dropout=0.1)
    losses_a = train_head(head_a, emb_a[train_idx], train_y, device)
    print(f"  Loss: {losses_a[0]:.4f} -> {losses_a[-1]:.4f}")
    metrics_a = evaluate_head(head_a, emb_a[test_idx], test_y, device)
    print(f"  ACC={metrics_a['ACC']:.2f}%  F1={metrics_a['F1']:.2f}%  "
          f"MCC={metrics_a['MCC']:.2f}%")

    # --- Strategy B: train + test ---
    print(f"\n{'=' * 50}")
    print(f"  Training GatedHead — Strategy B (longest-ORF)")
    print(f"{'=' * 50}")
    head_b = GatedHead(hidden_size=768, mlp_hidden=256, dropout=0.1)
    losses_b = train_head(head_b, emb_b[train_idx], train_y, device)
    print(f"  Loss: {losses_b[0]:.4f} -> {losses_b[-1]:.4f}")
    metrics_b = evaluate_head(head_b, emb_b[test_idx], test_y, device)
    print(f"  ACC={metrics_b['ACC']:.2f}%  F1={metrics_b['F1']:.2f}%  "
          f"MCC={metrics_b['MCC']:.2f}%")

    # --- Strategy C: train on frame-0, test with max over 3 frames ---
    print(f"\n{'=' * 50}")
    print(f"  Training GatedHead — Strategy C (3-frame max)")
    print(f"{'=' * 50}")
    head_c = GatedHead(hidden_size=768, mlp_hidden=256, dropout=0.1)
    losses_c = train_head(head_c, emb_c_frames[train_idx, 0, :], train_y, device)
    print(f"  Loss: {losses_c[0]:.4f} -> {losses_c[-1]:.4f}")

    test_scores_c, test_frames_c = score_3frame_max(
        head_c, emb_c_frames[test_idx], device
    )
    preds_c = torch.tensor([1 if s > 0.5 else 0 for s in test_scores_c], dtype=torch.long)
    metrics_c = compute_metrics(preds_c, test_y.long())
    print(f"  ACC={metrics_c['ACC']:.2f}%  F1={metrics_c['F1']:.2f}%  "
          f"MCC={metrics_c['MCC']:.2f}%")

    results = {
        "A_first_atg": metrics_a,
        "B_longest_orf": metrics_b,
        "C_3frame_max": metrics_c,
    }

    # --- Zero-shot with pretrained MLPHead ---
    if HEAD_PATH.exists():
        print(f"\n{'=' * 50}")
        print(f"  Zero-shot: pretrained MLPHead v1")
        print(f"{'=' * 50}")

        pretrained = MLPHead(hidden_size=768, mlp_hidden=256, dropout=0.1)
        pretrained.load_state_dict(
            torch.load(HEAD_PATH, map_location="cpu", weights_only=True)
        )

        zs_a = evaluate_head(pretrained, emb_a[test_idx], test_y, device)
        print(f"  A_zeroshot: ACC={zs_a['ACC']:.2f}%  F1={zs_a['F1']:.2f}%  MCC={zs_a['MCC']:.2f}%")
        results["A_zeroshot"] = zs_a

        zs_b = evaluate_head(pretrained, emb_b[test_idx], test_y, device)
        print(f"  B_zeroshot: ACC={zs_b['ACC']:.2f}%  F1={zs_b['F1']:.2f}%  MCC={zs_b['MCC']:.2f}%")
        results["B_zeroshot"] = zs_b

        zs_scores_c, zs_frames_c = score_3frame_max(
            pretrained, emb_c_frames[test_idx], device
        )
        zs_preds_c = torch.tensor([1 if s > 0.5 else 0 for s in zs_scores_c], dtype=torch.long)
        zs_c = compute_metrics(zs_preds_c, test_y.long())
        print(f"  C_zeroshot: ACC={zs_c['ACC']:.2f}%  F1={zs_c['F1']:.2f}%  MCC={zs_c['MCC']:.2f}%")
        results["C_zeroshot"] = zs_c

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY ({nv} seqs, test={len(test_idx)})")
    print(f"{'=' * 70}")
    print(f"{'Strategy':<20} {'ACC':>7} {'F1':>7} {'MCC':>7}  "
          f"{'TP':>5} {'TN':>5} {'FP':>5} {'FN':>5}")
    print("-" * 70)
    order = ["A_first_atg", "B_longest_orf", "C_3frame_max",
             "A_zeroshot", "B_zeroshot", "C_zeroshot"]
    for name in order:
        if name not in results:
            continue
        m = results[name]
        print(f"{name:<20} {m['ACC']:>6.2f}% {m['F1']:>6.2f}% {m['MCC']:>6.2f}%  "
              f"{m['TP']:>5} {m['TN']:>5} {m['FP']:>5} {m['FN']:>5}")

    # --- Per-sequence scores for all N sequences ---
    print("\nScoring all sequences for per-seq JSON dump ...")
    all_scores_a = score_all(head_a, emb_a, device)
    all_scores_b = score_all(head_b, emb_b, device)
    all_scores_c, all_frames_c = score_3frame_max(head_c, emb_c_frames, device)

    per_seq = []
    for i in range(nv):
        per_seq.append({
            "sequence_id": f_ids[i],
            "label": f_labels[i],
            "score_A": round(all_scores_a[i].item(), 4),
            "score_B": round(all_scores_b[i].item(), 4),
            "score_C": round(all_scores_c[i], 4),
            "frame_B": f_frames_b[i],
            "frame_C": all_frames_c[i],
        })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "rna_challenge_retest.json"
    with open(out_path, "w") as f:
        json.dump({
            "summary": results,
            "config": {
                "seed": SEED, "epochs": EPOCHS, "lr": LR,
                "test_ratio": TEST_RATIO, "n_valid": nv,
                "n_train": len(train_idx), "n_test": len(test_idx),
            },
            "per_sequence": per_seq,
        }, f, indent=2)

    elapsed = time.time() - t_start
    print(f"\nResults saved to: {out_path}")
    print(f"Total runtime: {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print("Done.")


if __name__ == "__main__":
    main()
