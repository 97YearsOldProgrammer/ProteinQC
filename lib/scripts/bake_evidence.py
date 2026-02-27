"""Bake all biological evidence offline for GRPO v2 training.

Runs CaLM scorer, perplexity scorer, translate, and Pfam scanner in batch
on all RNA Challenge sequences. Outputs one JSONL line per sequence.

Supports incremental resume: if output file exists, skips already-baked IDs.

Usage:
    bake-evidence --data data/rnachallenge/rnachallenge.tsv
    bake-evidence --skip-perplexity --skip-pfam   # CaLM-only, ~15 min
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _extract_species(sequence_id: str) -> str:
    """Extract species from RNA Challenge sequence ID string."""
    sid = sequence_id
    if "Homo sapiens" in sid or sid.startswith("ENST0") or "NONHSAG" in sid:
        return "Homo sapiens"
    if "Mus musculus" in sid or sid.startswith("ENSMUST0") or "NONMMUT" in sid:
        return "Mus musculus"
    if "Drosophila melanogaster" in sid or "NONDMET" in sid:
        return "Drosophila melanogaster"
    if "Drosophila pseudoobscura" in sid:
        return "Drosophila pseudoobscura"
    if "Drosophila" in sid:
        return "Drosophila sp."
    if "Caenorhabditis elegans" in sid:
        return "Caenorhabditis elegans"
    if "Danio rerio" in sid or sid.startswith("ENSDART0"):
        return "Danio rerio"
    if "Arabidopsis" in sid or "NONATHG" in sid:
        return "Arabidopsis thaliana"
    if "Oryza" in sid or "LOC_Os" in sid or sid.startswith("Os"):
        return "Oryza sativa"
    if "GRMZM" in sid:
        return "Zea mays"
    if "NONRATT" in sid:
        return "Rattus norvegicus"
    return "unknown"


def _load_sequences(tsv_path: Path) -> list[dict]:
    """Load sequence_id, sequence, label, species from RNA Challenge TSV."""
    import csv

    items: list[dict] = []
    with open(tsv_path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            raw_label = row["label"].strip()
            label = {"1": "coding", "0": "noncoding"}.get(raw_label)
            if label is None:
                raise ValueError(f"Unknown label '{raw_label}' for {row['sequence_id']}")
            seq_id = row["sequence_id"].strip()
            items.append({
                "sequence_id": seq_id,
                "sequence": row["sequence"].strip(),
                "label": label,
                "species": _extract_species(seq_id),
            })
    return items


def _load_done_ids(output_path: Path) -> set[str]:
    """Read already-baked sequence IDs from existing output JSONL."""
    done: set[str] = set()
    if not output_path.exists():
        return done
    with open(output_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                done.add(obj["sequence_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description="Bake biological evidence for GRPO v2")

    parser.add_argument("--data", default="data/rnachallenge/rnachallenge.tsv",
                        help="RNA Challenge TSV path")
    parser.add_argument("--output", default="data/rnachallenge/evidence_baked.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--calm-dir", default="models/calm",
                        help="CaLM model directory")
    parser.add_argument("--head-path", default="models/heads/mlp_head_v1.pt",
                        help="MLP head weights path")
    parser.add_argument("--pfam-db", default=None,
                        help="Pfam-A.hmm path (optional)")
    parser.add_argument("--skip-perplexity", action="store_true",
                        help="Skip perplexity scoring (faster)")
    parser.add_argument("--skip-pfam", action="store_true",
                        help="Skip Pfam domain scanning")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for CaLM scoring")

    args = parser.parse_args()

    data_path = Path(args.data)
    output_path = Path(args.output)

    if not data_path.exists():
        print(f"Data not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    # Load sequences
    print(f"Loading sequences from {data_path}...", flush=True)
    all_seqs = _load_sequences(data_path)
    print(f"  Total sequences: {len(all_seqs)}", flush=True)

    # Check for incremental resume
    done_ids = _load_done_ids(output_path)
    remaining = [s for s in all_seqs if s["sequence_id"] not in done_ids]
    if done_ids:
        print(f"  Already baked: {len(done_ids)}, remaining: {len(remaining)}", flush=True)
    if not remaining:
        print("  All sequences already baked. Nothing to do.", flush=True)
        return

    # Initialize tools
    calm_scorer = None
    perplexity_scorer = None
    pfam_scanner = None

    # CaLM scorer (always needed)
    calm_dir = Path(args.calm_dir)
    head_path = Path(args.head_path)
    if not (calm_dir / "model.safetensors").exists():
        print(f"CaLM model not found: {calm_dir}", file=sys.stderr)
        sys.exit(1)
    if not head_path.exists():
        print(f"MLP head not found: {head_path}", file=sys.stderr)
        sys.exit(1)

    print("Loading CaLM scorer...", flush=True)
    from proteinqc.tools.calm_scorer import CaLMScorer
    calm_scorer = CaLMScorer(calm_dir, head_path)

    # Perplexity scorer (optional)
    if not args.skip_perplexity:
        print("Loading perplexity scorer...", flush=True)
        from proteinqc.tools.perplexity_scorer import PerplexityScorer
        perplexity_scorer = PerplexityScorer(calm_dir)

    # Pfam scanner (optional)
    if not args.skip_pfam and args.pfam_db:
        pfam_path = Path(args.pfam_db)
        if pfam_path.exists():
            print(f"Loading Pfam scanner: {pfam_path}", flush=True)
            from proteinqc.tools.pfam_scanner import PfamScanner
            pfam_scanner = PfamScanner(pfam_path)
        else:
            print(f"Pfam DB not found, skipping: {pfam_path}", flush=True)

    # Translation import
    from proteinqc.tools.translate import translate

    # CaLM max: 1026 tokens = 1024 codons + CLS + EOS = 3072 bp
    CALM_MAX_BP = 3072

    # Bake in batches
    output_path.parent.mkdir(parents=True, exist_ok=True)
    batch_size = args.batch_size
    total = len(remaining)
    start_time = time.time()
    n_truncated = sum(1 for s in remaining if len(s["sequence"]) > CALM_MAX_BP)

    print(f"\nBaking {total} sequences...", flush=True)
    print(f"  CaLM: yes | Perplexity: {not args.skip_perplexity} | "
          f"Pfam: {pfam_scanner is not None}", flush=True)
    if n_truncated:
        print(f"  Truncating {n_truncated} sequences to {CALM_MAX_BP} bp for CaLM", flush=True)

    with open(output_path, "a") as out_fh:
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch = remaining[batch_start:batch_end]
            sequences = [item["sequence"] for item in batch]

            # Truncate for CaLM scoring (full sequence kept for translation/Pfam)
            calm_sequences = [seq[:CALM_MAX_BP] for seq in sequences]

            # CaLM scores (batch, on truncated sequences)
            calm_scores = calm_scorer.batch_score(calm_sequences)

            # Perplexity scores (sequential — each creates N masked copies)
            ppl_scores: list[float | None] = [None] * len(batch)
            if perplexity_scorer is not None:
                ppl_scores = perplexity_scorer.batch_score(sequences)

            # Translations
            translations = [translate(seq) for seq in sequences]

            # Pfam domains (batch on translations)
            pfam_results: list[list[str]] = [[] for _ in batch]
            if pfam_scanner is not None:
                proteins_for_scan = []
                scan_indices = []
                for idx, protein in enumerate(translations):
                    if len(protein) >= 10:
                        proteins_for_scan.append(protein)
                        scan_indices.append(idx)

                if proteins_for_scan:
                    hits_list = pfam_scanner.scan(proteins_for_scan)
                    for scan_idx, hits in zip(scan_indices, hits_list):
                        pfam_results[scan_idx] = [
                            f"{h.domain_name}(E={h.e_value:.1e})"
                            for h in hits
                        ]

            # Write JSONL
            for i, item in enumerate(batch):
                record = {
                    "sequence_id": item["sequence_id"],
                    "sequence": item["sequence"],
                    "label": item["label"],
                    "species": item["species"],
                    "calm_score": calm_scores[i],
                    "perplexity": ppl_scores[i],
                    "translation": translations[i] if translations[i] else None,
                    "pfam_domains": pfam_results[i],
                }
                out_fh.write(json.dumps(record) + "\n")
            out_fh.flush()

            elapsed = time.time() - start_time
            done = batch_end
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            print(
                f"  [{done}/{total}] {100 * done / total:.1f}% "
                f"({rate:.1f} seq/s, ETA {eta / 60:.1f} min)",
                flush=True,
            )

    elapsed = time.time() - start_time
    print(f"\nBaking complete: {total} sequences in {elapsed / 60:.1f} min", flush=True)
    print(f"Output: {output_path}", flush=True)


if __name__ == "__main__":
    main()
