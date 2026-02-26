"""Build agent training episodes from GENCODE annotation data.

Parses downloaded GENCODE v49 FASTA files into a balanced set of
coding and non-coding ORF examples for RL agent training.

Coding examples: longest ORF from protein-coding transcripts (pc_transcripts).
Non-coding examples: longest ORF from lncRNA transcripts.
"""

from __future__ import annotations

import gzip
import json
import random
from pathlib import Path
from typing import Optional

MIN_CODONS = 30
MAX_CODONS = 1000


def _parse_fasta_gz(path: Path) -> dict[str, str]:
    """Parse a gzipped FASTA file into {header_id: sequence} dict.

    Uses the first field of the header (up to '|' or whitespace) as ID.
    """
    sequences: dict[str, str] = {}
    current_id: Optional[str] = None
    chunks: list[str] = []

    with gzip.open(path, "rt") as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if current_id is not None:
                    sequences[current_id] = "".join(chunks)
                # GENCODE headers: >ENST00000456328.2|ENSG...
                raw_id = line[1:].split("|")[0].split()[0]
                current_id = raw_id
                chunks = []
            else:
                chunks.append(line)
        if current_id is not None:
            sequences[current_id] = "".join(chunks)

    return sequences


def _find_longest_orf(sequence: str) -> Optional[str]:
    """Find the longest ATG-initiated ORF in a DNA sequence.

    Scans all three reading frames. For each ATG, extends to the first
    in-frame stop codon. If no stop codon is found before the end of
    the sequence, the ORF extends to the last complete codon (run-off).

    Returns the ORF DNA sequence or None if no ATG found.
    """
    seq = sequence.upper().replace("U", "T")
    stop_codons = {"TAA", "TAG", "TGA"}
    best_orf: Optional[str] = None
    best_len = 0

    for frame in range(3):
        i = frame
        while i <= len(seq) - 3:
            codon = seq[i : i + 3]
            if codon != "ATG":
                i += 3
                continue
            # Found ATG — scan for stop
            orf_start = i
            j = i + 3
            while j <= len(seq) - 3:
                c = seq[j : j + 3]
                if c in stop_codons:
                    j += 3  # include stop codon in ORF
                    break
                j += 3
            # If no stop found, ORF extends to last complete codon (run-off)
            orf_seq = seq[orf_start:j]
            orf_codons = len(orf_seq) // 3
            if orf_codons > best_len:
                best_len = orf_codons
                best_orf = orf_seq
            i = j  # skip past this ORF
    return best_orf


class DataBuilder:
    """Build agent training episodes from GENCODE data.

    Args:
        gencode_dir: Directory containing downloaded GENCODE files.
        output_dir: Directory to write episodes.jsonl.
    """

    def __init__(self, gencode_dir: Path | str, output_dir: Path | str):
        self.gencode_dir = Path(gencode_dir).resolve()
        self.output_dir = Path(output_dir)

    def build(
        self,
        max_coding: int = 5000,
        max_noncoding: int = 5000,
        seed: int = 42,
    ) -> Path:
        """Parse GENCODE data, extract ORFs, write balanced episodes JSONL.

        Args:
            max_coding: Maximum coding examples to include.
            max_noncoding: Maximum non-coding examples to include.
            seed: Random seed for sampling.

        Returns:
            Path to the output episodes.jsonl file.
        """
        rng = random.Random(seed)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / "episodes.jsonl"

        coding = self._extract_coding(max_coding, rng)
        noncoding = self._extract_noncoding(max_noncoding, rng)

        # Combine and shuffle
        episodes = coding + noncoding
        rng.shuffle(episodes)

        with open(out_path, "w") as fh:
            for ep in episodes:
                fh.write(json.dumps(ep) + "\n")

        print(f"Wrote {len(episodes)} episodes ({len(coding)} coding, "
              f"{len(noncoding)} noncoding) to {out_path}")
        return out_path

    def _extract_coding(
        self, max_count: int, rng: random.Random
    ) -> list[dict]:
        """Extract longest ORFs from protein-coding transcripts.

        Uses pc_transcripts FASTA directly. The longest ATG-initiated ORF
        in each transcript is assumed to be the CDS (high confidence for
        GENCODE-annotated protein-coding transcripts).
        """
        pc_fasta = self._find_file("pc_transcripts.fa.gz")
        if pc_fasta is None:
            print("Warning: protein-coding FASTA not found, "
                  "skipping coding examples")
            return []

        print("Parsing protein-coding transcripts...")
        transcripts = _parse_fasta_gz(pc_fasta)
        print(f"  Loaded {len(transcripts)} transcripts")

        coding: list[dict] = []
        for tid, sequence in transcripts.items():
            orf_seq = _find_longest_orf(sequence)
            if orf_seq is None:
                continue

            n_codons = len(orf_seq) // 3
            if n_codons < MIN_CODONS or n_codons > MAX_CODONS:
                continue

            coding.append({
                "transcript_id": tid,
                "sequence": orf_seq,
                "label": "coding",
            })

        if len(coding) > max_count:
            coding = rng.sample(coding, max_count)

        print(f"  Extracted {len(coding)} coding ORFs")
        return coding

    def _extract_noncoding(
        self, max_count: int, rng: random.Random
    ) -> list[dict]:
        """Extract longest ORFs from lncRNA transcripts."""
        lncrna_fasta = self._find_file("lncRNA_transcripts.fa.gz")
        if lncrna_fasta is None:
            lncrna_fasta = self._find_file("lncrna.fa.gz")
        if lncrna_fasta is None:
            print("Warning: lncRNA FASTA not found, skipping noncoding examples")
            return []

        print("Parsing lncRNA transcripts...")
        transcripts = _parse_fasta_gz(lncrna_fasta)
        print(f"  Loaded {len(transcripts)} lncRNA transcripts")

        noncoding: list[dict] = []
        for tid, sequence in transcripts.items():
            orf_seq = _find_longest_orf(sequence)
            if orf_seq is None:
                continue

            n_codons = len(orf_seq) // 3
            if n_codons < MIN_CODONS or n_codons > MAX_CODONS:
                continue

            noncoding.append({
                "transcript_id": tid,
                "sequence": orf_seq,
                "label": "noncoding",
            })

        if len(noncoding) > max_count:
            noncoding = rng.sample(noncoding, max_count)

        print(f"  Extracted {len(noncoding)} noncoding ORFs")
        return noncoding

    def _find_file(self, suffix: str) -> Optional[Path]:
        """Find a file in gencode_dir matching the given suffix.

        Searches the top-level directory and one level of subdirectories.
        Skips hidden files. Deterministic: picks shortest filename on ties.
        """
        candidates: list[Path] = []
        for f in self.gencode_dir.iterdir():
            if f.is_file() and f.name.endswith(suffix) and not f.name.startswith("."):
                candidates.append(f)
        for d in self.gencode_dir.iterdir():
            if d.is_dir() and not d.name.startswith("."):
                for f in d.iterdir():
                    if f.is_file() and f.name.endswith(suffix) and not f.name.startswith("."):
                        candidates.append(f)
        if not candidates:
            return None
        candidates.sort(key=lambda p: (len(p.name), p.name))
        return candidates[0]
