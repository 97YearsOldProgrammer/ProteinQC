"""Pfam domain scanner using pyhmmer (native Python HMMER3 bindings).

Runs hmmsearch against Pfam-A HMM database to identify known protein
domains in translated ORF sequences. Uses pyhmmer for in-process
search — no subprocess, no temp files, no HMMER binary required.

pip install pyhmmer
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pyhmmer
from pyhmmer.easel import Alphabet, DigitalSequenceBlock, TextSequence
from pyhmmer.plan7 import HMMFile

_VALID_AA = re.compile(r"^[A-Za-z*]+$")


@dataclass(frozen=True)
class DomainHit:
    """A single Pfam domain hit.

    Attributes:
        domain_id: Pfam accession (e.g. "PF00069.29")
        domain_name: Human-readable name (e.g. "Pkinase")
        e_value: Domain-level independent E-value
        score: Domain bit score
        is_antifam: True if hit comes from AntiFam database
        query_name: Name of the query sequence that produced this hit
        ali_from: Alignment start position in query (1-based)
        ali_to: Alignment end position in query (1-based)
    """

    domain_id: str
    domain_name: str
    e_value: float
    score: float
    is_antifam: bool
    query_name: str
    ali_from: int
    ali_to: int


class PfamScanner:
    """Scan protein sequences against Pfam-A using pyhmmer hmmsearch.

    Preloads all HMMs into memory (~2GB for Pfam-A) for fast repeated
    searches. Uses Pfam gathering thresholds for domain inclusion.

    Args:
        pfam_db_path: Path to Pfam-A.hmm file.
        e_value_threshold: Not used when gathering thresholds are
            available (kept for API compatibility).
    """

    def __init__(
        self,
        pfam_db_path: Path | str,
        e_value_threshold: float = 1e-5,
    ):
        self.pfam_db = Path(pfam_db_path)
        self.e_value_threshold = e_value_threshold
        self._verify_db(self.pfam_db)
        self._alphabet = Alphabet.amino()
        self._hmms: list | None = None

    def _load_hmms(self) -> None:
        """Preload all HMMs from database into memory."""
        if self._hmms is not None:
            return
        print(f"Loading HMMs from {self.pfam_db} ...", flush=True)
        with HMMFile(str(self.pfam_db)) as hmm_file:
            self._hmms = list(hmm_file)
        print(f"Loaded {len(self._hmms):,} HMMs", flush=True)

    def scan(
        self,
        protein_sequences: list[str],
    ) -> list[list[DomainHit]]:
        """Run hmmsearch on protein sequences against all Pfam HMMs.

        Args:
            protein_sequences: Amino acid sequences (single-letter codes).

        Returns:
            List of hit lists, one per input sequence (same order).
            Each inner list contains DomainHit objects sorted by E-value.
        """
        if not protein_sequences:
            return []

        self._load_hmms()

        sanitized = [_sanitize_sequence(s) for s in protein_sequences]
        names = [f"seq_{i}" for i in range(len(sanitized))]

        # Build digital sequences into a shared block (no per-thread copy)
        seqs = DigitalSequenceBlock(
            self._alphabet,
            [
                TextSequence(name=name, sequence=seq).digitize(
                    self._alphabet
                )
                for name, seq in zip(names, sanitized)
            ],
        )

        hits_by_name: dict[str, list[DomainHit]] = {n: [] for n in names}
        total_domains = 0
        hmm_count = 0
        n_hmms = len(self._hmms)

        for top_hits in pyhmmer.hmmsearch(
            self._hmms, seqs, cpus=0, bit_cutoffs="gathering"
        ):
            hmm_count += 1
            hmm_acc = top_hits.query.accession or ""
            hmm_name = top_hits.query.name

            for hit in top_hits:
                if not hit.included:
                    continue
                seq_name = hit.name
                for domain in hit.domains:
                    if not domain.included:
                        continue
                    total_domains += 1
                    hits_by_name[seq_name].append(
                        DomainHit(
                            domain_id=hmm_acc,
                            domain_name=hmm_name,
                            e_value=domain.i_evalue,
                            score=domain.score,
                            is_antifam=False,
                            query_name=seq_name,
                            ali_from=domain.alignment.target_from,
                            ali_to=domain.alignment.target_to,
                        )
                    )

            if hmm_count % 2000 == 0:
                print(
                    f"  {hmm_count:,}/{n_hmms:,} HMMs "
                    f"({100 * hmm_count / n_hmms:.0f}%), "
                    f"{total_domains:,} hits so far",
                    file=sys.stderr,
                    flush=True,
                )

        print(
            f"Done: {n_hmms:,} HMMs x {len(seqs):,} sequences "
            f"-> {total_domains:,} domain hits",
            flush=True,
        )

        return [
            sorted(hits_by_name[n], key=lambda h: h.e_value) for n in names
        ]

    @staticmethod
    def _verify_db(db_path: Path) -> None:
        """Verify HMM database file exists."""
        if not db_path.exists():
            raise FileNotFoundError(f"HMM database not found: {db_path}")


def _sanitize_sequence(seq: str) -> str:
    """Strip whitespace and validate amino acid characters."""
    cleaned = seq.strip().replace("\n", "").replace("\r", "")
    if not cleaned:
        raise ValueError("Empty protein sequence")
    if not _VALID_AA.match(cleaned):
        bad = set(cleaned) - set(
            "ACDEFGHIKLMNPQRSTVWXYacdefghiklmnpqrstvwxy*"
        )
        raise ValueError(f"Invalid characters in protein sequence: {bad}")
    return cleaned
