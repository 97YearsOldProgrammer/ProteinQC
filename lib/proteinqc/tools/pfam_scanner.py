"""Pfam domain scanner using HMMER3 hmmscan.

Runs hmmscan against Pfam-A and optionally AntiFam HMM databases to
identify known protein domains in translated ORF sequences. AntiFam
hits flag spurious/non-functional translations.

Requires HMMER3 installed (conda install -c bioconda hmmer).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_VALID_AA = re.compile(r"^[A-Za-z*]+$")


@dataclass(frozen=True)
class DomainHit:
    """A single Pfam/AntiFam domain hit from hmmscan.

    Attributes:
        domain_id: Pfam accession (e.g. "PF00069") or AntiFam ID
        domain_name: Human-readable name (e.g. "Pkinase")
        e_value: Domain-level E-value (independent, not full-sequence)
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
    """Scan protein sequences against Pfam-A and AntiFam HMM databases.

    Wraps HMMER3 hmmscan with structured output parsing.

    Args:
        pfam_db_path: Path to Pfam-A.hmm (must be hmmpress'd).
        antifam_db_path: Path to AntiFam.hmm (optional, must be hmmpress'd).
        e_value_threshold: Max domain E-value to report (default: 1e-5).
    """

    def __init__(
        self,
        pfam_db_path: Path | str,
        antifam_db_path: Optional[Path | str] = None,
        e_value_threshold: float = 1e-5,
    ):
        self.pfam_db = Path(pfam_db_path)
        self.antifam_db = Path(antifam_db_path) if antifam_db_path else None
        self.e_value_threshold = e_value_threshold

        self._verify_hmmscan()
        self._verify_db(self.pfam_db)
        if self.antifam_db:
            self._verify_db(self.antifam_db)

    def scan(self, protein_sequences: list[str]) -> list[list[DomainHit]]:
        """Run hmmscan on translated protein sequences.

        Args:
            protein_sequences: Amino acid sequences (single-letter codes).

        Returns:
            List of hit lists, one per input sequence (same order).
            Each inner list contains DomainHit objects sorted by E-value.
        """
        if not protein_sequences:
            return []

        # Sanitize sequences at system boundary
        sanitized = [_sanitize_sequence(s) for s in protein_sequences]

        # Build name→index mapping for output ordering
        names = [f"seq_{i}" for i in range(len(sanitized))]
        hits_by_name: dict[str, list[DomainHit]] = {n: [] for n in names}

        # Run against Pfam-A
        pfam_hits = self._run_hmmscan(
            sanitized, names, self.pfam_db, is_antifam=False
        )
        for hit in pfam_hits:
            hits_by_name[hit.query_name].append(hit)

        # Optionally run against AntiFam
        if self.antifam_db:
            antifam_hits = self._run_hmmscan(
                sanitized, names, self.antifam_db, is_antifam=True
            )
            for hit in antifam_hits:
                hits_by_name[hit.query_name].append(hit)

        # Preserve input order, sort each list by E-value
        return [
            sorted(hits_by_name[n], key=lambda h: h.e_value)
            for n in names
        ]

    def _run_hmmscan(
        self,
        sequences: list[str],
        names: list[str],
        db_path: Path,
        is_antifam: bool,
    ) -> list[DomainHit]:
        """Execute hmmscan and parse domtblout output."""
        fasta_path: Optional[str] = None
        domtblout_path: Optional[str] = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".fasta", delete=False
            ) as fasta_f:
                fasta_path = fasta_f.name
                for name, seq in zip(names, sequences):
                    fasta_f.write(f">{name}\n{seq}\n")

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".domtblout", delete=False
            ) as out_f:
                domtblout_path = out_f.name

            cmd = [
                "hmmscan",
                "--domtblout", domtblout_path,
                "--domE", str(self.e_value_threshold),
                "--noali",
                "--cpu", "1",
                str(db_path),
                fasta_path,
            ]
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=300,
            )
            return self._parse_domtblout(domtblout_path, is_antifam)
        finally:
            if fasta_path:
                Path(fasta_path).unlink(missing_ok=True)
            if domtblout_path:
                Path(domtblout_path).unlink(missing_ok=True)

    def _parse_domtblout(
        self, path: str, is_antifam: bool
    ) -> list[DomainHit]:
        """Parse HMMER3 --domtblout format.

        domtblout columns (space-separated, first 22 are fixed):
        0: target name (domain)
        1: target accession
        2: tlen
        3: query name
        4: query accession
        5: qlen
        6: full E-value
        7: full score
        8: full bias
        9: domain # (this domain)
        10: domain # (of)
        11: domain c-Evalue
        12: domain i-Evalue
        13: domain score
        14: domain bias
        15: hmm from
        16: hmm to
        17: ali from
        18: ali to
        19: env from
        20: env to
        21: acc
        22+: description
        """
        hits: list[DomainHit] = []
        with open(path) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                fields = line.split()
                if len(fields) < 22:
                    continue
                hits.append(
                    DomainHit(
                        domain_id=fields[1],      # target accession
                        domain_name=fields[0],     # target name
                        e_value=float(fields[12]),  # domain i-Evalue
                        score=float(fields[13]),    # domain score
                        is_antifam=is_antifam,
                        query_name=fields[3],       # query name
                        ali_from=int(fields[17]),   # alignment start
                        ali_to=int(fields[18]),     # alignment end
                    )
                )
        return hits

    @staticmethod
    def _verify_hmmscan():
        """Check that hmmscan is on PATH."""
        if shutil.which("hmmscan") is None:
            raise RuntimeError(
                "hmmscan not found. Install HMMER3: "
                "conda install -c bioconda hmmer"
            )

    @staticmethod
    def _verify_db(db_path: Path):
        """Verify HMM database and its hmmpress index files exist."""
        if not db_path.exists():
            raise FileNotFoundError(f"HMM database not found: {db_path}")
        h3m = db_path.with_suffix(db_path.suffix + ".h3m")
        if not h3m.exists():
            raise FileNotFoundError(
                f"HMM index missing ({h3m.name}). Run: hmmpress {db_path}"
            )


def _sanitize_sequence(seq: str) -> str:
    """Strip whitespace and validate amino acid characters."""
    cleaned = seq.strip().replace("\n", "").replace("\r", "")
    if not cleaned:
        raise ValueError("Empty protein sequence")
    if not _VALID_AA.match(cleaned):
        bad_chars = set(cleaned) - set("ACDEFGHIKLMNPQRSTVWXYacdefghiklmnpqrstvwxy*")
        raise ValueError(f"Invalid characters in protein sequence: {bad_chars}")
    return cleaned
