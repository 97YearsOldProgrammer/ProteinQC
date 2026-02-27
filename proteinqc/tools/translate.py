"""DNA → protein translation utility.

Translates DNA sequences to amino acid sequences using Biopython's
CodonTable for genetic code lookup. Stops at first stop codon.
"""

from __future__ import annotations

from Bio.Data.CodonTable import unambiguous_dna_by_id


def translate(dna_sequence: str, genetic_code_id: int = 1) -> str:
    """Translate a DNA sequence to an amino acid sequence.

    Uses the specified NCBI genetic code. Translation stops at the
    first in-frame stop codon. Incomplete trailing codons are ignored.

    Args:
        dna_sequence: DNA sequence (T not U, uppercase recommended).
                      Length should be a multiple of 3 for full codons.
        genetic_code_id: NCBI genetic code ID (default: 1 = standard).

    Returns:
        Amino acid sequence (single-letter codes). Empty string if
        the sequence is too short or starts with a stop codon.
    """
    seq = dna_sequence.upper().replace("U", "T")
    if genetic_code_id not in unambiguous_dna_by_id:
        raise ValueError(
            f"Unknown genetic code ID {genetic_code_id}. "
            f"Valid: {sorted(unambiguous_dna_by_id.keys())}"
        )
    table = unambiguous_dna_by_id[genetic_code_id]
    forward = table.forward_table
    stop_codons = set(table.stop_codons)

    amino_acids: list[str] = []
    for i in range(0, len(seq) - 2, 3):
        codon = seq[i : i + 3]
        if codon in stop_codons:
            break
        aa = forward.get(codon)
        if aa is None:
            amino_acids.append("X")  # ambiguous/unknown codon
        else:
            amino_acids.append(aa)

    return "".join(amino_acids)


def translate_all_frames(dna_sequence: str, genetic_code_id: int = 1) -> list[str]:
    """Translate a DNA sequence in all three forward reading frames.

    Args:
        dna_sequence: DNA sequence (T not U).
        genetic_code_id: NCBI genetic code ID (default: 1 = standard).

    Returns:
        List of 3 protein sequences, one per frame (offsets 0, 1, 2).
    """
    return [
        translate(dna_sequence[frame:], genetic_code_id)
        for frame in range(3)
    ]
