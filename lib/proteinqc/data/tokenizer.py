"""Pure codon tokenizer for CaLM — no HuggingFace dependencies.

Loads vocab.txt (131 codon-level tokens) and encodes DNA/RNA sequences
into input_ids + attention_mask tensors matching HuggingFace tokenizer output.
"""

from pathlib import Path

import torch


# Special token IDs (matching vocab.txt line order)
PAD_ID = 0   # <pad>
CLS_ID = 1   # <cls>
EOS_ID = 2   # <eos>
UNK_ID = 3   # <unk>


class CodonTokenizer:
    """Codon-level tokenizer for CaLM.

    Loads vocab from vocab.txt, converts DNA→RNA, splits into 3-char codons,
    and encodes to integer IDs with <cls>/<eos> framing.

    Args:
        vocab_path: Path to vocab.txt (131 lines, one token per line)
    """

    def __init__(self, vocab_path: Path | str):
        self.vocab_path = Path(vocab_path)
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self._load_vocab()

    def _load_vocab(self):
        with open(self.vocab_path) as f:
            for idx, line in enumerate(f):
                token = line.strip()
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def encode(self, sequence: str) -> list[int]:
        """Encode a single DNA/RNA sequence to token IDs.

        Converts T→U, splits into 3-char codons, maps to vocab IDs.
        Adds <cls> prefix and <eos> suffix.

        Args:
            sequence: DNA or RNA sequence (must be codon-aligned, len % 3 == 0)

        Returns:
            List of token IDs: [<cls>, codon1, codon2, ..., <eos>]
        """
        seq = sequence.upper().replace("T", "U")
        ids = [CLS_ID]
        for i in range(0, len(seq), 3):
            codon = seq[i : i + 3]
            if len(codon) == 3:
                ids.append(self.token_to_id.get(codon, UNK_ID))
        ids.append(EOS_ID)
        return ids

    def batch_encode(
        self,
        sequences: list[str],
        device: torch.device | str = "cpu",
    ) -> dict[str, torch.Tensor]:
        """Encode a batch of sequences with padding.

        Args:
            sequences: List of DNA/RNA sequences
            device: Target device for output tensors

        Returns:
            Dict with 'input_ids' and 'attention_mask' tensors,
            both of shape [batch, max_seq_len].
        """
        encoded = [self.encode(seq) for seq in sequences]
        max_len = max(len(ids) for ids in encoded)

        input_ids = torch.full(
            (len(sequences), max_len), PAD_ID, dtype=torch.long, device=device,
        )
        attention_mask = torch.zeros(
            len(sequences), max_len, dtype=torch.long, device=device,
        )

        for i, ids in enumerate(encoded):
            length = len(ids)
            input_ids[i, :length] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :length] = 1

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to codon string (for debugging)."""
        tokens = []
        for token_id in ids:
            token = self.id_to_token.get(token_id, "<unk>")
            if token in ("<pad>", "<cls>", "<eos>", "<unk>", "<mask>", "<null>"):
                continue
            tokens.append(token)
        return "".join(tokens)
