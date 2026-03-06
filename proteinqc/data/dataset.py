"""Batching infrastructure for CaLM token-budget training and inference."""

from __future__ import annotations

import torch


# ── Batching infrastructure ──────────────────────────────────────

PAD_BUCKETS = list(range(64, 2113, 64))  # 33 bucket sizes, covers seqs up to 2112


def _bucket_pad(length: int) -> int:
    """Round up length to nearest bucket boundary for torch.compile shape stability."""
    for b in PAD_BUCKETS:
        if length <= b:
            return b
    return PAD_BUCKETS[-1]


def _length_sorted_chunks(source, sort_size: int = 256):
    """Accumulate samples, sort by length, yield in length order."""
    buf: list[dict] = []
    for sample in source:
        buf.append(sample)
        if len(buf) >= sort_size:
            buf.sort(key=lambda s: len(s["input_ids"]))
            yield from buf
            buf = []
    if buf:
        buf.sort(key=lambda s: len(s["input_ids"]))
        yield from buf


def token_budget_batcher(
    source, budget: int, max_batch: int, collator, sort_size: int = 256,
):
    """Yield collated batches fitting within token budget.

    Total tokens per batch = max_padded_length * batch_size <= budget.
    Groups similar lengths together via length-sorted chunking.
    """
    batch: list[dict] = []
    max_pad = 0

    for sample in _length_sorted_chunks(source, sort_size):
        padded = _bucket_pad(min(len(sample["input_ids"]), MAX_SEQ_LEN))
        new_max = max(max_pad, padded)

        if batch and (new_max * (len(batch) + 1) > budget or len(batch) >= max_batch):
            yield collator(batch)
            batch = [sample]
            max_pad = padded
        else:
            batch.append(sample)
            max_pad = new_max

    if batch:
        yield collator(batch)


MAX_SEQ_LEN = 2048  # ALiBi has no position limit; 2048 covers 96% of data, fits 4090


def collate_binary(samples: list[dict]) -> dict[str, torch.Tensor]:
    """Pad input_ids to bucket boundary, create attention_mask, stack labels."""
    max_len = min(max(len(s["input_ids"]) for s in samples), MAX_SEQ_LEN)
    pad_len = _bucket_pad(max_len)

    input_ids = torch.zeros(len(samples), pad_len, dtype=torch.long)
    attention_mask = torch.zeros(len(samples), pad_len, dtype=torch.long)
    labels = torch.tensor([s["label"] for s in samples], dtype=torch.long)

    for i, s in enumerate(samples):
        length = min(len(s["input_ids"]), MAX_SEQ_LEN)
        input_ids[i, :length] = s["input_ids"][:length]
        attention_mask[i, :length] = 1

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def pre_tokenize(sequences: list[str], labels: list[int], tokenizer) -> list[dict]:
    """Encode all sequences once upfront.

    Returns list of {input_ids: 1D tensor, label: int}.
    Eliminates per-batch string tokenization from the training hot path.
    """
    samples = []
    for seq, label in zip(sequences, labels):
        ids = tokenizer.encode(seq)
        samples.append({
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "label": label,
        })
    return samples


