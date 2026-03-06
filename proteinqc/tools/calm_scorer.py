"""CaLM-based ORF scorer using frozen encoder + classification head.

Wraps CaLMEncoder (frozen) + classification head (GatedHead or MLPHead)
for scoring ORF candidates. Head type auto-detected from state dict keys.
Uses TOKEN_BUDGET=8192 adaptive batching (same pattern as benchmark.py).
Model loaded once on construction, reused across all calls.

Supports two weight formats:
- File (.pt): standalone head state dict (backward compat)
- Directory (LoRA checkpoint): adapter_config.json + adapter_model.pt + head.pt
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch

TOKEN_BUDGET = 8_192  # max total tokens per batch (MPS memory constraint)
BATCH_MAX = 16        # absolute max batch size


class CaLMScorer:
    """Score DNA sequences for coding potential using frozen CaLM + head.

    Args:
        model_dir: Path to CaLM model directory (config.json + model.safetensors)
        head_weights_path: Path to head weights — either a .pt file (standalone
            head) or a directory containing a LoRA checkpoint (adapter_config.json,
            adapter_model.pt, head.pt).
        device: Compute device. Auto-selects MPS/CUDA/CPU if None.
    """

    def __init__(
        self,
        model_dir: Path | str,
        head_weights_path: Path | str,
        device: Optional[torch.device] = None,
        position_type: str = "rotary",
    ):
        from proteinqc.data.tokenizer import CodonTokenizer
        from proteinqc.models.calm_encoder import CaLMEncoder

        self.device = device or _select_device()
        self.model_dir = Path(model_dir)
        self.head_weights_path = Path(head_weights_path)

        self.tokenizer = CodonTokenizer(self.model_dir / "vocab.txt")

        head_path = self.head_weights_path
        is_lora = head_path.is_dir() and (head_path / "adapter_config.json").exists()

        if is_lora:
            # Auto-detect position_type from training log if available
            training_log = head_path / "training_log.json"
            if training_log.exists():
                with open(training_log) as f:
                    log_data = json.load(f)
                position_type = log_data.get("position_type", position_type)

            self.encoder, self.head = _load_lora_checkpoint(
                self.model_dir, head_path, self.device,
                position_type=position_type,
            )
        else:
            self.encoder = CaLMEncoder(
                self.model_dir, freeze=True, position_type=position_type,
            ).to(self.device)
            state = torch.load(
                head_path, map_location=self.device, weights_only=True,
            )
            self.head = _build_head(state)
            self.head.load_state_dict(state)
            self.head = self.head.to(self.device)

        self.encoder.train(False)
        self.head.train(False)

    def batch_score(self, sequences: list[str]) -> list[float]:
        """Score DNA sequences for coding potential.

        Uses adaptive token-budget batching: sequences sorted by length,
        grouped so total tokens per batch stays under TOKEN_BUDGET.
        Input order is preserved in output.

        Args:
            sequences: DNA sequences (T not U), ideally codon-aligned (len % 3 == 0).

        Returns:
            Coding probabilities in [0, 1], same order as input.
        """
        if not sequences:
            return []

        n = len(sequences)
        scores = [0.0] * n
        sorted_indices = sorted(range(n), key=lambda i: len(sequences[i]))

        i = 0
        while i < n:
            max_codons = len(sequences[sorted_indices[i]]) // 3 + 2
            adaptive_bs = max(1, TOKEN_BUDGET // max_codons)
            adaptive_bs = min(adaptive_bs, BATCH_MAX, n - i)

            batch_idx = sorted_indices[i : i + adaptive_bs]
            batch_seqs = [sequences[j] for j in batch_idx]

            encoded = self.tokenizer.batch_encode(batch_seqs, device=self.device)

            with torch.no_grad():
                cls_emb = self.encoder(
                    encoded["input_ids"], encoded["attention_mask"]
                )
                logits = self.head(cls_emb).squeeze(-1)
                probs = torch.sigmoid(logits).cpu().tolist()

            if isinstance(probs, float):
                probs = [probs]

            for j, orig_idx in enumerate(batch_idx):
                scores[orig_idx] = probs[j]

            if self.device.type == "mps" and max_codons > 500:
                torch.mps.empty_cache()

            i += adaptive_bs

        return scores


def _load_lora_checkpoint(
    model_dir: Path, ckpt_dir: Path, device: torch.device,
    position_type: str = "rotary",
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Load CaLM encoder with merged LoRA weights + classification head.

    Merges LoRA adapters directly into base weights: W' = W + (alpha/r) * B @ A.
    No PEFT dependency needed at inference time.
    """
    from proteinqc.models.calm_encoder import CaLMEncoder

    with open(ckpt_dir / "adapter_config.json") as f:
        cfg = json.load(f)

    encoder = CaLMEncoder(model_dir, freeze=True, position_type=position_type)

    lora_state = torch.load(
        ckpt_dir / "adapter_model.pt", map_location="cpu", weights_only=True,
    )

    scaling = cfg["lora_alpha"] / cfg["r"]
    encoder_state = encoder.state_dict()

    for key in list(lora_state.keys()):
        if "lora_A" not in key:
            continue
        # e.g. layers.0.q_proj.lora_A.default.weight
        base_key = key.replace(".lora_A.default.weight", ".weight")
        b_key = key.replace("lora_A", "lora_B")

        if base_key not in encoder_state:
            continue

        lora_a = lora_state[key]   # (r, in_features)
        lora_b = lora_state[b_key]  # (out_features, r)
        delta = (lora_b @ lora_a) * scaling
        encoder_state[base_key] = encoder_state[base_key] + delta

    encoder.load_state_dict(encoder_state)
    encoder = encoder.to(device)

    head_state = torch.load(
        ckpt_dir / "head.pt", map_location=device, weights_only=True,
    )
    head = _build_head(head_state)
    head.load_state_dict(head_state)
    head = head.to(device)

    return encoder, head


def _build_head(state_dict: dict[str, torch.Tensor]) -> torch.nn.Module:
    """Auto-detect head type from state dict keys and build matching module."""
    from proteinqc.models.classification_heads import GatedHead, MLPHead

    if any(k.startswith("gate.") for k in state_dict):
        return GatedHead(hidden_size=768, mlp_hidden=256, dropout=0.0)
    return MLPHead(hidden_size=768, mlp_hidden=256, dropout=0.0)


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
