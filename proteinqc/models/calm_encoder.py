"""Pure PyTorch CaLM encoder — no transformers, no multimolecule.

Pre-LN BERT architecture with RoPE positional embeddings.
Loads pre-trained CaLM weights from safetensors and provides
[CLS] embedding extraction for downstream classification.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate pairs: [-x2, x1] for RoPE."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _build_rope_cache(
    seq_len: int,
    head_dim: int,
    device: torch.device,
    base: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables for rotary position embeddings.

    Returns (cos, sin) each of shape [1, 1, seq_len, head_dim].
    """
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # [seq_len, head_dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)  # [seq_len, head_dim]
    return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]


def _apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors.

    q, k: [batch, heads, seq, head_dim]
    cos, sin: [1, 1, seq, head_dim]
    """
    seq_len = q.shape[2]
    cos = cos[:, :, :seq_len, :]
    sin = sin[:, :, :seq_len, :]
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


class TransformerLayer(nn.Module):
    """Single Pre-LN transformer layer matching CaLM architecture.

    Pre-LN Attention:
        LayerNorm → Q,K,V → RoPE → SDPA → OutProj → Residual
    Pre-LN FFN:
        LayerNorm → Linear(768→3072) → GELU → Linear(3072→768) → Residual
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        layer_norm_eps: float,
        attn_dropout: float,
        hidden_dropout: float,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Pre-LN for attention
        self.attn_ln = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout_p = attn_dropout
        self.hidden_dropout1 = nn.Dropout(hidden_dropout)

        # Pre-LN for FFN
        self.ffn_ln = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.ffn_up = nn.Linear(hidden_size, intermediate_size)
        self.ffn_down = nn.Linear(intermediate_size, hidden_size)
        self.hidden_dropout2 = nn.Dropout(hidden_dropout)

    def _reshape_for_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[batch, seq, hidden] → [batch, heads, seq, head_dim]."""
        b, s, _ = x.shape
        return x.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        # --- Pre-LN Attention ---
        h = self.attn_ln(x)
        q = self._reshape_for_heads(self.q_proj(h))
        k = self._reshape_for_heads(self.k_proj(h))
        v = self._reshape_for_heads(self.v_proj(h))

        q, k = _apply_rope(q, k, cos, sin)

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
        )
        # [batch, heads, seq, head_dim] → [batch, seq, hidden]
        attn_out = attn_out.transpose(1, 2).contiguous().view(x.shape)
        attn_out = self.hidden_dropout1(self.out_proj(attn_out))
        x = x + attn_out

        # --- Pre-LN FFN ---
        h = self.ffn_ln(x)
        h = F.gelu(self.ffn_up(h))
        h = self.hidden_dropout2(self.ffn_down(h))
        x = x + h

        return x


class CaLMLMHead(nn.Module):
    """Linear(768→768) → GELU → LayerNorm → Linear(768→131) + bias."""

    def __init__(self, hidden_size: int, vocab_size: int, layer_norm_eps: float):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """[batch, seq, hidden] → [batch, seq, vocab] logits."""
        x = self.dense(hidden_states)
        x = F.gelu(x)
        x = self.layer_norm(x)
        x = self.decoder(x) + self.bias
        return x


_log = logging.getLogger(__name__)


class CaLMEncoder(nn.Module):
    """Pure PyTorch CaLM BERT encoder (12-layer Pre-LN, RoPE, 131 codon vocab)."""

    def __init__(
        self,
        model_dir: Path | str,
        freeze: bool = True,
        load_lm_head: bool = False,
    ):
        super().__init__()
        self.model_dir = Path(model_dir)
        self.config = self._load_config()

        hidden_size = self.config["hidden_size"]
        num_heads = self.config["num_attention_heads"]
        intermediate_size = self.config["intermediate_size"]
        num_layers = self.config["num_hidden_layers"]
        layer_norm_eps = self.config["layer_norm_eps"]
        attn_dropout = self.config["attention_dropout"]
        hidden_dropout = self.config["hidden_dropout"]

        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads

        # Embedding
        self.word_embeddings = nn.Embedding(
            self.config["vocab_size"],
            hidden_size,
            padding_idx=self.config["pad_token_id"],
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                layer_norm_eps=layer_norm_eps,
                attn_dropout=attn_dropout,
                hidden_dropout=hidden_dropout,
            )
            for _ in range(num_layers)
        ])

        # Final LayerNorm (applied after all transformer layers)
        self.emb_layer_norm_after = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        # Optional LM head for MLM / perplexity scoring
        self.lm_head: Optional[CaLMLMHead] = None
        if load_lm_head:
            self.lm_head = CaLMLMHead(
                hidden_size=hidden_size,
                vocab_size=self.config["vocab_size"],
                layer_norm_eps=layer_norm_eps,
            )

        # RoPE cache (lazily built on first forward)
        self._rope_cos: Optional[torch.Tensor] = None
        self._rope_sin: Optional[torch.Tensor] = None
        self._rope_seq_len: int = 0

        # Load pre-trained weights
        self._load_weights()

        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def _load_config(self) -> dict:
        with open(self.model_dir / "config.json") as f:
            return json.load(f)

    def _load_weights(self):
        """Load and map safetensors weights to our architecture."""
        raw = load_file(str(self.model_dir / "model.safetensors"))
        mapped = self._map_weight_keys(raw)

        if self.lm_head is not None:
            self._map_lm_head_keys(raw, mapped)

        missing, unexpected = self.load_state_dict(mapped, strict=False)
        if missing:
            raise RuntimeError(f"Missing keys in weight loading: {missing}")
        if unexpected:
            _log.warning("Unused keys in weight loading (%d)", len(unexpected))

    def _map_weight_keys(self, raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Map safetensors keys → our state_dict keys."""
        mapped: dict[str, torch.Tensor] = {}

        # Embeddings
        mapped["word_embeddings.weight"] = raw["model.embeddings.word_embeddings.weight"]

        # Final LayerNorm
        mapped["emb_layer_norm_after.weight"] = raw["model.encoder.emb_layer_norm_after.weight"]
        mapped["emb_layer_norm_after.bias"] = raw["model.encoder.emb_layer_norm_after.bias"]

        # Transformer layers
        for i in range(self.config["num_hidden_layers"]):
            src = f"model.encoder.layer.{i}"
            dst = f"layers.{i}"

            # Attention Pre-LN
            mapped[f"{dst}.attn_ln.weight"] = raw[f"{src}.attention.layer_norm.weight"]
            mapped[f"{dst}.attn_ln.bias"] = raw[f"{src}.attention.layer_norm.bias"]

            # QKV projections
            mapped[f"{dst}.q_proj.weight"] = raw[f"{src}.attention.self.query.weight"]
            mapped[f"{dst}.q_proj.bias"] = raw[f"{src}.attention.self.query.bias"]
            mapped[f"{dst}.k_proj.weight"] = raw[f"{src}.attention.self.key.weight"]
            mapped[f"{dst}.k_proj.bias"] = raw[f"{src}.attention.self.key.bias"]
            mapped[f"{dst}.v_proj.weight"] = raw[f"{src}.attention.self.value.weight"]
            mapped[f"{dst}.v_proj.bias"] = raw[f"{src}.attention.self.value.bias"]

            # Output projection
            mapped[f"{dst}.out_proj.weight"] = raw[f"{src}.attention.output.dense.weight"]
            mapped[f"{dst}.out_proj.bias"] = raw[f"{src}.attention.output.dense.bias"]

            # FFN Pre-LN
            mapped[f"{dst}.ffn_ln.weight"] = raw[f"{src}.layer_norm.weight"]
            mapped[f"{dst}.ffn_ln.bias"] = raw[f"{src}.layer_norm.bias"]

            # FFN
            mapped[f"{dst}.ffn_up.weight"] = raw[f"{src}.intermediate.dense.weight"]
            mapped[f"{dst}.ffn_up.bias"] = raw[f"{src}.intermediate.dense.bias"]
            mapped[f"{dst}.ffn_down.weight"] = raw[f"{src}.output.dense.weight"]
            mapped[f"{dst}.ffn_down.bias"] = raw[f"{src}.output.dense.bias"]

        return mapped

    def _map_lm_head_keys(
        self,
        raw: dict[str, torch.Tensor],
        mapped: dict[str, torch.Tensor],
    ):
        """Map LM head weights from safetensors into our CaLMLMHead."""
        mapped["lm_head.dense.weight"] = raw["lm_head.transform.dense.weight"]
        mapped["lm_head.dense.bias"] = raw["lm_head.transform.dense.bias"]
        mapped["lm_head.layer_norm.weight"] = raw["lm_head.transform.layer_norm.weight"]
        mapped["lm_head.layer_norm.bias"] = raw["lm_head.transform.layer_norm.bias"]
        mapped["lm_head.decoder.weight"] = raw["lm_head.decoder.weight"]
        mapped["lm_head.bias"] = raw["lm_head.bias"]

    def _ensure_rope(self, seq_len: int, device: torch.device):
        """Build or extend RoPE cos/sin cache if needed."""
        if seq_len > self._rope_seq_len or self._rope_cos is None:
            self._rope_cos, self._rope_sin = _build_rope_cache(
                seq_len, self.head_dim, device
            )
            self._rope_seq_len = seq_len

    def _encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encoder forward pass → hidden states [batch, seq, hidden]."""
        device = input_ids.device
        seq_len = input_ids.shape[1]

        if attention_mask is None:
            attention_mask = input_ids.ne(self.config["pad_token_id"]).long()

        x = self.word_embeddings(input_ids)
        x = x * attention_mask.unsqueeze(-1).to(x.dtype)

        sdpa_mask = attention_mask[:, None, None, :].bool()  # [B,1,1,S] for SDPA broadcast

        self._ensure_rope(seq_len, device)
        cos = self._rope_cos.to(device=device, dtype=x.dtype)
        sin = self._rope_sin.to(device=device, dtype=x.dtype)

        for layer in self.layers:
            x = layer(x, sdpa_mask, cos, sin)

        return self.emb_layer_norm_after(x)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """[CLS] embeddings [batch, hidden] from codon sequences."""
        return self._encode(input_ids, attention_mask)[:, 0, :]

    def forward_mean_pool(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Mean-pooled embeddings [batch, hidden] over non-padding tokens."""
        full = self._encode(input_ids, attention_mask)  # [B, S, H]
        mask = attention_mask.unsqueeze(-1).to(full.dtype)  # [B, S, 1]
        summed = (full * mask).sum(dim=1)  # [B, H]
        lengths = mask.sum(dim=1).clamp(min=1)  # [B, 1]
        return summed / lengths

    def forward_mlm(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """MLM logits [batch, seq, vocab]. Requires load_lm_head=True."""
        if self.lm_head is None:
            raise RuntimeError(
                "LM head not loaded. Construct CaLMEncoder with load_lm_head=True."
            )
        hidden = self._encode(input_ids, attention_mask)
        return self.lm_head(hidden)

    @property
    def device(self) -> torch.device:
        return self.word_embeddings.weight.device
