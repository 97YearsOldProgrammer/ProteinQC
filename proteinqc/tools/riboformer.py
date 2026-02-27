"""PyTorch Riboformer: translation efficiency prediction from codon sequence + ribo-seq.

Two-branch architecture fusing sequence embeddings (Conv2D tower) with ribo-seq
coverage (Conv1D tower), each passed through a transformer block, then merged
via element-wise multiplication for final prediction.

Converted from TensorFlow/Keras original. Weight loading handled separately.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class RiboformerConfig:
    """Configuration for the Riboformer model."""

    wsize: int = 40
    vocab_size: int = 64
    embed_dim: int = 8
    num_heads: int = 8  # original TF used 10 but 8 divides both 32 and 8
    mlp_dim: int = 64
    dropout_rate: float = 0.4
    activation: str = "relu"
    conv2d_filters: int = 32
    conv2d_kernel: int = 5
    conv2d_layers: int = 5
    conv1d_filters: tuple[int, ...] = (32, 32, 32, 32, 8)
    conv1d_kernel: int = 9


class TokenAndPositionEmbedding(nn.Module):
    """Learnable token + position embedding for codon sequences."""

    def __init__(self, vocab_size: int, embed_dim: int, max_len: int) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_len, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return self.token_emb(x) + self.pos_emb(positions)


class ConvBnRelu2D(nn.Module):
    """Single Conv2D + BatchNorm + ReLU block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        padding = kernel_size // 2  # 'same' padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


class ConvBnRelu1D(nn.Module):
    """Single Conv1D + BatchNorm + ReLU block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        padding = kernel_size // 2  # 'same' padding
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


class TransformerBlock(nn.Module):
    """Post-LN transformer block with no residual around MLP.

    Architecture:
        mha_out = MHA(x, x, x)
        mha_out = dropout(mha_out)
        normed  = layernorm(mha_out + x)   # residual + post-LN
        out     = mlp(normed)               # no residual around MLP
    """

    def __init__(
        self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float
    ) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        mha_out, weights = self.mha(x, x, x)
        mha_out = self.dropout(mha_out)
        normed = self.layernorm(mha_out + x)
        out = self.mlp(normed)
        return out, weights


class RiboformerPyTorch(nn.Module):
    """Riboformer translation efficiency predictor.

    Two-branch architecture:
        Branch 1 (sequence): Embedding -> Conv2D tower -> Transformer -> Dense
        Branch 2 (ribo-seq): Conv1D tower -> Transformer -> Dense
        Fusion: element-wise multiply -> Dense(1, relu)

    Args:
        config: Model configuration dataclass.
    """

    def __init__(self, config: RiboformerConfig | None = None) -> None:
        super().__init__()
        cfg = config or RiboformerConfig()
        self.config = cfg

        # Branch 1: sequence pathway
        self.embedding = TokenAndPositionEmbedding(
            cfg.vocab_size, cfg.embed_dim, cfg.wsize
        )
        self.conv2d_tower = nn.ModuleList()
        in_ch = 1
        for _ in range(cfg.conv2d_layers):
            self.conv2d_tower.append(
                ConvBnRelu2D(in_ch, cfg.conv2d_filters, cfg.conv2d_kernel)
            )
            in_ch = cfg.conv2d_filters

        # TF: reduce_mean(axis=-1) on (B,40,8,32) reduces the 32 FILTERS → (B,40,8)
        # PyTorch equivalent: mean(dim=1) on (B,32,40,8) → (B,40,8)
        # This keeps embed_dim=8, matching TF MHA weight shapes (8, 80)
        self.seq_transformer = TransformerBlock(
            embed_dim=cfg.embed_dim,
            num_heads=cfg.num_heads,
            ff_dim=cfg.mlp_dim,
            dropout=cfg.dropout_rate,
        )
        self.seq_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cfg.wsize * cfg.embed_dim, 32),  # 40*8=320, matches TF Dense kernel (320,32)
            nn.ReLU(),
        )

        # Branch 2: ribo-seq coverage pathway
        self.conv1d_tower = nn.ModuleList()
        in_ch = 1
        for out_ch in cfg.conv1d_filters:
            self.conv1d_tower.append(
                ConvBnRelu1D(in_ch, out_ch, cfg.conv1d_kernel)
            )
            in_ch = out_ch
        last_conv1d_out = cfg.conv1d_filters[-1]

        self.exp_transformer = TransformerBlock(
            embed_dim=last_conv1d_out,
            num_heads=cfg.num_heads,
            ff_dim=cfg.mlp_dim,
            dropout=cfg.dropout_rate,
        )
        self.exp_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cfg.wsize * last_conv1d_out, 32),
            nn.ReLU(),
        )

        # Fusion
        self.final_dense = nn.Sequential(
            nn.Linear(32, 1),
            nn.ReLU(),
        )

    def forward(self, seq: Tensor, exp: Tensor) -> tuple[Tensor, None]:
        """Forward pass through both branches and fusion.

        Args:
            seq: Codon index tensor, shape (B, wsize), dtype long.
            exp: Ribo-seq coverage tensor, shape (B, wsize), dtype float.

        Returns:
            Tuple of (prediction tensor (B, 1), None) for API compatibility.
        """
        # Branch 1: sequence
        x = self.embedding(seq)                     # (B, 40, 8)
        x = x.unsqueeze(1)                          # (B, 1, 40, 8) — channels-first for Conv2d
        for conv in self.conv2d_tower:
            x = conv(x)                             # (B, 32, 40, 8)
        x = x.mean(dim=1)                           # (B, 40, 8) — mean over 32 filters (TF axis=-1)
        x, _ = self.seq_transformer(x)              # (B, 40, 8)
        x = self.seq_head(x)                        # (B, 32)

        # Branch 2: ribo-seq coverage
        y = exp.unsqueeze(1)                         # (B, 1, 40) — channels-first for Conv1d
        for conv in self.conv1d_tower:
            y = conv(y)                              # final: (B, 8, 40)
        y = y.permute(0, 2, 1)                       # (B, 40, 8) — seq-len first for transformer
        y, _ = self.exp_transformer(y)               # (B, 40, 8)
        y = self.exp_head(y)                         # (B, 32)

        # Fusion
        out = self.final_dense(x * y)               # (B, 1)
        return out, None
