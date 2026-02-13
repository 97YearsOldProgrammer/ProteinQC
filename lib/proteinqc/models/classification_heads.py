"""Classification heads for binary RNA classification (coding vs non-coding).

Three architectures:
1. LinearHead: Single linear layer (logistic regression)
2. MLPHead: Deep 3-layer MLP with GELU and dropout
3. GatedHead: Gated mixture of LinearHead and MLPHead with learned routing
"""

import torch
import torch.nn as nn


class LinearHead(nn.Module):
    """Simple linear classification head (logistic regression).

    Single linear layer from [CLS] embedding to binary logits.
    Handles easy-to-classify sequences with clear ORF signals.

    Args:
        hidden_size: Input embedding dimension (default: 768 for CaLM)
    """

    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            hidden_state: [batch, hidden_size] embeddings from encoder

        Returns:
            logits: [batch, 1] raw logits for binary classification
        """
        return self.classifier(hidden_state)


class MLPHead(nn.Module):
    """Deep MLP classification head with 3 layers.

    Nonlinear path for ambiguous sequences (short ORFs, lncRNAs with
    coding-like features). Uses GELU activation and dropout.

    Args:
        hidden_size: Input embedding dimension (default: 768 for CaLM)
        mlp_hidden: Hidden layer size (default: 256)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        hidden_size: int = 768,
        mlp_hidden: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            hidden_state: [batch, hidden_size] embeddings from encoder

        Returns:
            logits: [batch, 1] raw logits for binary classification
        """
        return self.classifier(hidden_state)


class GatedHead(nn.Module):
    """Gated mixture-of-experts head with learned routing.

    Routes between LinearHead (shortcut) and MLPHead (complex path) using
    a learned sigmoid gate. The gate decides per-sample which path to use.

    Architecture:
        hidden_state → Gate (Linear→Sigmoid) → g ∈ [0,1]
                    ↓                           ↓
                Shortcut                      MLP
                (Linear)                   (3-layer)
                    ↓                           ↓
               (1-g) × shortcut_out  +  g × mlp_out → logits

    Args:
        hidden_size: Input embedding dimension (default: 768 for CaLM)
        mlp_hidden: MLP hidden layer size (default: 256)
        dropout: Dropout probability (default: 0.1)
        balance_loss_weight: Weight for gate balance regularization (default: 0.01)
    """

    def __init__(
        self,
        hidden_size: int = 768,
        mlp_hidden: int = 256,
        dropout: float = 0.1,
        balance_loss_weight: float = 0.01,
    ):
        super().__init__()
        self.balance_loss_weight = balance_loss_weight

        # Gate: learns when to use shortcut vs MLP
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

        # Shortcut path: simple linear (like logistic regression)
        self.shortcut = nn.Linear(hidden_size, 1)

        # MLP path: deep nonlinear
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

        # Track gate values for analysis
        self.last_gate_values = None

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Forward pass with soft routing.

        Args:
            hidden_state: [batch, hidden_size] embeddings from encoder

        Returns:
            logits: [batch, 1] raw logits for binary classification
        """
        g = self.gate(hidden_state)  # [batch, 1] in [0, 1]
        shortcut_out = self.shortcut(hidden_state)  # [batch, 1]
        mlp_out = self.mlp(hidden_state)  # [batch, 1]

        # Soft routing: interpolate between shortcut and MLP
        logits = (1 - g) * shortcut_out + g * mlp_out

        # Store gate values for analysis
        self.last_gate_values = g.detach()

        return logits

    def get_balance_loss(self) -> torch.Tensor:
        """Compute gate balance regularization loss.

        Encourages the gate to use both paths equally on average.
        Prevents collapse to always-shortcut or always-MLP.

        Returns:
            balance_loss: Scalar tensor |mean(g) - 0.5|
        """
        if self.last_gate_values is None:
            return torch.tensor(0.0)

        mean_gate = self.last_gate_values.mean()
        balance_loss = torch.abs(mean_gate - 0.5)
        return self.balance_loss_weight * balance_loss

    def get_gate_stats(self) -> dict[str, float]:
        """Get statistics on gate routing behavior.

        Returns:
            stats: Dictionary with mean, std, min, max of gate values
        """
        if self.last_gate_values is None:
            return {}

        g = self.last_gate_values
        return {
            "gate_mean": g.mean().item(),
            "gate_std": g.std().item(),
            "gate_min": g.min().item(),
            "gate_max": g.max().item(),
            "pct_shortcut": (g < 0.5).float().mean().item() * 100,
            "pct_mlp": (g >= 0.5).float().mean().item() * 100,
        }
