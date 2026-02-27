"""Evaluation metrics for binary classification.

Implements standard metrics used in RNA coding potential benchmarks:
- Accuracy (ACC)
- Precision (PRE)
- Recall (REC)
- F1 Score (FSC)
- Matthews Correlation Coefficient (MCC)
"""

import numpy as np
import torch


def compute_binary_metrics(
    y_true: torch.Tensor | np.ndarray,
    y_pred: torch.Tensor | np.ndarray,
) -> dict[str, float]:
    """Compute all binary classification metrics.

    Args:
        y_true: Ground truth labels [N] (0 or 1)
        y_pred: Predicted labels [N] (0 or 1)

    Returns:
        metrics: Dictionary with ACC, PRE, REC, F1, MCC
    """
    # Convert to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    # Confusion matrix components
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    # Accuracy
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    # Precision
    pre = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Recall
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # F1 Score
    f1 = 2 * pre * rec / (pre + rec) if (pre + rec) > 0 else 0.0

    # Matthews Correlation Coefficient
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = numerator / denominator if denominator > 0 else 0.0

    return {
        "ACC": acc * 100,  # Report as percentage
        "PRE": pre * 100,
        "REC": rec * 100,
        "F1": f1 * 100,
        "MCC": mcc * 100,
    }


def compute_metrics_from_logits(
    y_true: torch.Tensor,
    logits: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute metrics from raw logits.

    Args:
        y_true: Ground truth labels [N] (0 or 1)
        logits: Raw model logits [N, 1] or [N]
        threshold: Classification threshold (default: 0.0 for sigmoid output)

    Returns:
        metrics: Dictionary with ACC, PRE, REC, F1, MCC
    """
    # Flatten logits if needed
    if logits.dim() > 1:
        logits = logits.squeeze(-1)

    # Apply sigmoid and threshold
    probs = torch.sigmoid(logits)
    y_pred = (probs > threshold).long()

    return compute_binary_metrics(y_true, y_pred)
