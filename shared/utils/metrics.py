"""Shared Metrics — Evaluate classification performance."""

import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)


def compute_metrics(y_true, y_pred, y_prob=None):
    """Compute all standard classification metrics.

    Args:
        y_true: List/array of ground-truth integer labels.
        y_pred: List/array of predicted integer labels.
        y_prob: Optional list/array of class probabilities (shape N x K).
                Required for ROC-AUC.

    Returns:
        dict with accuracy, precision, recall, f1, roc_auc,
        confusion_matrix, and classification_report.
    """
    # Handle edge case: single class present
    n_classes = len(set(y_true) | set(y_pred))

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted",
                                           zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted",
                                     zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average="weighted",
                                   zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, zero_division=0, output_dict=True
        ),
    }

    # ROC-AUC (multi-class one-vs-rest)
    if y_prob is not None and n_classes >= 2:
        y_prob_arr = np.array(y_prob)
        if y_prob_arr.ndim == 2 and y_prob_arr.shape[1] >= 2:
            try:
                metrics["roc_auc"] = float(
                    roc_auc_score(y_true, y_prob_arr, multi_class="ovr")
                )
            except ValueError:
                metrics["roc_auc"] = None
        else:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    return metrics
