"""
Evaluation metrics for binary medical image classification.

Metrics tracked:
  - Accuracy
  - Recall (Sensitivity)  ← primary metric for pneumonia detection
  - Precision
  - F1 Score
  - AUC-ROC
  - AUPRC (Area Under Precision-Recall Curve)
  - Confusion Matrix

All functions accept numpy arrays or torch tensors.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from typing import Dict, Tuple, Optional


# Core metric computation

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    prefix: str = "",
) -> Dict[str, float]:
    """
    Compute all classification metrics.

    Args:
        y_true : Ground truth labels (0 or 1), shape [N]
        y_pred : Predicted labels (0 or 1), shape [N]
        y_prob : Predicted probability for class 1 (PNEUMONIA), shape [N]
        prefix : Optional string prefix for metric keys (e.g. optimizer name)

    Returns:
        Dict of metric_name → float value
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.cpu().numpy()

    p = f"{prefix}_" if prefix else ""

    metrics = {
        f"{p}accuracy":  accuracy_score(y_true, y_pred),
        f"{p}recall":    recall_score(y_true, y_pred, zero_division=0),
        f"{p}precision": precision_score(y_true, y_pred, zero_division=0),
        f"{p}f1":        f1_score(y_true, y_pred, zero_division=0),
        f"{p}auc_roc":   roc_auc_score(y_true, y_prob),
        f"{p}auprc":     average_precision_score(y_true, y_prob),
    }

    return metrics


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Returns 2×2 confusion matrix [[TN, FP], [FN, TP]]."""
    return confusion_matrix(y_true, y_pred)


def compute_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (fpr, tpr, thresholds) for ROC curve plotting."""
    return roc_curve(y_true, y_prob)


def compute_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (precision, recall, thresholds) for PR curve plotting."""
    return precision_recall_curve(y_true, y_prob)


# Epoch-level aggregator

class MetricTracker:
    """
    Tracks per-epoch metrics across the full training run.
    Stores history for all optimizers for comparison plots.

    Usage:
        tracker = MetricTracker(optimizer_name="lipschitz_momentum")
        tracker.update(epoch=1, loss=0.42, metrics={...})
        tracker.get_history("recall")  # → list of recall values per epoch
    """

    def __init__(self, optimizer_name: str):
        self.optimizer_name = optimizer_name
        self.history: Dict[str, list] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "accuracy": [],
            "recall": [],
            "precision": [],
            "f1": [],
            "auc_roc": [],
            "auprc": [],
        }
        # Optimizer-specific trajectories
        self.beta_trajectory: list = []       # β_t per epoch (list of lists for per-layer)
        self.lipschitz_trajectory: list = []  # L_t per epoch

    def update(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        metrics: Dict[str, float],
        beta_values: Optional[list] = None,
        lipschitz_values: Optional[list] = None,
    ) -> None:
        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)

        for key in ["accuracy", "recall", "precision", "f1", "auc_roc", "auprc"]:
            val = metrics.get(key, metrics.get(f"{key}", 0.0))
            self.history[key].append(val)

        if beta_values is not None:
            self.beta_trajectory.append(beta_values)
        if lipschitz_values is not None:
            self.lipschitz_trajectory.append(lipschitz_values)

    def get_history(self, metric: str) -> list:
        return self.history.get(metric, [])

    def best_epoch(self, metric: str = "recall") -> Tuple[int, float]:
        """Returns (epoch_number, best_value) for a given metric."""
        vals = self.history[metric]
        best_idx = int(np.argmax(vals))
        return self.history["epoch"][best_idx], vals[best_idx]

    def convergence_epoch(
        self,
        metric: str = "recall",
        threshold: float = 0.90,
    ) -> Optional[int]:
        """
        Returns the first epoch where metric crosses the threshold.
        Used for the '30% convergence reduction' comparison.
        """
        for epoch, val in zip(self.history["epoch"], self.history[metric]):
            if val >= threshold:
                return epoch
        return None

    def summary(self) -> Dict[str, float]:
        """Returns best values across all tracked metrics."""
        return {
            "best_recall":    max(self.history["recall"],    default=0),
            "best_f1":        max(self.history["f1"],        default=0),
            "best_auc_roc":   max(self.history["auc_roc"],   default=0),
            "best_auprc":     max(self.history["auprc"],     default=0),
            "best_accuracy":  max(self.history["accuracy"],  default=0),
            "final_val_loss": self.history["val_loss"][-1]   if self.history["val_loss"] else 0,
        }

    def print_summary(self) -> None:
        s = self.summary()
        best_epoch, best_recall = self.best_epoch("recall")
        conv_epoch = self.convergence_epoch("recall", threshold=0.90)

        print(f"\n{'─'*55}")
        print(f"  Optimizer : {self.optimizer_name}")
        print(f"{'─'*55}")
        print(f"  Best Recall     : {s['best_recall']:.4f}  (epoch {best_epoch})")
        print(f"  Best F1         : {s['best_f1']:.4f}")
        print(f"  Best AUC-ROC    : {s['best_auc_roc']:.4f}")
        print(f"  Best AUPRC      : {s['best_auprc']:.4f}")
        print(f"  Best Accuracy   : {s['best_accuracy']:.4f}")
        print(f"  Final Val Loss  : {s['final_val_loss']:.4f}")
        if conv_epoch:
            print(f"  Converged (≥0.90 recall) at epoch : {conv_epoch}")
        else:
            print(f"  Did not reach 0.90 recall threshold")
        print(f"{'─'*55}\n")


# Inference helper

def run_inference(
    model: torch.nn.Module,
    loader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run full inference over a DataLoader.

    Returns:
        y_true : shape [N]
        y_pred : shape [N]  (argmax predictions)
        y_prob : shape [N]  (softmax probability for class 1 / PNEUMONIA)
    """
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs  = torch.softmax(logits, dim=1)[:, 1]   # P(PNEUMONIA)
            preds  = logits.argmax(dim=1)

            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()
    y_prob = torch.cat(all_probs).numpy()

    return y_true, y_pred, y_prob