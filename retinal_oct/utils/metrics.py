"""Evaluation metrics for 4-class retinal OCT classification."""

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
from sklearn.preprocessing import label_binarize


CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]
NUM_CLASSES = 4


def compute_metrics(y_true, y_pred, y_prob, prefix=""):
    """Compute macro-averaged classification metrics."""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.cpu().numpy()

    p = f"{prefix}_" if prefix else ""

    y_true_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))

    try:
        auc_roc = roc_auc_score(y_true_bin, y_prob, multi_class="ovr", average="macro")
    except ValueError:
        auc_roc = 0.0

    try:
        auprc = average_precision_score(y_true_bin, y_prob, average="macro")
    except ValueError:
        auprc = 0.0

    metrics = {
        f"{p}accuracy":  accuracy_score(y_true, y_pred),
        f"{p}recall":    recall_score(y_true, y_pred, average="macro", zero_division=0),
        f"{p}precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        f"{p}f1":        f1_score(y_true, y_pred, average="macro", zero_division=0),
        f"{p}auc_roc":   auc_roc,
        f"{p}auprc":     auprc,
    }

    return metrics


def compute_per_class_metrics(y_true, y_pred, y_prob):
    """Per-class recall, precision, F1. Returns dict keyed by class name."""
    recalls    = recall_score(y_true, y_pred, average=None, zero_division=0)
    precisions = precision_score(y_true, y_pred, average=None, zero_division=0)
    f1s        = f1_score(y_true, y_pred, average=None, zero_division=0)

    return {
        cls: {"recall": float(recalls[i]), "precision": float(precisions[i]), "f1": float(f1s[i])}
        for i, cls in enumerate(CLASS_NAMES)
    }


def compute_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))


def compute_roc_curve(y_true, y_prob, class_idx=0):
    """Returns (fpr, tpr, thresholds) for one class (OvR)."""
    y_true_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    return roc_curve(y_true_bin[:, class_idx], y_prob[:, class_idx])


def compute_pr_curve(y_true, y_prob, class_idx=0):
    y_true_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    return precision_recall_curve(y_true_bin[:, class_idx], y_prob[:, class_idx])


class MetricTracker:
    def __init__(self, optimizer_name):
        self.optimizer_name = optimizer_name
        self.history = {
            "epoch": [], "train_loss": [], "val_loss": [],
            "accuracy": [], "recall": [], "precision": [],
            "f1": [], "auc_roc": [], "auprc": [],
        }
        self.beta_trajectory = []
        self.lipschitz_trajectory = []

    def update(self, epoch, train_loss, val_loss, metrics,
               beta_values=None, lipschitz_values=None):
        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)

        for key in ["accuracy", "recall", "precision", "f1", "auc_roc", "auprc"]:
            self.history[key].append(metrics.get(key, 0.0))

        if beta_values is not None:
            self.beta_trajectory.append(beta_values)
        if lipschitz_values is not None:
            self.lipschitz_trajectory.append(lipschitz_values)

    def get_history(self, metric):
        return self.history.get(metric, [])

    def best_epoch(self, metric="recall"):
        vals = self.history[metric]
        best_idx = int(np.argmax(vals))
        return self.history["epoch"][best_idx], vals[best_idx]

    def convergence_epoch(self, metric="recall", threshold=0.85):
        for epoch, val in zip(self.history["epoch"], self.history[metric]):
            if val >= threshold:
                return epoch
        return None

    def summary(self):
        return {
            "best_recall":    max(self.history["recall"],    default=0),
            "best_f1":        max(self.history["f1"],        default=0),
            "best_auc_roc":   max(self.history["auc_roc"],   default=0),
            "best_auprc":     max(self.history["auprc"],     default=0),
            "best_accuracy":  max(self.history["accuracy"],  default=0),
            "final_val_loss": self.history["val_loss"][-1] if self.history["val_loss"] else 0,
        }

    def print_summary(self):
        s = self.summary()
        best_epoch, best_recall = self.best_epoch("recall")
        conv_epoch = self.convergence_epoch("recall", threshold=0.85)

        print(f"\n{'─'*55}")
        print(f"  Optimizer : {self.optimizer_name}")
        print(f"{'─'*55}")
        print(f"  Best Macro Recall  : {s['best_recall']:.4f}  (epoch {best_epoch})")
        print(f"  Best Macro F1      : {s['best_f1']:.4f}")
        print(f"  Best AUC-ROC (macro): {s['best_auc_roc']:.4f}")
        print(f"  Best AUPRC (macro)  : {s['best_auprc']:.4f}")
        print(f"  Best Accuracy       : {s['best_accuracy']:.4f}")
        print(f"  Final Val Loss      : {s['final_val_loss']:.4f}")
        if conv_epoch:
            print(f"  Converged (≥0.85 macro recall) at epoch : {conv_epoch}")
        else:
            print(f"  Did not reach 0.85 macro recall threshold")
        print(f"{'─'*55}\n")


def run_inference(model, loader, device):
    """Run inference over a DataLoader. Returns (y_true, y_pred, y_prob)."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs  = torch.softmax(logits, dim=1)   # [B, 4]
            preds  = logits.argmax(dim=1)

            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()
    y_prob = torch.cat(all_probs, dim=0).numpy()

    return y_true, y_pred, y_prob
