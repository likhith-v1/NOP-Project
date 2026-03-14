"""Post-training evaluation for a single optimizer checkpoint."""

import os
import sys
import argparse
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Support running directly: python chest_xray_deprecated/evaluate.py
if __name__ == "__main__" and __package__ is None:
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

from chest_xray_deprecated.utils.dataloader import load_config, build_dataloaders, build_datasets, get_class_weights
from chest_xray_deprecated.utils.metrics import (
    compute_metrics, compute_confusion_matrix,
    compute_roc_curve, compute_pr_curve
)
from chest_xray_deprecated.models.densenet import build_model
from chest_xray_deprecated.train import get_device

DPI = 300
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def evaluate(optimizer_name, cfg):
    device    = get_device(cfg)
    ckpt_dir  = Path(cfg["training"]["checkpoint_dir"])
    plot_dir  = Path(cfg["evaluation"]["plot_dir"]) / optimizer_name
    plot_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / f"{optimizer_name}_best.pth"
    if not ckpt_path.exists():
        print(f"  [ERROR] Checkpoint not found: {ckpt_path}")
        return

    model = build_model(cfg).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"\n  Loaded checkpoint: {ckpt_path} (epoch {ckpt['epoch']})")

    _, _, test_loader = build_dataloaders(cfg, seed=cfg["project"]["seed"])
    train_ds          = build_datasets(cfg)["train"]
    class_weights     = get_class_weights(train_ds).to(device)
    criterion         = torch.nn.CrossEntropyLoss(weight=class_weights)

    all_labels, all_preds, all_probs = [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels_dev = labels.to(device)
            logits = model(images)
            loss   = criterion(logits, labels_dev)
            probs  = torch.softmax(logits, dim=1)[:, 1]
            preds  = logits.argmax(dim=1)

            total_loss  += loss.item()
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()
    y_prob = torch.cat(all_probs).numpy()

    metrics = compute_metrics(y_true, y_pred, y_prob)
    test_loss = total_loss / len(test_loader)

    class_names = cfg["data"]["class_names"]
    print(f"\n  {'─'*45}")
    print(f"  Test Results — {optimizer_name}")
    print(f"  {'─'*45}")
    for k, v in metrics.items():
        print(f"    {k:<15}: {v:.4f}")
    print(f"    {'test_loss':<15}: {test_loss:.4f}")
    print(f"  {'─'*45}\n")

    cm = compute_confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5,
    )
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {optimizer_name}")
    tn, fp, fn, tp = cm.ravel()
    ax.text(0.5, -0.12,
            f"TN={tn}  FP={fp}  FN={fn}  TP={tp}  |  "
            f"Sensitivity={tp/(tp+fn):.3f}  Specificity={tn/(tn+fp):.3f}",
            transform=ax.transAxes, ha="center", fontsize=9, color="gray")
    plt.tight_layout()
    fig.savefig(plot_dir / "confusion_matrix.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(plot_dir / "confusion_matrix.pdf", dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Confusion matrix → {plot_dir}/confusion_matrix.png")

    fpr, tpr, _ = compute_roc_curve(y_true, y_prob)
    auc = metrics["auc_roc"]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, lw=2, color="#E63946", label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {optimizer_name}"); ax.legend()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    plt.tight_layout()
    fig.savefig(plot_dir / "roc_curve.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(plot_dir / "roc_curve.pdf", dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] ROC curve → {plot_dir}/roc_curve.png")

    prec_vals, rec_vals, _ = compute_pr_curve(y_true, y_prob)
    auprc = metrics["auprc"]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(rec_vals, prec_vals, lw=2, color="#2A9D8F", label=f"AUPRC = {auprc:.4f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {optimizer_name}"); ax.legend()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    plt.tight_layout()
    fig.savefig(plot_dir / "pr_curve.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(plot_dir / "pr_curve.pdf", dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] PR curve → {plot_dir}/pr_curve.png")

    result = {
        "optimizer": optimizer_name,
        "checkpoint_epoch": int(ckpt["epoch"]),
        "test_loss": test_loss,
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
    }
    with open(plot_dir / "test_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"  [Results] Saved → {plot_dir}/test_results.json\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", type=str, default="lipschitz_momentum",
                        choices=["lipschitz_momentum", "heavy_ball", "nesterov", "adam"])
    parser.add_argument("--config", type=str, default="chest_xray_deprecated/configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    evaluate(args.optimizer, cfg)
