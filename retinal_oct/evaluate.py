"""Post-training evaluation for a single optimizer checkpoint — Retinal OCT."""

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
from sklearn.preprocessing import label_binarize

# Add repo root to sys.path when running directly
if __name__ == "__main__" and __package__ is None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo_root)

from retinal_oct.utils.dataloader import load_config, build_dataloaders, build_datasets, get_class_weights
from retinal_oct.utils.metrics import (
    compute_metrics, compute_confusion_matrix,
    compute_roc_curve, compute_pr_curve,
    compute_per_class_metrics, CLASS_NAMES, NUM_CLASSES,
)
from retinal_oct.models.densenet import build_model
from retinal_oct.train import get_device

DPI = 300
CLASS_COLORS = ["#E63946", "#457B9D", "#2A9D8F", "#F4A261"]
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def evaluate(optimizer_name, cfg):
    device   = get_device(cfg)
    ckpt_dir = Path(cfg["training"]["checkpoint_dir"])
    plot_dir = Path(cfg["evaluation"]["plot_dir"]) / optimizer_name
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
            images     = images.to(device)
            labels_dev = labels.to(device)
            logits     = model(images)
            loss       = criterion(logits, labels_dev)
            probs      = torch.softmax(logits, dim=1)
            preds      = logits.argmax(dim=1)

            total_loss  += loss.item()
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()
    y_prob = torch.cat(all_probs, dim=0).numpy()

    metrics     = compute_metrics(y_true, y_pred, y_prob)
    per_class_m = compute_per_class_metrics(y_true, y_pred, y_prob)
    test_loss   = total_loss / len(test_loader)

    # Print metrics
    print(f"\n  Test Results — {optimizer_name} [Retinal OCT]")
    for k, v in metrics.items():
        print(f"  {k:<20} {v:>8.4f}")
    print(f"  {'test_loss':<20} {test_loss:>8.4f}")
    print(f"\n  Per-class breakdown:")
    for cls, m in per_class_m.items():
        print(f"  {cls:<10} {m['recall']:>8.4f} {m['precision']:>10.4f} {m['f1']:>8.4f}")
    print(f"\n")

    cm = compute_confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=ax, linewidths=0.5,
    )
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {optimizer_name}\n(Retinal OCT, 4-class)")
    plt.tight_layout()
    fig.savefig(plot_dir / "confusion_matrix.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(plot_dir / "confusion_matrix.pdf", dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Confusion matrix → {plot_dir}/confusion_matrix.png")

    y_true_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    for i, cls in enumerate(CLASS_NAMES):
        fpr, tpr, _ = compute_roc_curve(y_true, y_prob, class_idx=i)
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
        ax.plot(fpr, tpr, lw=2, color=CLASS_COLORS[i], label=f"{cls}  (AUC={auc:.3f})")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Per-class ROC Curves (OvR) — {optimizer_name}")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    plt.tight_layout()
    fig.savefig(plot_dir / "roc_curves.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(plot_dir / "roc_curves.pdf", dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] ROC curves → {plot_dir}/roc_curves.png")

    fig, ax = plt.subplots(figsize=(7, 6))
    for i, cls in enumerate(CLASS_NAMES):
        prec_vals, rec_vals, _ = compute_pr_curve(y_true, y_prob, class_idx=i)
        from sklearn.metrics import average_precision_score
        ap = average_precision_score(y_true_bin[:, i], y_prob[:, i])
        ax.plot(rec_vals, prec_vals, lw=2, color=CLASS_COLORS[i],
                label=f"{cls}  (AP={ap:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"Per-class Precision-Recall Curves — {optimizer_name}")
    ax.legend()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    plt.tight_layout()
    fig.savefig(plot_dir / "pr_curves.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(plot_dir / "pr_curves.pdf", dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] PR curves → {plot_dir}/pr_curves.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    recalls = [per_class_m[cls]["recall"] for cls in CLASS_NAMES]
    bars    = ax.bar(CLASS_NAMES, recalls, color=CLASS_COLORS, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, recalls):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    ax.axhline(np.mean(recalls), color="gray", ls="--", lw=1.5,
               label=f"Macro mean = {np.mean(recalls):.3f}")
    ax.set_ylabel("Recall"); ax.set_ylim(0, 1.15)
    ax.set_title(f"Per-class Recall — {optimizer_name}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(plot_dir / "per_class_recall.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(plot_dir / "per_class_recall.pdf", dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Per-class recall → {plot_dir}/per_class_recall.png")

    result = {
        "optimizer":       optimizer_name,
        "checkpoint_epoch": int(ckpt["epoch"]),
        "test_loss":        test_loss,
        "metrics":          metrics,
        "per_class":        per_class_m,
        "confusion_matrix": cm.tolist(),
    }
    with open(plot_dir / "test_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"  [Results] Saved → {plot_dir}/test_results.json\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", type=str, default="lipschitz_momentum",
                        choices=["lipschitz_momentum", "heavy_ball", "nesterov", "adam"])
    parser.add_argument("--config", type=str, default="retinal_oct/configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    evaluate(args.optimizer, cfg)
