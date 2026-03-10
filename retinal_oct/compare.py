"""
Runs all 4 optimizers sequentially and generates comparison plots
for the report — Retinal OCT (4-class).

Usage:
    python retinal_oct/compare.py                # runs all 4
    python retinal_oct/compare.py --plot-only    # plots from existing logs
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional

# Support running this script directly (e.g., `python retinal_oct/compare.py`).
# When executed this way, sys.path includes `retinal_oct/` but not the repo root.
# Add the repo root to sys.path so `import retinal_oct.*` works.
if __name__ == "__main__" and __package__ is None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo_root)

from retinal_oct.utils.dataloader import load_config, build_dataloaders, build_datasets, get_class_weights
from retinal_oct.utils.metrics import compute_metrics, compute_roc_curve
from retinal_oct.models.densenet import build_model
from retinal_oct.train import train, get_device
import torch
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score

COLORS = {
    "lipschitz_momentum": "#E63946",
    "heavy_ball":         "#457B9D",
    "nesterov":           "#2A9D8F",
    "adam":               "#F4A261",
}
LABELS = {
    "lipschitz_momentum": "LBM (Ours)",
    "heavy_ball":         "Heavy-Ball (β=0.9)",
    "nesterov":           "Nesterov (β=0.9)",
    "adam":               "Adam",
}
LINESTYLES = {
    "lipschitz_momentum": "-",
    "heavy_ball":         "--",
    "nesterov":           "-.",
    "adam":               ":",
}
CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]
DPI = 300
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": DPI,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_logs(log_dir: Path, optimizers: List[str]) -> Dict[str, dict]:
    logs = {}
    for opt in optimizers:
        path = log_dir / f"{opt}.json"
        if path.exists():
            with open(path) as f:
                logs[opt] = json.load(f)
        else:
            print(f"  [WARNING] Log not found: {path}")
    return logs


# Plot 1 — Loss Curves

def plot_loss_curves(logs: Dict, plot_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training & Validation Loss — Retinal OCT", fontweight="bold")
    for opt, log in logs.items():
        ax1.plot(log["history"]["epoch"], log["history"]["train_loss"],
                 color=COLORS[opt], ls=LINESTYLES[opt], lw=2, label=LABELS[opt])
        ax2.plot(log["history"]["epoch"], log["history"]["val_loss"],
                 color=COLORS[opt], ls=LINESTYLES[opt], lw=2, label=LABELS[opt])
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Training Loss"); ax1.legend()
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss"); ax2.set_title("Validation Loss"); ax2.legend()
    plt.tight_layout()
    _save(fig, plot_dir, "loss_curves")


# Plot 2 — Macro Recall (PRIMARY)

def plot_recall_curves(logs: Dict, plot_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title("Macro Recall vs. Epoch — Retinal OCT (Primary Metric)", fontweight="bold")
    for opt, log in logs.items():
        ax.plot(log["history"]["epoch"], log["history"]["recall"],
                color=COLORS[opt], ls=LINESTYLES[opt], lw=2.5, label=LABELS[opt])
    ax.axhline(0.85, color="gray", ls="--", lw=1, label="0.85 threshold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Macro Recall")
    ax.set_ylim(0, 1.05); ax.legend()
    plt.tight_layout()
    _save(fig, plot_dir, "recall_curves")


# Plot 3 — β_t Trajectory (LBM)

def plot_beta_trajectory(logs: Dict, plot_dir: Path) -> None:
    if "lipschitz_momentum" not in logs:
        return
    log = logs["lipschitz_momentum"]
    beta_series = []
    for epoch_betas in log.get("beta_trajectory", []):
        if isinstance(epoch_betas, list):
            beta_series.extend(epoch_betas)
        else:
            beta_series.append(epoch_betas)
    if not beta_series:
        return

    steps = np.arange(len(beta_series))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title("Dynamic Momentum β_t Over Training Steps — Retinal OCT (LBM)", fontweight="bold")
    ax.plot(steps, beta_series, color=COLORS["lipschitz_momentum"], lw=1.5, alpha=0.8)
    ax.axhline(0.9, color="gray", ls="--", lw=1.2, label="Fixed β=0.9 (baselines)")
    ax.fill_between(steps, beta_series, 0.9,
                    where=[b > 0.9 for b in beta_series],
                    alpha=0.15, color=COLORS["lipschitz_momentum"], label="Above fixed β")
    ax.fill_between(steps, beta_series, 0.9,
                    where=[b < 0.9 for b in beta_series],
                    alpha=0.15, color="steelblue", label="Below fixed β (damped)")
    ax.set_xlabel("Training Step"); ax.set_ylabel("β_t")
    ax.set_ylim(0.80, 1.01); ax.legend()
    plt.tight_layout()
    _save(fig, plot_dir, "beta_trajectory")


# Plot 4 — L_t Trajectory (LBM)

def plot_lipschitz_trajectory(logs: Dict, plot_dir: Path) -> None:
    if "lipschitz_momentum" not in logs:
        return
    log = logs["lipschitz_momentum"]
    L_series = []
    for epoch_L in log.get("lipschitz_trajectory", []):
        if isinstance(epoch_L, list):
            L_series.extend(epoch_L)
        else:
            L_series.append(epoch_L)
    if not L_series:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title("Estimated Lipschitz Constant L_t — Retinal OCT (LBM)", fontweight="bold")
    ax.semilogy(np.arange(len(L_series)), L_series,
                color=COLORS["lipschitz_momentum"], lw=1.5, alpha=0.8)
    ax.set_xlabel("Training Step"); ax.set_ylabel("L_t (log scale)")
    plt.tight_layout()
    _save(fig, plot_dir, "lipschitz_trajectory")


# Plot 5 — Convergence Bar Chart

def plot_convergence_comparison(logs: Dict, plot_dir: Path, threshold: float = 0.85) -> None:
    convergence_epochs = {}
    for opt, log in logs.items():
        conv = None
        for ep, rec in zip(log["history"]["epoch"], log["history"]["recall"]):
            if rec >= threshold:
                conv = ep
                break
        convergence_epochs[opt] = conv

    values  = [v if v is not None else 30 for v in convergence_epochs.values()]
    labels  = [LABELS[o] for o in convergence_epochs]
    colors  = [COLORS[o] for o in convergence_epochs]
    hatches = ["" if o == "lipschitz_momentum" else "///" for o in convergence_epochs]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(f"Epochs to Macro Recall ≥ {threshold} — Retinal OCT", fontweight="bold")
    bars = ax.bar(labels, values, color=colors, hatch=hatches, edgecolor="white", linewidth=1.5)
    for bar, val, opt in zip(bars, values, convergence_epochs):
        label = str(val) if convergence_epochs[opt] is not None else "N/A"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                label, ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Epoch"); ax.set_ylim(0, max(values) + 4)
    ax.tick_params(axis="x", rotation=10)
    plt.tight_layout()
    _save(fig, plot_dir, "convergence_comparison")


# Plot 6 — Summary Metrics Bar

def plot_metrics_summary(logs: Dict, plot_dir: Path) -> None:
    metrics_to_plot = ["recall", "precision", "f1", "auc_roc", "auprc", "accuracy"]
    optimizers = list(logs.keys())
    x = np.arange(len(metrics_to_plot))
    width = 0.18

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_title("Test Set Metrics Comparison — Retinal OCT (Macro)", fontweight="bold")
    for i, opt in enumerate(optimizers):
        test_m = logs[opt].get("test_metrics", {})
        vals   = [test_m.get(m, 0) for m in metrics_to_plot]
        offset = (i - len(optimizers) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=LABELS[opt],
               color=COLORS[opt], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper().replace("_", "-") for m in metrics_to_plot])
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.1)
    ax.legend(); ax.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.5)
    plt.tight_layout()
    _save(fig, plot_dir, "metrics_summary")


# Plot 7 — ROC Curves (from checkpoints)

def plot_roc_curves_from_logs(logs: Dict, cfg: dict, plot_dir: Path) -> None:
    """Plot per-class macro-averaged ROC curves by loading each optimizer's checkpoint."""
    device = get_device(cfg)
    _, _, test_loader = build_dataloaders(cfg, seed=cfg["project"]["seed"])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_title("ROC-AUC Comparison — Retinal OCT (Macro-avg)", fontweight="bold")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")

    num_classes = len(CLASS_NAMES)

    for opt in logs.keys():
        ckpt_path = Path(cfg["training"]["checkpoint_dir"]) / f"{opt}_best.pth"
        if not ckpt_path.exists():
            print(f"  [WARNING] Checkpoint not found for {opt}: {ckpt_path}")
            continue

        model = build_model(cfg).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        all_labels, all_probs = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                logits = model(images)
                probs = torch.softmax(logits, dim=1)
                all_labels.append(labels.cpu())
                all_probs.append(probs.cpu())

        y_true = torch.cat(all_labels).numpy()
        y_prob = torch.cat(all_probs, dim=0).numpy()
        y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

        # Compute macro-average ROC by averaging per-class FPR/TPR
        from sklearn.metrics import roc_curve as sk_roc_curve
        all_fpr = np.linspace(0, 1, 200)
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(num_classes):
            fpr_i, tpr_i, _ = sk_roc_curve(y_true_bin[:, i], y_prob[:, i])
            mean_tpr += np.interp(all_fpr, fpr_i, tpr_i)
        mean_tpr /= num_classes

        macro_auc = roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr")

        ax.plot(all_fpr, mean_tpr, lw=2, color=COLORS[opt], ls=LINESTYLES[opt],
                label=f"{LABELS[opt]}  (AUC={macro_auc:.3f})")

    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.legend(loc="lower right")
    plt.tight_layout()
    _save(fig, plot_dir, "roc_summary")


# Summary table

def print_comparison_table(logs: Dict) -> None:
    metrics = ["recall", "precision", "f1", "auc_roc", "auprc", "accuracy"]
    header  = f"{'Optimizer':<25}" + "".join(f"{m.upper():<12}" for m in metrics)
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for opt, log in logs.items():
        test_m = log.get("test_metrics", {})
        row = f"{LABELS[opt]:<25}" + "".join(f"{test_m.get(m, 0):<12.4f}" for m in metrics)
        print(row)
    print("=" * len(header) + "\n")


def _save(fig, plot_dir: Path, name: str) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ["png", "pdf"]:
        fig.savefig(plot_dir / f"{name}.{fmt}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] Saved → {plot_dir / name}.png")


# Main

def main(plot_only: bool = False, config_path: str = "retinal_oct/configs/config.yaml") -> None:
    cfg        = load_config(config_path)
    optimizers = cfg["comparison"]["optimizers_to_run"]
    log_dir    = Path(cfg["training"]["log_dir"])
    plot_dir   = Path(cfg["evaluation"]["plot_dir"])

    if not plot_only:
        print("\n" + "=" * 60)
        print("  COMPARISON RUN — All 4 Optimizers  [Retinal OCT]")
        print("=" * 60)
        for opt in optimizers:
            train(opt, cfg)

    print("\n  Generating plots from logs...")
    logs = load_logs(log_dir, optimizers)
    if not logs:
        print("  No logs found. Run training first.")
        return

    plot_loss_curves(logs, plot_dir)
    plot_recall_curves(logs, plot_dir)
    plot_beta_trajectory(logs, plot_dir)
    plot_lipschitz_trajectory(logs, plot_dir)
    plot_convergence_comparison(logs, plot_dir)
    plot_metrics_summary(logs, plot_dir)
    plot_roc_curves_from_logs(logs, cfg, plot_dir)
    print_comparison_table(logs)

    print(f"\n  All plots saved to: {plot_dir}")
    print("  ✓ Comparison complete.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--config", type=str, default="retinal_oct/configs/config.yaml",
                        help="Path to config YAML")
    args = parser.parse_args()
    main(plot_only=args.plot_only, config_path=args.config)
