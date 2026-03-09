"""
Training loop for a single optimizer run.

Usage:
    python train.py --optimizer lipschitz_momentum
    python train.py --optimizer heavy_ball
    python train.py --optimizer nesterov
    python train.py --optimizer adam

All results are saved to results/logs/<optimizer_name>.json
Best checkpoints saved to results/checkpoints/<optimizer_name>_best.pth
"""

import os
import sys
import json
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from tqdm import tqdm

# Local imports
from chest_xray.utils.dataloader import build_dataloaders, load_config, get_class_weights
from chest_xray.utils.metrics import MetricTracker, compute_metrics, run_inference
from chest_xray.models.densenet import build_model
from optimizers.lipschitz_momentum import LipschitzMomentumOptimizer


# Reproducibility


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


# Device setup


def get_device(cfg: dict) -> torch.device:
    requested = cfg["project"]["device"]
    if requested == "mps" and torch.backends.mps.is_available():
        print("  [Device] Apple MPS (Metal) — using GPU acceleration on Apple Silicon")
        return torch.device("mps")
    elif requested == "cuda" and torch.cuda.is_available():
        print(f"  [Device] CUDA — {torch.cuda.get_device_name(0)} detected")
        return torch.device("cuda")
    else:
        print("  [Device] CPU (MPS/CUDA unavailable) — training will be slow")
        return torch.device("cpu")


# Optimizer factory


def build_optimizer(optimizer_name: str, model: nn.Module, cfg: dict):
    ocfg = cfg["optimizers"][optimizer_name]

    if optimizer_name == "lipschitz_momentum":
        return LipschitzMomentumOptimizer(
            model.parameters(),
            lr=ocfg["lr"],
            beta_min=ocfg["beta_min"],
            beta_max=ocfg["beta_max"],
            eps=ocfg["eps"],
            weight_decay=ocfg["weight_decay"],
            power_iter_steps=ocfg["power_iter_steps"],
            hvp_epsilon=ocfg["hvp_epsilon"],
            per_layer=ocfg["per_layer"],
            lm_lambda_init=ocfg["lm_damping"]["lambda_init"],
            lm_lambda_up=ocfg["lm_damping"]["lambda_up"],
            lm_lambda_down=ocfg["lm_damping"]["lambda_down"],
            lm_lambda_min=ocfg["lm_damping"]["lambda_min"],
            lm_lambda_max=ocfg["lm_damping"]["lambda_max"],
        )

    elif optimizer_name == "heavy_ball":
        return torch.optim.SGD(
            model.parameters(),
            lr=ocfg["lr"],
            momentum=ocfg["momentum"],
            weight_decay=ocfg["weight_decay"],
            nesterov=False,
        )

    elif optimizer_name == "nesterov":
        return torch.optim.SGD(
            model.parameters(),
            lr=ocfg["lr"],
            momentum=ocfg["momentum"],
            weight_decay=ocfg["weight_decay"],
            nesterov=True,
        )

    elif optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=ocfg["lr"],
            betas=tuple(ocfg["betas"]),
            eps=ocfg["eps"],
            weight_decay=ocfg["weight_decay"],
        )

    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


# One training epoch


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    optimizer_name: str,
    use_hvp: bool = False,
    hvp_interval: int = 10,
) -> float:
    """
    Run one training epoch.

    For LipschitzMomentumOptimizer:
      - Every `hvp_interval` steps: use full HVP-based spectral norm (exact)
      - Other steps: use gradient-ratio approximation (fast)

    Returns mean training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = len(loader)

    pbar = tqdm(loader, desc=f"  Epoch {epoch:>3} [train]", leave=False)

    for step, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        # Forward
        logits = model(images)
        loss = criterion(logits, labels)

        # Backward
        loss.backward(retain_graph=(use_hvp and step % hvp_interval == 0))

        # Gradient clipping (prevents extreme steps, important with dynamic β)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step
        if (
            use_hvp
            and step % hvp_interval == 0
            and isinstance(optimizer, LipschitzMomentumOptimizer)
        ):
            # Full power-iteration step every hvp_interval batches
            optimizer.step_with_hvp(loss, model)
        else:
            if isinstance(optimizer, LipschitzMomentumOptimizer):
                optimizer.step(current_loss=loss.item())
            else:
                optimizer.step()

        total_loss += loss.item()

        # Update progress bar
        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                **(
                    {
                        "β": f"{optimizer.get_current_beta():.3f}",
                        "λ": f"{optimizer.get_current_lambda():.1e}",
                    }
                    if isinstance(optimizer, LipschitzMomentumOptimizer)
                    else {}
                ),
            }
        )

    return total_loss / num_batches


# Validation


def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """Returns (val_loss, metrics_dict, y_true, y_pred, y_prob)."""
    model.eval()
    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)

            total_loss += loss.item()
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()
    y_prob = torch.cat(all_probs).numpy()

    val_loss = total_loss / len(loader)
    metrics = compute_metrics(y_true, y_pred, y_prob)

    return val_loss, metrics, y_true, y_pred, y_prob


# Main training function


def train(optimizer_name: str, cfg: dict) -> MetricTracker:
    seed = cfg["project"]["seed"]
    set_seed(seed)

    device = get_device(cfg)
    epochs = cfg["training"]["epochs"]
    freeze_epochs = cfg["model"]["freeze_features_epochs"]
    patience = cfg["training"]["early_stopping_patience"]
    ckpt_dir = Path(cfg["training"]["checkpoint_dir"])
    log_dir = Path(cfg["training"]["log_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  Training with: {optimizer_name.upper()}")
    print(f"{'=' * 60}")

    # Data
    train_loader, val_loader, test_loader = build_dataloaders(cfg, seed=seed)

    # Model
    freeze = freeze_epochs > 0
    model = build_model(cfg, freeze_features=freeze).to(device)
    model.print_param_summary()

    # Loss
    # Compute class weights from training set
    from utils.dataloader import build_datasets

    train_ds = build_datasets(cfg)["train"]
    class_weights = get_class_weights(train_ds).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer & Scheduler
    optimizer = build_optimizer(optimizer_name, model, cfg)
    scfg = cfg["scheduler"]
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=scfg["T_max"],
        eta_min=scfg["eta_min"],
    )

    # Tracker
    tracker = MetricTracker(optimizer_name)
    best_recall = 0.0
    patience_counter = 0
    is_lbm = isinstance(optimizer, LipschitzMomentumOptimizer)

    # Training loop
    for epoch in range(1, epochs + 1):
        # Unfreeze backbone after warmup
        if freeze and epoch == freeze_epochs + 1:
            model.unfreeze_backbone()
            print(f"\n  [Epoch {epoch}] Backbone unfrozen — full fine-tuning begins.")

        t0 = time.time()

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch=epoch,
            optimizer_name=optimizer_name,
            use_hvp=is_lbm,
            hvp_interval=10,
        )

        val_loss, metrics, y_true, y_pred, y_prob = validate(
            model, val_loader, criterion, device
        )

        scheduler.step()
        elapsed = time.time() - t0

        # Log trajectories for LBM
        beta_vals = optimizer.beta_log[-len(train_loader) :] if is_lbm else None
        L_vals = optimizer.lipschitz_log[-len(train_loader) :] if is_lbm else None

        tracker.update(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            metrics=metrics,
            beta_values=beta_vals,
            lipschitz_values=L_vals,
        )

        recall = metrics["recall"]
        print(
            f"  Epoch {epoch:>3}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"recall={recall:.4f} | "
            f"f1={metrics['f1']:.4f} | "
            f"auc={metrics['auc_roc']:.4f} | "
            f"{elapsed:.1f}s"
            + (
                f" | β={optimizer.get_current_beta():.3f} "
                f"L={optimizer.get_current_lipschitz():.3f}"
                if is_lbm
                else ""
            )
        )

        # Checkpoint
        if recall > best_recall:
            best_recall = recall
            patience_counter = 0
            ckpt_path = ckpt_dir / f"{optimizer_name}_best.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "recall": recall,
                    "metrics": metrics,
                },
                ckpt_path,
            )
            print(f"    ✓ Checkpoint saved (recall={best_recall:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"\n  Early stopping at epoch {epoch} "
                    f"(no improvement for {patience} epochs)"
                )
                break

    # Final test evaluation
    print(f"\n  Loading best checkpoint for test evaluation...")
    ckpt = torch.load(ckpt_dir / f"{optimizer_name}_best.pth", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_loss, test_metrics, y_true, y_pred, y_prob = validate(
        model, test_loader, criterion, device
    )

    print(f"\n  [TEST RESULTS — {optimizer_name}]")
    for k, v in test_metrics.items():
        print(f"    {k:<15}: {v:.4f}")

    # Save log
    log_data = {
        "optimizer": optimizer_name,
        "history": tracker.history,
        "test_metrics": test_metrics,
        "beta_trajectory": tracker.beta_trajectory if is_lbm else [],
        "lipschitz_trajectory": tracker.lipschitz_trajectory if is_lbm else [],
    }
    log_path = log_dir / f"{optimizer_name}.json"
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)
    print(f"\n  Log saved → {log_path}")

    tracker.print_summary()
    return tracker


# Entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train with a single optimizer")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="lipschitz_momentum",
        choices=["lipschitz_momentum", "heavy_ball", "nesterov", "adam"],
        help="Optimizer to use for training",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="chest_xray/configs/config.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(args.optimizer, cfg)
