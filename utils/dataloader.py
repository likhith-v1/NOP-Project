"""
Data loading pipeline for Chest X-Ray (Pneumonia) dataset.

Dataset source : Kaggle — paulmooney/chest-xray-pneumonia
Expected layout:
    datasets/chest_xray/
        train/  NORMAL/  PNEUMONIA/
        val/    NORMAL/  PNEUMONIA/
        test/   NORMAL/  PNEUMONIA/

Key features:
  - Weighted random sampler to counter 3:1 class imbalance
  - Train / val / test augmentation strategies
  - MPS-compatible (pin_memory=False)
  - Reproducible via seed
"""

import os
import yaml
import numpy as np
from pathlib import Path
from typing import Tuple, Dict

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms


# Helpers


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def seed_worker(worker_id: int) -> None:
    """Ensure each DataLoader worker is deterministic."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)


def get_class_weights(dataset: datasets.ImageFolder) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from dataset.
    Returns a 1D tensor of shape [num_classes].
    """
    targets = np.array(dataset.targets)
    class_counts = np.bincount(targets)
    total = len(targets)
    # weight_i = total / (num_classes * count_i)
    weights = total / (len(class_counts) * class_counts)
    return torch.tensor(weights, dtype=torch.float32)


def get_sample_weights(dataset: datasets.ImageFolder) -> torch.Tensor:
    """
    Assign per-sample weights for WeightedRandomSampler.
    Each sample gets the weight of its class.
    """
    targets = np.array(dataset.targets)
    class_counts = np.bincount(targets)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[targets]
    return torch.tensor(sample_weights, dtype=torch.float32)


# Transforms


def build_train_transform(cfg: dict) -> transforms.Compose:
    """
    Aggressive augmentation for training.
    Chest X-Rays are grayscale but stored as RGB — we keep 3 channels
    for compatibility with DenseNet-121 pretrained weights.
    """
    aug = cfg["data"]["augmentation"]
    norm = cfg["data"]["normalize"]
    size = cfg["data"]["image_size"]

    cj = aug.get("color_jitter", {})

    transform_list = [
        transforms.Resize((size + 20, size + 20)),  # slightly oversized for crop
        transforms.RandomCrop(size),
    ]

    if aug.get("random_horizontal_flip", True):
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

    rot = aug.get("random_rotation_degrees", 10)
    if rot > 0:
        transform_list.append(transforms.RandomRotation(degrees=rot))

    if cj:
        transform_list.append(
            transforms.ColorJitter(
                brightness=cj.get("brightness", 0.2),
                contrast=cj.get("contrast", 0.2),
            )
        )

    transform_list += [
        transforms.ToTensor(),
        transforms.Normalize(mean=norm["mean"], std=norm["std"]),
    ]

    erasing_prob = aug.get("random_erasing_prob", 0.1)
    if erasing_prob > 0:
        # Applied after ToTensor
        transform_list.append(
            transforms.RandomErasing(p=erasing_prob, scale=(0.02, 0.1))
        )

    return transforms.Compose(transform_list)


def build_eval_transform(cfg: dict) -> transforms.Compose:
    """
    Deterministic transform for val / test — no augmentation.
    Centre crop only.
    """
    norm = cfg["data"]["normalize"]
    size = cfg["data"]["image_size"]

    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm["mean"], std=norm["std"]),
        ]
    )


# Dataset + DataLoader factory


def build_datasets(cfg: dict) -> Dict[str, datasets.ImageFolder]:
    """
    Build ImageFolder datasets for train / val / test splits.
    The Kaggle chest-xray dataset's val split is tiny (16 images).
    We keep it as-is; evaluation happens primarily on test.
    """
    root = Path(cfg["data"]["root"])

    train_tf = build_train_transform(cfg)
    eval_tf = build_eval_transform(cfg)

    dataset_dict = {
        "train": datasets.ImageFolder(root / "train", transform=train_tf),
        "val": datasets.ImageFolder(root / "val", transform=eval_tf),
        "test": datasets.ImageFolder(root / "test", transform=eval_tf),
    }

    # Sanity check — class order must be consistent
    for split, ds in dataset_dict.items():
        assert ds.class_to_idx == {"NORMAL": 0, "PNEUMONIA": 1}, (
            f"[{split}] Unexpected class ordering: {ds.class_to_idx}. "
            "Check your dataset folder names."
        )

    return dataset_dict


def build_dataloaders(
    cfg: dict,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader, test_loader).

    Train loader uses WeightedRandomSampler to handle class imbalance.
    Val and test loaders are sequential (no shuffle).
    MPS compatibility: pin_memory=False.
    """
    datasets_dict = build_datasets(cfg)

    g = torch.Generator()
    g.manual_seed(seed)

    # ── Train ──────────────────────────────────
    train_ds = datasets_dict["train"]

    if cfg["data"].get("use_weighted_sampler", True):
        sample_weights = get_sample_weights(train_ds)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=g,
        )
        shuffle = False  # mutually exclusive with sampler
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["data"]["batch_size_train"],
        sampler=sampler,
        shuffle=shuffle,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],  # False for MPS
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=True,  # avoids stray 1-sample batches disrupting HVP
    )

    # Val
    val_loader = DataLoader(
        datasets_dict["val"],
        batch_size=cfg["data"]["batch_size_val"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
    )

    # Test
    test_loader = DataLoader(
        datasets_dict["test"],
        batch_size=cfg["data"]["batch_size_test"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
    )

    return train_loader, val_loader, test_loader


# Dataset statistics helper


def print_dataset_stats(cfg: dict) -> None:
    """Print class distribution for all splits."""
    datasets_dict = build_datasets(cfg)

    print("\n" + "=" * 50)
    print("  Dataset Statistics: Chest X-Ray (Pneumonia)")
    print("=" * 50)

    for split, ds in datasets_dict.items():
        targets = np.array(ds.targets)
        counts = np.bincount(targets)
        total = len(targets)
        print(f"\n  [{split.upper()}]  Total: {total}")
        for idx, cls in enumerate(ds.classes):
            pct = 100 * counts[idx] / total
            bar = "█" * int(pct / 2)
            print(f"    {cls:<12} {counts[idx]:>5}  ({pct:5.1f}%)  {bar}")

    # Class weights for loss function
    train_ds = datasets_dict["train"]
    cw = get_class_weights(train_ds)
    print(f"\n  Loss class weights → NORMAL: {cw[0]:.4f} | PNEUMONIA: {cw[1]:.4f}")
    print("=" * 50 + "\n")


# Quick test

if __name__ == "__main__":
    cfg = load_config("configs/config.yaml")

    print_dataset_stats(cfg)

    train_loader, val_loader, test_loader = build_dataloaders(cfg, seed=42)

    # Grab one batch and verify shapes
    images, labels = next(iter(train_loader))
    print(f"  Train batch → images: {images.shape} | labels: {labels.shape}")
    print(
        f"  Pixel range after norm → min: {images.min():.3f} | max: {images.max():.3f}"
    )
    print(
        f"  Label distribution in batch → "
        f"NORMAL: {(labels == 0).sum().item()} | PNEUMONIA: {(labels == 1).sum().item()}"
    )
