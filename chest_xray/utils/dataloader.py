"""Data loading pipeline for Chest X-Ray (Pneumonia) dataset."""

import os
import yaml
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms


def load_config(config_path="chest_xray/configs/config.yaml"):
    """Load YAML config, falling back to repo root if not found."""
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        # Try resolving relative to the repo root (two levels up from this file)
        repo_root = Path(__file__).resolve().parents[2]
        alt = repo_root / config_path
        if alt.exists():
            cfg_path = alt
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)


def get_class_weights(dataset):
    """Inverse-frequency class weights -> tensor of shape [num_classes]."""
    targets = np.array(dataset.targets)
    class_counts = np.bincount(targets)
    total = len(targets)
    # weight_i = total / (num_classes * count_i)
    weights = total / (len(class_counts) * class_counts)
    return torch.tensor(weights, dtype=torch.float32)


def get_sample_weights(dataset):
    """Per-sample weights for WeightedRandomSampler."""
    targets = np.array(dataset.targets)
    class_counts = np.bincount(targets)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[targets]
    return torch.tensor(sample_weights, dtype=torch.float32)


def build_train_transform(cfg):
    """Training augmentation transform."""
    aug = cfg["data"]["augmentation"]
    norm = cfg["data"]["normalize"]
    size = cfg["data"]["image_size"]

    cj = aug.get("color_jitter", {})

    transform_list = [
        transforms.Resize((size + 20, size + 20)),
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
        transform_list.append(
            transforms.RandomErasing(p=erasing_prob, scale=(0.02, 0.1))
        )

    return transforms.Compose(transform_list)


def build_eval_transform(cfg):
    """Deterministic transform for val/test."""
    norm = cfg["data"]["normalize"]
    size = cfg["data"]["image_size"]

    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm["mean"], std=norm["std"]),
        ]
    )


def build_datasets(cfg):
    """Build ImageFolder datasets for train/val/test."""
    root = Path(cfg["data"]["root"])

    train_tf = build_train_transform(cfg)
    eval_tf = build_eval_transform(cfg)

    dataset_dict = {
        "train": datasets.ImageFolder(root / "train", transform=train_tf),
        "val": datasets.ImageFolder(root / "val", transform=eval_tf),
        "test": datasets.ImageFolder(root / "test", transform=eval_tf),
    }

    for split, ds in dataset_dict.items():
        assert ds.class_to_idx == {"NORMAL": 0, "PNEUMONIA": 1}, (
            f"Unexpected class ordering in {split}: {ds.class_to_idx}"
        )

    return dataset_dict


def build_dataloaders(cfg, seed=42):
    """Returns (train_loader, val_loader, test_loader)."""
    datasets_dict = build_datasets(cfg)

    g = torch.Generator()
    g.manual_seed(seed)


    train_ds = datasets_dict["train"]

    if cfg["data"].get("use_weighted_sampler", True):
        sample_weights = get_sample_weights(train_ds)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=g,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["data"]["batch_size_train"],
        sampler=sampler,
        shuffle=shuffle,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
        persistent_workers=cfg["data"].get("persistent_workers", False),
        prefetch_factor=cfg["data"].get("prefetch_factor", 2),
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=True,
    )

    val_loader = DataLoader(
        datasets_dict["val"],
        batch_size=cfg["data"]["batch_size_val"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
        persistent_workers=cfg["data"].get("persistent_workers", False),
        prefetch_factor=cfg["data"].get("prefetch_factor", 2),
    )

    test_loader = DataLoader(
        datasets_dict["test"],
        batch_size=cfg["data"]["batch_size_test"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
        persistent_workers=cfg["data"].get("persistent_workers", False),
        prefetch_factor=cfg["data"].get("prefetch_factor", 2),
    )

    return train_loader, val_loader, test_loader


def print_dataset_stats(cfg):
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

    train_ds = datasets_dict["train"]
    cw = get_class_weights(train_ds)
    print(f"\n  Loss class weights → NORMAL: {cw[0]:.4f} | PNEUMONIA: {cw[1]:.4f}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    cfg = load_config("configs/config.yaml")
    print_dataset_stats(cfg)

    train_loader, val_loader, test_loader = build_dataloaders(cfg, seed=42)
    images, labels = next(iter(train_loader))
    print(f"  Train batch: {images.shape}, labels: {labels.shape}")
