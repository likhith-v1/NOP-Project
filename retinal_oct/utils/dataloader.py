"""Data loading pipeline for Retinal OCT dataset (kermany2018)."""

import os
import yaml
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms


def load_config(config_path="retinal_oct/configs/config.yaml"):
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        repo_root = Path(__file__).resolve().parents[2]
        alt = repo_root / config_path
        if alt.exists():
            cfg_path = alt
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)


def _get_targets(dataset):
    """Extract targets from ImageFolder or Subset."""
    if hasattr(dataset, "targets"):
        return np.array(dataset.targets)
    return np.array([dataset.dataset.targets[i] for i in dataset.indices])


def get_class_weights(dataset):
    """Inverse-frequency class weights -> tensor of shape [num_classes]."""
    targets = _get_targets(dataset)
    class_counts = np.bincount(targets)
    total = len(targets)
    weights = total / (len(class_counts) * class_counts)
    return torch.tensor(weights, dtype=torch.float32)


def get_sample_weights(dataset):
    targets = _get_targets(dataset)
    class_counts = np.bincount(targets)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[targets]
    return torch.tensor(sample_weights, dtype=torch.float32)


def build_train_transform(cfg):
    aug  = cfg["data"]["augmentation"]
    norm = cfg["data"]["normalize"]
    size = cfg["data"]["image_size"]
    cj   = aug.get("color_jitter", {})

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
    norm = cfg["data"]["normalize"]
    size = cfg["data"]["image_size"]
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm["mean"], std=norm["std"]),
    ])


EXPECTED_CLASS_TO_IDX = {"CNV": 0, "DME": 1, "DRUSEN": 2, "NORMAL": 3}


def build_datasets(cfg):
    root     = Path(cfg["data"]["root"])
    train_tf = build_train_transform(cfg)
    eval_tf  = build_eval_transform(cfg)

    val_dir = root / "val"
    if val_dir.is_dir() and any(val_dir.iterdir()):
        dataset_dict = {
            "train": datasets.ImageFolder(root / "train", transform=train_tf),
            "val":   datasets.ImageFolder(val_dir,        transform=eval_tf),
            "test":  datasets.ImageFolder(root / "test",  transform=eval_tf),
        }
    else:
        full_train = datasets.ImageFolder(root / "train", transform=train_tf)
        n = len(full_train)
        val_frac = cfg["data"].get("val_fraction", 0.1)
        n_val = int(n * val_frac)
        n_train = n - n_val
        gen = torch.Generator().manual_seed(cfg["project"].get("seed", 42))
        train_idx, val_idx = torch.utils.data.random_split(
            range(n), [n_train, n_val], generator=gen
        )
        val_ds = datasets.ImageFolder(root / "train", transform=eval_tf)
        dataset_dict = {
            "train": Subset(full_train, train_idx.indices),
            "val":   Subset(val_ds, val_idx.indices),
            "test":  datasets.ImageFolder(root / "test", transform=eval_tf),
        }

    for split, ds in dataset_dict.items():
        actual = ds.class_to_idx if hasattr(ds, "class_to_idx") else ds.dataset.class_to_idx
        assert actual == EXPECTED_CLASS_TO_IDX, (
            f"Unexpected class ordering in {split}: {actual}"
        )

    return dataset_dict


def build_dataloaders(cfg, seed=42):
    datasets_dict = build_datasets(cfg)

    # Pinned memory improves H2D transfer for CUDA, but is not useful on CPU/MPS.
    effective_pin_memory = bool(cfg["data"].get("pin_memory", True) and torch.cuda.is_available())

    g = torch.Generator()
    g.manual_seed(seed)

    train_ds = datasets_dict["train"]

    if cfg["data"].get("use_weighted_sampler", False):
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
        pin_memory=effective_pin_memory,
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
        pin_memory=effective_pin_memory,
        persistent_workers=cfg["data"].get("persistent_workers", False),
        prefetch_factor=cfg["data"].get("prefetch_factor", 2),
    )

    test_loader = DataLoader(
        datasets_dict["test"],
        batch_size=cfg["data"]["batch_size_test"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=effective_pin_memory,
        persistent_workers=cfg["data"].get("persistent_workers", False),
        prefetch_factor=cfg["data"].get("prefetch_factor", 2),
    )

    return train_loader, val_loader, test_loader


def print_dataset_stats(cfg):
    datasets_dict = build_datasets(cfg)
    class_names   = cfg["data"]["class_names"]

    print("\n" + "=" * 55)
    print("  Dataset Statistics: Retinal OCT (kermany2018)")
    print("=" * 55)

    for split, ds in datasets_dict.items():
        targets = _get_targets(ds)
        classes = ds.classes if hasattr(ds, "classes") else ds.dataset.classes
        counts  = np.bincount(targets, minlength=len(classes))
        total   = len(targets)
        print(f"\n  [{split.upper()}]  Total: {total}")
        for idx, cls in enumerate(classes):
            pct = 100 * counts[idx] / total
            bar = "█" * int(pct / 4)
            print(f"    {cls:<10} {counts[idx]:>6}  ({pct:5.1f}%)  {bar}")

    train_ds = datasets_dict["train"]
    cw = get_class_weights(train_ds)
    print(f"\n  Loss class weights →")
    for i, cls in enumerate(class_names):
        print(f"    {cls:<10}: {cw[i]:.4f}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    cfg = load_config("retinal_oct/configs/config.yaml")
    print_dataset_stats(cfg)
    train_loader, val_loader, test_loader = build_dataloaders(cfg, seed=42)
    images, labels = next(iter(train_loader))
    print(f"  Train batch → images: {images.shape} | labels: {labels.shape}")
    print(f"  Unique labels in batch: {labels.unique().tolist()}")