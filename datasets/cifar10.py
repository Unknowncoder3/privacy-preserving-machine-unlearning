"""CIFAR-10 loading and deterministic retain/forget splitting."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_transforms(train: bool = True):
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616)),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])


def load_cifar10(data_root: str | Path):
    root = Path(data_root)
    train = datasets.CIFAR10(root=root, train=True, download=True,
                              transform=get_transforms(True))
    test = datasets.CIFAR10(root=root, train=False, download=True,
                             transform=get_transforms(False))
    return train, test


def make_split(train_dataset, forget_fraction: float, seed: int,
               manifest_path: str | Path | None = None) -> Tuple[Subset, Subset, list[int], list[int]]:
    if not 0 < forget_fraction < 1:
        raise ValueError("forget_fraction must be between 0 and 1")

    n = len(train_dataset)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=generator).tolist()
    forget_size = int(round(n * forget_fraction))
    forget_indices = sorted(indices[:forget_size])
    retain_indices = sorted(indices[forget_size:])

    retain = Subset(train_dataset, retain_indices)
    forget = Subset(train_dataset, forget_indices)

    if manifest_path:
        path = Path(manifest_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "dataset": "CIFAR-10",
            "seed": seed,
            "forget_fraction": forget_fraction,
            "total": n,
            "retain_count": len(retain_indices),
            "forget_count": len(forget_indices),
            "retain_indices": retain_indices,
            "forget_indices": forget_indices,
        }, indent=2))
    return retain, forget, retain_indices, forget_indices


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int = 2):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available())
