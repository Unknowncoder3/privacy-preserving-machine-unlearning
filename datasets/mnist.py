"""Local MNIST IDX loader with deterministic retain/forget splitting.

The loader intentionally reads the four standard MNIST IDX files directly so
the experiment does not depend on an internet connection or torchvision's
download/processing cache.
"""
from __future__ import annotations

import gzip
import json
import random
import struct
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

MNIST_CLASSES = tuple(str(i) for i in range(10))
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _open_binary(path: Path):
    return gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")


def _read_idx(path: Path) -> np.ndarray:
    with _open_binary(path) as f:
        magic = struct.unpack(">I", f.read(4))[0]
        data_type = (magic >> 8) & 0xFF
        dimensions = magic & 0xFF
        if data_type != 0x08:
            raise ValueError(f"Unsupported IDX data type in {path}: {data_type}")

        shape = tuple(struct.unpack(">I", f.read(4))[0] for _ in range(dimensions))
        raw = f.read()

    expected = int(np.prod(shape))
    if len(raw) != expected:
        raise ValueError(
            f"Invalid IDX file {path}: expected {expected} bytes, got {len(raw)}"
        )
    return np.frombuffer(raw, dtype=np.uint8).reshape(shape)


def _find_file(root: Path, stem: str) -> Path:
    # MNIST archives are commonly distributed with either the standard
    # hyphenated names or names using a dot before the IDX dimension.
    variants = (
        stem,
        stem.replace("-idx", ".idx"),
        f"{stem}.gz",
        f"{stem.replace('-idx', '.idx')}.gz",
    )
    for name in variants:
        candidate = root / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Missing MNIST file: {stem}(.gz) in {root}. "
        "Expected the standard MNIST IDX files."
    )


class LocalMNIST(Dataset):
    """Minimal MNIST Dataset backed by local IDX files."""

    def __init__(self, root: str | Path, train: bool = True, transform=None):
        self.root = Path(root)
        self.transform = transform
        prefix = "train" if train else "t10k"

        image_path = _find_file(self.root, f"{prefix}-images-idx3-ubyte")
        label_path = _find_file(self.root, f"{prefix}-labels-idx1-ubyte")

        self.images = _read_idx(image_path)
        self.targets = _read_idx(label_path).astype(np.int64)

        if self.images.ndim != 3 or self.images.shape[1:] != (28, 28):
            raise ValueError(f"Unexpected MNIST image shape: {self.images.shape}")
        if len(self.images) != len(self.targets):
            raise ValueError("MNIST image and label counts do not match.")

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        image = self.images[index]
        target = int(self.targets[index])
        image = transforms.functional.to_pil_image(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def get_transforms(train: bool = True):
    # MNIST is intentionally kept deterministic. Reproducibility matters more
    # than augmentation for this unlearning benchmark.
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])


def load_mnist(data_root: str | Path):
    root = Path(data_root)
    # Accept either data/MNIST or a direct MNIST directory.
    if (root / "MNIST").is_dir():
        root = root / "MNIST"

    train = LocalMNIST(root, train=True, transform=get_transforms(True))
    test = LocalMNIST(root, train=False, transform=get_transforms(False))
    return train, test


def make_split(
    train_dataset,
    forget_fraction: float,
    seed: int,
    manifest_path: str | Path | None = None,
) -> Tuple[Subset, Subset, list[int], list[int]]:
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
            "dataset": "MNIST",
            "seed": seed,
            "forget_fraction": forget_fraction,
            "total": n,
            "retain_count": len(retain_indices),
            "forget_count": len(forget_indices),
            "retain_indices": retain_indices,
            "forget_indices": forget_indices,
        }, indent=2))

    return retain, forget, retain_indices, forget_indices


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int = 0):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
