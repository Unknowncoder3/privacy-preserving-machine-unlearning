"""Download CIFAR-10 and create a reproducible retain/forget split."""
from pathlib import Path
import yaml

from datasets import load_cifar10, make_split, seed_everything


def main():
    cfg = yaml.safe_load(Path("configs/default.yaml").read_text())
    seed = int(cfg["seed"])
    seed_everything(seed)
    train, test = load_cifar10(cfg["paths"]["data"])
    retain, forget, retain_idx, forget_idx = make_split(
        train,
        cfg["cifar10"]["forget_fraction"],
        seed,
        Path(cfg["paths"]["artifacts"]) / "split_manifest.json",
    )
    print(f"CIFAR-10 train: {len(train)}")
    print(f"CIFAR-10 test:  {len(test)}")
    print(f"Retain set:     {len(retain)}")
    print(f"Forget set:     {len(forget)}")
    print(f"Manifest:       {Path(cfg['paths']['artifacts']) / 'split_manifest.json'}")


if __name__ == "__main__":
    main()
