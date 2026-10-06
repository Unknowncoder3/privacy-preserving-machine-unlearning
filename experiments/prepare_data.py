"""Validate local MNIST and create a reproducible retain/forget split."""
from pathlib import Path
import yaml

from datasets import load_mnist, make_split, seed_everything


def main():
    cfg = yaml.safe_load(Path("configs/default.yaml").read_text())
    seed = int(cfg["seed"])
    seed_everything(seed)

    train, test = load_mnist(cfg["paths"]["data"])
    retain, forget, _, _ = make_split(
        train,
        cfg["mnist"]["forget_fraction"],
        seed,
        Path(cfg["paths"]["artifacts"]) / "split_manifest.json",
    )

    print(f"MNIST train: {len(train)}")
    print(f"MNIST test:  {len(test)}")
    print(f"Retain set:  {len(retain)}")
    print(f"Forget set:  {len(forget)}")
    print(f"Manifest:    {Path(cfg['paths']['artifacts']) / 'split_manifest.json'}")


if __name__ == "__main__":
    main()
