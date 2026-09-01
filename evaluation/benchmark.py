"""Evaluate saved checkpoints on retain, forget, and test sets."""
from __future__ import annotations
from pathlib import Path
import json
import yaml
import torch

from datasets import load_cifar10, make_loader, make_split
from models import CIFARResNet18
from evaluation import evaluate, confidence_statistics


def evaluate_checkpoint(checkpoint_path, loaders, device, num_classes):
    model = CIFARResNet18(num_classes).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    out = {}
    for name, loader in loaders.items():
        out[name] = evaluate(model, loader, device)
        out[name].update(confidence_statistics(model, loader, device))
    return out


def main():
    cfg = yaml.safe_load(Path("configs/default.yaml").read_text())
    train, test = load_cifar10(cfg["paths"]["data"])
    retain, forget, _, _ = make_split(train, cfg["cifar10"]["forget_fraction"], int(cfg["seed"]), Path(cfg["paths"]["artifacts"]) / "split_manifest.json")
    b, w = cfg["training"]["batch_size"], cfg["training"]["num_workers"]
    loaders = {"retain": make_loader(retain, b, False, w), "forget": make_loader(forget, b, False, w), "test": make_loader(test, b, False, w)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(cfg["paths"]["checkpoints"])
    results = {}
    for name in ("original", "retrained", "unlearned_combined"):
        path = root / f"{name}.pt"
        if path.exists():
            results[name] = evaluate_checkpoint(path, loaders, device, cfg["cifar10"]["num_classes"])
    out = Path(cfg["paths"]["results"]); out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
