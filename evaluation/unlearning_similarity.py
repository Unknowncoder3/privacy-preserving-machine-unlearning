"""Evaluate how closely the unlearned model matches the retrained oracle."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from datasets import load_mnist, make_loader, make_split
from models import MNISTCNN
from evaluation.similarity import compare_logits


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(path: Path, device, num_classes: int):
    model = MNISTCNN(num_classes).to(device)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


@torch.no_grad()
def collect_logits(model, loader, device):
    outputs = []
    labels = []
    for x, y in loader:
        outputs.append(model(x.to(device)).cpu())
        labels.append(y)
    return torch.cat(outputs), torch.cat(labels)


def main():
    cfg = yaml.safe_load(Path("configs/default.yaml").read_text())
    train, test = load_mnist(cfg["paths"]["data"])
    retain, forget, _, _ = make_split(
        train,
        cfg["mnist"]["forget_fraction"],
        int(cfg["seed"]),
        Path(cfg["paths"]["artifacts"]) / "split_manifest.json",
    )

    batch = cfg["training"]["batch_size"]
    workers = cfg["training"]["num_workers"]
    loaders = {
        "retain": make_loader(retain, batch, False, workers),
        "forget": make_loader(forget, batch, False, workers),
        "test": make_loader(test, batch, False, workers),
    }

    device = get_device()
    root = Path(cfg["paths"]["checkpoints"])
    unlearned_path = root / "unlearned_combined.pt"
    retrained_path = root / "retrained.pt"
    original_path = root / "original.pt"

    for path in (unlearned_path, retrained_path, original_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {path}")

    unlearned = load_model(
        unlearned_path, device, cfg["mnist"]["num_classes"]
    )
    retrained = load_model(
        retrained_path, device, cfg["mnist"]["num_classes"]
    )
    original = load_model(
        original_path, device, cfg["mnist"]["num_classes"]
    )

    results = {}
    for split, loader in loaders.items():
        unlearned_logits, labels = collect_logits(unlearned, loader, device)
        retrained_logits, _ = collect_logits(retrained, loader, device)
        original_logits, _ = collect_logits(original, loader, device)

        results[split] = {
            "unlearned_vs_retrained": compare_logits(
                unlearned_logits, retrained_logits
            ),
            "unlearned_vs_original": compare_logits(
                unlearned_logits, original_logits
            ),
            "samples": int(labels.numel()),
        }

    out = Path(cfg["paths"]["results"])
    out.mkdir(parents=True, exist_ok=True)
    path = out / "unlearning_similarity.json"
    path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
