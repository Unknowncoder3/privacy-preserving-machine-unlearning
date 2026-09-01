"""Run a confidence-based membership inference evaluation."""
from pathlib import Path
import json
import yaml
import torch

from datasets import load_cifar10, make_loader, make_split
from models import CIFARResNet18
from attacks import confidence_scores, evaluate_confidence_attack


def load_model(path, device, classes):
    model = CIFARResNet18(classes).to(device)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return model


def main():
    cfg = yaml.safe_load(Path("configs/default.yaml").read_text())
    train, test = load_cifar10(cfg["paths"]["data"])
    retain, forget, _, _ = make_split(train, cfg["cifar10"]["forget_fraction"], int(cfg["seed"]), Path(cfg["paths"]["artifacts"]) / "split_manifest.json")
    b, w = cfg["training"]["batch_size"], cfg["training"]["num_workers"]
    forget_loader = make_loader(forget, b, False, w)
    # Test data acts as a simple non-member reference population.
    nonmember_loader = make_loader(test, b, False, w)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(cfg["paths"]["checkpoints"])
    results = {}
    for name in ("original", "retrained", "unlearned_combined"):
        path = root / f"{name}.pt"
        if not path.exists():
            continue
        model = load_model(path, device, cfg["cifar10"]["num_classes"])
        results[name] = evaluate_confidence_attack(
            confidence_scores(model, forget_loader, device),
            confidence_scores(model, nonmember_loader, device),
        )
    out = Path(cfg["paths"]["results"]); out.mkdir(parents=True, exist_ok=True)
    (out / "mia.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
