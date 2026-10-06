"""Run a confidence-based membership inference evaluation."""
from pathlib import Path
import json
import yaml
import torch

from datasets import load_mnist, make_loader, make_split
from models import MNISTCNN
from attacks import confidence_scores, evaluate_confidence_attack


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(path, device, classes):
    model = MNISTCNN(classes).to(device)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return model


def main():
    cfg = yaml.safe_load(Path("configs/default.yaml").read_text())
    train, test = load_mnist(cfg["paths"]["data"])
    _, forget, _, _ = make_split(
        train, cfg["mnist"]["forget_fraction"], int(cfg["seed"]),
        Path(cfg["paths"]["artifacts"]) / "split_manifest.json"
    )

    b, w = cfg["training"]["batch_size"], cfg["training"]["num_workers"]
    forget_loader = make_loader(forget, b, False, w)
    nonmember_loader = make_loader(test, b, False, w)

    device = get_device()
    root = Path(cfg["paths"]["checkpoints"])
    results = {}

    for name in ("original", "retrained", "unlearned_combined"):
        path = root / f"{name}.pt"
        if not path.exists():
            continue
        model = load_model(path, device, cfg["mnist"]["num_classes"])
        results[name] = evaluate_confidence_attack(
            confidence_scores(model, forget_loader, device),
            confidence_scores(model, nonmember_loader, device),
        )

    out = Path(cfg["paths"]["results"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "mia.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
