"""Run the proposed combined unlearning method."""
from pathlib import Path
import yaml
import torch
from torch.optim import SGD
from tqdm import tqdm

from datasets import load_cifar10, make_loader, make_split, seed_everything
from models import CIFARResNet18
from evaluation import evaluate, confidence_statistics
from unlearning import selective_forgetting_step


def load_model(path, device, num_classes, cfg):
    model = CIFARResNet18(num_classes).to(device)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    return model


def main():
    cfg = yaml.safe_load(Path("configs/default.yaml").read_text())
    seed_everything(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, test_ds = load_cifar10(cfg["paths"]["data"])
    retain, forget, _, _ = make_split(
        train_ds, cfg["cifar10"]["forget_fraction"], int(cfg["seed"]),
        Path(cfg["paths"]["artifacts"]) / "split_manifest.json"
    )
    batch = cfg["unlearning"]["batch_size"]
    workers = cfg["training"]["num_workers"]
    retain_loader = make_loader(retain, batch, True, workers)
    forget_loader = make_loader(forget, batch, True, workers)
    test_loader = make_loader(test_ds, batch, False, workers)

    original_path = Path(cfg["paths"]["checkpoints"]) / "original.pt"
    if not original_path.exists():
        raise FileNotFoundError("Train original.pt first with experiments/train_original.py")

    model = load_model(original_path, device, cfg["cifar10"]["num_classes"], cfg)
    teacher = load_model(original_path, device, cfg["cifar10"]["num_classes"], cfg)
    for p in teacher.parameters():
        p.requires_grad_(False)

    optimizer = SGD(model.parameters(), lr=cfg["unlearning"]["learning_rate"], momentum=0.9)
    forget_iter = iter(forget_loader)
    steps = 0
    for epoch in range(cfg["unlearning"]["epochs"]):
        for retain_batch in tqdm(retain_loader, desc=f"Unlearning {epoch+1}"):
            try:
                forget_batch = next(forget_iter)
            except StopIteration:
                forget_iter = iter(forget_loader)
                forget_batch = next(forget_iter)
            stats = selective_forgetting_step(
                model, teacher, retain_batch, forget_batch, optimizer, device,
                temperature=cfg["unlearning"]["temperature"],
                kd_weight=cfg["unlearning"]["kd_weight"],
                forget_weight=cfg["unlearning"]["forget_weight"],
                gradient_threshold=cfg["unlearning"]["gradient_threshold"],
            )
            steps += 1
        test_metrics = evaluate(model, test_loader, device)
        print(f"epoch={epoch+1:02d} test_acc={test_metrics['accuracy']:.4f} "
              f"retain_ce={stats['retain_ce']:.4f} forget_ce={stats['forget_ce']:.4f}")

    out = Path(cfg["paths"]["checkpoints"])
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": cfg, "steps": steps}, out / "unlearned_combined.pt")
    print("Saved", out / "unlearned_combined.pt")
    print("Test:", evaluate(model, test_loader, device))
    print("Forget:", evaluate(model, forget_loader, device))
    print("Forget confidence:", confidence_statistics(model, forget_loader, device))
    print("Retain:", evaluate(model, retain_loader, device))


if __name__ == "__main__":
    main()
