"""Train the original ResNet-18 on all CIFAR-10 training samples."""
from pathlib import Path
import time
import yaml
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from datasets import load_cifar10, make_loader, seed_everything
from models import CIFARResNet18
from evaluation import evaluate


def get_device():
    """Prefer CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(model, loader, optimizer, device):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    running, total = 0.0, 0
    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        running += loss.item() * y.size(0)
        total += y.size(0)
    return running / total


def main():
    cfg = yaml.safe_load(Path("configs/default.yaml").read_text())
    seed_everything(int(cfg["seed"]))
    device = get_device()
    print(f"Using device: {device}")

    train_ds, test_ds = load_cifar10(cfg["paths"]["data"])
    loader = make_loader(
        train_ds,
        cfg["training"]["batch_size"],
        True,
        cfg["training"]["num_workers"],
    )
    test_loader = make_loader(
        test_ds,
        cfg["training"]["batch_size"],
        False,
        cfg["training"]["num_workers"],
    )

    model = CIFARResNet18(cfg["cifar10"]["num_classes"]).to(device)
    optimizer = SGD(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        momentum=cfg["training"]["momentum"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"])

    start = time.perf_counter()
    for epoch in range(cfg["training"]["epochs"]):
        loss = train(model, loader, optimizer, device)
        scheduler.step()
        metrics = evaluate(model, test_loader, device)
        print(
            f"epoch={epoch + 1:02d} train_loss={loss:.4f} "
            f"test_acc={metrics['accuracy']:.4f}"
        )
    elapsed = time.perf_counter() - start

    out = Path(cfg["paths"]["checkpoints"])
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": cfg,
            "train_seconds": elapsed,
        },
        out / "original.pt",
    )
    print(f"Saved {out / 'original.pt'} in {elapsed / 60:.2f} min")


if __name__ == "__main__":
    main()
