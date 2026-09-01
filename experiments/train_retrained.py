"""Train the retrained oracle on retain samples only."""
from pathlib import Path
import time
import yaml
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from datasets import load_cifar10, make_loader, make_split, seed_everything
from models import CIFARResNet18
from evaluation import evaluate


def main():
    cfg = yaml.safe_load(Path("configs/default.yaml").read_text())
    seed_everything(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, test_ds = load_cifar10(cfg["paths"]["data"])
    retain, forget, _, _ = make_split(
        train_ds, cfg["cifar10"]["forget_fraction"], int(cfg["seed"]),
        Path(cfg["paths"]["artifacts"]) / "split_manifest.json"
    )
    loader = make_loader(retain, cfg["training"]["batch_size"], True, cfg["training"]["num_workers"])
    test_loader = make_loader(test_ds, cfg["training"]["batch_size"], False, cfg["training"]["num_workers"])
    model = CIFARResNet18(cfg["cifar10"]["num_classes"]).to(device)
    opt = SGD(model.parameters(), lr=cfg["training"]["learning_rate"], momentum=cfg["training"]["momentum"], weight_decay=cfg["training"]["weight_decay"])
    sched = CosineAnnealingLR(opt, T_max=cfg["training"]["epochs"])
    criterion = torch.nn.CrossEntropyLoss()
    start = time.perf_counter()
    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        for x, y in tqdm(loader, leave=False):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
        sched.step()
        print(f"epoch={epoch+1:02d} test_acc={evaluate(model, test_loader, device)['accuracy']:.4f}")
    elapsed = time.perf_counter() - start
    out = Path(cfg["paths"]["checkpoints"])
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": cfg, "train_seconds": elapsed}, out / "retrained.pt")
    print(f"Saved {out / 'retrained.pt'} in {elapsed/60:.2f} min")


if __name__ == "__main__":
    main()
