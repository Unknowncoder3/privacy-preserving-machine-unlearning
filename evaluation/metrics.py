"""Classification and unlearning evaluation metrics."""
from __future__ import annotations

import time
import torch


def evaluate(model, loader, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    criterion = torch.nn.CrossEntropyLoss()
    start = time.perf_counter()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += criterion(logits, y).item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    return {
        "loss": loss_sum / max(total, 1),
        "accuracy": correct / max(total, 1),
        "samples": total,
        "seconds": time.perf_counter() - start,
    }


def confidence_statistics(model, loader, device):
    model.eval()
    confidences = []
    with torch.no_grad():
        for x, _ in loader:
            probs = torch.softmax(model(x.to(device)), dim=1)
            confidences.append(probs.max(dim=1).values.cpu())
    if not confidences:
        return {"mean_confidence": 0.0}
    values = torch.cat(confidences)
    return {"mean_confidence": values.mean().item()}
