"""Reusable gradient-ascent forgetting primitive."""
from __future__ import annotations

import torch
import torch.nn as nn


def forget_batch_gradient_ascent(model, batch, optimizer, device, weight=1.0):
    """Perform one gradient-ascent step on the forget loss.

    Returns the scalar forget loss. This is intentionally provided as a
    baseline; the proposed method uses a selective mask plus retain objective.
    """
    x, y = batch[0].to(device), batch[1].to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(x), y)
    (-weight * loss).backward()
    optimizer.step()
    return float(loss.detach().item())
