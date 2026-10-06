"""Stable selective-gradient + retain-set knowledge-distillation unlearning.

The forget objective is a bounded uniform-target objective rather than raw
cross-entropy ascent. This avoids exploding logits while explicitly reducing
confidence on the designated forget set.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def kd_loss(student_logits, teacher_logits, temperature: float):
    t = float(temperature)
    return F.kl_div(
        F.log_softmax(student_logits / t, dim=1),
        F.softmax(teacher_logits / t, dim=1),
        reduction="batchmean",
    ) * (t * t)


def uniform_forget_loss(logits: torch.Tensor) -> torch.Tensor:
    """KL(uniform || model prediction), minimized when predictions are uniform."""
    log_probs = F.log_softmax(logits, dim=1)
    classes = logits.size(1)
    uniform = torch.full_like(log_probs, 1.0 / classes)
    return F.kl_div(log_probs, uniform, reduction="batchmean")


def selective_forgetting_step(
    model,
    teacher,
    retain_batch,
    forget_batch,
    optimizer,
    device,
    temperature=4.0,
    kd_weight=0.5,
    forget_weight=0.1,
    gradient_threshold=0.25,
    max_grad_norm=1.0,
):
    """One stable combined unlearning step.

    Retain objective = CE + KD.
    Forget objective = KL(uniform || prediction) on forget samples.

    The forget gradient is selectively masked and normalized to the retain
    gradient scale, then applied with a small bounded weight. Final gradient
    clipping prevents a single batch from destabilizing the model.
    """
    model.train()
    teacher.eval()
    rx, ry = (v.to(device) for v in retain_batch)
    fx, _ = (v.to(device) for v in forget_batch)
    criterion = torch.nn.CrossEntropyLoss()

    # Forget-sensitive gradient.
    optimizer.zero_grad(set_to_none=True)
    forget_logits = model(fx)
    forget_loss = uniform_forget_loss(forget_logits)
    forget_grads = torch.autograd.grad(
        forget_loss, tuple(model.parameters()), retain_graph=False, allow_unused=True
    )

    masks = []
    q = min(max(float(gradient_threshold), 0.0), 1.0)
    for g in forget_grads:
        if g is None:
            masks.append(None)
            continue
        threshold = torch.quantile(g.detach().abs().flatten(), q)
        masks.append((g.detach().abs() >= threshold).to(g.dtype))

    # Retain objective.
    optimizer.zero_grad(set_to_none=True)
    student_logits = model(rx)
    with torch.no_grad():
        teacher_logits = teacher(rx)
    retain_ce = criterion(student_logits, ry)
    retain_kd = kd_loss(student_logits, teacher_logits, temperature)
    retain_loss = retain_ce + kd_weight * retain_kd
    retain_grads = torch.autograd.grad(
        retain_loss, tuple(model.parameters()), retain_graph=False, allow_unused=True
    )

    # Match the masked forget gradient to the retain-gradient scale.
    retain_sq = torch.zeros((), device=device)
    forget_sq = torch.zeros((), device=device)
    masked_forget = []
    for rg, fg, mask in zip(retain_grads, forget_grads, masks):
        if rg is not None:
            retain_sq = retain_sq + rg.detach().pow(2).sum()
        if fg is not None and mask is not None:
            mfg = fg.detach() * mask
            masked_forget.append(mfg)
            forget_sq = forget_sq + mfg.pow(2).sum()
        else:
            masked_forget.append(None)

    scale = torch.sqrt(retain_sq + 1e-12) / torch.sqrt(forget_sq + 1e-12)

    optimizer.zero_grad(set_to_none=True)
    for p, rg, mfg in zip(model.parameters(), retain_grads, masked_forget):
        if rg is None and mfg is None:
            p.grad = None
        elif rg is None:
            p.grad = forget_weight * scale * mfg
        elif mfg is None:
            p.grad = rg
        else:
            p.grad = rg + forget_weight * scale * mfg

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()

    return {
        "retain_ce": float(retain_ce.detach().item()),
        "retain_kd": float(retain_kd.detach().item()),
        "forget_uniform": float(forget_loss.detach().item()),
    }
