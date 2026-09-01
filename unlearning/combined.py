"""Combined selective gradient forgetting + retain-set knowledge distillation.

The method starts from an already-trained model. For each step it computes a
forget-set gradient, masks only high-magnitude forget-sensitive coordinates,
and applies gradient ascent on those coordinates while optimizing the retain
objective (classification + teacher distillation) normally.
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


def selective_forgetting_step(model, teacher, retain_batch, forget_batch,
                              optimizer, device, temperature=4.0,
                              kd_weight=1.0, forget_weight=1.0,
                              gradient_threshold=0.5):
    """One combined unlearning step.

    `gradient_threshold` is a per-parameter quantile. A value of 0.5 keeps
    roughly the upper half of forget-gradient magnitudes for tensors that are
    non-empty. This is deliberately simple and inspectable for the baseline
    research implementation; the threshold is exposed for ablation studies.
    """
    model.train()
    teacher.eval()
    rx, ry = (v.to(device) for v in retain_batch)
    fx, fy = (v.to(device) for v in forget_batch)

    criterion = torch.nn.CrossEntropyLoss()

    # Forget gradient: retain only coordinates with large magnitude.
    optimizer.zero_grad(set_to_none=True)
    forget_logits = model(fx)
    forget_loss = criterion(forget_logits, fy)
    forget_grads = torch.autograd.grad(
        forget_loss, tuple(model.parameters()), retain_graph=False, allow_unused=True
    )

    masks = []
    q = min(max(float(gradient_threshold), 0.0), 1.0)
    for p, g in zip(model.parameters(), forget_grads):
        if g is None:
            masks.append(None)
            continue
        threshold = torch.quantile(g.detach().abs().flatten(), q)
        masks.append((g.detach().abs() >= threshold).to(g.dtype))

    # Retain objective: supervised loss + teacher distillation.
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

    # Directly assign the combined gradient: retain gradient minus selected
    # forget gradient. Subtracting the forget gradient is gradient ascent on
    # the forget loss, encouraging the model to reduce confidence on forgotten
    # examples while preserving the retain objective.
    optimizer.zero_grad(set_to_none=True)
    for p, rg, fg, mask in zip(model.parameters(), retain_grads, forget_grads, masks):
        if rg is None and fg is None:
            p.grad = None
        elif rg is None:
            p.grad = -forget_weight * fg * mask
        elif fg is None:
            p.grad = rg
        else:
            p.grad = rg - forget_weight * fg * mask
    optimizer.step()

    return {
        "retain_ce": float(retain_ce.detach().item()),
        "retain_kd": float(retain_kd.detach().item()),
        "forget_ce": float(forget_loss.detach().item()),
    }
