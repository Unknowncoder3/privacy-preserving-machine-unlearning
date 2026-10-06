"""Stable selective-gradient + retain-protected projected forgetting.

The proposed experimental variant keeps the bounded uniform-target forget
objective, but removes the component of the forget gradient that conflicts
with the retain objective. This allows stronger forgetting updates without
deliberately moving against the retain gradient.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import math


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


def bounded_ce_forget_loss(logits: torch.Tensor, targets: torch.Tensor, target_ce: float) -> torch.Tensor:
    """Bounded CE-ascent objective for label-aware forgetting.

    Minimizing negative capped cross-entropy performs gradient ascent on forget
    samples until their CE reaches target_ce; after that the gradient is zero.
    """
    ce = F.cross_entropy(logits, targets, reduction="none")
    cap = torch.full_like(ce, float(target_ce))
    return -torch.minimum(ce, cap).mean()


def project_conflicting_gradient(
    forget_grads,
    retain_grads,
    eps: float = 1e-12,
):
    """Remove the retain-conflicting component of a forget gradient.

    Both gradients are gradients of objectives to minimize. If their global
    dot product is negative, following the forget gradient would increase the
    retain objective. In that case, remove its projection onto the retain
    gradient. If they are aligned or orthogonal, leave the forget gradient
    unchanged.
    """
    retain_sq = torch.zeros((), device=next(
        (g.device for g in retain_grads if g is not None),
        torch.device("cpu"),
    ))
    dot = torch.zeros_like(retain_sq)

    for rg, fg in zip(retain_grads, forget_grads):
        if rg is not None:
            retain_sq = retain_sq + rg.detach().pow(2).sum()
        if rg is not None and fg is not None:
            dot = dot + (fg.detach() * rg.detach()).sum()

    if dot.item() >= 0.0 or retain_sq.item() <= eps:
        return [g.detach() if g is not None else None for g in forget_grads], False, float(dot.item())

    coefficient = dot / (retain_sq + eps)
    projected = []
    for rg, fg in zip(retain_grads, forget_grads):
        if fg is None:
            projected.append(None)
        elif rg is None:
            projected.append(fg.detach())
        else:
            projected.append(fg.detach() - coefficient * rg.detach())
    return projected, True, float(dot.item())


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
    forget_objective="uniform",
    forget_ce_target=None,
    gradient_threshold=0.25,
    max_grad_norm=1.0,
    project_conflicts=False,
):
    """One stable combined unlearning step.

    Retain objective = CE + KD.
    Forget objective = KL(uniform || prediction) on forget samples.

    The forget gradient is selectively masked, optionally projected away from
    the retain-conflicting component, normalized to the retain-gradient scale,
    and applied with a bounded weight. Final clipping prevents instability.
    """
    model.train()
    teacher.eval()
    rx, ry = (v.to(device) for v in retain_batch)
    fx, fy = (v.to(device) for v in forget_batch)
    criterion = torch.nn.CrossEntropyLoss()

    optimizer.zero_grad(set_to_none=True)
    forget_logits = model(fx)
    if forget_objective == "uniform":
        forget_loss = uniform_forget_loss(forget_logits)
    elif forget_objective == "bounded_ce":
        target_ce = math.log(forget_logits.size(1)) if forget_ce_target is None else float(forget_ce_target)
        forget_loss = bounded_ce_forget_loss(forget_logits, fy, target_ce)
    else:
        raise ValueError(f"Unknown forget_objective: {forget_objective}")
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

    masked_forget = []
    forget_sq = torch.zeros((), device=device)
    for fg, mask in zip(forget_grads, masks):
        if fg is not None and mask is not None:
            mfg = fg.detach() * mask
            masked_forget.append(mfg)
            forget_sq = forget_sq + mfg.pow(2).sum()
        else:
            masked_forget.append(None)

    projected = False
    conflict_dot = 0.0
    if project_conflicts:
        masked_forget, projected, conflict_dot = project_conflicting_gradient(
            masked_forget, retain_grads
        )

    retain_sq = torch.zeros((), device=device)
    for rg in retain_grads:
        if rg is not None:
            retain_sq = retain_sq + rg.detach().pow(2).sum()

    forget_sq = torch.zeros((), device=device)
    for mfg in masked_forget:
        if mfg is not None:
            forget_sq = forget_sq + mfg.pow(2).sum()

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
        "forget_uniform": float(forget_loss.detach().item()) if forget_objective == "uniform" else None,
        "forget_loss": float(forget_loss.detach().item()),
        "forget_objective": forget_objective,
        "projection_applied": bool(projected),
        "forget_retain_dot": float(conflict_dot),
    }
