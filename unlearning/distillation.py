"""Knowledge-distillation utilities for retain-set knowledge preservation."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def retain_distillation_loss(student_logits: torch.Tensor,
                             teacher_logits: torch.Tensor,
                             temperature: float = 4.0) -> torch.Tensor:
    t = float(temperature)
    return F.kl_div(
        F.log_softmax(student_logits / t, dim=1),
        F.softmax(teacher_logits / t, dim=1),
        reduction="batchmean",
    ) * (t ** 2)
