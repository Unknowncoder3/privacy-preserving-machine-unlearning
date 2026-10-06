"""Compare unlearned behavior with the original and retrained reference models.

The retrained model is treated as the practical oracle: it was trained from
scratch without the forget set. This module reports raw, interpretable
behavioral distances rather than inventing a single arbitrary score.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _flatten_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits.detach().float().flatten(1)


def _probabilities(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits.detach().float(), dim=1).clamp_min(1e-8)


def js_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """Mean Jensen-Shannon divergence between two logit distributions."""
    p = _probabilities(p_logits)
    q = _probabilities(q_logits)
    m = 0.5 * (p + q)
    return 0.5 * (
        F.kl_div(m.log(), p, reduction="none").sum(dim=1)
        + F.kl_div(m.log(), q, reduction="none").sum(dim=1)
    ).mean()


def prediction_agreement(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    p = p_logits.argmax(dim=1)
    q = q_logits.argmax(dim=1)
    return (p == q).float().mean()


def cosine_similarity(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    p = _flatten_logits(p_logits)
    q = _flatten_logits(q_logits)
    return F.cosine_similarity(p, q, dim=1).mean()


def probability_mae(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    p = _probabilities(p_logits)
    q = _probabilities(q_logits)
    return (p - q).abs().mean()


def mean_confidence(logits: torch.Tensor) -> torch.Tensor:
    return _probabilities(logits).max(dim=1).values.mean()


def compare_logits(unlearned_logits: torch.Tensor, reference_logits: torch.Tensor) -> dict:
    """Return interpretable distances between an unlearned model and a reference."""
    return {
        "prediction_agreement": float(
            prediction_agreement(unlearned_logits, reference_logits).item()
        ),
        "js_divergence": float(
            js_divergence(unlearned_logits, reference_logits).item()
        ),
        "probability_mae": float(
            probability_mae(unlearned_logits, reference_logits).item()
        ),
        "logit_cosine_similarity": float(
            cosine_similarity(unlearned_logits, reference_logits).item()
        ),
        "unlearned_mean_confidence": float(mean_confidence(unlearned_logits).item()),
        "reference_mean_confidence": float(mean_confidence(reference_logits).item()),
    }
