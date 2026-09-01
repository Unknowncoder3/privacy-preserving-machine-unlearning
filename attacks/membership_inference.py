"""Simple black-box membership inference evaluator.

This module uses model confidence as the attack score. It compares member
samples against held-out non-members and reports ROC-AUC when both groups are
available. This is an evaluation baseline, not a formal privacy guarantee.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score


def confidence_scores(model, loader, device):
    model.eval()
    scores = []
    with torch.no_grad():
        for x, _ in loader:
            p = torch.softmax(model(x.to(device)), dim=1)
            scores.extend(p.max(dim=1).values.cpu().numpy().tolist())
    return np.asarray(scores, dtype=np.float64)


def evaluate_confidence_attack(member_scores, nonmember_scores):
    y = np.concatenate([np.ones(len(member_scores)), np.zeros(len(nonmember_scores))])
    scores = np.concatenate([member_scores, nonmember_scores])
    auc = float(roc_auc_score(y, scores))
    threshold = float(np.median(scores))
    pred = (scores >= threshold).astype(int)
    acc = float(accuracy_score(y, pred))
    return {"auc": auc, "accuracy_at_median_threshold": acc, "threshold": threshold}
