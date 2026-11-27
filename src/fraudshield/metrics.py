"""Fraud-specific evaluation metrics and cost-based threshold selection."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve

from .config import CostModel


def recall_at_fpr(y: np.ndarray, p: np.ndarray, max_fpr: float = 0.01) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    ok = fpr <= max_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0


def decision_cost(y: np.ndarray, p: np.ndarray, amount: np.ndarray, threshold: float, cost: CostModel) -> float:
    alert = p >= threshold
    missed = (y == 1) & ~alert
    false_alert = (y == 0) & alert
    return float(
        (amount[missed] + cost.chargeback_fee).sum()
        + alert.sum() * cost.review_cost
        + false_alert.sum() * cost.friction_cost
    )


def no_model_cost(y: np.ndarray, amount: np.ndarray, cost: CostModel) -> float:
    """Cost when every transaction is approved (all fraud becomes chargebacks)."""
    return float((amount[y == 1] + cost.chargeback_fee).sum())


def select_threshold(y: np.ndarray, p: np.ndarray, amount: np.ndarray, cost: CostModel) -> tuple[float, float]:
    """Return (threshold, cost) minimising expected business cost over a dense candidate grid."""
    grid = np.unique(np.concatenate([np.linspace(0.005, 0.995, 199), np.quantile(p, np.linspace(0.9, 0.9995, 200))]))
    costs = np.array([decision_cost(y, p, amount, t, cost) for t in grid])
    best = int(np.argmin(costs))
    return float(grid[best]), float(costs[best])


def evaluate_scores(
    y: np.ndarray, p: np.ndarray, amount: np.ndarray, threshold: float, cost: CostModel
) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    alert = p >= threshold
    tp = int(((y == 1) & alert).sum())
    fp = int(((y == 0) & alert).sum())
    fn = int(((y == 1) & ~alert).sum())
    tn = int(((y == 0) & ~alert).sum())
    baseline = no_model_cost(y, amount, cost)
    model_cost = decision_cost(y, p, amount, threshold, cost)
    fraud_amount = float(amount[y == 1].sum())
    caught_amount = float(amount[(y == 1) & alert].sum())
    return {
        "n": int(len(y)),
        "fraud_rate": float(y.mean()),
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "recall_at_1pct_fpr": recall_at_fpr(y, p, 0.01),
        "brier": float(brier_score_loss(y, p)),
        "threshold": float(threshold),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "alert_rate": float(alert.mean()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "fraud_amount_caught_pct": caught_amount / max(fraud_amount, 1e-9),
        "cost_no_model": baseline,
        "cost_with_model": model_cost,
        "expected_savings": baseline - model_cost,
        "savings_pct": (baseline - model_cost) / max(baseline, 1e-9),
    }
