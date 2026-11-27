"""Drift monitoring job: score a new period with the registered model and compare to its training reference."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .config import CostModel, n_threads
from .drift import drift_report
from .features import FEATURES, build_features
from .metrics import evaluate_scores
from .model import FraudModel


def current_features(history: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time features for ``current`` using ``history`` as prior card context (no look-ahead)."""
    both = pd.concat([history, current], ignore_index=True)
    feats = build_features(both)
    return feats.iloc[len(history) :].reset_index(drop=True)


def run_drift(
    model: FraudModel,
    metadata: dict,
    profile: dict,
    history: pd.DataFrame,
    current: pd.DataFrame,
    cost: CostModel | None = None,
) -> dict:
    cost = cost or CostModel(**metadata.get("cost_model", {}))
    current = current.reset_index(drop=True)
    feats = current_features(history, current)
    X = feats[FEATURES].to_numpy(dtype=float)
    with threadpool_limits(limits=n_threads()):
        scores = model.predict_proba(X)
    report = drift_report(profile, feats, scores, model.threshold, profile.get("alert_rate"))
    report["generated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    report["model_version"] = metadata.get("version")
    report["current_period"] = [str(current["timestamp"].min()), str(current["timestamp"].max())]
    if "is_fraud" in current and current["is_fraud"].nunique() > 1:
        y = current["is_fraud"].to_numpy(int)
        perf = evaluate_scores(y, scores, current["amount"].to_numpy(float), model.threshold, cost)
        ref = metadata.get("metrics", {}).get(metadata.get("model_name", ""), {}).get("test", {})
        keys = ("fraud_rate", "pr_auc", "roc_auc", "recall_at_1pct_fpr", "precision", "recall", "savings_pct")
        report["performance"] = {
            "reference_test": {k: ref.get(k) for k in keys},
            "current": {k: perf[k] for k in keys},
        }
        ftype = current["fraud_type"].to_numpy()
        alert = scores >= model.threshold
        report["performance"]["current_per_type_recall"] = {
            t: round(float(alert[ftype == t].mean()), 4) for t in sorted(set(ftype) - {""})
        }
    report["top_drifted_features"] = [r["feature"] for r in report["features"][:5]]
    report["score_quantiles_current"] = np.quantile(scores, [0.5, 0.9, 0.99]).round(5).tolist()
    return report
