"""Population Stability Index (PSI) drift monitoring for features and model scores."""

from __future__ import annotations

import numpy as np
import pandas as pd

PSI_STABLE = 0.10
PSI_SIGNIFICANT = 0.25
_EPS = 1e-4
_MAX_CATEGORIES = 12
# Features that drift by construction and would drown real signal: card history length grows
# monotonically with calendar time, so it is reported but excluded from the retrain decision.
STRUCTURAL = {"history_count_log": "grows monotonically with time (cards accumulate history)"}


def _status(psi: float) -> str:
    if psi < PSI_STABLE:
        return "stable"
    if psi < PSI_SIGNIFICANT:
        return "moderate"
    return "significant"


def profile_column(values: np.ndarray, n_bins: int = 10) -> dict:
    """Binning spec + reference proportions. Low-cardinality columns are binned by value."""
    values = np.asarray(values, dtype=float)
    uniq = np.unique(values)
    if len(uniq) <= _MAX_CATEGORIES:
        props = [float((values == u).mean()) for u in uniq]
        return {"kind": "categorical", "values": uniq.tolist(), "props": props}
    edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)[1:-1]))
    counts = np.bincount(np.searchsorted(edges, values, side="right"), minlength=len(edges) + 1)
    return {"kind": "numeric", "edges": edges.tolist(), "props": (counts / counts.sum()).tolist()}


def _current_props(spec: dict, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if spec["kind"] == "categorical":
        ref_vals = np.asarray(spec["values"])
        props = [(values == u).mean() for u in ref_vals]
        other = 1.0 - float(np.sum(props))  # mass on values never seen in reference
        return np.asarray([*props, max(other, 0.0)])
    edges = np.asarray(spec["edges"])
    counts = np.bincount(np.searchsorted(edges, values, side="right"), minlength=len(edges) + 1)
    return counts / max(counts.sum(), 1)


def psi(spec: dict, values: np.ndarray) -> float:
    ref = np.asarray(spec["props"], dtype=float)
    cur = _current_props(spec, values)
    if spec["kind"] == "categorical":
        ref = np.append(ref, 0.0)
    ref = np.clip(ref, _EPS, None)
    cur = np.clip(cur, _EPS, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def build_reference_profile(features: pd.DataFrame, scores: np.ndarray) -> dict:
    return {
        "n": int(len(features)),
        "features": {c: profile_column(features[c].to_numpy()) for c in features.columns},
        "score": profile_column(scores),
        "score_mean": float(np.mean(scores)),
    }


def drift_report(
    profile: dict,
    features: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    reference_alert_rate: float | None = None,
) -> dict:
    rows = []
    for name, spec in profile["features"].items():
        if name not in features:
            continue
        value = psi(spec, features[name].to_numpy())
        row = {"feature": name, "psi": round(value, 4), "status": _status(value)}
        if name in STRUCTURAL:
            row["status"] = "structural"
            row["note"] = STRUCTURAL[name]
        rows.append(row)
    rows.sort(key=lambda r: (r["status"] == "structural", -r["psi"]))
    score_psi = psi(profile["score"], scores)
    n_sig = sum(r["status"] == "significant" for r in rows)
    n_mod = sum(r["status"] == "moderate" for r in rows)
    if n_sig or score_psi >= PSI_SIGNIFICANT:
        action = "retrain: significant drift in features or score distribution"
    elif n_mod or score_psi >= PSI_STABLE:
        action = "investigate: moderate drift; monitor closely and schedule retraining"
    else:
        action = "none: distributions are stable"
    return {
        "reference_n": profile["n"],
        "current_n": int(len(features)),
        "score_psi": round(score_psi, 4),
        "score_status": _status(score_psi),
        "score_mean_reference": round(profile["score_mean"], 5),
        "score_mean_current": round(float(np.mean(scores)), 5),
        "alert_rate_reference": None if reference_alert_rate is None else round(reference_alert_rate, 5),
        "alert_rate_current": round(float((scores >= threshold).mean()), 5),
        "n_significant": n_sig,
        "n_moderate": n_mod,
        "recommended_action": action,
        "features": rows,
    }
