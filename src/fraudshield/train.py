"""Training pipeline: features -> time split -> fit -> calibrate -> cost-optimal threshold -> register."""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from . import __version__
from .config import CostModel, n_threads
from .drift import build_reference_profile
from .features import FEATURES, build_features
from .metrics import evaluate_scores, select_threshold
from .model import FraudModel, PlattCalibrator, make_estimator, typical_legit_profile
from .registry import ModelRegistry

log = logging.getLogger(__name__)
CANDIDATES = ("logreg", "hgb")


@dataclass
class Split:
    df: pd.DataFrame
    X: np.ndarray
    y: np.ndarray
    amount: np.ndarray

    def subset(self, mask: np.ndarray) -> Split:
        return Split(self.df[mask].reset_index(drop=True), self.X[mask], self.y[mask], self.amount[mask])


@dataclass
class TrainingResult:
    models: dict[str, FraudModel]
    metrics: dict[str, dict]
    best: str
    splits: dict[str, Split]
    reference_profile: dict
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def best_model(self) -> FraudModel:
        return self.models[self.best]


def time_split(df: pd.DataFrame, fractions=(0.6, 0.2)) -> dict[str, np.ndarray]:
    """Chronological split on the timestamp: past -> train, next -> valid (calibration/threshold), last -> test."""
    ts = pd.to_datetime(df["timestamp"])
    q_train = ts.quantile(fractions[0])
    q_valid = ts.quantile(fractions[0] + fractions[1])
    train = (ts <= q_train).to_numpy()
    valid = ((ts > q_train) & (ts <= q_valid)).to_numpy()
    test = (ts > q_valid).to_numpy()
    return {"train": train, "valid": valid, "test": test}


def prepare(df: pd.DataFrame) -> Split:
    df = df.sort_values(["timestamp", "txn_id"], kind="mergesort").reset_index(drop=True)
    feats = build_features(df)
    return Split(df, feats[FEATURES].to_numpy(dtype=float), df["is_fraud"].to_numpy(int), df["amount"].to_numpy(float))


def per_type_recall(split: Split, p: np.ndarray, threshold: float) -> dict[str, dict]:
    out = {}
    alert = p >= threshold
    ftype = split.df["fraud_type"].to_numpy()
    for t in sorted(set(ftype) - {""}):
        m = ftype == t
        out[t] = {"n": int(m.sum()), "recall": float(alert[m].mean())}
    return out


def fit_candidate(name: str, train: Split, valid: Split, cost: CostModel, seed: int = 42) -> FraudModel:
    est = make_estimator(name, seed)
    est.fit(train.X, train.y)
    model = FraudModel(name=name, estimator=est, calibrator=PlattCalibrator())
    model.calibrator.fit(model.raw_proba(valid.X), valid.y)
    p_valid = model.predict_proba(valid.X)
    model.threshold, _ = select_threshold(valid.y, p_valid, valid.amount, cost)
    model.baseline = typical_legit_profile(train.X, train.y)
    return model


def run_training(
    df: pd.DataFrame, cost: CostModel | None = None, seed: int = 42, candidates=CANDIDATES
) -> TrainingResult:
    with threadpool_limits(limits=n_threads()):
        return _run_training(df, cost or CostModel(), seed, candidates)


def _run_training(df: pd.DataFrame, cost: CostModel, seed: int, candidates) -> TrainingResult:
    t0 = time.perf_counter()
    data = prepare(df)
    masks = time_split(data.df)
    splits = {k: data.subset(m) for k, m in masks.items()}
    timings = {"features_s": time.perf_counter() - t0}
    models, metrics = {}, {}
    for name in candidates:
        t1 = time.perf_counter()
        model = fit_candidate(name, splits["train"], splits["valid"], cost, seed)
        timings[f"fit_{name}_s"] = time.perf_counter() - t1
        models[name] = model
        metrics[name] = {}
        for part in ("valid", "test"):
            s = splits[part]
            p = model.predict_proba(s.X)
            metrics[name][part] = evaluate_scores(s.y, p, s.amount, model.threshold, cost)
        metrics[name]["test_raw_brier"] = float(np.mean((model.raw_proba(splits["test"].X) - splits["test"].y) ** 2))
        metrics[name]["test_per_type_recall"] = per_type_recall(
            splits["test"], model.predict_proba(splits["test"].X), model.threshold
        )
        log.info("%s: valid PR-AUC %.3f", name, metrics[name]["valid"]["pr_auc"])
    # model selection on the validation split only; the test split is reported, never used to choose
    best = max(candidates, key=lambda n: metrics[n]["valid"]["expected_savings"])
    # drift reference = the most recent pre-deployment window (validation): card histories are mature
    # there, so the profile reflects steady-state serving traffic rather than the cold-start months.
    ref = splits["valid"]
    ref_scores = models[best].predict_proba(ref.X)
    profile = build_reference_profile(pd.DataFrame(ref.X, columns=FEATURES), ref_scores)
    profile["alert_rate"] = float((ref_scores >= models[best].threshold).mean())
    timings["total_s"] = time.perf_counter() - t0
    return TrainingResult(models, metrics, best, splits, profile, timings)


def data_fingerprint(df: pd.DataFrame) -> str:
    h = hashlib.sha256(pd.util.hash_pandas_object(df[["txn_id", "amount", "is_fraud"]], index=False).values)
    return h.hexdigest()[:16]


def train_and_register(
    df: pd.DataFrame, registry_dir, cost: CostModel | None = None, seed: int = 42
) -> tuple[str, TrainingResult]:
    cost = cost or CostModel()
    result = run_training(df, cost, seed)
    best = result.best_model
    splits = result.splits
    metadata = {
        "model_name": best.name,
        "package_version": __version__,
        "sklearn_version": sklearn.__version__,
        "threshold": best.threshold,
        "calibration": {"method": "platt", "a": best.calibrator.a, "b": best.calibrator.b},
        "features": FEATURES,
        "cost_model": cost.__dict__,
        "data": {
            "fingerprint": data_fingerprint(df),
            "rows": int(len(df)),
            "fraud_rate": float(df["is_fraud"].mean()),
            "train_period": [str(splits["train"].df["timestamp"].min()), str(splits["train"].df["timestamp"].max())],
            "valid_period": [str(splits["valid"].df["timestamp"].min()), str(splits["valid"].df["timestamp"].max())],
            "test_period": [str(splits["test"].df["timestamp"].min()), str(splits["test"].df["timestamp"].max())],
        },
        "selection": {"criterion": "max expected savings on validation split", "candidates": list(result.models)},
        "metrics": result.metrics,
        "timings": result.timings,
        "seed": seed,
    }
    version = ModelRegistry(registry_dir).register(best, metadata, result.reference_profile)
    return version, result
