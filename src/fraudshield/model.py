"""Model wrappers: candidate estimators, probability calibration and per-prediction contributions."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_GROUPS, FEATURES

_EPS = 1e-6


def make_estimator(name: str, seed: int = 42):
    """Candidate models. Both use class re-weighting for the ~1.5% positive rate; calibration undoes the bias."""
    if name == "logreg":
        return make_pipeline(
            StandardScaler(), LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000, random_state=seed)
        )
    if name == "hgb":
        return HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=400,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=1.0,
            class_weight="balanced",
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30,
            random_state=seed,
        )
    raise ValueError(f"unknown model {name!r}")


class PlattCalibrator:
    """Sigmoid calibration on the logit of the raw (class-weighted) score, fitted on a held-out split.

    Implemented directly (rather than CalibratedClassifierCV + FrozenEstimator) so the artefact is
    stable across scikit-learn versions and the fitted parameters are inspectable in the registry.
    """

    def __init__(self) -> None:
        self.a = 1.0
        self.b = 0.0

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, _EPS, 1 - _EPS)
        return np.log(p / (1 - p))

    def fit(self, raw: np.ndarray, y: np.ndarray) -> PlattCalibrator:
        lr = LogisticRegression(C=1e4, max_iter=1000)
        lr.fit(self._logit(raw).reshape(-1, 1), y)
        self.a, self.b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        return self

    def transform(self, raw: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(self.a * self._logit(raw) + self.b)))


@dataclass
class FraudModel:
    name: str
    estimator: object
    calibrator: PlattCalibrator
    threshold: float = 0.5
    baseline: np.ndarray = field(default_factory=lambda: np.zeros(len(FEATURES)))
    feature_names: list[str] = field(default_factory=lambda: list(FEATURES))

    def raw_proba(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba(np.asarray(X, dtype=float))[:, 1]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.calibrator.transform(self.raw_proba(X))

    def contributions(self, x: np.ndarray) -> dict[str, float]:
        """Leave-one-feature-group-out contributions for one row.

        contribution(g) = p(x) - p(x with group g reset to a typical legitimate value). Positive values
        push towards fraud. All perturbations are scored in a single vectorised predict call so the
        overhead stays at ~1 extra model call per explanation.
        """
        x = np.asarray(x, dtype=float).reshape(1, -1)
        groups = list(FEATURE_GROUPS)
        batch = np.repeat(x, len(groups) + 1, axis=0)
        col = {f: i for i, f in enumerate(self.feature_names)}
        for r, g in enumerate(groups, start=1):
            idx = [col[f] for f in FEATURE_GROUPS[g]]
            batch[r, idx] = self.baseline[idx]
        p = self.predict_proba(batch)
        return {g: float(p[0] - p[r]) for r, g in enumerate(groups, start=1)}


def typical_legit_profile(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Reference point for explanations: median legitimate transaction (mode for one-hot groups)."""
    legit = X[y == 0]
    base = np.median(legit, axis=0)
    col = {f: i for i, f in enumerate(FEATURES)}
    cat_idx = [col[f] for f in FEATURE_GROUPS["merchant_category"]]
    base[cat_idx] = 0.0
    base[cat_idx[int(np.argmax(legit[:, cat_idx].sum(axis=0)))]] = 1.0
    return base
