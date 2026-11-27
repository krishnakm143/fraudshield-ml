import numpy as np

from fraudshield.config import CostModel
from fraudshield.drift import profile_column, psi
from fraudshield.metrics import decision_cost, recall_at_fpr, select_threshold
from fraudshield.registry import ModelRegistry


def test_hgb_beats_baseline_and_is_useful(trained):
    _, _, result = trained
    m = result.metrics
    assert m["hgb"]["test"]["pr_auc"] > 0.6
    assert m["hgb"]["test"]["roc_auc"] > 0.95
    assert m["hgb"]["test"]["pr_auc"] >= m["logreg"]["test"]["pr_auc"] - 0.02
    for name in m:
        assert m[name]["test"]["expected_savings"] > 0


def test_calibration_improves_brier(trained):
    _, _, result = trained
    for name in result.metrics:  # class-weighted raw scores are over-confident; Platt fixes it
        assert result.metrics[name]["test"]["brier"] < result.metrics[name]["test_raw_brier"]


def test_threshold_minimises_cost():
    rng = np.random.default_rng(0)
    y = (rng.random(4000) < 0.03).astype(int)
    p = np.clip(0.6 * y + rng.normal(0.2, 0.15, 4000), 0, 1)
    amount = rng.uniform(10, 500, 4000)
    cost = CostModel()
    t, c = select_threshold(y, p, amount, cost)
    for other in (0.05, 0.3, 0.6, 0.9):
        assert c <= decision_cost(y, p, amount, other, cost) + 1e-9
    assert recall_at_fpr(y, p, 0.01) > 0.3


def test_psi_detects_shift_and_ignores_noise():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, 20000)
    spec = profile_column(ref)
    assert psi(spec, rng.normal(0, 1, 5000)) < 0.02
    assert psi(spec, rng.normal(0.8, 1, 5000)) > 0.25
    binary = profile_column((rng.random(5000) < 0.2).astype(float))
    assert binary["kind"] == "categorical"
    assert psi(binary, (rng.random(5000) < 0.5).astype(float)) > 0.1


def test_registry_versioning_roundtrip(trained, tmp_path):
    _, _, result = trained
    reg = ModelRegistry(tmp_path / "reg")
    v1 = reg.register(result.best_model, {"model_name": result.best})
    v2 = reg.register(result.best_model, {"model_name": result.best})
    assert (v1, v2) == ("v0001", "v0002") and reg.latest_version() == "v0002"
    model, meta = reg.load(v1)
    X = result.splits["test"].X[:50]
    np.testing.assert_allclose(model.predict_proba(X), result.best_model.predict_proba(X))
    assert meta["version"] == "v0001"


def test_contributions_explain_fraud(trained):
    _, _, result = trained
    model = result.best_model
    test = result.splits["test"]
    p = model.predict_proba(test.X)
    i = int(np.argmax(p))
    contrib = model.contributions(test.X[i])
    assert max(contrib.values()) > 0.05  # something meaningfully drives the top-scored txn
    assert contrib == model.contributions(test.X[i])  # deterministic
