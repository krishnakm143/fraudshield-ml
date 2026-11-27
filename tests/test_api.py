import time

import numpy as np

TXN = {
    "card_id": "C00042",
    "timestamp": "2026-03-01T02:14:00",
    "amount": 1899.0,
    "merchant_category": "electronics",
    "channel": "online",
    "lat": 44.43,
    "lon": 26.10,
    "country": "RO",
    "device_id": "dev_unknown_1",
}


def test_health_and_model(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["model_version"] == "v0001"
    m = client.get("/model").json()
    assert m["model_name"] in {"hgb", "logreg"}
    assert 0 < m["threshold"] < 1
    assert set(m["candidates"]) == {"hgb", "logreg"}
    assert m["feature_store_cards"] > 100


def test_score_with_explanation_and_summary(client):
    r = client.post("/score", params={"summary": True, "commit": False}, json=TXN)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["fraud_probability"] <= 1
    assert body["decision"] in {"approve", "review"}
    assert len(body["explanation"]) == 5
    s = body["analyst_summary"]
    assert s["risk_level"] in {"low", "medium", "high", "critical"}
    assert body["summary_source"] == "offline-heuristic"


def test_suspicious_scores_higher_than_routine(client):
    routine = {
        **TXN,
        "amount": 23.5,
        "merchant_category": "grocery",
        "channel": "pos",
        "timestamp": "2026-03-01T13:05:00",
        "device_id": "card_present",
        "country": "US",
    }
    p_bad = client.post("/score", params={"commit": False}, json=TXN).json()["fraud_probability"]
    p_ok = client.post("/score", params={"commit": False}, json=routine).json()["fraud_probability"]
    assert p_bad > p_ok


def test_card_testing_burst_raises_risk_via_online_state(client):
    """Velocity features update in real time: the 5th tiny charge in minutes looks worse than the 1st."""
    probes = [
        {
            **TXN,
            "card_id": "C00077",
            "amount": 1.0 + i,
            "merchant_category": "digital_goods",
            "timestamp": f"2026-03-02T10:{i * 3:02d}:00",
            "device_id": "dev_probe",
        }
        for i in range(5)
    ]
    out = client.post("/score/batch", json={"transactions": probes}).json()
    assert out["n"] == 5
    probs = [r["fraud_probability"] for r in out["results"]]
    assert probs[-1] > probs[0]


def test_validation_errors(client):
    assert client.post("/score", json={**TXN, "amount": -5}).status_code == 422
    assert client.post("/score", json={**TXN, "merchant_category": "casino"}).status_code == 422
    assert client.post("/score", json={**TXN, "unexpected": 1}).status_code == 422


def test_score_latency_under_20ms(client):
    for _ in range(5):
        client.post("/score", params={"commit": False}, json=TXN)
    lat = []
    for _ in range(50):
        t0 = time.perf_counter()
        r = client.post("/score", params={"commit": False}, json=TXN)
        lat.append((time.perf_counter() - t0) * 1000)
        assert r.status_code == 200
    server_side = r.json()["latency_ms"]
    assert server_side < 20
    assert np.median(lat) < 20  # end-to-end through the ASGI stack


def test_drift_endpoint_flags_next_month(client):
    r = client.get("/drift")
    assert r.status_code == 200
    rep = r.json()
    assert rep["current_n"] > 0 and rep["features"]
    assert rep["n_significant"] + rep["n_moderate"] >= 1
    assert rep["features"][0]["psi"] >= rep["features"][-1]["psi"] or rep["features"][-1]["status"] == "structural"
    assert "performance" in rep and rep["recommended_action"]
