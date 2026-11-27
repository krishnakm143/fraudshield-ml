import numpy as np
import pandas as pd

from fraudshield.features import FEATURES, CardStateStore, build_features


def test_offline_online_parity(small_df):
    """Training features (vectorised) == serving features (incremental store): no train/serve skew."""
    df = small_df[small_df.card_id.isin(small_df.card_id.unique()[:150])]
    offline = build_features(df).set_index(df["txn_id"])
    store = CardStateStore()
    rows = {}
    for txn in df.sort_values(["timestamp", "txn_id"]).to_dict("records"):
        rows[txn["txn_id"]] = store.score_and_update(txn)
    online = pd.DataFrame.from_dict(rows, orient="index")[FEATURES].loc[offline.index]
    np.testing.assert_allclose(online.to_numpy(), offline[FEATURES].to_numpy(), atol=1e-6)


def test_features_are_point_in_time(small_df):
    """Appending future transactions must not change any earlier transaction's features."""
    cut = small_df["timestamp"].quantile(0.5)
    past = small_df[small_df.timestamp <= cut]
    f_past = build_features(past)
    f_full = build_features(small_df).loc[past.index]
    pd.testing.assert_frame_equal(f_past, f_full)


def test_velocity_features_hand_computed():
    base = pd.Timestamp("2026-01-10 12:00:00")
    rows = [
        # amount, minutes offset, lat, lon
        (50.0, 0, 40.71, -74.01),
        (1.0, 10, 40.71, -74.01),
        (2.0, 20, 40.71, -74.01),
        (900.0, 30, 51.51, -0.13),  # London 30 minutes after New York
    ]
    df = pd.DataFrame(
        {
            "txn_id": [f"t{i}" for i in range(4)],
            "timestamp": [base + pd.Timedelta(minutes=m) for _, m, _, _ in rows],
            "card_id": "C1",
            "amount": [a for a, *_ in rows],
            "merchant_category": ["grocery", "digital_goods", "digital_goods", "jewelry"],
            "channel": ["pos", "online", "online", "pos"],
            "lat": [r[2] for r in rows],
            "lon": [r[3] for r in rows],
            "country": ["US", "US", "US", "GB"],
            "device_id": ["card_present", "d1", "d1", "card_present"],
        }
    )
    f = build_features(df)
    last = f.iloc[3]
    assert last["txn_count_1h"] == 3
    assert last["small_txn_count_1h"] == 2
    assert np.expm1(last["speed_prev_kmh_log"]) > 5000  # impossible travel
    assert last["is_foreign"] == 1 and last["is_new_category"] == 1
    assert f.iloc[1]["is_new_device"] == 1 and f.iloc[2]["is_new_device"] == 0
    assert f.iloc[0]["history_count_log"] == 0
