import pandas as pd

from fraudshield.data import COLUMNS, FRAUD_TYPES, generate_transactions


def test_generator_is_deterministic():
    a = generate_transactions(n_cards=120, days=30, seed=3)
    b = generate_transactions(n_cards=120, days=30, seed=3)
    pd.testing.assert_frame_equal(a, b)
    c = generate_transactions(n_cards=120, days=30, seed=4)
    assert not a["amount"].equals(c["amount"])


def test_generator_schema_and_fraud_mix(small_df):
    assert list(small_df.columns) == COLUMNS
    assert small_df["txn_id"].is_unique
    assert small_df["timestamp"].is_monotonic_increasing
    rate = small_df["is_fraud"].mean()
    assert 0.01 < rate < 0.02
    assert set(small_df.loc[small_df.is_fraud == 1, "fraud_type"]) == set(FRAUD_TYPES)
    assert (small_df.loc[small_df.is_fraud == 0, "fraud_type"] == "").all()


def test_card_testing_pattern_is_bursty(small_df):
    ct = small_df[small_df.fraud_type == "card_testing"]
    assert (ct["amount"] < 10).mean() > 0.6  # mostly tiny probe charges
    assert (ct["channel"] == "online").all()
