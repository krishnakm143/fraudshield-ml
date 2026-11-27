from __future__ import annotations

import pytest

from fraudshield.data import generate_next_month, generate_transactions

N_CARDS = 1200


@pytest.fixture(scope="session")
def small_df():
    return generate_transactions(n_cards=N_CARDS, days=60, seed=7)


@pytest.fixture(scope="session")
def next_month_df():
    return generate_next_month(n_cards=N_CARDS, seed=7)


@pytest.fixture(scope="session")
def trained(tmp_path_factory, small_df):
    """Train once into a temporary registry; shared by model/API tests."""
    from fraudshield.train import train_and_register

    root = tmp_path_factory.mktemp("fs")
    version, result = train_and_register(small_df, root / "registry")
    return root, version, result


@pytest.fixture(scope="session")
def client(trained, small_df, next_month_df, tmp_path_factory):
    root, _, _ = trained
    data = root / "transactions.csv.gz"
    nm = root / "next_month.csv.gz"
    small_df.to_csv(data, index=False)
    next_month_df.to_csv(nm, index=False)
    mp = pytest.MonkeyPatch()
    mp.setenv("FRAUDSHIELD_REGISTRY", str(root / "registry"))
    mp.setenv("FRAUDSHIELD_DATA", str(data))
    mp.setenv("FRAUDSHIELD_NEXT_MONTH", str(nm))
    mp.setenv("FRAUDSHIELD_REPORTS", str(root / "reports"))
    mp.setenv("LLM_OFFLINE", "1")
    from fastapi.testclient import TestClient

    from fraudshield import api

    api._state["service"] = None
    with TestClient(api.app) as c:
        yield c
    api._state["service"] = None
    mp.undo()
