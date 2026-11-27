"""Point-in-time correct feature engineering.

Every behavioural feature for a transaction is computed from that card's *strictly earlier*
transactions only (processing order = timestamp, then txn_id). Two implementations share one
definition:

* :func:`build_features` - vectorised offline path used for training / batch evaluation.
* :class:`CardStateStore` - incremental online path used by the API (an in-process feature store).

``tests/test_features.py`` replays a log through the online store and asserts both produce the
same matrix, which is the guard against training/serving skew and label leakage.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import CATEGORIES

HOUR = 3600
DAY = 86400
SMALL_AMOUNT = 5.0
NO_HISTORY_GAP = 30 * DAY

NUMERIC_FEATURES: list[str] = [
    "amount_log",
    "hour_sin",
    "hour_cos",
    "is_night",
    "is_online",
    "secs_since_prev_log",
    "dist_prev_km_log",
    "speed_prev_kmh_log",
    "dist_home_km_log",
    "txn_count_1h",
    "txn_count_24h",
    "amount_sum_24h_log",
    "small_txn_count_1h",
    "amount_ratio_mean_log",
    "amount_zscore",
    "is_new_category",
    "is_new_device",
    "is_foreign",
    "history_count_log",
]
CATEGORY_FEATURES: list[str] = [f"cat_{c}" for c in CATEGORIES]
FEATURES: list[str] = NUMERIC_FEATURES + CATEGORY_FEATURES

# Feature groups used for explanations (one-hot columns are explained together).
FEATURE_GROUPS: dict[str, list[str]] = {f: [f] for f in NUMERIC_FEATURES}
FEATURE_GROUPS["hour_of_day"] = ["hour_sin", "hour_cos"]
del FEATURE_GROUPS["hour_sin"], FEATURE_GROUPS["hour_cos"]
FEATURE_GROUPS["merchant_category"] = CATEGORY_FEATURES

_EARTH_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype=float)) for x in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * _EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _epoch_seconds(ts: pd.Series) -> np.ndarray:
    return pd.to_datetime(ts).astype("datetime64[s]").astype("int64").to_numpy()


def _finish(
    amount,
    hour,
    online,
    gap,
    dist_prev,
    dist_home,
    c1h,
    c24h,
    s24h,
    small1h,
    n,
    mean_amt,
    mean_la,
    std_la,
    new_cat,
    new_dev,
    foreign,
    category,
) -> dict[str, np.ndarray]:
    """Shared final transformation so offline and online paths cannot diverge."""
    amount = np.asarray(amount, dtype=float)
    la = np.log1p(amount)
    speed = dist_prev / (np.maximum(gap, 60.0) / HOUR)
    ratio = np.where(n > 0, amount / np.maximum(mean_amt, 0.01), 1.0)
    z = np.where(n > 1, (la - mean_la) / (std_la + 0.25), 0.0)
    out = {
        "amount_log": la,
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "is_night": ((hour >= 0) & (hour < 5)).astype(float),
        "is_online": np.asarray(online, dtype=float),
        "secs_since_prev_log": np.log1p(gap),
        "dist_prev_km_log": np.log1p(dist_prev),
        "speed_prev_kmh_log": np.log1p(np.minimum(speed, 20000.0)),
        "dist_home_km_log": np.log1p(dist_home),
        "txn_count_1h": np.asarray(c1h, dtype=float),
        "txn_count_24h": np.asarray(c24h, dtype=float),
        "amount_sum_24h_log": np.log1p(s24h),
        "small_txn_count_1h": np.asarray(small1h, dtype=float),
        "amount_ratio_mean_log": np.log(np.clip(ratio, 1e-3, 1e4)),
        "amount_zscore": np.clip(z, -10, 10),
        "is_new_category": np.asarray(new_cat, dtype=float),
        "is_new_device": np.asarray(new_dev, dtype=float),
        "is_foreign": np.asarray(foreign, dtype=float),
        "history_count_log": np.log1p(n),
    }
    category = np.asarray(category)
    for c in CATEGORIES:
        out[f"cat_{c}"] = (category == c).astype(float)
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised point-in-time features. Returns a frame aligned to ``df.index`` with ``FEATURES`` columns."""
    order = np.lexsort((df["txn_id"].to_numpy(), _epoch_seconds(df["timestamp"]), df["card_id"].to_numpy()))
    d = df.iloc[order]
    t = _epoch_seconds(d["timestamp"])
    t_rel = t - t.min()
    card_code, _ = pd.factorize(d["card_id"])
    card_code = card_code.astype(np.int64)
    m = len(d)
    idx = np.arange(m)
    big = np.int64(t_rel.max() + 10 * DAY)
    key = card_code * big + t_rel
    first_idx = np.searchsorted(card_code, card_code, side="left")  # start of this card's block
    n = idx - first_idx  # number of prior transactions
    has_prev = n > 0

    amount = d["amount"].to_numpy(dtype=float)
    lat = d["lat"].to_numpy(dtype=float)
    lon = d["lon"].to_numpy(dtype=float)
    la = np.log1p(amount)

    prev = np.maximum(idx - 1, 0)
    gap = np.where(has_prev, t - t[prev], NO_HISTORY_GAP).astype(float)
    dist_prev = np.where(has_prev, haversine_km(lat, lon, lat[prev], lon[prev]), 0.0)

    def prior_sum(x: np.ndarray) -> np.ndarray:
        c0 = np.concatenate([[0.0], np.cumsum(x)])
        return c0[idx] - c0[first_idx]

    nz = np.maximum(n, 1)
    mean_amt = prior_sum(amount) / nz
    mean_la = prior_sum(la) / nz
    var_la = np.maximum(prior_sum(la * la) / nz - mean_la**2, 0.0)
    std_la = np.sqrt(var_la)
    home_lat = prior_sum(lat) / nz
    home_lon = prior_sum(lon) / nz
    dist_home = np.where(has_prev, haversine_km(lat, lon, home_lat, home_lon), 0.0)

    def window(seconds: int, weights: np.ndarray | None = None) -> np.ndarray:
        j = np.searchsorted(key, key - seconds, side="left")
        j = np.maximum(j, first_idx)
        if weights is None:
            return (idx - j).astype(float)
        c0 = np.concatenate([[0.0], np.cumsum(weights)])
        return c0[idx] - c0[j]

    c1h = window(HOUR)
    c24h = window(DAY)
    s24h = window(DAY, amount)
    small1h = window(HOUR, (amount < SMALL_AMOUNT).astype(float))

    new_cat = (~d.duplicated(["card_id", "merchant_category"])).to_numpy()
    online = (d["channel"] == "online").to_numpy()
    new_dev = np.zeros(m, dtype=bool)
    on = d[online]
    new_dev[online] = (~on.duplicated(["card_id", "device_id"])).to_numpy()
    home_country = d.groupby("card_id", sort=False)["country"].transform("first").to_numpy()
    foreign = d["country"].to_numpy() != home_country

    ts = pd.to_datetime(d["timestamp"])
    hour = (ts.dt.hour + ts.dt.minute / 60.0).to_numpy()
    feats = _finish(
        amount,
        hour,
        online,
        gap,
        dist_prev,
        dist_home,
        c1h,
        c24h,
        s24h,
        small1h,
        n,
        mean_amt,
        mean_la,
        std_la,
        new_cat,
        new_dev,
        foreign,
        d["merchant_category"].to_numpy(),
    )
    out = pd.DataFrame(feats, index=d.index)[FEATURES]
    return out.loc[df.index]


@dataclass
class CardState:
    n: int = 0
    sum_amt: float = 0.0
    sum_la: float = 0.0
    sumsq_la: float = 0.0
    sum_lat: float = 0.0
    sum_lon: float = 0.0
    prev_t: int | None = None
    prev_lat: float = 0.0
    prev_lon: float = 0.0
    home_country: str | None = None
    window: deque = field(default_factory=deque)  # (t, amount) within the last 24h
    categories: set = field(default_factory=set)
    devices: set = field(default_factory=set)


class CardStateStore:
    """Incremental per-card state: an in-process stand-in for an online feature store (e.g. Redis)."""

    def __init__(self) -> None:
        self.cards: dict[str, CardState] = {}

    def __len__(self) -> int:
        return len(self.cards)

    @staticmethod
    def _t(ts) -> int:
        return int(pd.Timestamp(ts).value // 1_000_000_000)

    def features(self, txn: dict) -> dict[str, float]:
        s = self.cards.get(txn["card_id"]) or CardState()
        t = self._t(txn["timestamp"])
        amount = float(txn["amount"])
        lat, lon = float(txn["lat"]), float(txn["lon"])
        online = txn["channel"] == "online"
        if s.n:
            gap = float(t - s.prev_t)
            dist_prev = float(haversine_km(lat, lon, s.prev_lat, s.prev_lon))
            dist_home = float(haversine_km(lat, lon, s.sum_lat / s.n, s.sum_lon / s.n))
            mean_amt, mean_la = s.sum_amt / s.n, s.sum_la / s.n
            std_la = math.sqrt(max(s.sumsq_la / s.n - mean_la**2, 0.0))
        else:
            gap, dist_prev, dist_home, mean_amt, mean_la, std_la = float(NO_HISTORY_GAP), 0.0, 0.0, 0.0, 0.0, 0.0
        c1h = c24h = small1h = 0
        s24h = 0.0
        for wt, wa in s.window:
            if wt >= t - DAY:
                c24h += 1
                s24h += wa
                if wt >= t - HOUR:
                    c1h += 1
                    small1h += wa < SMALL_AMOUNT
        ts = pd.Timestamp(txn["timestamp"])
        feats = _finish(
            np.array([amount]),
            np.array([ts.hour + ts.minute / 60.0]),
            np.array([online]),
            np.array([gap]),
            np.array([dist_prev]),
            np.array([dist_home]),
            np.array([c1h]),
            np.array([c24h]),
            np.array([s24h]),
            np.array([small1h]),
            np.array([s.n]),
            np.array([mean_amt]),
            np.array([mean_la]),
            np.array([std_la]),
            np.array([txn["merchant_category"] not in s.categories]),
            np.array([online and txn["device_id"] not in s.devices]),
            np.array([s.home_country is not None and txn["country"] != s.home_country]),
            np.array([txn["merchant_category"]]),
        )
        return {k: float(feats[k][0]) for k in FEATURES}

    def update(self, txn: dict) -> None:
        s = self.cards.setdefault(txn["card_id"], CardState())
        t = self._t(txn["timestamp"])
        amount = float(txn["amount"])
        la = math.log1p(amount)
        s.n += 1
        s.sum_amt += amount
        s.sum_la += la
        s.sumsq_la += la * la
        s.sum_lat += float(txn["lat"])
        s.sum_lon += float(txn["lon"])
        s.prev_t, s.prev_lat, s.prev_lon = t, float(txn["lat"]), float(txn["lon"])
        if s.home_country is None:
            s.home_country = txn["country"]
        s.window.append((t, amount))
        while s.window and s.window[0][0] < t - DAY:
            s.window.popleft()
        s.categories.add(txn["merchant_category"])
        if txn["channel"] == "online":
            s.devices.add(txn["device_id"])

    def score_and_update(self, txn: dict, commit: bool = True) -> dict[str, float]:
        f = self.features(txn)
        if commit:
            self.update(txn)
        return f

    @classmethod
    def from_history(cls, df: pd.DataFrame) -> CardStateStore:
        store = cls()
        d = df.sort_values(["timestamp", "txn_id"], kind="mergesort")
        cols = ["card_id", "timestamp", "amount", "lat", "lon", "country", "merchant_category", "channel", "device_id"]
        for row in d[cols].itertuples(index=False):
            store.update(row._asdict())
        return store
