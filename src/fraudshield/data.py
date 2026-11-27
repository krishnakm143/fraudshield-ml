"""Deterministic synthetic card-transaction generator.

The generator models a population of cardholders with stable habits (home city, spend level,
preferred merchant categories, devices, active hours) and injects five fraud typologies that
real fraud teams see every day:

* ``card_testing``      - bursts of tiny online charges minutes apart, sometimes followed by a cash-out
* ``geo_velocity``      - card-present use in a far-away city minutes after a genuine purchase
* ``night_high_value``  - large purchases between 01:00 and 04:59 in high-resale categories
* ``new_category``      - high-value spend in a category the cardholder has never used
* ``account_takeover``  - a new device in a foreign country running a string of escalating orders

Legitimate behaviour deliberately overlaps with every typology (genuine travel, $0.99 app-store
charges, phone upgrades, the odd late-night purchase) so the problem is not trivially separable.

Everything is driven by ``numpy.random.default_rng`` with explicit seeds: the same arguments
always yield byte-identical data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CATEGORIES: list[str] = [
    "grocery",
    "restaurants",
    "fuel",
    "retail",
    "travel",
    "entertainment",
    "utilities",
    "health",
    "digital_goods",
    "electronics",
    "jewelry",
    "gift_cards",
    "crypto_exchange",
]

# Relative popularity for legitimate spend and a typical amount multiplier per category.
_CAT_POPULARITY = np.array([10, 8, 6, 7, 1.0, 4, 3, 3, 4, 0.8, 0.15, 0.3, 0.1])
_CAT_AMOUNT = np.array([1.0, 0.8, 1.0, 1.4, 6.0, 0.9, 2.2, 1.3, 0.3, 4.5, 6.0, 2.0, 5.0])
_CAT_ONLINE = np.array([0.08, 0.15, 0.0, 0.35, 0.8, 0.4, 0.9, 0.2, 1.0, 0.6, 0.3, 0.7, 1.0])

FRAUD_TYPES: list[str] = [
    "card_testing",
    "geo_velocity",
    "night_high_value",
    "new_category",
    "account_takeover",
]

# (name, lat, lon, country)
CITIES: list[tuple[str, float, float, str]] = [
    ("New York", 40.71, -74.01, "US"),
    ("Los Angeles", 34.05, -118.24, "US"),
    ("Chicago", 41.88, -87.63, "US"),
    ("Houston", 29.76, -95.37, "US"),
    ("Phoenix", 33.45, -112.07, "US"),
    ("Philadelphia", 39.95, -75.17, "US"),
    ("San Antonio", 29.42, -98.49, "US"),
    ("San Diego", 32.72, -117.16, "US"),
    ("Dallas", 32.78, -96.80, "US"),
    ("Seattle", 47.61, -122.33, "US"),
    ("Denver", 39.74, -104.99, "US"),
    ("Boston", 42.36, -71.06, "US"),
    ("Atlanta", 33.75, -84.39, "US"),
    ("Miami", 25.76, -80.19, "US"),
    ("Toronto", 43.65, -79.38, "CA"),
    ("London", 51.51, -0.13, "GB"),
    ("Paris", 48.86, 2.35, "FR"),
    ("Berlin", 52.52, 13.40, "DE"),
    ("Mumbai", 19.08, 72.88, "IN"),
    ("Singapore", 1.35, 103.82, "SG"),
    ("Lagos", 6.52, 3.38, "NG"),
    ("Sao Paulo", -23.55, -46.63, "BR"),
    ("Bucharest", 44.43, 26.10, "RO"),
    ("Manila", 14.60, 120.98, "PH"),
]
_DOMESTIC = [i for i, c in enumerate(CITIES) if c[3] == "US"]
_FOREIGN = [i for i, c in enumerate(CITIES) if c[3] != "US"]

COLUMNS = [
    "txn_id",
    "timestamp",
    "card_id",
    "amount",
    "merchant_id",
    "merchant_category",
    "channel",
    "lat",
    "lon",
    "country",
    "device_id",
    "is_fraud",
    "fraud_type",
]


@dataclass(frozen=True)
class Shift:
    """Distribution shift applied to a generated period (used for the 'next month' drift scenario)."""

    amount_scale: float = 1.0
    online_boost: float = 0.0
    night_boost: float = 0.0
    category_boost: dict[str, float] = field(default_factory=dict)  # multiplies legit category propensity
    stealth_rate: float = 0.30  # share of attacks that mimic the cardholder (known device, home location)
    fraud_mix: dict[str, float] = field(default_factory=dict)


DEFAULT_FRAUD_MIX = {
    "card_testing": 0.22,
    "geo_velocity": 0.20,
    "night_high_value": 0.18,
    "new_category": 0.18,
    "account_takeover": 0.22,
}

NEXT_MONTH_SHIFT = Shift(
    amount_scale=1.45,  # holiday season + inflation
    online_boost=0.35,  # shift to e-commerce
    night_boost=0.06,
    category_boost={"electronics": 4.0, "gift_cards": 6.0, "jewelry": 4.0, "retail": 1.8, "travel": 2.0},
    stealth_rate=0.55,  # fraudsters adapt: more attacks look like the genuine cardholder
    fraud_mix={  # fraud ring pivots to account takeover and card testing
        "card_testing": 0.30,
        "geo_velocity": 0.10,
        "night_high_value": 0.10,
        "new_category": 0.10,
        "account_takeover": 0.40,
    },
)


@dataclass
class _Card:
    card_id: str
    home: int
    mean_log_amount: float
    sigma: float
    cat_probs: np.ndarray
    online_pref: float
    night_pref: float
    devices: list[str]
    rate: float
    upgrade_day: float | None
    trip: tuple[float, float, int] | None  # (start_day, end_day, city_idx)


def _profiles(n_cards: int, days: int, seed: int) -> list[_Card]:
    rng = np.random.default_rng([seed, 0])
    cards: list[_Card] = []
    for i in range(n_cards):
        home = int(rng.choice(_DOMESTIC)) if rng.random() < 0.88 else int(rng.choice(_FOREIGN))
        n_pref = int(rng.integers(3, 7))
        pref = rng.choice(len(CATEGORIES), size=n_pref, replace=False, p=_CAT_POPULARITY / _CAT_POPULARITY.sum())
        probs = np.full(len(CATEGORIES), 0.006)
        probs[pref] = _CAT_POPULARITY[pref] * rng.uniform(0.5, 1.5, size=n_pref)
        probs /= probs.sum()
        n_dev = 1 if rng.random() < 0.85 else 2
        devices = [f"dev_{i:05d}_{k}" for k in range(n_dev)]
        trip = None
        if rng.random() < 0.10:
            start = float(rng.uniform(5, max(days - 3, 6)))
            dest = int(rng.choice([c for c in range(len(CITIES)) if c != home]))
            trip = (start, start + float(rng.uniform(2, 9)), dest)
        cards.append(
            _Card(
                card_id=f"C{i:05d}",
                home=home,
                mean_log_amount=float(rng.normal(np.log(38), 0.45)),
                sigma=float(rng.uniform(0.55, 0.95)),
                cat_probs=probs,
                online_pref=float(rng.beta(2, 5)),
                night_pref=float(rng.choice([0.01, 0.03, 0.12], p=[0.6, 0.3, 0.1])),
                devices=devices,
                rate=float(rng.gamma(3.0, 1.0 / 3.0)),
                upgrade_day=float(rng.uniform(5, days)) if rng.random() < 0.12 else None,
                trip=trip,
            )
        )
    return cards


def _jitter(rng: np.random.Generator, n: int, scale: float = 0.08) -> np.ndarray:
    return rng.normal(0, scale, size=n)


def _legit(cards: list[_Card], days: int, txn_rate: float, shift: Shift, rng: np.random.Generator) -> dict:
    cols: dict[str, list] = {k: [] for k in ("t", "card", "amount", "cat", "online", "lat", "lon", "country", "device")}
    for card in cards:
        n = int(rng.poisson(card.rate * txn_rate * days))
        if n == 0:
            continue
        day = np.floor(rng.uniform(0, days, size=n))
        night = rng.random(n) < min(card.night_pref + shift.night_boost, 0.5)
        hour = np.where(night, rng.uniform(0, 5, size=n), np.clip(rng.normal(14, 3.5, size=n), 6, 23.99))
        t = (day + hour / 24.0) * 86400.0
        cat_probs = card.cat_probs
        if shift.category_boost:
            boost = np.array([shift.category_boost.get(c, 1.0) for c in CATEGORIES])
            cat_probs = cat_probs * boost / (cat_probs * boost).sum()
        cat = rng.choice(len(CATEGORIES), size=n, p=cat_probs)
        p_online = np.clip(_CAT_ONLINE[cat] * (0.5 + card.online_pref), 0, 1)
        p_online = np.where(_CAT_ONLINE[cat] > 0, 1 - (1 - p_online) * (1 - shift.online_boost), 0.0)
        online = rng.random(n) < p_online
        amount = np.exp(rng.normal(card.mean_log_amount, card.sigma, size=n)) * _CAT_AMOUNT[cat] * shift.amount_scale
        # genuine micro-payments (app stores, streaming) overlap with card-testing amounts
        micro = (cat == CATEGORIES.index("digital_goods")) & (rng.random(n) < 0.35)
        amount = np.where(micro, rng.choice([0.99, 1.99, 2.99, 4.99], size=n), amount)
        # genuine big-ticket purchases (a new laptop, a flight) overlap with high-value fraud
        big = rng.random(n) < 0.012
        amount = np.where(big, amount * rng.uniform(4, 12, size=n), amount)
        city = np.full(n, card.home)
        # cross-border e-commerce / VPN / roaming: online location that is far from home
        roam = online & (rng.random(n) < 0.04)
        city = np.where(roam, rng.integers(0, len(CITIES), size=n), city)
        if card.trip is not None:
            s, e, dest = card.trip
            in_trip = (t >= s * 86400) & (t < e * 86400) & ~online
            city = np.where(in_trip, dest, city)
        lat = np.array([CITIES[c][1] for c in city]) + _jitter(rng, n)
        lon = np.array([CITIES[c][2] for c in city]) + _jitter(rng, n)
        country = [CITIES[c][3] for c in city]
        dev_idx = rng.integers(0, len(card.devices), size=n)
        device = []
        for k in range(n):
            if not online[k]:
                device.append("card_present")
            elif rng.random() < 0.02:  # new browser / cleared cookies
                device.append(f"{card.devices[0]}_b{int(rng.integers(1000))}")
            elif card.upgrade_day is not None and t[k] >= card.upgrade_day * 86400:
                device.append(f"{card.devices[0]}_new")
            else:
                device.append(card.devices[dev_idx[k]])
        cols["t"].extend(t.tolist())
        cols["card"].extend([card.card_id] * n)
        cols["amount"].extend(amount.tolist())
        cols["cat"].extend(cat.tolist())
        cols["online"].extend(online.tolist())
        cols["lat"].extend(lat.tolist())
        cols["lon"].extend(lon.tolist())
        cols["country"].extend(country)
        cols["device"].extend(device)
    cols["fraud_type"] = [""] * len(cols["t"])
    return cols


def _append(cols: dict, t, card, amount, cat, online, city, device, ftype, rng) -> None:
    cols["t"].append(float(t))
    cols["card"].append(card.card_id)
    cols["amount"].append(float(amount))
    cols["cat"].append(int(cat))
    cols["online"].append(bool(online))
    cols["lat"].append(CITIES[city][1] + float(rng.normal(0, 0.05)))
    cols["lon"].append(CITIES[city][2] + float(rng.normal(0, 0.05)))
    cols["country"].append(CITIES[city][3])
    cols["device"].append(device)
    cols["fraud_type"].append(ftype)


def _inject_fraud(cols: dict, cards: list[_Card], days: int, n_target: int, shift: Shift, rng) -> None:
    mix = shift.fraud_mix or DEFAULT_FRAUD_MIX
    names = list(mix)
    probs = np.array([mix[k] for k in names], dtype=float)
    probs /= probs.sum()
    by_card: dict[str, list[float]] = {}
    for t, c in zip(cols["t"], cols["card"], strict=True):
        by_card.setdefault(c, []).append(t)
    eligible = [c for c in cards if len(by_card.get(c.card_id, [])) >= 3]
    cat = {name: CATEGORIES.index(name) for name in CATEGORIES}
    scale = shift.amount_scale
    injected = 0
    ep = 0
    while injected < n_target:
        ep += 1
        card = eligible[int(rng.integers(len(eligible)))]
        ftype = names[int(rng.choice(len(names), p=probs))]
        start = float(rng.uniform(2, days - 0.5)) * 86400
        far = [c for c in range(len(CITIES)) if c != card.home]
        # a share of attacks (30% by default) is 'stealthy': known device, home location, believable amounts
        stealth = rng.random() < shift.stealth_rate
        fraud_dev = card.devices[0] if stealth else f"dev_fr_{ep:05d}"
        if ftype == "card_testing":
            city = int(rng.choice(far)) if rng.random() < 0.5 and not stealth else card.home
            n = int(rng.integers(2, 7))
            t = start
            for _ in range(n):
                t += float(rng.uniform(60, 1800 if stealth else 600))
                c = (
                    cat["digital_goods"]
                    if rng.random() < 0.7
                    else int(rng.choice([cat["retail"], cat["entertainment"]]))
                )
                _append(cols, t, card, rng.uniform(0.5, 9.0), c, True, city, fraud_dev, ftype, rng)
            injected += n
            if rng.random() < 0.5:
                t += float(rng.uniform(300, 3600))
                big = rng.uniform(250, 1500) * scale
                _append(
                    cols,
                    t,
                    card,
                    big,
                    cat[str(rng.choice(["electronics", "gift_cards"]))],
                    True,
                    city,
                    fraud_dev,
                    ftype,
                    rng,
                )
                injected += 1
        elif ftype == "geo_velocity":
            anchor = float(rng.choice(by_card[card.card_id]))
            city = int(
                rng.choice(
                    [
                        c
                        for c in far
                        if abs(CITIES[c][1] - CITIES[card.home][1]) + abs(CITIES[c][2] - CITIES[card.home][2]) > 15
                    ]
                )
            )
            n = int(rng.integers(1, 4))
            t = anchor
            for _ in range(n):
                t += float(rng.uniform(5, 90 if not stealth else 600)) * 60
                c = int(rng.choice([cat["retail"], cat["electronics"], cat["restaurants"], cat["jewelry"]]))
                amt = np.exp(card.mean_log_amount) * _CAT_AMOUNT[c] * rng.uniform(1.0, 3.0) * scale
                _append(cols, t, card, amt, c, False, city, "card_present", ftype, rng)
            injected += n
        elif ftype == "night_high_value":
            day = np.floor(start / 86400)
            n = int(rng.integers(1, 3))
            for _ in range(n):
                t = (day + rng.uniform(1, 5) / 24) * 86400
                c = int(rng.choice([cat["electronics"], cat["jewelry"], cat["gift_cards"], cat["retail"]]))
                amt = np.exp(card.mean_log_amount) * rng.uniform(2 if stealth else 5, 15) * scale
                online = rng.random() < 0.5
                _append(cols, t, card, amt, c, online, card.home, fraud_dev if online else "card_present", ftype, rng)
            injected += n
        elif ftype == "new_category":
            unused = [
                k
                for k in (cat["jewelry"], cat["gift_cards"], cat["crypto_exchange"], cat["electronics"])
                if card.cat_probs[k] < 0.01
            ]
            c_new = int(rng.choice(unused)) if unused else cat["crypto_exchange"]
            n = int(rng.integers(1, 3))
            t = start
            for _ in range(n):
                t += float(rng.uniform(60, 7200))
                amt = np.exp(card.mean_log_amount) * rng.uniform(1.5 if stealth else 3, 10) * scale
                online = rng.random() < _CAT_ONLINE[c_new]
                _append(
                    cols,
                    t,
                    card,
                    amt,
                    c_new,
                    online,
                    card.home,
                    card.devices[0] if online else "card_present",
                    ftype,
                    rng,
                )
            injected += n
        else:  # account_takeover
            city = (
                card.home
                if stealth
                else (int(rng.choice(_FOREIGN)) if rng.random() < 0.6 else int(rng.choice(_DOMESTIC)))
            )
            n = int(rng.integers(3, 7))
            t = start
            amt = np.exp(card.mean_log_amount) * rng.uniform(0.8, 2.0) * scale
            for _ in range(n):
                t += float(rng.uniform(5, 240)) * 60
                amt *= rng.uniform(1.0, 1.5)
                c = int(rng.choice([cat["electronics"], cat["gift_cards"], cat["retail"], cat["travel"]]))
                _append(cols, t, card, amt, c, True, city, fraud_dev, ftype, rng)
            injected += n


def generate_transactions(
    n_cards: int = 2500,
    days: int = 90,
    start: str = "2026-01-01",
    fraud_rate: float = 0.015,
    txn_rate: float = 0.22,
    seed: int = 42,
    stream: int = 1,
    shift: Shift | None = None,
    id_prefix: str = "tx",
) -> pd.DataFrame:
    """Generate a labelled transaction log sorted by time.

    ``seed`` fixes the card population; ``stream`` selects an independent transaction stream for
    the same population (so a 'next month' file can reuse the cardholders of the training file).
    """
    shift = shift or Shift()
    cards = _profiles(n_cards, days, seed)
    rng = np.random.default_rng([seed, stream])
    cols = _legit(cards, days, txn_rate, shift, rng)
    n_legit = len(cols["t"])
    n_fraud = int(round(n_legit * fraud_rate / (1 - fraud_rate)))
    _inject_fraud(cols, cards, days, n_fraud, shift, rng)

    base = pd.Timestamp(start)
    df = pd.DataFrame(
        {
            "timestamp": base + pd.to_timedelta(np.round(np.asarray(cols["t"])), unit="s"),
            "card_id": cols["card"],
            "amount": np.round(np.asarray(cols["amount"]), 2),
            "merchant_category": [CATEGORIES[c] for c in cols["cat"]],
            "channel": np.where(np.asarray(cols["online"]), "online", "pos"),
            "lat": np.round(np.asarray(cols["lat"]), 4),
            "lon": np.round(np.asarray(cols["lon"]), 4),
            "country": cols["country"],
            "device_id": cols["device"],
            "fraud_type": cols["fraud_type"],
        }
    )
    df = df[df["timestamp"] < base + pd.Timedelta(days=days)]
    df["amount"] = df["amount"].clip(lower=0.5)
    df["is_fraud"] = (df["fraud_type"] != "").astype(int)
    mrng = np.random.default_rng([seed, stream, 7])
    merch_no = mrng.integers(0, 400, size=len(df))
    fraud_merch = mrng.integers(400, 460, size=len(df))  # small pool of merchants abused by fraud rings
    df["merchant_id"] = [
        f"M_{c[:4]}_{(f if fr else m):03d}"
        for c, m, f, fr in zip(df["merchant_category"], merch_no, fraud_merch, df["is_fraud"].to_numpy(), strict=True)
    ]
    df = df.sort_values(["timestamp", "card_id"], kind="mergesort").reset_index(drop=True)
    df["txn_id"] = [f"{id_prefix}{i:07d}" for i in range(len(df))]
    return df[COLUMNS]


def generate_next_month(n_cards: int = 2500, seed: int = 42, fraud_rate: float = 0.022) -> pd.DataFrame:
    """Same cardholders, the following month, with a realistic distribution shift."""
    return generate_transactions(
        n_cards=n_cards,
        days=30,
        start="2026-04-01",
        fraud_rate=fraud_rate,
        seed=seed,
        stream=2,
        shift=NEXT_MONTH_SHIFT,
        id_prefix="nm",
    )


def load_transactions(path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"], keep_default_na=False)
    df["is_fraud"] = df["is_fraud"].astype(int)
    return df
