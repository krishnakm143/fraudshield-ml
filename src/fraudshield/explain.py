"""Per-prediction explanations: model-agnostic contributions + an analyst-facing narrative.

The numbers come from :meth:`FraudModel.contributions` (leave-one-feature-group-out against a
typical legitimate transaction). The narrative is written by Claude when an API key is configured;
otherwise a deterministic template produces the same structured output, so the analyst UI never
depends on an external call.
"""

from __future__ import annotations

import json
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .llm import LLMClient

_LABELS = {
    "amount_log": "transaction amount",
    "hour_of_day": "time of day",
    "is_night": "night-time (00:00-05:00)",
    "is_online": "card-not-present channel",
    "secs_since_prev_log": "time since previous transaction",
    "dist_prev_km_log": "distance from previous transaction",
    "speed_prev_kmh_log": "implied travel speed",
    "dist_home_km_log": "distance from usual location",
    "txn_count_1h": "transactions in last hour",
    "txn_count_24h": "transactions in last 24h",
    "amount_sum_24h_log": "spend in last 24h",
    "small_txn_count_1h": "small (<$5) charges in last hour",
    "amount_ratio_mean_log": "amount vs card's average",
    "amount_zscore": "amount unusualness (z-score)",
    "is_new_category": "first purchase in this merchant category",
    "is_new_device": "unseen device",
    "is_foreign": "foreign country",
    "history_count_log": "length of card history",
    "merchant_category": "merchant category",
}


def describe(group: str, f: dict[str, float], txn: dict) -> str:
    """Human-readable value for a feature group, inverting the log transforms."""
    e = lambda k: math.expm1(f[k])  # noqa: E731
    match group:
        case "amount_log":
            return f"${e('amount_log'):,.2f}"
        case "hour_of_day":
            return str(txn.get("timestamp", ""))[11:16] or "n/a"
        case "secs_since_prev_log":
            s = e("secs_since_prev_log")
            return f"{s / 60:.0f} min" if s < 3 * 3600 else f"{s / 3600:.1f} h"
        case "dist_prev_km_log" | "dist_home_km_log":
            return f"{e(group):,.0f} km"
        case "speed_prev_kmh_log":
            return f"{e(group):,.0f} km/h"
        case "amount_sum_24h_log":
            return f"${e(group):,.2f}"
        case "amount_ratio_mean_log":
            return f"{math.exp(f[group]):.1f}x"
        case "amount_zscore":
            return f"{f[group]:+.1f}"
        case "history_count_log":
            return f"{e(group):.0f} prior txns"
        case "merchant_category":
            return str(txn.get("merchant_category", "n/a"))
        case _ if group.startswith("is_"):
            return "yes" if f[group] >= 0.5 else "no"
        case _:
            return f"{f[group]:g}"


def top_contributions(contrib: dict[str, float], features: dict[str, float], txn: dict, k: int = 5) -> list[dict]:
    ranked = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:k]
    return [
        {
            "feature": g,
            "label": _LABELS.get(g, g),
            "value": describe(g, features, txn),
            "contribution": round(c, 4),
            "direction": "raises risk" if c > 0 else "lowers risk",
        }
        for g, c in ranked
    ]


class AnalystSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk_level: Literal["low", "medium", "high", "critical"]
    headline: str = Field(description="One sentence an analyst can read in 3 seconds")
    key_signals: list[str] = Field(description="2-4 concrete signals, each citing the observed value")
    likely_pattern: Literal[
        "card_testing", "geo_velocity", "night_high_value", "new_category", "account_takeover", "none", "unclear"
    ]
    recommended_action: str


def risk_level(prob: float, threshold: float) -> str:
    if prob >= max(0.8, 3 * threshold):
        return "critical"
    if prob >= threshold:
        return "high"
    if prob >= threshold / 3:
        return "medium"
    return "low"


def _guess_pattern(f: dict[str, float], drivers: list[dict]) -> str:
    raising = {d["feature"] for d in drivers if d["contribution"] > 0}
    if not raising:
        return "none"
    if f["small_txn_count_1h"] >= 2 or ("small_txn_count_1h" in raising and f["txn_count_1h"] >= 2):
        return "card_testing"
    if f["speed_prev_kmh_log"] > math.log1p(1000) and raising & {"speed_prev_kmh_log", "dist_prev_km_log"}:
        return "geo_velocity"
    if f["is_new_device"] and (f["is_foreign"] or f["txn_count_24h"] >= 2):
        return "account_takeover"
    if f["is_night"] and f["amount_ratio_mean_log"] > math.log(3):
        return "night_high_value"
    if f["is_new_category"] and "merchant_category" in raising | {"is_new_category"}:
        return "new_category"
    return "unclear"


_ACTIONS = {
    "critical": "Decline and block the card; contact the cardholder through a verified channel.",
    "high": "Hold the transaction and send a step-up verification (3-D Secure / push approval) to the cardholder.",
    "medium": "Approve but add the card to the watch list; review if another alert fires within 24h.",
    "low": "Approve; no action required.",
}


def fallback_summary(prob: float, threshold: float, features: dict, drivers: list[dict], txn: dict) -> AnalystSummary:
    level = risk_level(prob, threshold)
    pattern = _guess_pattern(features, drivers)
    raising = [d for d in drivers if d["contribution"] > 0][:4]
    signals = [f"{d['label']}: {d['value']} (+{d['contribution']:.2f})" for d in raising] or [
        "no feature materially raises risk versus a typical legitimate transaction"
    ]
    pattern_txt = "" if pattern in {"none", "unclear"} else f", consistent with {pattern.replace('_', ' ')}"
    category = txn["merchant_category"].replace("_", " ")
    article = "an" if category[0] in "aeiou" else "a"
    headline = (
        f"{level.capitalize()} risk ({prob:.0%}) on ${float(txn['amount']):,.2f} at {article} "
        f"{category} merchant{pattern_txt}."
    )
    return AnalystSummary(
        risk_level=level,
        headline=headline,
        key_signals=signals,
        likely_pattern=pattern,
        recommended_action=_ACTIONS[level],
    )


SYSTEM = (
    "You are a senior card-fraud analyst. Given a model score and feature contributions, write a terse, "
    "factual triage note. Only cite signals present in the input; never invent facts. Contributions are the "
    "change in fraud probability when that feature is reset to a typical legitimate value."
)


def analyst_summary(
    llm: LLMClient, prob: float, threshold: float, features: dict, drivers: list[dict], txn: dict
) -> AnalystSummary:
    payload = {
        "fraud_probability": round(prob, 4),
        "alert_threshold": round(threshold, 4),
        "transaction": {k: str(v) for k, v in txn.items()},
        "top_contributions": drivers,
    }
    return llm.structured(
        system=SYSTEM,
        prompt="Write the triage note for this transaction:\n" + json.dumps(payload, indent=2),
        schema=AnalystSummary,
        fallback=lambda: fallback_summary(prob, threshold, features, drivers, txn),
        effort="low",
    )
