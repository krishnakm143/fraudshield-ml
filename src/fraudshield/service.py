"""Scoring service: registered model + online feature store + explanations, shared by API and CLI."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .config import Settings, get_settings, n_threads
from .data import load_transactions
from .explain import analyst_summary, risk_level, top_contributions
from .features import FEATURES, CardStateStore
from .llm import LLMClient
from .monitor import run_drift
from .registry import ModelRegistry

log = logging.getLogger(__name__)


class FraudService:
    def __init__(self, settings: Settings | None = None, llm: LLMClient | None = None) -> None:
        self.settings = settings or get_settings()
        self.registry = ModelRegistry(self.settings.registry_dir)
        self.model, self.metadata = self.registry.load()
        self.llm = llm or LLMClient()
        self.store = CardStateStore()
        self._history: pd.DataFrame | None = None
        self._drift: dict | None = None
        self._limiter = threadpool_limits(limits=n_threads())  # process-wide; keeps p50 latency low
        if self.settings.data_path.exists():
            t0 = time.perf_counter()
            self._history = load_transactions(self.settings.data_path)
            self.store = CardStateStore.from_history(self._history)
            log.info("feature store warmed with %d cards in %.2fs", len(self.store), time.perf_counter() - t0)
        self.model.contributions(self.model.baseline)  # warm-up so the first request is not an outlier

    @property
    def threshold(self) -> float:
        return self.model.threshold

    def score(self, txn: dict, explain: bool = True, summary: bool = False, commit: bool = True) -> dict:
        t0 = time.perf_counter()
        feats = self.store.score_and_update(txn, commit=commit)
        x = np.fromiter((feats[f] for f in FEATURES), dtype=float, count=len(FEATURES))
        prob = float(self.model.predict_proba(x.reshape(1, -1))[0])
        out = {
            "txn_id": txn.get("txn_id"),
            "card_id": txn["card_id"],
            "fraud_probability": round(prob, 6),
            "threshold": round(self.threshold, 6),
            "decision": "review" if prob >= self.threshold else "approve",
            "risk_level": risk_level(prob, self.threshold),
            "model_version": self.metadata["version"],
        }
        if explain or summary:
            drivers = top_contributions(self.model.contributions(x), feats, txn)
            out["explanation"] = drivers
            if summary:
                out["analyst_summary"] = analyst_summary(
                    self.llm, prob, self.threshold, feats, drivers, txn
                ).model_dump()
                out["summary_source"] = self.llm.mode
        out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 3)
        return out

    def score_batch(self, txns: list[dict], explain: bool = False, commit: bool = True) -> list[dict]:
        order = sorted(range(len(txns)), key=lambda i: (pd.Timestamp(txns[i]["timestamp"]), str(txns[i].get("txn_id"))))
        results: list[dict | None] = [None] * len(txns)
        for i in order:  # chronological so velocity features see earlier txns in the same batch
            results[i] = self.score(txns[i], explain=explain, commit=commit)
        return results  # type: ignore[return-value]

    def model_info(self) -> dict:
        m = self.metadata
        name = m["model_name"]
        return {
            "version": m["version"],
            "model_name": name,
            "registered_at": m["registered_at"],
            "threshold": m["threshold"],
            "calibration": m["calibration"],
            "features": m["features"],
            "cost_model": m["cost_model"],
            "data": m["data"],
            "test_metrics": m["metrics"][name]["test"],
            "candidates": {k: v["test"]["pr_auc"] for k, v in m["metrics"].items()},
            "available_versions": self.registry.versions(),
            "feature_store_cards": len(self.store),
            "llm_mode": self.llm.mode,
        }

    def drift(self, refresh: bool = False) -> dict:
        path = Path(self.settings.reports_dir) / "drift_report.json"
        if self._drift is not None and not refresh:
            return self._drift
        if path.exists() and not refresh:
            report = json.loads(path.read_text())
            if report.get("model_version") == self.metadata["version"]:
                self._drift = report
                return report
        if self._history is None or not self.settings.next_month_path.exists():
            raise FileNotFoundError("no drift report and no current-period data to compute one")
        current = load_transactions(self.settings.next_month_path)
        profile = self.registry.load_reference_profile(self.metadata["version"])
        self._drift = run_drift(self.model, self.metadata, profile, self._history, current)
        return self._drift
