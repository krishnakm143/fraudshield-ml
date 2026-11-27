"""FastAPI service: real-time scoring, batch scoring, model metadata and drift report."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from . import __version__
from .data import CATEGORIES
from .service import FraudService

log = logging.getLogger(__name__)


class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    txn_id: str | None = None
    card_id: str = Field(examples=["C00042"])
    timestamp: datetime = Field(examples=["2026-03-31T02:14:00"])
    amount: float = Field(gt=0, examples=[899.0])
    merchant_id: str | None = None
    merchant_category: str = Field(examples=["electronics"])
    channel: Literal["online", "pos"] = "pos"
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    country: str = Field(min_length=2, max_length=2, examples=["US"])
    device_id: str = "card_present"

    def as_record(self) -> dict:
        d = self.model_dump()
        d["timestamp"] = self.timestamp.replace(tzinfo=None)
        if d["merchant_category"] not in CATEGORIES:
            raise HTTPException(422, f"merchant_category must be one of {CATEGORIES}")
        return d


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transactions: list[Transaction] = Field(min_length=1, max_length=5000)
    explain: bool = False
    commit: bool = True


_state: dict[str, FraudService | None] = {"service": None}


def get_service() -> FraudService:
    if _state["service"] is None:
        try:
            _state["service"] = FraudService()
        except FileNotFoundError as exc:
            raise HTTPException(503, f"no model available: {exc}. Run `fraudshield train` first.") from exc
    return _state["service"]  # type: ignore[return-value]


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_service()
    except HTTPException as exc:  # start anyway; /health reports the problem
        log.warning(exc.detail)
    yield


app = FastAPI(
    title="FraudShield",
    version=__version__,
    description="Real-time card fraud scoring with explanations and drift monitoring.",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    svc = _state["service"]
    return {
        "status": "ok" if svc else "degraded",
        "model_loaded": svc is not None,
        "model_version": svc.metadata["version"] if svc else None,
        "version": __version__,
    }


@app.post("/score")
def score(
    txn: Transaction,
    explain: bool = Query(True, description="Include per-feature contributions"),
    summary: bool = Query(False, description="Include analyst summary (Claude, or offline template)"),
    commit: bool = Query(True, description="Update the card's online feature state with this transaction"),
) -> dict:
    return get_service().score(txn.as_record(), explain=explain, summary=summary, commit=commit)


@app.post("/score/batch")
def score_batch(req: BatchRequest) -> dict:
    svc = get_service()
    results = svc.score_batch([t.as_record() for t in req.transactions], explain=req.explain, commit=req.commit)
    n_review = sum(r["decision"] == "review" for r in results)
    return {"n": len(results), "n_review": n_review, "model_version": svc.metadata["version"], "results": results}


@app.get("/model")
def model() -> dict:
    return get_service().model_info()


@app.get("/drift")
def drift(refresh: bool = False) -> dict:
    try:
        return get_service().drift(refresh=refresh)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
