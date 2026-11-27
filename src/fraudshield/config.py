"""Runtime configuration (environment-driven, with sensible local defaults)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _path(env: str, default: Path) -> Path:
    return Path(os.getenv(env, str(default)))


@dataclass(frozen=True)
class CostModel:
    """Business cost of decisions, used to pick the alert threshold.

    * a missed fraud costs the transaction amount (chargeback write-off) plus a fixed handling fee
    * every alert (true or false) costs an analyst review; false alerts also add customer friction
    """

    review_cost: float = 6.0
    friction_cost: float = 4.0
    chargeback_fee: float = 20.0


@dataclass(frozen=True)
class Settings:
    data_path: Path = field(
        default_factory=lambda: _path("FRAUDSHIELD_DATA", PROJECT_ROOT / "data" / "transactions.csv.gz")
    )
    next_month_path: Path = field(
        default_factory=lambda: _path("FRAUDSHIELD_NEXT_MONTH", PROJECT_ROOT / "data" / "next_month.csv.gz")
    )
    registry_dir: Path = field(
        default_factory=lambda: _path("FRAUDSHIELD_REGISTRY", PROJECT_ROOT / "models" / "registry")
    )
    reports_dir: Path = field(default_factory=lambda: _path("FRAUDSHIELD_REPORTS", PROJECT_ROOT / "reports"))


def n_threads() -> int:
    """OpenMP/BLAS threads for fitting and scoring.

    Defaults to 1: on small/medium tabular data (and inside CPU-quota containers) multi-threaded
    HistGradientBoosting is often *slower* due to oversubscription, and single-row scoring latency
    is dominated by thread start-up. Override with FRAUDSHIELD_THREADS.
    """
    return max(int(os.getenv("FRAUDSHIELD_THREADS", "1")), 1)


def get_settings() -> Settings:
    return Settings()
