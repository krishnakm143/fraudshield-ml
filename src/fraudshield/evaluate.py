"""Offline evaluation: trains both candidates on the time split and prints test-set metrics.

python -m fraudshield.evaluate [--data PATH] [--json reports/metrics.json]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .config import CostModel, get_settings, n_threads
from .data import generate_transactions, load_transactions
from .features import FEATURES, CardStateStore
from .train import TrainingResult, run_training


def latency_benchmark(result: TrainingResult, n: int = 500) -> dict:
    """End-to-end single-transaction latency: online features + calibrated model (+ explanation)."""
    model = result.best_model
    hist = result.splits["train"].df
    store = CardStateStore.from_history(hist)
    txns = result.splits["valid"].df.head(n).to_dict("records")
    out = {}
    with threadpool_limits(limits=n_threads()):
        for explain in (False, True):
            lat = []
            for txn in txns:
                t0 = time.perf_counter()
                f = store.score_and_update(txn, commit=False)
                x = np.array([[f[k] for k in FEATURES]])
                model.predict_proba(x)
                if explain:
                    model.contributions(x[0])
                lat.append((time.perf_counter() - t0) * 1000)
            key = "with_explanation" if explain else "score_only"
            out[key] = {"p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95))}
    return out


def format_report(result: TrainingResult, latency: dict | None = None) -> str:
    keys = [
        ("pr_auc", "PR-AUC", "{:.3f}"),
        ("roc_auc", "ROC-AUC", "{:.4f}"),
        ("recall_at_1pct_fpr", "Recall @ 1% FPR", "{:.3f}"),
        ("brier", "Brier (calibrated)", "{:.4f}"),
        ("threshold", "Cost-optimal threshold", "{:.3f}"),
        ("precision", "Precision @ threshold", "{:.3f}"),
        ("recall", "Recall @ threshold", "{:.3f}"),
        ("alert_rate", "Alert rate", "{:.2%}"),
        ("fraud_amount_caught_pct", "Fraud $ caught", "{:.1%}"),
        ("expected_savings", "Expected savings ($)", "{:,.0f}"),
        ("savings_pct", "Savings vs no model", "{:.1%}"),
    ]
    names = list(result.models)
    test = result.splits["test"]
    lines = [
        f"Test split: {test.df['timestamp'].min():%Y-%m-%d} .. {test.df['timestamp'].max():%Y-%m-%d}  "
        f"n={len(test.y):,}  fraud={test.y.sum()} ({test.y.mean():.2%})",
        "",
        f"{'metric':<26}" + "".join(f"{n:>14}" for n in names),
        "-" * (26 + 14 * len(names)),
    ]
    for k, label, fmt in keys:
        lines.append(f"{label:<26}" + "".join(f"{fmt.format(result.metrics[n]['test'][k]):>14}" for n in names))
    lines.append(
        f"{'Brier (uncalibrated)':<26}" + "".join(f"{result.metrics[n]['test_raw_brier']:>14.4f}" for n in names)
    )
    lines += ["", f"Selected on validation savings: {result.best}", "", "Recall by fraud pattern (test):"]
    for t, v in result.metrics[result.best]["test_per_type_recall"].items():
        lines.append(f"  {t:<20} n={v['n']:<4} recall={v['recall']:.2f}")
    if latency:
        lines += ["", "Single-transaction latency (features + model):"]
        for k, v in latency.items():
            lines.append(f"  {k:<18} p50={v['p50_ms']:.2f} ms  p95={v['p95_ms']:.2f} ms")
    lines.append(f"\nTraining wall time: {result.timings['total_s']:.1f}s")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="fraudshield evaluate", description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=None, help="transactions CSV (default: bundled sample)")
    ap.add_argument("--json", type=Path, default=None, help="also write metrics JSON here")
    ap.add_argument("--no-latency", action="store_true")
    args = ap.parse_args(argv)
    path = args.data or get_settings().data_path
    df = load_transactions(path) if path.exists() else generate_transactions()
    result = run_training(df, CostModel())
    latency = None if args.no_latency else latency_benchmark(result)
    print(format_report(result, latency))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {"best": result.best, "metrics": result.metrics, "latency": latency, "timings": result.timings}
        args.json.write_text(json.dumps(payload, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
