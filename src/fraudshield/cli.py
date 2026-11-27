"""`fraudshield` command line: generate | train | evaluate | drift | serve."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import CostModel, get_settings


def _generate(args) -> int:
    from .data import generate_next_month, generate_transactions

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = generate_transactions(n_cards=args.cards, seed=args.seed)
    nm = generate_next_month(n_cards=args.cards, seed=args.seed)
    df.to_csv(out / "transactions.csv.gz", index=False)
    nm.to_csv(out / "next_month.csv.gz", index=False)
    df.head(200).to_csv(out / "sample_transactions.csv", index=False)
    print(f"transactions: {len(df):,} rows, fraud {df.is_fraud.mean():.2%} -> {out / 'transactions.csv.gz'}")
    print(f"next month:   {len(nm):,} rows, fraud {nm.is_fraud.mean():.2%} -> {out / 'next_month.csv.gz'}")
    return 0


def _load(path: Path):
    from .data import generate_transactions, load_transactions

    if path.exists():
        return load_transactions(path)
    print(f"{path} not found; generating synthetic data in memory", file=sys.stderr)
    return generate_transactions()


def _train(args) -> int:
    from .evaluate import format_report
    from .train import train_and_register

    s = get_settings()
    cost = CostModel(review_cost=args.review_cost, friction_cost=args.friction_cost, chargeback_fee=args.chargeback_fee)
    version, result = train_and_register(_load(Path(args.data or s.data_path)), args.registry or s.registry_dir, cost)
    print(format_report(result))
    print(f"\nRegistered {result.best} as {version} in {args.registry or s.registry_dir}")
    return 0


def _evaluate(args) -> int:
    from .evaluate import main

    argv = (["--data", args.data] if args.data else []) + (["--json", args.json] if args.json else [])
    return main(argv)


def _drift(args) -> int:
    from .data import load_transactions
    from .monitor import run_drift
    from .registry import ModelRegistry

    s = get_settings()
    reg = ModelRegistry(args.registry or s.registry_dir)
    model, meta = reg.load()
    history = _load(Path(args.data or s.data_path))
    cur_path = Path(args.current or s.next_month_path)
    if cur_path.exists():
        current = load_transactions(cur_path)
    else:
        from .data import generate_next_month

        current = generate_next_month()
    report = run_drift(model, meta, reg.load_reference_profile(), history, current)
    out = Path(args.out or Path(s.reports_dir) / "drift_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=float))
    print(
        f"Score PSI {report['score_psi']:.3f} ({report['score_status']}); "
        f"{report['n_significant']} significant / {report['n_moderate']} moderate feature drifts"
    )
    print(f"{'feature':<24}{'PSI':>8}  status")
    for r in report["features"][: args.top]:
        print(f"{r['feature']:<24}{r['psi']:>8.3f}  {r['status']}")
    if "performance" in report:
        ref, cur = report["performance"]["reference_test"], report["performance"]["current"]
        print("\nPerformance       reference   current")
        for k in ("fraud_rate", "pr_auc", "recall_at_1pct_fpr", "recall", "precision"):
            print(f"  {k:<18}{ref[k]:>9.3f}{cur[k]:>10.3f}")
    print(f"\nRecommended action: {report['recommended_action']}\nReport written to {out}")
    return 0


def _serve(args) -> int:
    import uvicorn

    uvicorn.run("fraudshield.api:app", host=args.host, port=args.port, log_level="info")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="fraudshield", description="Real-time card fraud detection toolkit")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate deterministic synthetic transaction data")
    g.add_argument("--out", default="data")
    g.add_argument("--cards", type=int, default=2500)
    g.add_argument("--seed", type=int, default=42)
    g.set_defaults(func=_generate)

    t = sub.add_parser("train", help="train candidates, calibrate, pick threshold, register best model")
    t.add_argument("--data")
    t.add_argument("--registry")
    t.add_argument("--review-cost", type=float, default=CostModel.review_cost)
    t.add_argument("--friction-cost", type=float, default=CostModel.friction_cost)
    t.add_argument("--chargeback-fee", type=float, default=CostModel.chargeback_fee)
    t.set_defaults(func=_train)

    e = sub.add_parser("evaluate", help="print test-set metrics for both candidates")
    e.add_argument("--data")
    e.add_argument("--json")
    e.set_defaults(func=_evaluate)

    d = sub.add_parser("drift", help="PSI drift report for a new period vs the training reference")
    d.add_argument("--data", help="history (training-period) transactions")
    d.add_argument("--current", help="new-period transactions")
    d.add_argument("--registry")
    d.add_argument("--out")
    d.add_argument("--top", type=int, default=10)
    d.set_defaults(func=_drift)

    s = sub.add_parser("serve", help="run the FastAPI service")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=_serve)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
