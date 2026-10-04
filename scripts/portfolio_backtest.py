#!/usr/bin/env python3
"""Offline portfolio backtest of the screener → rankings/portfolio_*.

Fits a pooled model on the early part of history only, then runs the
screener as a real portfolio over the later, unseen part: top-N by Buy
Score from the top 5% of the liquid universe on the 10-day model, reviewed
every 10 trading days, next-open fills, NSE costs, circuit locks, liquidity floor,
cash at the risk-free rate — compared with Nifty benchmarks.

    python scripts/portfolio_backtest.py                       # full universe
    python scripts/portfolio_backtest.py --max 300             # smoke test
    python scripts/portfolio_backtest.py --top-n 15 --rebalance 20
    python scripts/portfolio_backtest.py --horizon 1 --rebalance 5 --absolute  # old screener
    python scripts/portfolio_backtest.py --model Ensemble      # slow, closer to app

Then commit/sync the rankings/ folder; the Scanner tab shows the result.
Do NOT run this inside a Streamlit request — it is deliberately offline.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from data import fetch_data, fetch_index  # noqa: E402
from execution import RISK_FREE_RATE  # noqa: E402
from model import SCREEN_HORIZON  # noqa: E402
from portfolio import (  # noqa: E402
    DEFAULT_CAPITAL,
    DEFAULT_MAX_RISK,
    DEFAULT_MIN_PROB,
    DEFAULT_MIN_QUANTILE,
    DEFAULT_REBALANCE_DAYS,
    DEFAULT_TOP_N,
    SURVIVORSHIP_NOTE,
    benchmark_curve,
    build_slim_frames,
    score_frames,
    series_stats,
    simulate_portfolio,
    time_split_fit,
)
from screener import (  # noqa: E402
    PORTFOLIO_EQUITY as OUT_EQUITY,
)
from screener import (  # noqa: E402
    PORTFOLIO_JSON as OUT_JSON,
)
from screener import (  # noqa: E402
    PORTFOLIO_TRADES as OUT_TRADES,
)
from screener import (  # noqa: E402
    RANKINGS_DIR,
)
from train_global import load_watchlist  # noqa: E402

BENCHMARKS = {
    "Nifty 500 (price index)": "^CRSLDX",
    "Nifty 50 ETF (total return)": "NIFTYBEES.NS",
}


def main():
    ap = argparse.ArgumentParser(description="Portfolio backtest of the screener")
    ap.add_argument("--stocks", default=str(ROOT / "stocks_universe.csv"))
    ap.add_argument("--max", type=int, default=None, help="Cap symbols (debug)")
    ap.add_argument("--batch-size", type=int, default=60)
    ap.add_argument("--model", default="fast", help="'fast' (XGBoost) or 'Ensemble'")
    ap.add_argument("--horizon", type=int, default=SCREEN_HORIZON,
                    help="Label horizon of the ranking model (default: the screener's)")
    ap.add_argument("--max-train-rows", type=int, default=1_500_000,
                    help="Subsample pooled training rows to bound memory")
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    ap.add_argument("--rebalance", type=int, default=DEFAULT_REBALANCE_DAYS,
                    help="Trading days between rebalances")
    ap.add_argument("--capital", type=float, default=DEFAULT_CAPITAL)
    ap.add_argument("--min-turnover-cr", type=float, default=1.0,
                    help="Minimum 20-day median traded value, ₹ crore/day")
    ap.add_argument("--max-risk", type=float, default=DEFAULT_MAX_RISK)
    ap.add_argument("--min-prob", type=float, default=DEFAULT_MIN_PROB,
                    help="Absolute entry cutoff (used only with --absolute)")
    ap.add_argument("--min-quantile", type=float, default=DEFAULT_MIN_QUANTILE,
                    help="Relative entry: probability in the top (1 - q) of the liquid universe")
    ap.add_argument("--absolute", action="store_true",
                    help="Use the old absolute cutoff (--min-prob) instead of --min-quantile")
    ap.add_argument("--require-edge", action="store_true",
                    help="Require trailing accuracy >= majority baseline")
    ap.add_argument("--out", default=str(RANKINGS_DIR))
    args = ap.parse_args()

    t0 = time.time()
    watch = load_watchlist(args.stocks)
    symbols = list(watch.values())[: args.max] if args.max else list(watch.values())
    print(f"Portfolio backtest on {len(symbols)} symbols from {args.stocks}\n")

    index_close = fetch_index()
    frames = build_slim_frames(symbols, index_close, horizons=(args.horizon,),
                               batch_size=args.batch_size,
                               log=lambda m: print(m, flush=True))
    if len(frames) < args.top_n * 2:
        sys.exit(f"Only {len(frames)} usable stocks — need at least {args.top_n * 2}.")

    print(f"\nFitting {args.model} model on the first {args.train_frac:.0%} of days ...")
    predictor, scaler, info = time_split_fit(frames, args.horizon, args.train_frac, args.model,
                                             max_train_rows=args.max_train_rows)
    print(f"  cut {info['cut_date'].date()}, trading from {info['test_start'].date()} "
          f"({info['n_train_rows']:,} training rows, {info['n_stocks']} stocks)")

    probs = score_frames(frames, predictor, scaler)

    def progress(k, total, d, n_held):
        if k % (args.rebalance * 10) == 0:
            print(f"  {d.date()}  day {k}/{total}  holding {n_held}", flush=True)

    print("\nSimulating ...")
    equity, trades, stats = simulate_portfolio(
        frames, probs, info["test_start"], capital=args.capital, top_n=args.top_n,
        rebalance_days=args.rebalance, min_prob=args.min_prob, max_risk=args.max_risk,
        min_turnover=args.min_turnover_cr * 1e7, horizon=args.horizon, progress=progress,
        min_quantile=None if args.absolute else args.min_quantile,
        require_edge=args.require_edge)

    curves = {"Screener portfolio": equity}
    bench_stats = {}
    for label, sym in BENCHMARKS.items():
        df = fetch_data(sym)
        if df is None or df.empty:
            print(f"  benchmark {sym} unavailable")
            continue
        curve = benchmark_curve(df["Close"], equity.index, args.capital)
        curves[label] = curve
        bench_stats[label] = {**series_stats(curve, args.capital), "symbol": sym}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(curves).rename_axis("date").to_csv(out / OUT_EQUITY)
    trades.to_csv(out / OUT_TRADES, index=False)
    meta = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_s": round(time.time() - t0, 1),
        "watchlist": Path(args.stocks).name,
        "n_symbols": len(symbols),
        "n_usable": len(frames),
        "params": {
            "model": info["model"], "horizon": args.horizon, "train_frac": args.train_frac,
            "cut_date": str(info["cut_date"].date()),
            "test_start": str(info["test_start"].date()),
            "top_n": args.top_n, "rebalance_days": args.rebalance, "capital": args.capital,
            "min_turnover_cr": args.min_turnover_cr, "max_risk": args.max_risk,
            "min_prob": args.min_prob, "risk_free_rate": RISK_FREE_RATE,
            "min_quantile": None if args.absolute else args.min_quantile,
            "require_edge": args.require_edge,
        },
        "stats": stats,
        "benchmarks": bench_stats,
        "caveats": [
            SURVIVORSHIP_NOTE,
            "The ranking model here is refit on early history only, so it is "
            "weaker than the deployed global model is on recent data — but "
            "this is the only honest way to test it.",
            "The Nifty 500 benchmark is a price index (excludes ~1.3%/yr of "
            "dividends); the Nifty 50 ETF includes dividends net of fees.",
            "Taxes (STCG at 20%) are not deducted.",
        ],
    }
    (out / OUT_JSON).write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    print(f"\nDone in {meta['elapsed_s']}s  ({stats['start']} to {stats['end']})")
    print(f"  Screener: CAGR {stats['cagr']:+.1%}  Sharpe {stats['sharpe']:.2f}  "
          f"MaxDD {stats['max_drawdown']:.1%}  trades {stats['n_trades']}  "
          f"hit {stats['hit_rate']:.0%}  costs {stats['costs_pct_per_year']:.1%}/yr  "
          f"turnover {stats['turnover_per_year']:.1f}x/yr")
    for label, b in bench_stats.items():
        print(f"  {label}: CAGR {b['cagr']:+.1%}  Sharpe {b['sharpe']:.2f}  "
              f"MaxDD {b['max_drawdown']:.1%}")
    print(f"  wrote {out / OUT_JSON}, {OUT_EQUITY}, {OUT_TRADES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
