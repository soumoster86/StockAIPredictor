#!/usr/bin/env python3
# =============================
# train_global.py
# =============================
"""Train ONE global model per horizon on the pooled history of every stock
in a watchlist CSV, and save the artifacts for the app to load at runtime.

Run this OFFLINE (on your laptop), then commit the global_models/ folder:

    python train_global.py                          # liquid default (stocks.csv)
    python train_global.py --stocks stocks_universe.csv  # full NSE dump
    python train_global.py --stocks my.csv --horizons 1 5

Why offline: training a full ensemble on tens of thousands of pooled rows
is far too heavy for a Streamlit Cloud request. Train once here, load in
milliseconds there.

Training uses a **time-ordered** pool split (not a random shuffle): early
calendar days train, later days evaluate, with an embargo matching the
label horizon so overlapping multi-day targets cannot leak across the cut.
The deployed artifact is then refit on all labeled history.

Improving the model over time = rerun this monthly as more REAL price
history accumulates. Do NOT feed the model its own predictions.
"""

import argparse
import csv
import json
import os
import sys
import time

from data import HORIZONS, add_features, fetch_index, fetch_many
from model import (
    GLOBAL_META_FILE,
    GLOBAL_MODEL_DIR,
    save_global_model,
    train_global_predictor,
)


def load_watchlist(path):
    rows = {}
    # utf-8-sig strips the Excel/Windows BOM so the first header parses as
    # "Name" not "﻿Name"; the explicit ﻿ replace mirrors
    # ui.services.load_stock_list as a belt-and-suspenders guard.
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            r = {k.replace("﻿", "").strip().lower(): (v or "").strip()
                 for k, v in r.items()}
            name, sym = r.get("name"), r.get("symbol", "").upper()
            if name and sym and sym not in rows.values():
                rows[name] = sym
    return rows


def build_frames(symbols, index_close, batch_size=40):
    """Fetch + feature-engineer every stock. Uses batched yfinance downloads
    to stay under rate limits on large universes."""
    frames = {}
    symbols = list(symbols)
    total = len(symbols)
    for start in range(0, total, batch_size):
        chunk = symbols[start:start + batch_size]
        print(f"  batch {start + 1}-{start + len(chunk)} / {total} ...",
              flush=True)
        try:
            batch = fetch_many(chunk)
        except Exception as e:
            print(f"    batch failed ({str(e)[:60]}); falling back per-symbol")
            batch = {}
            for sym in chunk:
                try:
                    from data import fetch_data
                    batch[sym] = fetch_data(sym)
                except Exception:
                    batch[sym] = None
                time.sleep(0.25)

        for sym in chunk:
            raw = batch.get(sym)
            if raw is None or getattr(raw, "empty", True) or len(raw) < 200:
                print(f"    {sym}: skip (no/short data)")
                continue
            try:
                frames[sym] = add_features(raw, index_close=index_close)
                print(f"    {sym}: {len(frames[sym])} rows")
            except Exception as e:
                print(f"    {sym}: skip ({str(e)[:50]})")
        time.sleep(0.5)  # pause between batches
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stocks", default="stocks.csv",
        help="Watchlist CSV (Name,Symbol). Prefer stocks_universe.csv for "
             "full-universe offline training; stocks.csv is the liquid app list.",
    )
    ap.add_argument("--horizons", type=int, nargs="*", default=HORIZONS)
    ap.add_argument("--model", default="Ensemble")
    ap.add_argument("--batch-size", type=int, default=40)
    args = ap.parse_args()

    if not os.path.exists(args.stocks):
        sys.exit(f"Watchlist not found: {args.stocks}")

    watch = load_watchlist(args.stocks)
    symbols = list(watch.values())
    print(f"Training global model on {len(symbols)} stocks "
          f"from {args.stocks}\n")

    print("Fetching NIFTY market context ...")
    index_close = fetch_index()
    print("  " + ("ok" if index_close is not None else
                  "unavailable — context features will be neutral") + "\n")

    print("Fetching + engineering features (batched):")
    frames = build_frames(symbols, index_close, batch_size=args.batch_size)
    if not frames:
        sys.exit("No stocks could be fetched — aborting.")
    print(f"\n{len(frames)} stocks usable.\n")

    meta = {
        "stocks": list(frames.keys()),
        "n_stocks": len(frames),
        "model": args.model,
        "trained_at": time.strftime("%Y-%m-%d %H:%M"),
        "split": "time",
        "source_watchlist": args.stocks,
        "horizons": {},
    }

    for h in args.horizons:
        print(f"Training horizon {h}d (time-ordered pool + embargo) ...",
              flush=True)
        try:
            predictor, scaler, m = train_global_predictor(
                frames, f"Target_{h}", model_type=args.model)
        except ValueError as e:
            print(f"  skipped: {e}")
            continue
        path = save_global_model(predictor, scaler, h)
        meta["horizons"][str(h)] = {
            "rows": m["n_rows"],
            "stocks": m["n_stocks"],
            "n_train": m.get("n_train"),
            "n_test": m.get("n_test"),
            "cut_date": m.get("cut_date"),
            "embargo_days": m.get("embargo_days"),
            "split": m.get("split", "time"),
            "accuracy": round(m["accuracy"], 4),
            "baseline": round(m["baseline_accuracy"], 4),
        }
        edge = m["accuracy"] - m["baseline_accuracy"]
        print(f"  saved {path}")
        print(f"  time-holdout accuracy {m['accuracy']:.3f} "
              f"vs baseline {m['baseline_accuracy']:.3f} "
              f"({edge:+.3f})  on {m['n_rows']:,} rows "
              f"(train={m.get('n_train')}, test={m.get('n_test')}, "
              f"cut={m.get('cut_date')}, embargo={m.get('embargo_days')}d)\n")

    with open(os.path.join(GLOBAL_MODEL_DIR, GLOBAL_META_FILE), "w") as f:
        json.dump(meta, f, indent=2)

    print("Done. Commit the global_models/ folder, push, and reboot the app.")
    print("The app will auto-prefer the global model where available.")
    print("\nReminder: metrics are from a pooled **time** holdout, but stocks "
          "in the app's watchlist were usually in the training universe. "
          "Trust each stock's Walk-Forward tab for a stricter verdict.")


if __name__ == "__main__":
    main()
