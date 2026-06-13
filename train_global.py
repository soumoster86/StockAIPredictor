#!/usr/bin/env python3
# =============================
# train_global.py
# =============================
"""Train ONE global model per horizon on the pooled history of every stock
in stocks.csv, and save the artifacts for the app to load at runtime.

Run this OFFLINE (on your laptop), then commit the global_models/ folder:

    python train_global.py                 # uses stocks.csv
    python train_global.py --stocks my.csv # a different watchlist
    python train_global.py --horizons 1 5  # only some horizons (faster)

Why offline: training a full ensemble on ~100k pooled rows is far too heavy
for a Streamlit Cloud request. Train once here, load in milliseconds there.

Improving the model over time = rerun this monthly as more REAL price
history accumulates. Do NOT feed the model its own predictions.
"""

import argparse
import csv
import json
import os
import sys
import time

from data import HORIZONS, add_features, fetch_data, fetch_index
from model import (
    GLOBAL_MODEL_DIR, GLOBAL_META_FILE,
    train_global_predictor, save_global_model,
)


def load_watchlist(path):
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r = {k.strip().lower(): (v or "").strip() for k, v in r.items()}
            name, sym = r.get("name"), r.get("symbol", "").upper()
            if name and sym and sym not in rows.values():
                rows[name] = sym
    return rows


def build_frames(symbols, index_close):
    """Fetch + feature-engineer every stock once. Returns {symbol: frame}."""
    frames = {}
    for i, sym in enumerate(symbols, 1):
        print(f"  [{i}/{len(symbols)}] {sym} ...", end=" ", flush=True)
        try:
            raw = fetch_data(sym)
            if raw is None or raw.empty or len(raw) < 200:
                print("skip (no/short data)")
                continue
            frames[sym] = add_features(raw, index_close=index_close)
            print(f"{len(frames[sym])} rows")
        except Exception as e:
            print(f"skip ({str(e)[:50]})")
        time.sleep(0.3)  # be gentle with the data source
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stocks", default="stocks.csv")
    ap.add_argument("--horizons", type=int, nargs="*", default=HORIZONS)
    ap.add_argument("--model", default="Ensemble")
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

    print("Fetching + engineering features per stock:")
    frames = build_frames(symbols, index_close)
    if not frames:
        sys.exit("No stocks could be fetched — aborting.")
    print(f"\n{len(frames)} stocks usable.\n")

    meta = {"stocks": list(frames.keys()), "n_stocks": len(frames),
            "model": args.model, "trained_at": time.strftime("%Y-%m-%d %H:%M"),
            "horizons": {}}

    for h in args.horizons:
        print(f"Training horizon {h}d ...", flush=True)
        try:
            predictor, scaler, m = train_global_predictor(
                frames, f"Target_{h}", model_type=args.model)
        except ValueError as e:
            print(f"  skipped: {e}")
            continue
        path = save_global_model(predictor, scaler, h)
        meta["horizons"][str(h)] = {
            "rows": m["n_rows"], "stocks": m["n_stocks"],
            "accuracy": round(m["accuracy"], 4),
            "baseline": round(m["baseline_accuracy"], 4),
        }
        edge = m["accuracy"] - m["baseline_accuracy"]
        print(f"  saved {path}")
        print(f"  pooled-val accuracy {m['accuracy']:.3f} "
              f"vs baseline {m['baseline_accuracy']:.3f} "
              f"({edge:+.3f})  on {m['n_rows']:,} rows\n")

    with open(os.path.join(GLOBAL_MODEL_DIR, GLOBAL_META_FILE), "w") as f:
        json.dump(meta, f, indent=2)

    print("Done. Commit the global_models/ folder, push, and reboot the app.")
    print("The app will auto-prefer the global model where available.")
    print("\nReminder: pooled-validation accuracy is optimistic (rows are "
          "shuffled across time). Trust the app's per-stock WALK-FORWARD tab "
          "for the leakage-free verdict.")


if __name__ == "__main__":
    main()
