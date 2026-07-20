# =============================
# screener.py — pure watchlist screening (no Streamlit)
# =============================
"""Batch scan helpers shared by the app and offline precompute.

Live path: ui/services.py wraps these with Streamlit cache/session.
Offline path: scripts/precompute_rankings.py writes rankings_latest.csv.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from data import add_features, fetch_index, fetch_many
from model import (
    buy_score, compute_risk_score, compute_trade_plan,
    find_support_resistance, global_model_available, load_global_model,
    quick_scan, quick_scan_global,
)

SCAN_BATCH = 80
RANKINGS_DIR = Path(__file__).resolve().parent / "rankings"
RANKINGS_CSV = "rankings_latest.csv"
RANKINGS_META = "rankings_meta.json"
# App treats precomputed files older than this as stale (hours)
DEFAULT_MAX_AGE_HOURS = 48

RESULT_COLUMNS = [
    "Name", "Symbol", "Price", "Day", "Screen", "Probability Up", "Rating",
    "Test Acc", "Baseline", "Model", "Risk", "Reward Risk",
    "To Support", "To Resistance", "Buy Score",
]


def normalize_stock_items(stock_items):
    """Preserve order, drop duplicate symbols."""
    seen, out = set(), []
    for name, sym in stock_items:
        if not sym or sym in seen:
            continue
        seen.add(sym)
        out.append((str(name), str(sym)))
    return out


def slice_scan_batch(stock_items, offset=0, batch_size=SCAN_BATCH):
    """Return (batch_items, next_offset, total, complete)."""
    items = normalize_stock_items(stock_items)
    total = len(items)
    offset = max(0, int(offset))
    size = max(int(batch_size), 1)
    batch = items[offset:offset + size]
    next_offset = min(offset + len(batch), total)
    complete = next_offset >= total
    return batch, next_offset, total, complete


def merge_scan_frames(existing, new_df):
    """Append a batch; newest Symbol wins; sort by Buy Score."""
    if new_df is None or (isinstance(new_df, pd.DataFrame) and new_df.empty):
        if existing is None:
            return pd.DataFrame()
        return existing if isinstance(existing, pd.DataFrame) else pd.DataFrame(existing)
    if existing is None or (isinstance(existing, pd.DataFrame) and existing.empty):
        out = new_df.copy()
    else:
        old = existing if isinstance(existing, pd.DataFrame) else pd.DataFrame(existing)
        out = pd.concat([old, new_df], ignore_index=True)
        if "Symbol" in out.columns:
            out = out.drop_duplicates(subset=["Symbol"], keep="last")
    if not out.empty and "Buy Score" in out.columns:
        out = out.sort_values(
            ["Buy Score", "Probability Up"], ascending=False,
        ).reset_index(drop=True)
    return out


def score_batch(batch_items, index_close=None, global_bundle=None, price_map=None):
    """Score one batch of (name, symbol). Returns (rows, failures).

    `price_map` optional {symbol: OHLCV DataFrame} to avoid re-fetch.
    """
    rows, failures = [], []
    if not batch_items:
        return rows, failures

    if price_map is None:
        price_map = fetch_many([sym for _, sym in batch_items]) or {}

    if global_bundle is None and global_model_available():
        global_bundle = load_global_model(1)

    for name, sym in batch_items:
        try:
            raw = price_map.get(sym, pd.DataFrame())
            if raw is None or getattr(raw, "empty", True) or len(raw) < 400:
                failures.append((sym, "no/short data"))
                continue
            d = add_features(raw, index_close=index_close)
            if global_bundle is not None:
                scan = quick_scan_global(d, bundle=global_bundle)
                if scan is None:
                    scan = quick_scan(d)
            else:
                scan = quick_scan(d)
            if scan is None:
                failures.append((sym, "too little history"))
                continue
            r = compute_risk_score(d)
            s = find_support_resistance(d)
            plan = compute_trade_plan(d, s.get("support"), s.get("resistance"))
            price = float(d["Close"].iloc[-1])
            prev = float(d["Close"].iloc[-2])
            to_sup = (price / s["support"] - 1) if s.get("support") else None
            to_res = (s["resistance"] / price - 1) if s.get("resistance") else None
            rr = plan.get("reward_risk")
            prob = scan["probability"]
            rows.append({
                "Name": name, "Symbol": sym,
                "Price": price, "Day": price / prev - 1,
                "Screen": scan["signal"], "Probability Up": prob,
                "Rating": scan["rating"],
                "Test Acc": scan["accuracy"], "Baseline": scan["baseline"],
                "Model": scan.get("model", ""),
                "Risk": r["score"],
                "Reward Risk": rr,
                "To Support": to_sup,
                "To Resistance": to_res,
                "Buy Score": buy_score(
                    prob, scan["accuracy"], scan["baseline"],
                    r["score"], rr, to_sup,
                ),
            })
        except Exception as e:
            failures.append((sym, str(e)[:60]))
    return rows, failures


def scan_universe(stock_items, batch_size=SCAN_BATCH, max_symbols=None,
                  pause_s=0.35, on_batch=None):
    """Walk the full list in batches. Returns (df, failures, meta)."""
    items = normalize_stock_items(stock_items)
    if max_symbols is not None:
        items = items[: max(int(max_symbols), 0)]

    index_close = fetch_index()
    bundle = load_global_model(1) if global_model_available() else None
    all_rows, fail_map = [], {}
    offset = 0
    total = len(items)
    t0 = time.time()

    while offset < total:
        batch, next_offset, _, _ = slice_scan_batch(
            items, offset=offset, batch_size=batch_size,
        )
        # Re-slice from full items via offset for correct batch
        batch = items[offset:next_offset]
        rows, fails = score_batch(
            batch, index_close=index_close, global_bundle=bundle,
        )
        all_rows.extend(rows)
        for s, r in fails:
            fail_map[s] = r
        if on_batch:
            on_batch(next_offset, total, len(rows), len(fails))
        offset = next_offset
        if offset < total and pause_s > 0:
            time.sleep(pause_s)

    df = pd.DataFrame(all_rows)
    if not df.empty and "Buy Score" in df.columns:
        df = df.sort_values(
            ["Buy Score", "Probability Up"], ascending=False,
        ).reset_index(drop=True)

    meta = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_requested": total,
        "n_scored": int(len(df)),
        "n_failed": len(fail_map),
        "batch_size": int(batch_size),
        "engine": "global" if bundle is not None else "per-stock",
        "elapsed_s": round(time.time() - t0, 1),
        "source": "precomputed",
    }
    return df, list(fail_map.items()), meta


def save_rankings(df, failures, meta, directory=None):
    """Write rankings CSV + meta JSON. Returns paths."""
    directory = Path(directory or RANKINGS_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / RANKINGS_CSV
    meta_path = directory / RANKINGS_META

    out = df.copy() if df is not None else pd.DataFrame(columns=RESULT_COLUMNS)
    # Stable column order when present
    cols = [c for c in RESULT_COLUMNS if c in out.columns]
    extra = [c for c in out.columns if c not in cols]
    out = out[cols + extra] if cols else out
    out.to_csv(csv_path, index=False)

    payload = dict(meta or {})
    payload["failures_sample"] = [
        {"symbol": s, "reason": r} for s, r in (failures or [])[:50]
    ]
    payload["n_failures_listed"] = min(50, len(failures or []))
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return csv_path, meta_path


def load_rankings(directory=None, max_age_hours=DEFAULT_MAX_AGE_HOURS):
    """Load precomputed rankings if present and fresh enough.

    Returns (df, meta) or (empty DataFrame, None) when missing/stale/invalid.
    """
    directory = Path(directory or RANKINGS_DIR)
    csv_path = directory / RANKINGS_CSV
    meta_path = directory / RANKINGS_META
    if not csv_path.exists():
        return pd.DataFrame(), None

    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}

    if max_age_hours is not None and meta.get("generated_at"):
        try:
            gen = datetime.strptime(
                meta["generated_at"], "%Y-%m-%dT%H:%M:%SZ",
            ).replace(tzinfo=timezone.utc)
            age_h = (datetime.now(timezone.utc) - gen).total_seconds() / 3600.0
            if age_h > float(max_age_hours):
                meta["stale"] = True
                meta["age_hours"] = round(age_h, 1)
                # Still return data but mark stale — caller decides
        except Exception:
            pass

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return pd.DataFrame(), None

    if df.empty or "Symbol" not in df.columns:
        return pd.DataFrame(), meta or None

    if "Buy Score" in df.columns:
        df = df.sort_values(
            ["Buy Score", "Probability Up"], ascending=False,
        ).reset_index(drop=True)

    meta = meta or {}
    meta["source"] = "precomputed"
    meta["path"] = str(csv_path)
    return df, meta


def filter_rankings_to_watchlist(df, stocks: dict):
    """Keep only symbols that appear in the current watchlist; reattach names."""
    if df is None or df.empty or not stocks:
        return pd.DataFrame()
    sym_to_name = {str(s).upper(): n for n, s in stocks.items()}
    out = df[df["Symbol"].astype(str).str.upper().isin(sym_to_name)].copy()
    if out.empty:
        return out
    out["Name"] = out["Symbol"].astype(str).str.upper().map(sym_to_name)
    if "Buy Score" in out.columns:
        out = out.sort_values(
            ["Buy Score", "Probability Up"], ascending=False,
        ).reset_index(drop=True)
    return out
