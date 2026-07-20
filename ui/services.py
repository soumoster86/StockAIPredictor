"""Cached data / model loaders and watchlist scan helpers."""
from pathlib import Path

import pandas as pd
import streamlit as st

from data import fetch_data, fetch_many, add_features, fetch_index
from model import (
    train_model, walk_forward, multi_horizon_forecast,
    global_model_available, load_global_model,
    predict_with_global, multi_horizon_global,
)
from screener import (
    SCAN_BATCH as _SCAN_BATCH,
    normalize_stock_items, slice_scan_batch, merge_scan_frames,
    score_batch, load_rankings, filter_rankings_to_watchlist,
    DEFAULT_MAX_AGE_HOURS,
)

ROOT = Path(__file__).resolve().parent.parent
# Full NSE-style dump is the default app watchlist (user request).
STOCKS_UNIVERSE_FILE = ROOT / "stocks_universe.csv"
# Optional shorter liquid list (Nifty-50-style) — kept for offline / upload use.
STOCKS_LIQUID_FILE = ROOT / "stocks.csv"
# Prefer full universe; fall back to stocks.csv if universe file is missing.
STOCKS_FILE = (
    STOCKS_UNIVERSE_FILE if STOCKS_UNIVERSE_FILE.exists() else STOCKS_LIQUID_FILE
)
DEFAULT_STOCKS = {"Reliance Industries": "RELIANCE.NS", "Infosys": "INFY.NS"}
# One screener batch size. Full universe is walked with offset + merge
# (Scan next batch) so Yahoo/Cloud limits stay manageable.
SCAN_BATCH = _SCAN_BATCH
# Back-compat alias used by older UI captions / tests
SCAN_MAX = SCAN_BATCH

# Session keys for multi-batch screener state
SCAN_FP = "scan_fp"
SCAN_DF = "scan_df"
SCAN_FAILS = "scan_failures"
SCAN_OFFSET = "scan_offset"
SCAN_TOTAL = "scan_total"
SCAN_ACTIVE = "scan_active"
SCAN_SOURCE = "scan_source"  # "live" | "precomputed"
SCAN_ASOF = "scan_asof"


@st.cache_data(ttl=3600)
def load_stock_list(file_or_path):
    # utf-8-sig handles Excel/Windows BOM so "Name" is not "\ufeffName"
    df = pd.read_csv(file_or_path, encoding="utf-8-sig")
    df.columns = [str(c).replace("\ufeff", "").strip().lower() for c in df.columns]
    if not {"name", "symbol"}.issubset(df.columns):
        raise ValueError("CSV must have 'Name' and 'Symbol' columns.")
    df = df.dropna(subset=["name", "symbol"])
    df["name"] = df["name"].astype(str).str.strip()
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df = df.drop_duplicates(subset=["symbol"]).drop_duplicates(subset=["name"])
    return dict(zip(df["name"], df["symbol"]))


@st.cache_data(ttl=3600, max_entries=20, show_spinner="Downloading price data...")
def get_data(symbol):
    return fetch_data(symbol)


@st.cache_data(ttl=3600, max_entries=1, show_spinner=False)
def get_index():
    """NIFTY Close series for market-context features (None if unavailable)."""
    return fetch_index()


@st.cache_data(ttl=3600, max_entries=2, show_spinner="Downloading watchlist data (one batched request)...")
def get_data_batch(symbols):
    return fetch_many(list(symbols))


@st.cache_resource(ttl=3600, max_entries=4, show_spinner="Training model...")
def get_trained(symbol, model_type, calibrate, use_global):
    data = add_features(get_data(symbol), index_close=get_index())
    if use_global:
        bundle = load_global_model(1)
        if bundle is not None:
            try:
                return (data,) + predict_with_global(data, bundle, calibrate)
            except ValueError:
                pass
    return (data,) + train_model(data, model_type, calibrate)


@st.cache_data(ttl=3600, max_entries=4, show_spinner="Training one model per horizon (1/3/5/10/20 days)...")
def get_horizons(symbol, model_type, use_global):
    data = add_features(get_data(symbol), index_close=get_index())
    if use_global and global_model_available():
        rows = multi_horizon_global(data)
        if any(v is not None for v in rows.values()):
            per_stock = None
            ordered = []
            for h, row in rows.items():
                if row is not None:
                    ordered.append(row)
                else:
                    if per_stock is None:
                        per_stock = multi_horizon_forecast(data, model_type)
                    match = per_stock[per_stock['Horizon'].str.startswith(str(h))]
                    if not match.empty:
                        ordered.append(match.iloc[0].to_dict())
            return pd.DataFrame(ordered)
    return multi_horizon_forecast(data, model_type)


@st.cache_data(ttl=3600, max_entries=2, show_spinner="Running walk-forward validation (trains one model per fold)...")
def run_walk_forward(symbol, model_type, calibrate):
    data = add_features(get_data(symbol), index_close=get_index())
    return walk_forward(data, model_type, calibrate=calibrate)


def cap_scan_items(stock_items, limit=SCAN_BATCH):
    """First `limit` unique symbols (legacy helper)."""
    return normalize_stock_items(stock_items)[: max(int(limit), 0)]


def watchlist_fingerprint(stocks: dict) -> tuple:
    """Stable id so we reset scan state when the watchlist changes."""
    return tuple(sorted((str(s).upper() for s in stocks.values())))


def reset_scan_session(stocks: dict) -> None:
    """Clear accumulated screener state for this watchlist."""
    items = normalize_stock_items(list(stocks.items()))
    st.session_state[SCAN_FP] = watchlist_fingerprint(stocks)
    st.session_state[SCAN_DF] = None
    st.session_state[SCAN_FAILS] = []
    st.session_state[SCAN_OFFSET] = 0
    st.session_state[SCAN_TOTAL] = len(items)
    st.session_state[SCAN_ACTIVE] = False
    st.session_state[SCAN_SOURCE] = None
    st.session_state[SCAN_ASOF] = None


def ensure_scan_session(stocks: dict) -> None:
    """Init or invalidate session when the stock list changes."""
    fp = watchlist_fingerprint(stocks)
    if st.session_state.get(SCAN_FP) != fp:
        reset_scan_session(stocks)


def scan_progress(stocks: dict) -> dict:
    """Snapshot of multi-batch progress for the UI."""
    ensure_scan_session(stocks)
    total = int(st.session_state.get(SCAN_TOTAL) or 0)
    offset = int(st.session_state.get(SCAN_OFFSET) or 0)
    df = st.session_state.get(SCAN_DF)
    n_ok = 0 if df is None or getattr(df, "empty", True) else len(df)
    fails = st.session_state.get(SCAN_FAILS) or []
    source = st.session_state.get(SCAN_SOURCE)
    # Precomputed seeds mark offset == total so complete=True
    complete = total > 0 and offset >= total
    if source == "precomputed" and n_ok > 0:
        complete = True
        offset = max(offset, total)
    return {
        "total": total,
        "offset": offset,
        "attempted": offset if source != "precomputed" else total,
        "succeeded": n_ok,
        "failed": len(fails),
        "remaining": 0 if source == "precomputed" else max(total - offset, 0),
        "complete": complete,
        "active": bool(st.session_state.get(SCAN_ACTIVE)),
        "batch_size": SCAN_BATCH,
        "source": source,
        "asof": st.session_state.get(SCAN_ASOF),
    }


@st.cache_data(ttl=3600, max_entries=48, show_spinner="Scanning batch (download + score)...")
def run_scan_batch(stock_items, offset=0, batch_size=SCAN_BATCH):
    """Scan one batch starting at `offset`. Cached per (list, offset, size).

    Returns (df, failures, next_offset, total, complete).
    """
    batch, next_offset, total, complete = slice_scan_batch(
        list(stock_items), offset=offset, batch_size=batch_size,
    )
    if not batch:
        return pd.DataFrame(), [], next_offset, total, True

    # Use cached fetch helpers from this module
    price_map = get_data_batch(tuple(sym for _, sym in batch))
    index_close = get_index()
    bundle = load_global_model(1) if global_model_available() else None
    rows, failures = score_batch(
        batch, index_close=index_close, global_bundle=bundle, price_map=price_map,
    )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            ["Buy Score", "Probability Up"], ascending=False,
        ).reset_index(drop=True)
    return df, failures, next_offset, total, complete


def run_scan(stock_items):
    """Legacy: scan the first SCAN_BATCH names only (no session merge)."""
    df, failures, _, _, _ = run_scan_batch(
        tuple(stock_items), offset=0, batch_size=SCAN_BATCH,
    )
    return df, failures


def advance_scan_session(stocks: dict, n_batches: int = 1) -> dict:
    """Run 1..n batches and merge into session. Returns progress dict."""
    ensure_scan_session(stocks)
    items = tuple(normalize_stock_items(list(stocks.items())))
    n_batches = max(1, min(int(n_batches), 10))
    st.session_state[SCAN_ACTIVE] = True
    st.session_state[SCAN_SOURCE] = "live"
    st.session_state[SCAN_ASOF] = None
    st.session_state[SCAN_TOTAL] = len(items)

    for _ in range(n_batches):
        offset = int(st.session_state.get(SCAN_OFFSET) or 0)
        if offset >= len(items):
            break
        df, fails, next_offset, total, complete = run_scan_batch(
            items, offset=offset, batch_size=SCAN_BATCH,
        )
        st.session_state[SCAN_DF] = merge_scan_frames(
            st.session_state.get(SCAN_DF), df,
        )
        prev_fails = st.session_state.get(SCAN_FAILS) or []
        fail_map = {s: r for s, r in prev_fails}
        for s, r in fails:
            fail_map[s] = r
        st.session_state[SCAN_FAILS] = list(fail_map.items())
        st.session_state[SCAN_OFFSET] = next_offset
        st.session_state[SCAN_TOTAL] = total
        if complete:
            break

    return scan_progress(stocks)


def get_scan_results():
    """Accumulated scan DataFrame (may be empty)."""
    df = st.session_state.get(SCAN_DF)
    if df is None:
        return pd.DataFrame()
    return df


@st.cache_data(ttl=600, max_entries=2, show_spinner=False)
def _cached_precomputed(max_age_hours=DEFAULT_MAX_AGE_HOURS):
    return load_rankings(max_age_hours=max_age_hours)


def precomputed_status(stocks: dict, max_age_hours=DEFAULT_MAX_AGE_HOURS):
    """Info about offline rankings file for the UI."""
    df, meta = _cached_precomputed(max_age_hours=max_age_hours)
    if meta is None and (df is None or df.empty):
        return {"available": False, "df": pd.DataFrame(), "meta": None}
    filtered = filter_rankings_to_watchlist(df, stocks)
    return {
        "available": not filtered.empty,
        "df": filtered,
        "meta": meta,
        "n_file": 0 if df is None else len(df),
        "n_watchlist": len(filtered),
        "stale": bool((meta or {}).get("stale")),
        "asof": (meta or {}).get("generated_at"),
        "age_hours": (meta or {}).get("age_hours"),
        "engine": (meta or {}).get("engine"),
    }


def seed_session_from_precomputed(stocks: dict, allow_stale: bool = True) -> bool:
    """Load offline rankings into session. Returns True if seeded."""
    ensure_scan_session(stocks)
    # Allow stale for manual "Load precomputed" — auto-seed skips stale
    max_age = None if allow_stale else DEFAULT_MAX_AGE_HOURS
    df, meta = load_rankings(max_age_hours=max_age)
    if meta and meta.get("stale") and not allow_stale:
        return False
    filtered = filter_rankings_to_watchlist(df, stocks)
    if filtered.empty:
        return False

    items = normalize_stock_items(list(stocks.items()))
    st.session_state[SCAN_DF] = filtered
    st.session_state[SCAN_FAILS] = []
    st.session_state[SCAN_TOTAL] = len(items)
    st.session_state[SCAN_OFFSET] = len(items)  # treat as fully covered
    st.session_state[SCAN_ACTIVE] = True
    st.session_state[SCAN_SOURCE] = "precomputed"
    st.session_state[SCAN_ASOF] = (meta or {}).get("generated_at")
    st.session_state[SCAN_FP] = watchlist_fingerprint(stocks)
    return True


def maybe_autoseed_precomputed(stocks: dict) -> bool:
    """If session empty, seed from fresh precomputed file once."""
    ensure_scan_session(stocks)
    if st.session_state.get(SCAN_ACTIVE) or int(st.session_state.get(SCAN_OFFSET) or 0) > 0:
        return False
    df = st.session_state.get(SCAN_DF)
    if df is not None and not getattr(df, "empty", True):
        return False
    return seed_session_from_precomputed(stocks, allow_stale=False)

