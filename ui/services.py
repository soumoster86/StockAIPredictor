"""Cached data / model loaders and watchlist scan helpers."""
from pathlib import Path

import pandas as pd
import streamlit as st

from data import fetch_data, fetch_many, add_features, fetch_index
from model import (
    train_model, walk_forward, multi_horizon_forecast,
    compute_risk_score, find_support_resistance, compute_trade_plan,
    quick_scan, quick_scan_global, buy_score,
    global_model_available, load_global_model,
    predict_with_global, multi_horizon_global,
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
# Hard cap for *screener* runs only (selectbox still loads the full list).
# Full-universe scans in one go blow Yahoo/Cloud limits; raise carefully.
SCAN_MAX = 80


@st.cache_data(ttl=3600)
def load_stock_list(file_or_path):
    df = pd.read_csv(file_or_path)
    df.columns = [c.strip().lower() for c in df.columns]
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


def cap_scan_items(stock_items, limit=SCAN_MAX):
    """Preserve order, drop duplicate symbols, hard-cap length for cloud safety."""
    seen, out = set(), []
    for name, sym in stock_items:
        if sym in seen:
            continue
        seen.add(sym)
        out.append((name, sym))
        if len(out) >= limit:
            break
    return out


@st.cache_data(ttl=3600, max_entries=2, show_spinner=False)
def run_scan(stock_items):
    """Scan the watchlist (capped at SCAN_MAX).

    Prefers the pre-trained global model when artifacts exist — inference only,
    no per-stock tree fit. Falls back to quick_scan otherwise.
    """
    stock_items = cap_scan_items(list(stock_items), SCAN_MAX)
    rows, failures = [], []
    seen = set()
    batch = get_data_batch(tuple(sym for _, sym in stock_items))
    global_bundle = load_global_model(1) if global_model_available() else None
    mode = "global" if global_bundle is not None else "per-stock tree"
    progress = st.progress(0.0, text=f"Scanning watchlist ({mode})...")
    for i, (name, sym) in enumerate(stock_items):
        progress.progress(
            (i + 1) / max(len(stock_items), 1),
            text=f"Scanning {sym} ({mode})...",
        )
        if sym in seen:
            continue
        seen.add(sym)
        try:
            raw = batch.get(sym, pd.DataFrame())
            if raw.empty or len(raw) < 400:
                failures.append((sym, "no/short data"))
                continue
            d = add_features(raw, index_close=get_index())
            if global_bundle is not None:
                scan = quick_scan_global(d, bundle=global_bundle)
                if scan is None:
                    scan = quick_scan(d)  # short history fallback
            else:
                scan = quick_scan(d)
            if scan is None:
                failures.append((sym, "too little history"))
                continue
            r = compute_risk_score(d)
            s = find_support_resistance(d)
            plan = compute_trade_plan(d, s.get("support"), s.get("resistance"))
            price = float(d['Close'].iloc[-1])
            prev = float(d['Close'].iloc[-2])
            to_sup = (price / s['support'] - 1) if s.get('support') else None
            to_res = (s['resistance'] / price - 1) if s.get('resistance') else None
            rr = plan.get("reward_risk")
            prob = scan['probability']
            rows.append({
                "Name": name, "Symbol": sym,
                "Price": price, "Day": price / prev - 1,
                "Screen": scan['signal'], "Probability Up": prob,
                "Rating": scan['rating'],
                "Test Acc": scan['accuracy'], "Baseline": scan['baseline'],
                "Model": scan.get('model', ''),
                "Risk": r['score'],
                "Reward Risk": rr,
                "To Support": to_sup,
                "To Resistance": to_res,
                "Buy Score": buy_score(
                    prob, scan['accuracy'], scan['baseline'],
                    r['score'], rr, to_sup,
                ),
            })
        except Exception as e:
            failures.append((sym, str(e)[:60]))
    progress.empty()
    df = pd.DataFrame(rows)
    if not df.empty:
        # Best long candidates first; non-BUY rows still listed by score
        df = df.sort_values(
            ["Buy Score", "Probability Up"], ascending=False,
        ).reset_index(drop=True)
    return df, failures
