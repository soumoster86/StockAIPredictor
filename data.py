# =============================
# data.py
# =============================
import numpy as np
import pandas as pd
import yfinance as yf

# Prediction horizons (trading days)
HORIZONS = [1, 3, 5, 10, 20]
NOISE_THRESHOLD = 0.002  # 1-day "meaningful move" threshold; scaled by sqrt(h)
INDEX_SYMBOL = "^NSEI"   # NIFTY 50 — market context for Indian stocks

# Single source of truth for model features. All are scale-free.
FEATURES = [
    # Trend
    'Close_MA20', 'MA20_MA50', 'MACD_hist',
    # Momentum
    'Return', 'Mom5', 'Mom10', 'Mom20', 'RSI',
    # Volatility (regime information)
    'Vol20', 'ATR_pct',
    # Volume
    'Vol_ratio',
    # Market context (NIFTY) — single names are heavily index-driven; these
    # let the model separate "this stock is weak" from "everything fell"
    'Nifty_Ret', 'Nifty_Mom20', 'Rel_Str5', 'Rel_Str20',
]


def _download_with_retry(retries=2, backoff=0.8, **kwargs):
    """yf.download with retry on exceptions (rate limits, transient network).
    Empty results are NOT retried — an empty frame usually means an invalid
    symbol, and retrying would just make typos feel slow."""
    import time
    for attempt in range(retries + 1):
        try:
            return yf.download(start="2020-01-01", auto_adjust=True,
                               progress=False, **kwargs)
        except Exception:
            if attempt == retries:
                return None
            time.sleep(backoff * (attempt + 1))
    return None


def fetch_data(symbol):
    """Download daily OHLCV data for one symbol. Empty DataFrame on failure."""
    data = _download_with_retry(tickers=symbol)
    if data is None or data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data


def fetch_many(symbols):
    """Download daily OHLCV for many symbols in ONE batched request instead
    of one request per symbol — the difference between 2 and 46 hits on a
    rate-limited shared cloud IP. Returns {symbol: DataFrame}; symbols that
    fail come back as empty DataFrames, never exceptions."""
    symbols = list(dict.fromkeys(symbols))  # preserve order, drop dups
    if not symbols:
        return {}
    if len(symbols) == 1:
        return {symbols[0]: fetch_data(symbols[0])}

    raw = _download_with_retry(tickers=symbols, group_by="ticker", threads=True)
    if raw is None or raw.empty:
        return {s: pd.DataFrame() for s in symbols}

    out = {}
    for s in symbols:
        try:
            df = raw[s].dropna(how="all")
        except KeyError:  # ticker entirely absent from the response
            df = pd.DataFrame()
        out[s] = df if df is not None and not df.empty else pd.DataFrame()
    return out


def fetch_index(symbol=INDEX_SYMBOL):
    """Close series of the market index, or None if it can't be fetched.
    Callers treat None as 'context unavailable': the app keeps working on
    neutral values, so an index rate-limit can never break anything."""
    df = fetch_data(symbol)
    if df is None or df.empty or 'Close' not in df.columns:
        return None
    return df['Close']


MAX_DAILY_MOVE = 0.50   # |close-to-close| beyond this is treated as a bad tick


def clean_ohlcv(data, max_move=MAX_DAILY_MOVE):
    """Defend the model from garbage input before any feature is computed.

    Yahoo data occasionally contains bad ticks, unadjusted splits, or
    zero/!positive prices that would silently distort every downstream
    feature. This gate:
      - drops rows with non-positive or missing OHLC prices,
      - repairs obviously broken bars where High/Low don't bracket the
        Open/Close (a data error, not a real bar),
      - neutralizes implausible single-day jumps (> max_move, e.g. a 90%
        gap from a bad print or an unadjusted split) by forward-filling
        that day's close, so one bad number can't poison MA/RSI/returns.

    Returns a cleaned copy. Conservative by design: it only touches values
    that are not credibly real, never normal volatility."""
    data = data.copy()

    price_cols = [c for c in ("Open", "High", "Low", "Close") if c in data.columns]
    # Non-positive or missing prices are unusable -> drop the row.
    valid = (data[price_cols] > 0).all(axis=1) & data[price_cols].notna().all(axis=1)
    data = data[valid]
    if data.empty:
        return data

    # Repair bars where High/Low fail to bracket Open/Close (pure data errors).
    if {"High", "Low", "Open", "Close"}.issubset(data.columns):
        data["High"] = data[["High", "Open", "Close"]].max(axis=1)
        data["Low"] = data[["Low", "Open", "Close"]].min(axis=1)

    # Neutralize implausible prices. Comparing only to the prior close
    # double-flags a spike AND its recovery; instead compare each close to a
    # robust local reference (centered rolling median), so only the genuine
    # outlier is replaced. A >50% deviation on a single name is almost always
    # a bad tick or unadjusted corporate action; real moves pass untouched.
    if "Close" in data.columns and len(data) > 3:
        c = data["Close"].astype(float)
        ref = c.rolling(5, center=True, min_periods=1).median()
        bad = (c / ref - 1).abs() > max_move
        if bad.any():
            fixed = c.copy()
            fixed[bad] = np.nan
            data["Close"] = fixed.ffill().bfill()

    if "Volume" in data.columns:
        data["Volume"] = data["Volume"].fillna(0).clip(lower=0)

    return data


def add_features(data, index_close=None):
    data = clean_ohlcv(data)
    close = data['Close']
    high = data['High']
    low = data['Low']

    # ----- Trend -----
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    data['Close_MA20'] = close / ma20 - 1
    data['MA20_MA50'] = ma20 / ma50 - 1

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    data['MACD_hist'] = (macd - macd_signal) / close

    # ----- Momentum -----
    data['Return'] = close.pct_change()
    data['Mom5'] = close.pct_change(5)
    data['Mom10'] = close.pct_change(10)
    data['Mom20'] = close.pct_change(20)

    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # ----- Volatility -----
    data['Vol20'] = data['Return'].rolling(20).std()

    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    data['ATR_pct'] = true_range.rolling(14).mean() / close

    # ----- Volume -----
    volume = data['Volume'].fillna(0)
    vol_ma = volume.rolling(20).mean().replace(0, np.nan)
    data['Vol_ratio'] = ((volume / vol_ma) - 1).clip(-1, 5).fillna(0)

    # ----- Market context (NIFTY) -----
    # Same-day values only — the index close is known the moment the stock
    # closes, so there is no lookahead. Dates align to the stock's calendar
    # with forward-fill, which also handles non-NSE tickers (different
    # holidays) cleanly.
    if index_close is not None and len(index_close) > 0:
        idx = index_close.reindex(data.index).ffill()
        data['Nifty_Ret'] = idx.pct_change()
        data['Nifty_Mom20'] = idx.pct_change(20)
        data['Rel_Str5'] = data['Mom5'] - idx.pct_change(5)
        data['Rel_Str20'] = data['Mom20'] - idx.pct_change(20)
    else:
        # Neutral fallback: a zero-variance column scales to zeros, so the
        # model simply learns nothing from these features instead of crashing.
        for col in ('Nifty_Ret', 'Nifty_Mom20', 'Rel_Str5', 'Rel_Str20'):
            data[col] = 0.0

    # ----- Multi-horizon targets -----
    # Target_h = 1 if the h-day-forward return exceeds a noise threshold that
    # scales with sqrt(h) (volatility grows with the square root of time).
    # The last h rows have no future to look at: they stay NaN — never 0 —
    # so they can be excluded from training but still used for prediction.
    for h in HORIZONS:
        fwd = close.pct_change(h).shift(-h)
        thr = NOISE_THRESHOLD * np.sqrt(h)
        data[f'Target_{h}'] = np.where(fwd.notna(), (fwd > thr).astype(float), np.nan)

    data['Target'] = data['Target_1']  # backward-compatible alias

    # Drop only the indicator warm-up period; keep tail rows whose *targets*
    # are NaN — their features are valid and needed for live prediction.
    return data.dropna(subset=FEATURES)
