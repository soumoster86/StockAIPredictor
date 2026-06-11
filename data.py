# =============================
# data.py
# =============================
import numpy as np
import pandas as pd
import yfinance as yf

# Prediction horizons (trading days)
HORIZONS = [1, 3, 5, 10, 20]
NOISE_THRESHOLD = 0.002  # 1-day "meaningful move" threshold; scaled by sqrt(h)

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
]


def fetch_data(symbol):
    """Download daily OHLCV data. Returns an empty DataFrame on any failure."""
    try:
        data = yf.download(symbol, start="2020-01-01", auto_adjust=True, progress=False)
    except Exception:
        return pd.DataFrame()

    if data is None or data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data


def add_features(data):
    data = data.copy()
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
