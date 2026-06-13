"""Feature engineering: warm-up handling, target masking, NIFTY context."""
import numpy as np
import pandas as pd

from data import add_features, FEATURES, HORIZONS


def synth_ohlcv(n=900, seed=7, drift=0.0004, vol=0.014):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-01", periods=n)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(drift, vol, n))), index=idx)
    return pd.DataFrame({
        "Open": close, "High": close * 1.01, "Low": close * 0.99,
        "Close": close,
        "Volume": rng.integers(1e5, 5e6, n).astype(float),
    }, index=idx)


def test_features_have_no_nans_after_warmup():
    d = add_features(synth_ohlcv())
    assert int(d[FEATURES].isna().sum().sum()) == 0
    assert int(np.isinf(d[FEATURES].to_numpy()).sum()) == 0


def test_target_tails_stay_nan_per_horizon():
    d = add_features(synth_ohlcv())
    for h in HORIZONS:
        assert int(d[f"Target_{h}"].isna().sum()) == h, f"horizon {h}"
    # Latest row keeps valid features for live prediction
    assert not d[FEATURES].tail(1).isna().any().any()


def test_nifty_relative_strength_recovers_planted_alpha():
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2021-01-01", periods=900)
    nifty = pd.Series(15000 * np.exp(np.cumsum(rng.normal(0.0004, 0.009, 900))), index=idx)
    stock_ret = nifty.pct_change().fillna(0).values + 0.0005 + rng.normal(0, 0.004, 900)
    close = pd.Series(100 * np.exp(np.cumsum(stock_ret)), index=idx)
    raw = pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99,
                        "Close": close, "Volume": np.full(900, 1e6)}, index=idx)
    d = add_features(raw, index_close=nifty)
    assert 0.005 < d["Rel_Str20"].mean() < 0.015  # planted alpha ~0.0005*20


def test_nifty_calendar_mismatch_is_ffilled():
    raw = synth_ohlcv()
    nifty = pd.Series(np.linspace(15000, 18000, 900), index=raw.index)
    holey = nifty.drop(nifty.sample(30, random_state=1).index)
    d = add_features(raw, index_close=holey)
    assert int(d[FEATURES].isna().sum().sum()) == 0


def test_nifty_fallback_is_neutral_and_backward_compatible():
    raw = synth_ohlcv()
    d_none = add_features(raw, index_close=None)
    ctx = d_none[["Nifty_Ret", "Nifty_Mom20", "Rel_Str5", "Rel_Str20"]]
    assert bool((ctx == 0).all().all())
    d_noarg = add_features(raw)
    assert len(d_noarg) == len(d_none)


def test_fetch_many_parses_batched_multiindex(monkeypatch_yf=None):
    """fetch_many must parse yfinance's (ticker, field) MultiIndex response,
    return empty frames for failed tickers, and never raise."""
    import data as data_mod
    idx = pd.bdate_range("2024-01-01", periods=10)
    cols = pd.MultiIndex.from_product(
        [["A.NS", "B.NS"], ["Open", "High", "Low", "Close", "Volume"]])
    vals = np.random.default_rng(0).uniform(90, 110, (10, 10))
    fake = pd.DataFrame(vals, index=idx, columns=cols)
    fake[("B.NS", "Close")] = np.nan  # B partially broken but present

    orig = data_mod.yf.download
    try:
        data_mod.yf.download = lambda *a, **k: fake
        out = data_mod.fetch_many(["A.NS", "B.NS", "MISSING.NS", "A.NS"])  # dup too
        assert set(out) == {"A.NS", "B.NS", "MISSING.NS"}
        assert len(out["A.NS"]) == 10 and "Close" in out["A.NS"].columns
        assert out["MISSING.NS"].empty  # absent ticker -> empty, no exception

        data_mod.yf.download = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("rate limited"))
        out2 = data_mod.fetch_many(["A.NS", "B.NS"])
        assert all(df.empty for df in out2.values())  # total failure -> all empty
    finally:
        data_mod.yf.download = orig
