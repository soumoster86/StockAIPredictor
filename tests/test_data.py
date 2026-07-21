"""Feature engineering: warm-up handling, target masking, NIFTY context."""
import numpy as np
import pandas as pd

from data import FEATURES, HORIZONS, add_features


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
        # The last h rows have no future -> always NaN (usable for prediction).
        # Interior rows may also be NaN now (ambiguous-move dead-band), so we
        # assert on the tail specifically rather than the total NaN count.
        assert bool(d[f"Target_{h}"].tail(h).isna().all()), f"horizon {h} tail"
        # The row immediately before the tail has a future -> must be labeled.
        assert not bool(pd.isna(d[f"Target_{h}"].iloc[-(h + 1)])), f"horizon {h} pre-tail"
    # Latest row keeps valid features for live prediction
    assert not d[FEATURES].tail(1).isna().any().any()


def test_labels_are_binary_and_have_both_classes():
    d = add_features(synth_ohlcv())
    for h in HORIZONS:
        vals = d[f"Target_{h}"].dropna().unique()
        assert set(vals).issubset({0.0, 1.0}), f"horizon {h} not binary"
        assert len(vals) == 2, f"horizon {h} is single-class"


def test_deadband_drops_ambiguous_interior_rows():
    """The dead-band must remove some interior (has-future) rows beyond the
    h no-future tail, and must shrink as the band width goes to zero."""
    import data as data_mod

    d = add_features(synth_ohlcv())
    interior_nans = int(d["Target_1"].iloc[:-1].isna().sum())
    assert interior_nans > 0, "dead-band dropped no interior rows"

    orig = data_mod.LABEL_DEADBAND_K
    try:
        data_mod.LABEL_DEADBAND_K = 0.0
        d0 = add_features(synth_ohlcv())
        assert int(d0["Target_1"].iloc[:-1].isna().sum()) == 0
    finally:
        data_mod.LABEL_DEADBAND_K = orig


def test_threshold_scales_with_stock_volatility():
    """The up-label threshold must rise with the stock's volatility: the
    smallest forward move that still earns a '1' should be larger for a
    high-vol stock than a calm one."""
    calm = add_features(synth_ohlcv(vol=0.008, seed=3))
    wild = add_features(synth_ohlcv(vol=0.030, seed=3))

    def min_up_move(d):
        fwd = d["Close"].pct_change().shift(-1)
        return float(fwd[d["Target_1"] == 1.0].min())

    assert min_up_move(wild) > min_up_move(calm)


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


def test_clean_ohlcv_neutralizes_bad_tick_keeps_recovery():
    from data import clean_ohlcv
    idx = pd.bdate_range("2024-01-01", periods=7)
    c = pd.Series([100, 101, 100, 500, 102, 103, 101], index=idx, dtype=float)
    raw = pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99,
                        "Close": c, "Volume": np.full(7, 1e6)}, index=idx)
    cleaned = clean_ohlcv(raw)
    assert cleaned["Close"].iloc[3] != 500          # spike neutralized
    assert cleaned["Close"].iloc[4] == 102.0        # recovery preserved


def test_clean_ohlcv_drops_nonpositive_and_repairs_bars():
    from data import clean_ohlcv
    idx = pd.bdate_range("2024-01-01", periods=5)
    c = pd.Series([100, 0, 101, np.nan, 102], index=idx, dtype=float)
    raw = pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c,
                        "Volume": np.full(5, 1e6)}, index=idx)
    assert len(clean_ohlcv(raw)) == 3               # 0 and NaN rows dropped

    idx2 = pd.bdate_range("2024-01-01", periods=2)
    bad = pd.DataFrame({"Open": [100, 100], "High": [101, 99],
                        "Low": [99, 101], "Close": [100, 100],
                        "Volume": [1e6, 1e6]}, index=idx2)
    fixed = clean_ohlcv(bad)
    assert (fixed["High"] >= fixed["Low"]).all()    # inverted bar repaired


def test_clean_ohlcv_preserves_normal_volatility():
    from data import clean_ohlcv
    idx = pd.bdate_range("2024-01-01", periods=7)
    normal = [100, 108, 99, 105, 110, 102, 107]     # real ±8% days
    c = pd.Series(normal, index=idx, dtype=float)
    raw = pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99,
                        "Close": c, "Volume": np.full(7, 1e6)}, index=idx)
    assert clean_ohlcv(raw)["Close"].tolist() == [float(x) for x in normal]


def test_add_features_survives_injected_spike():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2021-01-01", periods=700)
    c = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.014, 700))), index=idx)
    c.iloc[300] *= 3.0                              # 200% bad spike
    raw = pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c,
                        "Volume": rng.integers(1e5, 5e6, 700).astype(float)}, index=idx)
    d = add_features(raw)
    assert int(d[FEATURES].isna().sum().sum()) == 0
    assert d["Return"].abs().max() < 0.5            # spike didn't survive into features
