"""Pure strategy/analytics logic — every function with hand-checkable math."""
import numpy as np
import pandas as pd

from model import (
    build_positions, performance_stats, tune_thresholds, rating_from_prob,
    compute_risk_score, calibration_metrics, find_support_resistance,
    compute_trade_plan, position_size, predict,
)
from data import add_features
from data import FEATURES
from test_data import synth_ohlcv


def test_build_positions_hysteresis():
    probs = np.array([0.7, 0.55, 0.35, 0.55, 0.8])
    assert build_positions(probs, 0.6, 0.4).tolist() == [1, 1, 0, 0, 1]


def test_performance_stats_hand_check():
    pos = np.array([1.0, 1, 0, 0, 1])
    rets = np.array([0.01, -0.01, 0.02, -0.02, 0.01])
    s = performance_stats(pos, rets, cost=0.001)
    expected = np.prod(1 + np.array([0.009, -0.01, -0.001, 0.0, 0.009])) - 1
    assert abs(s["total_return"] - expected) < 1e-12
    assert s["n_trades"] == 2 and s["exposure"] == 0.6
    assert abs(s["win_rate"] - 2 / 3) < 1e-12


def test_tune_thresholds_finds_planted_edge():
    rng = np.random.default_rng(1)
    probs = rng.uniform(0.2, 0.8, 400)
    rets = np.where(probs > 0.6, 0.01, -0.002) + rng.normal(0, 0.001, 400)
    entry, _ = tune_thresholds(probs, rets)
    assert entry >= 0.6


def test_rating_bands():
    cases = [(0.85, "Strong Buy"), (0.80, "Buy"), (0.65, "Buy"),
             (0.60, "Neutral"), (0.45, "Neutral"), (0.40, "Sell")]
    for p, want in cases:
        assert rating_from_prob(p) == want, p


def test_predict_signal_uses_tuned_thresholds_not_rating_bands():
    class DummyPredictor:
        def __init__(self, prob):
            self.prob = prob

        def predict_last(self, _):
            return self.prob

    class IdentityScaler:
        def transform(self, x):
            return x

    data = pd.DataFrame({feat: [0.0] for feat in FEATURES})

    assert predict(DummyPredictor(0.58), IdentityScaler(), data, thresholds=(0.55, 0.45))[0] == "BUY"
    assert predict(DummyPredictor(0.50), IdentityScaler(), data, thresholds=(0.55, 0.45))[0] == "HOLD"
    assert predict(DummyPredictor(0.40), IdentityScaler(), data, thresholds=(0.55, 0.45))[0] == "SELL"


def test_risk_score_orders_calm_vs_wild():
    rng = np.random.default_rng(0)

    def make(vol):
        idx = pd.bdate_range("2023-01-01", periods=400)
        c = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, vol, 400))), index=idx)
        return add_features(pd.DataFrame({
            "Open": c, "High": c * (1 + vol), "Low": c * (1 - vol),
            "Close": c, "Volume": np.full(400, 1e6)}, index=idx))

    calm, wild = compute_risk_score(make(0.004)), compute_risk_score(make(0.045))
    assert calm["score"] < wild["score"] and calm["level"] == "Low"


def test_calibration_metrics_separate_good_from_overconfident():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.2, 0.8, 5000)
    y_good = (rng.uniform(size=5000) < p).astype(float)
    y_over = (rng.uniform(size=5000) < 0.5 + 0.3 * (p - 0.5)).astype(float)
    assert calibration_metrics(p, y_over)["ece"] > 2 * calibration_metrics(p, y_good)["ece"]


def _swingy_stock():
    idx = pd.bdate_range("2024-01-01", periods=400)
    t = np.arange(400)
    close = pd.Series(1512.5 + 62.5 * np.sin(t / 18.0)
                      + np.random.default_rng(7).normal(0, 2, 400), index=idx)
    close.iloc[-1] = 1525.0
    return add_features(pd.DataFrame({
        "Open": close, "High": close + 4, "Low": close - 4,
        "Close": close, "Volume": np.full(400, 1e6)}, index=idx))


def test_support_resistance_bracket_price():
    sr = find_support_resistance(_swingy_stock())
    assert sr["support"] < sr["price"] < sr["resistance"]
    assert 1430 < sr["support"] < 1480 and 1545 < sr["resistance"] < 1595


def test_trade_plan_arithmetic_and_fallbacks():
    d = _swingy_stock()
    sr = find_support_resistance(d)
    plan = compute_trade_plan(d, sr["support"], sr["resistance"])
    assert plan["stop"] < plan["entry"] < plan["target"]
    rr = (plan["target"] - plan["entry"]) / (plan["entry"] - plan["stop"])
    assert abs(rr - plan["reward_risk"]) < 1e-9
    plan2 = compute_trade_plan(d, None, None)
    assert "ATR" in plan2["stop_basis"] and "ATR" in plan2["target_basis"]


def test_position_size_formula_and_caps():
    ps = position_size(1_000_000, 1.0, 1525, 1480)
    assert ps["shares"] == 222 and ps["risk_amount"] == 10_000
    capped = position_size(100_000, 2.0, 1000, 999)
    assert capped["shares"] == 100 and capped["capped_by_capital"]
    assert position_size(100_000, 1.0, 1000, 1000) is None


def test_random_benchmark_ranks_skill_high_and_noise_mid():
    from model import random_signal_benchmark
    idx = pd.bdate_range("2024-01-01", periods=250)
    rng = np.random.default_rng(0)
    prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0, 0.012, 250))), index=idx)
    fwd = prices.pct_change().shift(-1).to_numpy()
    skilled = np.clip(np.where(fwd > 0, 0.8, 0.2) + rng.normal(0, 0.05, 250), 0, 1)
    r = random_signal_benchmark(skilled, prices, idx, (0.6, 0.4), n_random=400)
    assert r is not None and r["sharpe_percentile"] > 80
    pcts = []
    for s in range(20):
        p = np.random.default_rng(1000 + s).uniform(0, 1, 250)
        res = random_signal_benchmark(p, prices, idx, (0.6, 0.4), n_random=150, seed=7)
        if res:
            pcts.append(res["sharpe_percentile"])
    assert 35 < float(np.mean(pcts)) < 65


def test_random_benchmark_matches_exposure_and_edge_cases():
    from model import random_signal_benchmark
    idx = pd.bdate_range("2024-01-01", periods=200)
    rng = np.random.default_rng(3)
    prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0, 0.012, 200))), index=idx)
    probs = rng.uniform(0, 1, 200)
    r = random_signal_benchmark(probs, prices, idx, (0.6, 0.4), n_random=100)
    assert r is not None and 0 < r["exposure_days"] < r["total_days"]
    assert len(r["rand_sharpes"]) == 100
    assert random_signal_benchmark(np.full(200, 0.5), prices, idx, (0.6, 0.4)) is None
    assert random_signal_benchmark(np.array([0.8, 0.2, 0.9]),
                                   prices.head(3), idx[:3], (0.6, 0.4)) is None
    a = random_signal_benchmark(probs, prices, idx, (0.6, 0.4), seed=42)
    b = random_signal_benchmark(probs, prices, idx, (0.6, 0.4), seed=42)
    assert a["sharpe_percentile"] == b["sharpe_percentile"]
