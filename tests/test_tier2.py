"""Risk-free rate, plan-based backtest, and the screener portfolio backtest."""
import numpy as np
import pandas as pd

from data import add_features
from execution import RISK_FREE_RATE, daily_rate, equity_stats, simulate_bracket
from model import _trade_daily_returns, performance_stats, plan_backtest
from portfolio import select_portfolio, series_stats, simulate_portfolio, trailing_accuracy


def synthetic(n=420, seed=0, drift=0.0005, volume=2e6, start="2024-01-01"):
    """Featured OHLCV frame from a calm random walk."""
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + drift + rng.normal(0, 0.01, n))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * 1.006
    low = np.minimum(open_, close) * 0.994
    idx = pd.bdate_range(start, periods=n)
    raw = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close,
                        "Volume": volume}, index=idx)
    return add_features(raw)


# ---------------------------------------------------------------------------
# Risk-free rate
# ---------------------------------------------------------------------------

def test_daily_rate_compounds_to_annual():
    assert abs((1 + daily_rate(0.06)) ** 252 - 1.06) < 1e-12


def test_cash_earns_risk_free_and_sharpe_is_excess():
    n = 252
    flat = performance_stats(np.zeros(n), np.full(n, 0.01), rf=0.06)
    assert abs(flat["total_return"] - 0.06) < 1e-9          # all cash → T-bill return
    # 6%/yr of daily noise-free returns has zero excess return → Sharpe ≈ 0
    r = np.full(n, daily_rate(0.06)) + np.tile([1e-4, -1e-4], n // 2)
    assert abs(equity_stats(r, 0.06)["sharpe"]) < 1e-6
    assert equity_stats(r, 0.0)["sharpe"] > 5               # vs zero, looks great


def test_legacy_performance_stats_unchanged_without_rf():
    s = performance_stats(np.array([1.0, 0, 0]), np.array([0.01, 0.02, 0.03]), cost=0.0)
    assert abs(s["total_return"] - 0.01) < 1e-12


# ---------------------------------------------------------------------------
# Plan backtest
# ---------------------------------------------------------------------------

def test_trade_daily_returns_compound_to_net_trade_return():
    d = synthetic()
    res = {"fill_date": d.index[300], "exit_date": d.index[306], "fill_price": 101.0,
           "exit_price": 104.0, "buy_cost": 0.002, "sell_cost": 0.003}
    r = _trade_daily_returns(d["Close"], res)
    assert len(r) == 7
    assert abs(np.prod(1 + r) - 104 * 0.997 / (101 * 1.002)) < 1e-12
    same_day = _trade_daily_returns(d["Close"], {**res, "exit_date": d.index[300]})
    assert abs(same_day.iloc[0] - (104 * 0.997 / (101 * 1.002) - 1)) < 1e-12


def test_plan_backtest_trades_the_recommended_plan():
    d = synthetic(drift=0.004, seed=3)          # steady uptrend: targets get hit
    test_index = d.index[-100:]
    probs = np.full(100, 0.3)
    probs[5] = 0.8                              # one BUY signal
    stats, trades, equity = plan_backtest(probs, d, test_index, 0.6)
    assert len(trades) == 1
    t = trades.iloc[0]
    sig = test_index[5]
    assert t["signal_date"] == sig
    assert t["fill_date"] == d.index[d.index.get_loc(sig) + 1]
    assert t["fill_price"] == d["Open"].iloc[d.index.get_loc(sig) + 1]   # next open
    expected = simulate_bracket(d, sig, t["stop"], t["target"])
    assert t["status"] == expected["status"]
    assert abs(t["outcome_return"] - expected["outcome_return"]) < 1e-12
    # Days outside the trade earned cash interest; inside, the trade's path.
    cash_days = len(equity) - int(round(stats["exposure"] * len(equity)))
    want = (1 + t["outcome_return"]) * (1 + daily_rate(RISK_FREE_RATE)) ** cash_days - 1
    assert abs(stats["total_return"] - want) < 1e-9


def test_plan_backtest_takes_one_trade_at_a_time():
    d = synthetic(drift=0.0, seed=5)
    test_index = d.index[-80:]
    stats, trades, _ = plan_backtest(np.full(80, 0.9), d, test_index, 0.6)
    fills = pd.to_datetime(trades["fill_date"])
    exits = pd.to_datetime(trades["exit_date"])
    assert (fills.iloc[1:].to_numpy() > exits.iloc[:-1].to_numpy()).all()   # no overlap
    assert stats["n_trades"] + stats["n_open"] == len(trades)


# ---------------------------------------------------------------------------
# Portfolio backtest
# ---------------------------------------------------------------------------

def test_select_portfolio_hold_buffer():
    ranked = ["A", "B", "C", "D", "E", "F"]
    # C still ranks inside top_n × 2 = 4 → kept; F (rank 6) is dropped.
    assert select_portfolio(ranked, ["C", "F"], top_n=2) == ["C", "A"]
    assert select_portfolio(ranked, [], top_n=3) == ["A", "B", "C"]


def test_trailing_accuracy_uses_only_resolved_predictions():
    d = synthetic()
    probs = pd.DataFrame({"X": pd.Series(0.9, index=d.index)})   # always predicts up
    acc, base = trailing_accuracy({"X": d}, probs, d.index[0], horizon=1)
    y = d["Target_1"]
    k = 100
    known = y.iloc[:k].dropna()       # row k−1's 1-day outcome is known at day k's close
    assert abs(acc["X"].iloc[k] - (known == 1).mean()) < 1e-12
    assert acc["X"].iloc[:60].isna().all()                        # too few resolved yet


def _universe():
    frames = {s: synthetic(seed=i, volume=v) for i, (s, v) in
              enumerate([("A", 2e6), ("B", 50), ("C", 2e6), ("D", 2e6)])}
    idx = frames["A"].index
    probs = pd.DataFrame({"A": 0.70, "B": 0.90, "C": 0.40, "D": 0.65}, index=idx)
    return frames, probs, idx[-40]


def test_portfolio_picks_liquid_top_names_and_pays_costs():
    frames, probs, start = _universe()
    equity, trades, stats = simulate_portfolio(frames, probs, start, top_n=2,
                                               rebalance_days=5, max_risk=10)
    held = set(trades.loc[trades["status"] == "OPEN", "symbol"])
    assert held == {"A", "D"}            # B is illiquid (₹ thousands/day), C below 0.55
    assert "B" not in set(trades["symbol"]) and "C" not in set(trades["symbol"])
    assert (trades["shares"] % 1 == 0).all()
    assert stats["costs_paid"] > 0
    first = trades.groupby("symbol")["entry_date"].min()
    assert (first == equity.index[1]).all()        # bought at the open after day-0 signal


def test_portfolio_upper_circuit_delays_the_buy():
    frames, probs, start = _universe()
    d = frames["A"].copy()
    day1 = d.index[d.index.get_loc(start) + 1]
    px = float(d.loc[d.index[d.index.get_loc(start)], "Close"]) * 1.05
    d.loc[day1, ["Open", "High", "Low", "Close"]] = px     # locked +5% session
    frames["A"] = d
    _, trades, _ = simulate_portfolio(frames, probs, start, top_n=2, rebalance_days=5,
                                      max_risk=10)
    entry_a = trades.loc[trades["symbol"] == "A", "entry_date"].min()
    assert entry_a > day1                                  # retried the next session


def test_series_stats_cagr():
    idx = pd.bdate_range("2024-01-01", periods=253)
    eq = pd.Series(np.linspace(100, 110, 253), index=idx)
    s = series_stats(eq, 100)
    assert abs(s["total_return"] - 0.10) < 1e-12
    assert abs(s["cagr"] - (1.10 ** (252 / 253) - 1)) < 1e-12
