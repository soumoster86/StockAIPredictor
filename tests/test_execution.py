"""Execution realism: next-open fills, Indian costs, slippage, circuit locks."""
import numpy as np
import pandas as pd

from execution import (
    DP_CHARGE_INR,
    GENERIC_FEE_PER_SIDE,
    MAX_HALF_SPREAD,
    cost_profile_for,
    locked_sessions,
    market_frame,
    slippage,
    statutory_costs,
    valid_rows,
)
from journal import resolve_entry
from model import backtest, build_positions, performance_stats


def ohlcv(rows, volume=1e6, start="2026-05-04"):
    """rows: (Open, High, Low, Close)."""
    idx = pd.bdate_range(start, periods=len(rows))
    df = pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=idx)
    df["Volume"] = volume
    return df


# ---------------------------------------------------------------------------
# Costs
# ---------------------------------------------------------------------------

def test_cost_profile_detection():
    assert cost_profile_for("RELIANCE.NS") == "NSE"
    assert cost_profile_for("500325.BO") == "NSE"
    assert cost_profile_for("^NSEI") == "NSE"
    assert cost_profile_for("AAPL") == "GENERIC"
    assert cost_profile_for(None) == "GENERIC"


def test_nse_statutory_costs_hand_check():
    buy, sell = statutory_costs("NSE", notional=100_000)
    fees = (0.0000297 + 0.000001) * 1.18          # exchange + SEBI, with GST
    assert abs(buy - (0.001 + 0.00015 + fees)) < 1e-12
    assert abs(sell - (0.001 + fees + DP_CHARGE_INR / 100_000)) < 1e-12
    # ≈ 0.24% round trip before slippage — over 2x the old flat 0.1%/change
    assert 0.0023 < buy + sell < 0.0025
    assert statutory_costs("GENERIC") == (GENERIC_FEE_PER_SIDE, GENERIC_FEE_PER_SIDE)


def test_dp_charge_hurts_small_trades_more():
    _, sell_small = statutory_costs("NSE", notional=10_000)
    _, sell_big = statutory_costs("NSE", notional=1_000_000)
    assert sell_small > sell_big


def test_slippage_grows_as_liquidity_shrinks():
    rows = [(100, 101, 99, 100 + (i % 3)) for i in range(40)]
    liquid = slippage(ohlcv(rows, volume=5e7)).iloc[-1]     # ~₹500 Cr/day
    thin = slippage(ohlcv(rows, volume=2e3)).iloc[-1]       # ~₹2 lakh/day
    assert liquid < 0.001 < thin
    dead = slippage(ohlcv(rows, volume=0)).iloc[-1]         # no trades reported
    assert dead == MAX_HALF_SPREAD


# ---------------------------------------------------------------------------
# Market frame / circuits
# ---------------------------------------------------------------------------

def test_market_frame_splits_gap_and_session():
    df = ohlcv([(100, 101, 99, 100), (110, 116, 109, 115), (114, 115, 100, 103)])
    m = market_frame(df)
    assert abs(m["gap"].iloc[0] - 0.10) < 1e-12              # 100 → open 110
    assert abs(m["intraday"].iloc[0] - (115 / 110 - 1)) < 1e-12
    assert abs(m["gap"].iloc[1] - (114 / 115 - 1)) < 1e-12
    assert valid_rows(m).tolist() == [True, True, False]     # last row: no next day


def test_market_frame_series_is_legacy_close_to_close():
    close = pd.Series([100.0, 102.0, 101.0], index=pd.bdate_range("2026-01-05", periods=3))
    m = market_frame(close)
    assert m["gap"].iloc[0] == 0.0
    assert abs(m["intraday"].iloc[0] - 0.02) < 1e-12


def test_bad_open_print_does_not_fabricate_gap():
    df = ohlcv([(100, 101, 99, 100), (5, 103, 99, 102)])     # open of 5 is a bad tick
    assert market_frame(df)["gap"].iloc[0] == 0.0


def test_locked_sessions_detects_circuits():
    df = ohlcv([
        (100, 101, 99, 100),
        (105, 105, 105, 105),   # +5%, single price → upper circuit
        (105, 107, 103, 104),   # normal day
        (99, 99, 99, 99),       # −4.8%, single price → lower circuit
        (99, 99, 99, 99),       # flat single-price day: not a circuit move
    ])
    up, down = locked_sessions(df)
    assert up.tolist() == [False, True, False, False, False]
    assert down.tolist() == [False, False, False, True, False]
    m = market_frame(df)
    assert m["can_buy"].tolist() == [False, True, True, True, True]
    assert m["can_sell"].tolist() == [True, True, False, True, True]


# ---------------------------------------------------------------------------
# Strategy engine
# ---------------------------------------------------------------------------

def test_build_positions_respects_blocked_orders():
    probs = np.array([0.7, 0.6, 0.6, 0.3, 0.3])
    can_buy = np.array([False, True, True, True, True])
    can_sell = np.array([True, True, True, False, True])
    pos = build_positions(probs, 0.65, 0.4, can_buy, can_sell)
    # buy blocked day 0, intent still long → fills day 1; sell blocked day 3
    assert pos.tolist() == [0, 1, 1, 1, 0]


def test_next_open_performance_hand_check():
    pos = np.array([1.0, 1.0, 0.0])
    gap = np.array([0.05, 0.02, -0.03])
    intraday = np.array([0.01, -0.01, 0.04])
    s = performance_stats(pos, intraday, cost=0.002, gap=gap, sell_cost=0.003)
    # day0: missed the 5% gap (wasn't in yet), earn session, pay buy cost
    # day1: held through gap and session
    # day2: exit at open — eat the −3% gap, miss the session, pay sell cost
    expected = (1 + 0.01 - 0.002) * (1.02 * 0.99) * (1 - 0.03 - 0.003) - 1
    assert abs(s["total_return"] - expected) < 1e-12
    assert s["n_trades"] == 1
    assert abs(s["total_costs"] - 0.005) < 1e-12


def test_backtest_no_longer_credits_overnight_gaps_to_new_signals():
    """All the move happens overnight. A model that 'knows' tomorrow's
    close-to-close return made money under the old same-close fill; with a
    next-open fill, entering after the signal misses the gap it predicted."""
    n = 120
    gaps = np.where(np.arange(n) % 2 == 0, 0.02, -0.02)   # up, down, up, ...
    closes = 100 * np.cumprod(1 + gaps)
    rows = [(c, c, c, c) for c in closes]   # open == close: zero intraday move
    df = ohlcv(rows, volume=1e7)
    # Single-price bars would look like circuit locks; widen the ranges.
    df["High"] *= 1.01
    df["Low"] *= 0.99

    fwd = df["Close"].pct_change().shift(-1)
    probs = np.where(fwd > 0, 0.9, 0.1)[:-1]
    idx = df.index[:-1]

    legacy, _, _ = backtest(probs, df["Close"], idx, (0.6, 0.4))
    realistic, _, _ = backtest(probs, df, idx, (0.6, 0.4))
    assert legacy["total_return"] > 0.5        # look-ahead-style fantasy
    # Filled a session late, the position sits through exactly the down gaps.
    assert realistic["total_return"] < -0.5
    assert realistic["round_trip_cost"] > 0.0023


# ---------------------------------------------------------------------------
# Journal fills
# ---------------------------------------------------------------------------

REC = dict(signal_date="2026-05-01", symbol="TEST.NS", signal="BUY",
           entry=1000.0, stop=950.0, target=1100.0)


def test_journal_fills_at_next_open():
    r = resolve_entry(REC, ohlcv([(1010, 1020, 1000, 1015)] * 3))
    assert r["fill_price"] == 1010 and r["status"] == "OPEN"


def test_journal_gap_down_through_stop_exits_at_open():
    r = resolve_entry(REC, ohlcv([(1000, 1010, 990, 1000), (920, 930, 900, 910)]))
    assert r["status"] == "STOP HIT" and r["exit_price"] == 920   # not 950
    assert r["gross_return"] < 950 / 1000 - 1


def test_journal_gap_up_through_target_exits_at_open():
    r = resolve_entry(REC, ohlcv([(1000, 1010, 990, 1000), (1150, 1160, 1140, 1155)]))
    assert r["status"] == "TARGET HIT" and r["exit_price"] == 1150


def test_journal_no_fill_on_upper_circuit_or_void_plan():
    locked = ohlcv([(1050, 1050, 1050, 1050)], start="2026-05-04")
    prior = ohlcv([(1000, 1005, 995, 1000)], start="2026-05-01")
    assert resolve_entry(REC, pd.concat([prior, locked]))["status"] == "NO FILL"
    gapped = ohlcv([(940, 960, 930, 950)])                # opens below the stop
    assert resolve_entry(REC, gapped)["status"] == "NO FILL"


def test_journal_cannot_exit_on_lower_circuit():
    rows = [(1000, 1010, 990, 1000),
            (950, 950, 950, 950),      # −5% locked: stop "touched" but no buyers
            (930, 940, 925, 935)]      # first chance to sell: gap-open exit
    r = resolve_entry(REC, ohlcv(rows))
    assert r["status"] == "STOP HIT" and r["days"] == 3 and r["exit_price"] == 930
