"""Partial bars, history start, liquidity/surveillance filters, sizing cap."""
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import data as D
from execution import price_band_hint, traded_value
from model import position_size, rank_buy_candidates
from screener import load_surveillance

IST = ZoneInfo("Asia/Kolkata")
NY = ZoneInfo("America/New_York")


def bars(end="2026-10-05", n=5, close=100.0, volume=1e5):
    idx = pd.bdate_range(end=end, periods=n)
    return pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99,
                         "Close": close, "Volume": volume}, index=idx)


# ---------------------------------------------------------------------------
# Partial bars
# ---------------------------------------------------------------------------

def test_drops_todays_bar_while_nse_is_open():
    df = bars(end="2026-10-05")                     # Monday
    midday = datetime(2026, 10, 5, 11, 30, tzinfo=IST)
    out = D.drop_incomplete_bar(df, "RELIANCE.NS", now=midday)
    assert out.index[-1] == pd.Timestamp("2026-10-02")
    assert D.market_is_open("RELIANCE.NS", now=midday)


def test_keeps_todays_bar_after_close_and_yesterdays_bar_always():
    df = bars(end="2026-10-05")
    evening = datetime(2026, 10, 5, 17, 0, tzinfo=IST)
    assert len(D.drop_incomplete_bar(df, "INFY.NS", now=evening)) == len(df)
    next_morning = datetime(2026, 10, 6, 10, 0, tzinfo=IST)
    assert len(D.drop_incomplete_bar(df, "INFY.NS", now=next_morning)) == len(df)
    assert not D.market_is_open("INFY.NS", now=evening)


def test_session_rules_per_exchange():
    df = bars(end="2026-10-05")
    # 11:00 in New York is 20:30 in India: the NSE bar is final, the US one isn't.
    ny_morning = datetime(2026, 10, 5, 11, 0, tzinfo=NY)
    assert len(D.drop_incomplete_bar(df, "^NSEI", now=ny_morning)) == len(df)
    assert len(D.drop_incomplete_bar(df, "AAPL", now=ny_morning)) == len(df) - 1
    # Unknown exchange: untouched.
    assert len(D.drop_incomplete_bar(df, "VOD.L", now=ny_morning)) == len(df)
    saturday = datetime(2026, 10, 3, 11, 0, tzinfo=IST)
    assert not D.market_is_open("TCS.NS", now=saturday)


def test_data_start_defaults_to_2010():
    assert D.DATA_START <= "2012-01-01"


# ---------------------------------------------------------------------------
# Liquidity / price band
# ---------------------------------------------------------------------------

def test_traded_value_is_recent_median():
    df = bars(n=40, close=200.0, volume=50_000)       # ₹1 Cr/day
    assert abs(traded_value(df) - 1e7) < 1e-6
    assert np.isnan(traded_value(df.drop(columns="Volume")))


def test_price_band_hint_from_circuit_locks():
    df = bars(n=30, close=100.0)
    assert price_band_hint(df) is None                # never locked
    locked = df.copy()
    d = locked.index[-3]
    locked.loc[d, ["Open", "High", "Low", "Close"]] = 105.0   # +5% single-price day
    assert price_band_hint(locked) == 0.05
    big = df.copy()
    big.loc[d, ["Open", "High", "Low", "Close"]] = 120.0      # +20% band
    assert price_band_hint(big) == 0.20


def test_position_size_capped_by_liquidity():
    ps = position_size(10_000_000, 2.0, entry=100.0, stop=95.0, adv_value=1e6)
    # Risk formula: 200,000 / 5 = 40,000 shares; 2% of ₹10 lakh/day = 200 shares.
    assert ps["shares"] == 200 and ps["capped_by_liquidity"]
    free = position_size(10_000_000, 2.0, entry=100.0, stop=95.0)
    assert free["shares"] == 40_000 and not free["capped_by_liquidity"]


def _scan():
    return pd.DataFrame({
        "Symbol": ["A.NS", "B.NS", "C.NS", "D.NS"],
        "Screen": "BUY", "Probability Up": 0.7, "Risk": 4.0,
        "Test Acc": 0.6, "Baseline": 0.55,
        "Traded Value": [5.0, 0.2, 3.0, 8.0],          # ₹ Cr/day
        "Price Band": [None, None, 0.05, 0.20],
        "Buy Score": [90, 80, 70, 60],
    })


def test_rank_buy_candidates_tradability_filters():
    picks = rank_buy_candidates(_scan(), exclude_symbols=["D.NS"])
    assert picks["Symbol"].tolist() == ["A.NS"]        # B thin, C 5% band, D listed
    loose = rank_buy_candidates(_scan(), min_turnover_cr=0, exclude_tight_band=False)
    assert loose["Symbol"].tolist() == ["A.NS", "B.NS", "C.NS", "D.NS"]
    legacy = _scan().drop(columns=["Traded Value", "Price Band"])
    assert len(rank_buy_candidates(legacy)) == 4       # old files: filters skipped


def test_rank_buy_candidates_all_bands_unknown():
    scan = _scan()
    scan["Price Band"] = [None] * 4                     # object column of Nones
    assert len(rank_buy_candidates(scan, min_turnover_cr=0)) == 4


def test_load_surveillance_normalizes(tmp_path):
    p = tmp_path / "surveillance.csv"
    p.write_text("﻿Symbol,Reason\nXYZ,ASM Stage 2\nABC.NS,GSM\n", encoding="utf-8")
    assert load_surveillance(p) == {"XYZ.NS": "ASM Stage 2", "ABC.NS": "GSM"}
    assert load_surveillance(tmp_path / "missing.csv") == {}
