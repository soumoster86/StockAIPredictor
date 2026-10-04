# =============================
# execution.py
# =============================
"""Execution realism: what it actually costs, and when you can actually trade.

Pure numpy/pandas (no torch) so both model.py backtests and journal.py
resolution share one definition of a fill.

Three facts the old close-to-close backtest ignored:

  1. Timing — a signal computed from today's close can only be acted on at
     the NEXT session's open. The overnight gap belongs to whatever position
     you held yesterday, not to the new signal.
  2. Costs — Indian delivery trades pay STT on both legs, stamp duty on the
     buy, exchange/SEBI fees with GST, and a DP charge per sell, on top of a
     bid-ask spread and market impact that grow as liquidity shrinks.
  3. Circuits — a stock locked at its upper band has no sellers (you can't
     buy); locked at its lower band, no buyers (you can't exit).

The central object is the *market frame*: one row per signal date t, holding
the facts for an order placed at t's close and filled at t+1's open.
"""
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Cost model — NSE equity delivery (rates as of FY2025-26)
# ---------------------------------------------------------------------------
STT_RATE = 0.001             # 0.1% on buy AND sell (delivery)
STAMP_DUTY_BUY = 0.00015     # 0.015% on buy only
EXCHANGE_TXN = 0.0000297     # NSE transaction charge, 0.00297% per side
SEBI_FEE = 0.000001          # ₹10 per crore, per side
GST_RATE = 0.18              # on brokerage + exchange + SEBI fees
BROKERAGE_RATE = 0.0         # most discount brokers charge ₹0 on delivery
DP_CHARGE_INR = 15.93        # depository charge per scrip per sell day (incl. GST)

# Non-Indian tickers: venue costs unknown, so charge a flat conservative fee.
GENERIC_FEE_PER_SIDE = 0.0005

DEFAULT_NOTIONAL = 100_000   # ₹ per trade — sets the DP charge share and impact

# Idle cash isn't idle: it can sit in a liquid fund / T-bills. Days out of
# the market earn this, and Sharpe is measured against it. ~91-day T-bill
# yield; update when rates move.
RISK_FREE_RATE = 0.06
TRADING_DAYS = 252

# ---------------------------------------------------------------------------
# Slippage model — half-spread from liquidity + square-root market impact
# ---------------------------------------------------------------------------
REF_TURNOVER = 1e9           # ₹100 Cr/day: a liquid large-cap ...
REF_HALF_SPREAD = 0.0003     # ... crosses about 3 bps of half-spread
MIN_HALF_SPREAD = 0.0002
MAX_HALF_SPREAD = 0.02       # 2%: thin small-caps / no reported volume
IMPACT_K = 1.0               # square-root law: impact ≈ k · σ · sqrt(Q / ADV)
DEFAULT_SLIPPAGE = 0.0005    # used only when the data has no Volume at all
LIQUIDITY_WINDOW = 20

# ---------------------------------------------------------------------------
# Circuit-lock detection
# ---------------------------------------------------------------------------
LOCK_RANGE_TOL = 0.001       # High and Low within 0.1% → a single-price session
LOCK_MIN_MOVE = 0.019        # ≥ ~2% (the tightest NSE band) from prior close
MAX_GAP = 0.50               # an open > 50% from prior close is a bad print


def cost_profile_for(symbol):
    """'NSE' for Indian listings (and the NIFTY index), 'GENERIC' otherwise."""
    s = str(symbol or "").upper()
    return "NSE" if s.endswith((".NS", ".BO")) or s.startswith("^NSE") else "GENERIC"


def statutory_costs(profile="NSE", notional=DEFAULT_NOTIONAL):
    """(buy_fraction, sell_fraction) of trade value paid in fees and taxes,
    before any spread or impact."""
    if profile != "NSE":
        return GENERIC_FEE_PER_SIDE, GENERIC_FEE_PER_SIDE
    gst_base = BROKERAGE_RATE + EXCHANGE_TXN + SEBI_FEE
    per_side = gst_base * (1.0 + GST_RATE)
    buy = STT_RATE + STAMP_DUTY_BUY + per_side
    sell = STT_RATE + per_side + DP_CHARGE_INR / max(float(notional), 1.0)
    return float(buy), float(sell)


def slippage(df, notional=DEFAULT_NOTIONAL):
    """Per-bar one-way slippage fraction, known at that bar's close.

    half-spread = REF_HALF_SPREAD · sqrt(REF_TURNOVER / ADV), clipped
    impact      = IMPACT_K · σ_daily · sqrt(notional / ADV)

    ADV is the rolling median traded value (Close × Volume) through the bar,
    so there is no lookahead. A bar with no reported volume is treated as
    illiquid (maximum spread)."""
    close = df["Close"].astype(float)
    if "Volume" not in df.columns:
        return pd.Series(DEFAULT_SLIPPAGE, index=df.index)

    traded = (close * df["Volume"].astype(float)).where(lambda v: v > 0)
    adv = traded.rolling(LIQUIDITY_WINDOW, min_periods=5).median()
    half_spread = (REF_HALF_SPREAD * np.sqrt(REF_TURNOVER / adv)).clip(
        MIN_HALF_SPREAD, MAX_HALF_SPREAD)

    vol = close.pct_change().rolling(LIQUIDITY_WINDOW, min_periods=5).std()
    impact = IMPACT_K * vol * np.sqrt(float(notional) / adv)

    out = (half_spread + impact.fillna(0.0)).clip(upper=MAX_HALF_SPREAD * 2)
    return out.fillna(MAX_HALF_SPREAD)


def side_costs(df, profile="NSE", notional=DEFAULT_NOTIONAL):
    """(buy_cost, sell_cost) Series: all-in one-way cost fraction per bar."""
    buy, sell = statutory_costs(profile, notional)
    slip = slippage(df, notional)
    return slip + buy, slip + sell


def locked_sessions(df):
    """(up_locked, down_locked) boolean Series.

    A session is locked when it traded at a single price (High ≈ Low) after
    moving at least the tightest circuit band from the prior close — the
    daily-bar signature of a stock frozen at its upper or lower circuit."""
    if not {"High", "Low", "Close"}.issubset(df.columns):
        false = pd.Series(False, index=df.index)
        return false, false
    close = df["Close"].astype(float)
    single_price = (df["High"] - df["Low"]).abs() <= LOCK_RANGE_TOL * close
    move = close / close.shift(1) - 1.0
    return single_price & (move >= LOCK_MIN_MOVE), single_price & (move <= -LOCK_MIN_MOVE)


def market_frame(df, index=None, profile="NSE", notional=DEFAULT_NOTIONAL):
    """Execution facts for a signal formed at row t's close, filled at t+1's open.

    Columns:
      gap        Open[t+1] / Close[t] − 1   (earned by the position held INTO t+1)
      intraday   Close[t+1] / Open[t+1] − 1 (earned by the position decided at t)
      buy_cost   one-way all-in cost of buying at Open[t+1]
      sell_cost  one-way all-in cost of selling at Open[t+1]
      can_buy    False when t+1 is locked at its upper circuit
      can_sell   False when t+1 is locked at its lower circuit

    Accepts a Close Series too (legacy): with no Open the gap is zero and the
    intraday leg is the close-to-close return. The last row has no next
    session and comes back NaN; callers mask it out with `valid_rows`."""
    if isinstance(df, pd.Series):
        df = df.to_frame("Close")
    close = df["Close"].astype(float)
    next_close = close.shift(-1)

    if "Open" in df.columns:
        next_open = df["Open"].astype(float).shift(-1)
        # A bad open print must not fabricate a huge gap: fall back to no gap.
        bad = ((next_open / close - 1.0).abs() > MAX_GAP) | ~(next_open > 0)
        next_open = next_open.where(~bad, close).where(next_close.notna())
    else:
        next_open = close.where(next_close.notna())

    up, down = locked_sessions(df)
    buy_cost, sell_cost = side_costs(df, profile, notional)

    frame = pd.DataFrame({
        "gap": next_open / close - 1.0,
        "intraday": next_close / next_open - 1.0,
        "buy_cost": buy_cost,
        "sell_cost": sell_cost,
        "can_buy": ~up.shift(-1, fill_value=False).astype(bool),
        "can_sell": ~down.shift(-1, fill_value=False).astype(bool),
    }, index=df.index)
    if index is not None:
        frame = frame.reindex(index)
        frame[["can_buy", "can_sell"]] = frame[["can_buy", "can_sell"]].fillna(True)
    return frame


def valid_rows(frame):
    """Boolean mask of rows with a usable next session."""
    return (np.isfinite(frame["gap"].to_numpy(dtype=float))
            & np.isfinite(frame["intraday"].to_numpy(dtype=float)))


# ---------------------------------------------------------------------------
# Bracket orders (entry + stop + target) — shared by the journal and the
# plan backtest so both score a trade plan identically
# ---------------------------------------------------------------------------

def session_opens(future, entry_ref=None):
    """Session opens; without an Open column assume each day opens at the
    prior close (no gap), the first at `entry_ref`."""
    if "Open" in future.columns:
        return future["Open"].astype(float)
    return future["Close"].astype(float).shift(1).fillna(entry_ref)


def simulate_bracket(prices, signal_date, stop, target, max_days=20, profile="NSE",
                     notional=DEFAULT_NOTIONAL, entry_ref=None, locks=None, costs=None):
    """Simulate a long trade planned at `signal_date`'s close.

      - Fill at the NEXT session's open (the signal close isn't tradeable).
        If that session is locked at the upper circuit, or opens already
        beyond the stop or target (the plan is void), the result is NO FILL.
      - Each later session: an open through the stop exits at the OPEN
        (a gap down fills worse than the stop); an open through the target
        exits at the open (better). Otherwise an intraday touch exits at the
        level. Both touched on one day → STOP HIT (intraday order unknown,
        score conservatively). A session locked at the lower circuit can't
        be exited — the position carries to the next day.
      - After `max_days` with neither → EXPIRED at that day's close. Not
        enough days yet → OPEN, marked to the last close.
      - `outcome_return` is net of statutory costs and slippage (liquidity
        as known at the signal date); `gross_return` is the price move alone.

    `locks` / `costs` accept precomputed locked_sessions / side_costs output
    so a backtest simulating many trades on one stock computes them once."""
    signal_date = pd.Timestamp(signal_date)
    future = prices.loc[prices.index > signal_date].head(max_days)
    empty = {"days": 0, "outcome_return": np.nan, "exit_date": None, "fill_date": None,
             "fill_price": np.nan, "exit_price": np.nan, "gross_return": np.nan}
    if future.empty:
        return {"status": "OPEN", **empty}

    opens = session_opens(future, entry_ref)
    fill = float(opens.iloc[0])
    up_locked, down_locked = locks if locks is not None else locked_sessions(prices)
    up_locked = up_locked.reindex(future.index, fill_value=False)
    down_locked = down_locked.reindex(future.index, fill_value=False)
    if bool(up_locked.iloc[0]) or not (stop < fill < target):
        return {"status": "NO FILL", **empty}

    buy_costs, sell_costs = costs if costs is not None else side_costs(prices, profile, notional)
    known = prices.index <= signal_date
    pick = known.nonzero()[0][-1] if known.any() else 0
    buy_cost, sell_cost = float(buy_costs.iloc[pick]), float(sell_costs.iloc[pick])

    def done(status, days, exit_price, exit_date):
        net = exit_price * (1.0 - sell_cost) / (fill * (1.0 + buy_cost)) - 1.0
        return {"status": status, "days": days, "outcome_return": net,
                "exit_date": exit_date, "fill_date": future.index[0], "fill_price": fill,
                "exit_price": float(exit_price), "gross_return": exit_price / fill - 1.0,
                "buy_cost": buy_cost, "sell_cost": sell_cost}

    lows, highs = future["Low"].to_numpy(float), future["High"].to_numpy(float)
    for i, dt in enumerate(future.index):
        if bool(down_locked.iloc[i]):
            continue  # frozen at the lower circuit: no buyers, can't exit today
        day_open = float(opens.iloc[i])
        if i > 0 and day_open <= stop:
            return done("STOP HIT", i + 1, day_open, dt)
        if i > 0 and day_open >= target:
            return done("TARGET HIT", i + 1, day_open, dt)
        if lows[i] <= stop:  # checked first: same-day double-touch → STOP
            return done("STOP HIT", i + 1, stop, dt)
        if highs[i] >= target:
            return done("TARGET HIT", i + 1, target, dt)

    last_close = float(future["Close"].iloc[-1])
    if len(future) >= max_days:
        return done("EXPIRED", max_days, last_close, future.index[-1])
    return done("OPEN", len(future), last_close, None)


# ---------------------------------------------------------------------------
# Return statistics
# ---------------------------------------------------------------------------

def daily_rate(annual):
    """Compounded daily equivalent of an annual rate."""
    return (1.0 + float(annual)) ** (1.0 / TRADING_DAYS) - 1.0


def equity_stats(daily_returns, rf=0.0):
    """Equity curve, total return, excess-return Sharpe and max drawdown
    for a series of daily returns."""
    r = np.asarray(daily_returns, dtype=float)
    equity = np.cumprod(1.0 + r)
    excess = r - daily_rate(rf)
    std = excess.std()
    sharpe = float(excess.mean() / std * np.sqrt(TRADING_DAYS)) if std > 0 else float("nan")
    max_dd = float((equity / np.maximum.accumulate(equity) - 1.0).min()) if len(r) else 0.0
    return {
        "total_return": float(equity[-1] - 1.0) if len(r) else 0.0,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "equity": equity,
    }
