# =============================
# portfolio.py
# =============================
"""Portfolio-level backtest of the screener — the product as people use it.

The per-stock backtests answer "does the model time THIS stock?". The
screener's real promise is different: "each week, these are the best long
candidates in the universe". This module tests that promise honestly:

  1. Time split. A pooled model is fitted only on days before a cut; the
     simulation starts after the cut plus a purge gap, so no training label
     peeks into the test period. (The deployed global model is refit on ALL
     history, so it can never be backtested fairly — we fit our own.)
  2. Same ranking as the screener. At each rebalance close, every stock is
     scored point-in-time: probability > entry threshold, risk ≤ cap, model
     accuracy ≥ baseline (from predictions whose outcomes were already
     known), ranked by buy_score() with the same R:R / support inputs.
  3. Real trading. Equal-weight top N, integer shares, orders at the next
     open, NSE costs and liquidity slippage, circuit-locked sessions
     blocked (retried next day), a liquidity floor on traded value, a hold
     buffer so names aren't churned for small rank changes, and cash
     earning the risk-free rate.
  4. A real benchmark: the Nifty 500 / Nifty 50, not the same stock's
     buy-and-hold.

Known bias: the universe is today's listed stocks (survivorship), which
flatters every result. The output says so.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data import FEATURES
from execution import (
    RISK_FREE_RATE,
    TIGHT_BAND,
    TRADING_DAYS,
    cost_profile_for,
    daily_rate,
    equity_stats,
    locked_sessions,
    price_band_hint,
    side_costs,
)
from model import (
    DEFAULT_THRESHOLDS,
    SCREEN_REVIEW_DAYS,
    SCREEN_TOP_PCT,
    buy_score,
    compute_risk_score,
    compute_trade_plan,
    find_support_resistance,
    make_fast_predictor,
    make_predictor,
    pool_training_data,
)

DEFAULT_TOP_N = 10
DEFAULT_REBALANCE_DAYS = SCREEN_REVIEW_DAYS
DEFAULT_CAPITAL = 1_000_000
DEFAULT_MIN_TURNOVER = 1e7     # ₹1 Cr/day median traded value
DEFAULT_MAX_RISK = 8.0         # same cap as rank_buy_candidates
DEFAULT_MIN_PROB = DEFAULT_THRESHOLDS[0]
DEFAULT_MIN_QUANTILE = 1.0 - SCREEN_TOP_PCT   # the app's relative shortlist rule
HOLD_BUFFER = 2                # keep a holding while it ranks inside top N × this
MIN_ACCURACY_ROWS = 60         # resolved predictions before accuracy counts
MIN_HISTORY = 260              # rows before a stock can be scored (1y of S/R)

SURVIVORSHIP_NOTE = (
    "The universe is today's listed stocks. Companies that were delisted or "
    "collapsed during the test period are missing, which flatters the result."
)


# ---------------------------------------------------------------------------
# Honest model fit
# ---------------------------------------------------------------------------

def time_split_fit(frames, horizon=1, train_frac=0.6, model_type="fast",
                   max_train_rows=None, seed=0):
    """Fit a pooled model on the earliest `train_frac` of trading days.

    Returns (predictor, scaler, info). info["test_start"] is the first day
    the simulation may trade: after the cut plus a purge of 2·horizon + 5
    calendar days, so labels of the last training rows (which look `horizon`
    trading days ahead) end before testing begins. `max_train_rows`
    randomly subsamples the training rows to bound memory and time."""
    target_col = f"Target_{horizon}"
    X, y, day_ids, n_stocks = pool_training_data(frames, target_col)
    if X is None or len(X) < 500:
        raise ValueError("Not enough pooled history to fit a model.")

    unique_days = np.unique(day_ids)
    cut_i = min(max(int(len(unique_days) * train_frac), 1), len(unique_days) - 1)
    cut_day = int(unique_days[cut_i - 1])
    train = day_ids <= cut_day
    if max_train_rows and train.sum() > max_train_rows:
        keep = np.random.default_rng(seed).choice(
            np.flatnonzero(train), size=int(max_train_rows), replace=False)
        train = np.zeros(len(train), bool)
        train[np.sort(keep)] = True

    scaler = StandardScaler().fit(X[train])
    predictor = make_fast_predictor() if model_type == "fast" else make_predictor(model_type)
    predictor.fit(scaler.transform(X[train]), y[train], int(train.sum()))

    epoch = pd.Timestamp("1970-01-01")
    purge = 2 * horizon + 5
    info = {
        "cut_date": epoch + pd.Timedelta(days=cut_day),
        "test_start": epoch + pd.Timedelta(days=cut_day + purge + 1),
        "n_train_rows": int(train.sum()),
        "n_stocks": int(n_stocks),
        "horizon": int(horizon),
        "model": getattr(predictor, "name", model_type),
    }
    return predictor, scaler, info


def build_slim_frames(symbols, index_close, horizons=(1,), batch_size=60,
                      min_rows=400, log=print):
    """Fetch + feature-engineer every symbol in batches, keeping only the
    columns the backtest needs as float32 — the full universe since 2010
    otherwise needs several GB of RAM."""
    import time

    from data import add_features, fetch_many
    keep = (["Open", "High", "Low", "Close", "Volume"] + list(FEATURES)
            + [f"Target_{h}" for h in horizons])
    frames = {}
    for i in range(0, len(symbols), batch_size):
        chunk = symbols[i:i + batch_size]
        try:
            batch = fetch_many(chunk)
        except Exception as e:  # one bad batch shouldn't sink the run
            log(f"  batch {i}: {str(e)[:60]}")
            continue
        for sym in chunk:
            raw = batch.get(sym)
            if raw is None or raw.empty or len(raw) < min_rows:
                continue
            try:
                frames[sym] = add_features(raw, index_close=index_close)[keep].astype("float32")
            except Exception:
                pass
        log(f"  {i + len(chunk)}/{len(symbols)} fetched, {len(frames)} usable")
        time.sleep(0.4)
    return frames


def score_frames(frames, predictor, scaler):
    """Probability for every row of every stock → DataFrame dates × symbols."""
    cols = {}
    for sym, d in frames.items():
        if d is None or d.empty:
            continue
        p = predictor.predict_all(scaler.transform(d[FEATURES].values))
        cols[sym] = pd.Series(np.asarray(p, dtype=float), index=d.index)
    return pd.DataFrame(cols).sort_index()


def trailing_accuracy(frames, probs, start, horizon=1):
    """Point-in-time out-of-sample accuracy and majority baseline per stock.

    At date d, only predictions made on or after `start` whose `horizon`-day
    outcome was already known by d count. Fewer than MIN_ACCURACY_ROWS
    resolved predictions → NaN (unknown)."""
    acc, base = {}, {}
    for sym, d in frames.items():
        if sym not in probs.columns:
            continue
        y = d[f"Target_{horizon}"]
        p = probs[sym].reindex(d.index)
        ok = (d.index >= start) & y.notna().to_numpy() & p.notna().to_numpy()
        correct = pd.Series(np.where(ok, ((p > 0.5) == (y == 1)).astype(float), 0.0), d.index)
        ups = pd.Series(np.where(ok, (y == 1).astype(float), 0.0), d.index)
        n = pd.Series(ok.astype(float), d.index)
        # Known at d: rows up to d − horizon sessions.
        n_k = n.cumsum().shift(horizon)
        a = correct.cumsum().shift(horizon) / n_k
        up_rate = ups.cumsum().shift(horizon) / n_k
        enough = n_k >= MIN_ACCURACY_ROWS
        acc[sym] = a.where(enough)
        base[sym] = np.maximum(up_rate, 1 - up_rate).where(enough)
    return pd.DataFrame(acc).sort_index(), pd.DataFrame(base).sort_index()


# ---------------------------------------------------------------------------
# Screener replica
# ---------------------------------------------------------------------------

def rank_candidates(frames, d, probs_row, adv_row, acc_row, base_row, held=(),
                    top_n=DEFAULT_TOP_N, min_prob=DEFAULT_MIN_PROB,
                    max_risk=DEFAULT_MAX_RISK, min_turnover=DEFAULT_MIN_TURNOVER,
                    min_quantile=None, require_edge=True):
    """Screener ranking at date d using only data known at d's close.

    Entry is absolute (probability > `min_prob`, as the app's screener) or,
    with `min_quantile`, relative: probability in the top (1 − q) of that
    day's liquid universe — robust to the label's base rate drifting away
    from 50%. Returns symbols sorted by Buy Score (best first)."""
    liquid = adv_row.reindex(probs_row.index) >= min_turnover
    if min_quantile is not None:
        pool_probs = probs_row[liquid]
        if pool_probs.empty:
            return []
        min_prob = float(pool_probs.quantile(min_quantile))
        eligible = probs_row[(probs_row >= min_prob) & liquid]
    else:
        eligible = probs_row[(probs_row > min_prob) & liquid]
    eligible = eligible.sort_values(ascending=False)
    pool = list(dict.fromkeys(list(eligible.index[:3 * top_n])
                              + [h for h in held if h in eligible.index]))
    scored = []
    for sym in pool:
        acc, base = acc_row.get(sym, np.nan), base_row.get(sym, np.nan)
        if require_edge and np.isfinite(acc) and np.isfinite(base) and acc < base:
            continue  # require_edge, as rank_buy_candidates does
        hist = frames[sym].loc[:d]
        if len(hist) < MIN_HISTORY:
            continue
        risk = compute_risk_score(hist)["score"]
        if risk > max_risk:
            continue
        band = price_band_hint(hist)
        if band is not None and band <= TIGHT_BAND:
            continue  # surveillance-style 2%/5% band, as the screener excludes
        sr = find_support_resistance(hist)
        plan = compute_trade_plan(hist, sr["support"], sr["resistance"])
        price = float(hist["Close"].iloc[-1])
        to_sup = (price / sr["support"] - 1) if sr["support"] else None
        score = buy_score(float(eligible[sym]),
                          acc if np.isfinite(acc) else None,
                          base if np.isfinite(base) else None,
                          risk, plan["reward_risk"], to_sup)
        scored.append((score, sym))
    return [sym for _, sym in sorted(scored, reverse=True)]


def select_portfolio(ranked, held, top_n=DEFAULT_TOP_N, buffer=HOLD_BUFFER):
    """Keep holdings that still rank inside top_n × buffer; fill the rest
    with the best new names."""
    keep = [h for h in held if h in ranked[: top_n * buffer]][:top_n]
    new = [s for s in ranked if s not in keep][: top_n - len(keep)]
    return keep + new


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def _panel(frames, col, calendar):
    return pd.DataFrame({s: d[col] for s, d in frames.items()}).reindex(calendar)


def simulate_portfolio(frames, probs, start, end=None, capital=DEFAULT_CAPITAL,
                       top_n=DEFAULT_TOP_N, rebalance_days=DEFAULT_REBALANCE_DAYS,
                       min_prob=DEFAULT_MIN_PROB, max_risk=DEFAULT_MAX_RISK,
                       min_turnover=DEFAULT_MIN_TURNOVER, horizon=1,
                       rf=RISK_FREE_RATE, buffer=HOLD_BUFFER, progress=None,
                       min_quantile=None, require_edge=True):
    """Run the screener as a portfolio from `start` to `end`.

    Returns (equity Series, trades DataFrame, stats dict)."""
    frames = {s: d for s, d in frames.items() if s in probs.columns}
    calendar = probs.index[(probs.index >= pd.Timestamp(start))]
    if end is not None:
        calendar = calendar[calendar <= pd.Timestamp(end)]
    if len(calendar) < rebalance_days + 2:
        raise ValueError("Test window too short to simulate.")

    opens = _panel(frames, "Open", calendar)
    closes = _panel(frames, "Close", calendar)
    last_close = closes.ffill()
    traded = (_panel(frames, "Close", probs.index) * _panel(frames, "Volume", probs.index))
    adv = traded.where(traded > 0).rolling(20, min_periods=5).median().reindex(calendar)
    notional = capital / top_n
    up, down, buy_c, sell_c = {}, {}, {}, {}
    for s, d in frames.items():
        u, dn = locked_sessions(d)
        b, sl = side_costs(d, cost_profile_for(s), notional)
        up[s], down[s], buy_c[s], sell_c[s] = u, dn, b, sl
    up = pd.DataFrame(up).reindex(calendar, fill_value=False).astype(bool)
    down = pd.DataFrame(down).reindex(calendar, fill_value=False).astype(bool)
    buy_c = pd.DataFrame(buy_c).reindex(calendar).ffill()
    sell_c = pd.DataFrame(sell_c).reindex(calendar).ffill()
    acc, base = trailing_accuracy(frames, probs, calendar[0], horizon)
    acc, base = acc.reindex(calendar), base.reindex(calendar)
    p_cal = probs.reindex(calendar)

    rf_d = daily_rate(rf)
    cash = float(capital)
    shares, lots = {}, {}            # sym -> shares; sym -> (entry date, cost basis/share)
    pending_buys, pending_sells = {}, set()
    trades, equity, holdings_n, costs_paid, traded_value = [], [], [], 0.0, 0.0

    for k, d in enumerate(calendar):
        prev = calendar[k - 1] if k else d
        # 1) Orders from the last close execute at today's open.
        for s in list(pending_sells):
            o = opens.at[d, s]
            if not np.isfinite(o) or down.at[d, s]:
                continue                     # no print / lower circuit: retry tomorrow
            sc = float(sell_c.at[prev, s])
            proceeds = shares[s] * o
            cash += proceeds * (1 - sc)
            costs_paid += proceeds * sc
            traded_value += proceeds
            entry_d, basis = lots.pop(s)
            trades.append({"symbol": s, "entry_date": entry_d, "exit_date": d,
                           "entry_price": basis, "exit_price": o * (1 - sc),
                           "shares": shares[s], "net_return": o * (1 - sc) / basis - 1,
                           "status": "CLOSED"})
            del shares[s]
            pending_sells.discard(s)
        for s, target_value in list(pending_buys.items()):
            o = opens.at[d, s]
            if not np.isfinite(o) or up.at[d, s]:
                continue                     # no print / upper circuit: retry tomorrow
            bc = float(buy_c.at[prev, s])
            n = int(min(target_value, cash) // (o * (1 + bc)))
            if n > 0:
                spend = n * o
                cash -= spend * (1 + bc)
                costs_paid += spend * bc
                traded_value += spend
                shares[s] = n
                lots[s] = (d, o * (1 + bc))
            del pending_buys[s]

        # 2) Cash earns the risk-free rate; positions marked at the close.
        if k:
            cash *= 1 + rf_d
        value = cash + sum(n * last_close.at[d, s] for s, n in shares.items())
        equity.append(value)
        holdings_n.append(len(shares))

        # 3) Rebalance at the close: tomorrow's orders.
        if k % rebalance_days == 0 and k < len(calendar) - 1:
            ranked = rank_candidates(
                frames, d, p_cal.loc[d].dropna(), adv.loc[d], acc.loc[d], base.loc[d],
                held=list(shares), top_n=top_n, min_prob=min_prob,
                max_risk=max_risk, min_turnover=min_turnover,
                min_quantile=min_quantile, require_edge=require_edge)
            target = select_portfolio(ranked, list(shares), top_n, buffer)
            pending_sells = {s for s in shares if s not in target}
            pending_buys = {s: value / top_n for s in target if s not in shares}
            if progress:
                progress(k, len(calendar), d, len(shares))

    for s, n in shares.items():   # still held: mark to the last close
        entry_d, basis = lots[s]
        px = float(last_close.iloc[-1][s])
        trades.append({"symbol": s, "entry_date": entry_d, "exit_date": None,
                       "entry_price": basis, "exit_price": px, "shares": n,
                       "net_return": px / basis - 1, "status": "OPEN"})

    equity = pd.Series(equity, index=calendar, name="Screener portfolio")
    trades = pd.DataFrame(trades)
    stats = series_stats(equity, capital, rf)
    years = len(calendar) / TRADING_DAYS
    avg_equity = float(equity.mean())
    closed = trades[trades["status"] == "CLOSED"] if len(trades) else trades
    stats.update({
        "n_trades": int(len(closed)),
        "hit_rate": float((closed["net_return"] > 0).mean()) if len(closed) else float("nan"),
        "avg_trade_return": float(closed["net_return"].mean()) if len(closed) else float("nan"),
        "avg_holdings": float(np.mean(holdings_n)),
        "costs_paid": float(costs_paid),
        "costs_pct_per_year": float(costs_paid / avg_equity / years) if years else 0.0,
        "turnover_per_year": float(traded_value / avg_equity / years) if years else 0.0,
    })
    return equity, trades, stats


def series_stats(equity, start_value=None, rf=RISK_FREE_RATE):
    """CAGR, volatility, excess Sharpe and drawdown of a daily value series."""
    equity = equity.dropna()
    base = float(start_value) if start_value else float(equity.iloc[0])
    rets = equity.pct_change().fillna(equity.iloc[0] / base - 1).to_numpy()
    eq = equity_stats(rets, rf)
    years = len(equity) / TRADING_DAYS
    total = float(equity.iloc[-1] / base - 1)
    return {
        "total_return": total,
        "cagr": float((1 + total) ** (1 / years) - 1) if years > 0 and total > -1 else float("nan"),
        "volatility": float(np.std(rets) * math.sqrt(TRADING_DAYS)),
        "sharpe": eq["sharpe"],
        "max_drawdown": eq["max_drawdown"],
        "start": str(equity.index[0].date()),
        "end": str(equity.index[-1].date()),
    }


def benchmark_curve(close, calendar, start_value):
    """Buy-and-hold value of a benchmark, rebased to `start_value` on the
    first calendar day."""
    c = close.reindex(calendar).ffill().bfill()
    return c / c.iloc[0] * start_value
