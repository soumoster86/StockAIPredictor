# =============================
# journal.py
# =============================
"""Signal journal: log each signal the app produces, then score it against
what the market actually did. Backtests look backward; this is the app's
forward test.

Storage is one CSV **per authenticated user** under `journals/`, so multi-user
deploys do not share or clobber each other's forward-tests. Files are
human-readable, editable, and easy to download from the Journal tab.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
JOURNAL_DIR = ROOT / "journals"
LEGACY_JOURNAL_FILE = ROOT / "journal.csv"  # pre-per-user; read-only migrate
# Back-compat alias used by older docs / imports
JOURNAL_FILE = LEGACY_JOURNAL_FILE
MAX_HOLD_DAYS = 20  # trading days before an unresolved BUY plan expires

COLUMNS = [
    "signal_date", "symbol", "name", "model_type", "signal", "probability",
    "rating", "entry", "stop", "target", "reward_risk", "risk_score", "logged_at",
]


def safe_username(user):
    """Filesystem-safe username fragment (letters, digits, . _ -)."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(user or "anonymous").strip())
    s = s.strip("._-")[:64]
    return s or "anonymous"


def journal_path_for(user=None):
    """Return the journal CSV path for `user`.

    - With a username: `journals/<user>.csv` (created on first write).
    - Without: legacy `journal.csv` at the repo root (single-user / tests).
    """
    if user is None or str(user).strip() == "":
        return LEGACY_JOURNAL_FILE
    return JOURNAL_DIR / f"{safe_username(user)}.csv"


def _maybe_migrate_legacy(path):
    """If the per-user file is missing but legacy journal.csv exists and the
    user path is under journals/, seed an empty dir only — we do **not**
    auto-copy the shared legacy file (it may belong to another person)."""
    path = Path(path)
    if path.parent == JOURNAL_DIR:
        JOURNAL_DIR.mkdir(parents=True, exist_ok=True)


def load_journal(path=None, user=None):
    if path is None:
        path = journal_path_for(user)
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path)
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[COLUMNS]


def append_signal(record, path=None, user=None):
    """Append one signal. Deduped on (signal_date, symbol, model_type):
    logging the same stock twice in a day updates nothing and returns False."""
    if path is None:
        path = journal_path_for(user)
    path = Path(path)
    _maybe_migrate_legacy(path)
    df = load_journal(path)
    dup = (
        (df["signal_date"] == record["signal_date"])
        & (df["symbol"] == record["symbol"])
        & (df["model_type"] == record["model_type"])
    )
    if dup.any():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.concat([df, pd.DataFrame([record])[COLUMNS]], ignore_index=True)
    df.to_csv(path, index=False)
    return True


def resolve_entry(rec, prices, max_days=MAX_HOLD_DAYS):
    """Score one journal entry against subsequent price action.

    BUY: walk forward from the day after the signal. If the day's Low
    touches the stop -> STOP HIT; if the High touches the target ->
    TARGET HIT. If both happen the same day, assume STOP HIT (we can't
    know intraday order, so score conservatively). After `max_days` with
    neither -> EXPIRED at that day's close. Not enough days yet -> OPEN,
    with the unrealized return so far.

    SELL / HOLD: no stop/target to resolve — just record the forward
    return after `max_days` (CLOSED) or so far (OPEN). For SELL, a
    negative forward return means exiting was the right call."""
    signal_date = pd.Timestamp(rec["signal_date"])
    entry = float(rec["entry"])
    future = prices.loc[prices.index > signal_date].head(max_days)

    if future.empty:
        return {"status": "OPEN", "days": 0, "outcome_return": np.nan, "exit_date": None}

    if rec["signal"] == "BUY":
        stop, target = float(rec["stop"]), float(rec["target"])
        for i, (dt, row) in enumerate(future.iterrows(), start=1):
            hit_stop = float(row["Low"]) <= stop
            hit_target = float(row["High"]) >= target
            if hit_stop:  # checked first: same-day double-touch scores as STOP
                return {"status": "STOP HIT", "days": i,
                        "outcome_return": stop / entry - 1.0, "exit_date": dt}
            if hit_target:
                return {"status": "TARGET HIT", "days": i,
                        "outcome_return": target / entry - 1.0, "exit_date": dt}
        last_close = float(future["Close"].iloc[-1])
        if len(future) >= max_days:
            return {"status": "EXPIRED", "days": max_days,
                    "outcome_return": last_close / entry - 1.0,
                    "exit_date": future.index[-1]}
        return {"status": "OPEN", "days": len(future),
                "outcome_return": last_close / entry - 1.0, "exit_date": None}

    # SELL / HOLD — informational forward return only
    last_close = float(future["Close"].iloc[-1])
    status = "CLOSED" if len(future) >= max_days else "OPEN"
    return {"status": status, "days": len(future),
            "outcome_return": last_close / entry - 1.0,
            "exit_date": future.index[-1] if status == "CLOSED" else None}


def resolve_journal(journal_df, price_fetcher, max_days=MAX_HOLD_DAYS):
    """Resolve every entry. `price_fetcher(symbol)` must return an OHLC
    DataFrame (the app passes its cached data loader). Symbols that fail
    to fetch are marked NO DATA rather than crashing the tab."""
    if journal_df.empty:
        return journal_df.assign(status=[], days=[], outcome_return=[])

    results = []
    price_cache = {}
    for _, rec in journal_df.iterrows():
        sym = rec["symbol"]
        if sym not in price_cache:
            try:
                price_cache[sym] = price_fetcher(sym)
            except Exception:
                price_cache[sym] = pd.DataFrame()
        prices = price_cache[sym]
        if prices is None or prices.empty:
            results.append({"status": "NO DATA", "days": 0,
                            "outcome_return": np.nan, "exit_date": None})
        else:
            results.append(resolve_entry(rec, prices, max_days))

    out = journal_df.copy().reset_index(drop=True)
    res = pd.DataFrame(results)
    out[["status", "days", "outcome_return"]] = res[["status", "days", "outcome_return"]]
    return out


def scorecard(resolved_df):
    """Aggregate honesty report over resolved BUY signals."""
    buys = resolved_df[resolved_df["signal"] == "BUY"]
    done = buys[buys["status"].isin(["TARGET HIT", "STOP HIT", "EXPIRED"])]

    out = {
        "n_signals": int(len(resolved_df)),
        "n_buys": int(len(buys)),
        "n_resolved": int(len(done)),
        "n_open": int((buys["status"] == "OPEN").sum()),
    }
    if len(done) == 0:
        out.update({"target_rate": np.nan, "stop_rate": np.nan,
                    "win_rate": np.nan, "avg_return": np.nan, "avg_days": np.nan})
        return out

    out["target_rate"] = float((done["status"] == "TARGET HIT").mean())
    out["stop_rate"] = float((done["status"] == "STOP HIT").mean())
    out["win_rate"] = float((done["outcome_return"] > 0).mean())
    out["avg_return"] = float(done["outcome_return"].mean())
    out["avg_days"] = float(done["days"].mean())
    return out
