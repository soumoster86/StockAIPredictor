# =============================
# journal.py
# =============================
"""Signal journal: log each signal, score against real prices later.

Storage backends (configured via secrets / env):

  local     — journals/<user>.csv  (default; ephemeral on Streamlit Cloud)
  supabase  — hosted Postgres via PostgREST  (survives redeploys)

Public API is unchanged for tests and the UI:

  load_journal(user=...) / load_journal(path=...)
  append_signal(record, user=...) / append_signal(record, path=...)
  resolve_entry / resolve_journal / scorecard
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from execution import cost_profile_for, locked_sessions, side_costs

ROOT = Path(__file__).parent
JOURNAL_DIR = ROOT / "journals"
LEGACY_JOURNAL_FILE = ROOT / "journal.csv"
JOURNAL_FILE = LEGACY_JOURNAL_FILE  # back-compat alias
MAX_HOLD_DAYS = 20

COLUMNS = [
    "signal_date", "symbol", "name", "model_type", "signal", "probability",
    "rating", "entry", "stop", "target", "reward_risk", "risk_score", "logged_at",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def safe_username(user):
    """Filesystem-safe username fragment (letters, digits, . _ -)."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(user or "anonymous").strip())
    s = s.strip("._-")[:64]
    return s or "anonymous"


def journal_path_for(user=None):
    """Local CSV path for a user (even when remote backend is active — used
    for download/export fallbacks)."""
    if user is None or str(user).strip() == "":
        return LEGACY_JOURNAL_FILE
    return JOURNAL_DIR / f"{safe_username(user)}.csv"


def _secrets_section():
    """Read [journal] from Streamlit secrets if available."""
    try:
        import streamlit as st
        sec = st.secrets.get("journal", None)
        if sec is None:
            return {}
        return dict(sec)
    except Exception:
        return {}


def get_journal_config():
    """Return resolved journal config dict.

    Precedence: env vars > Streamlit secrets > defaults.
    """
    sec = _secrets_section()
    backend = (
        os.environ.get("JOURNAL_BACKEND")
        or sec.get("backend")
        or "local"
    ).strip().lower()

    cfg = {
        "backend": backend if backend in ("local", "supabase") else "local",
        "supabase_url": (
            os.environ.get("SUPABASE_URL")
            or sec.get("supabase_url")
            or sec.get("url")
            or ""
        ).rstrip("/"),
        "supabase_key": (
            os.environ.get("SUPABASE_KEY")
            or os.environ.get("SUPABASE_SERVICE_KEY")
            or sec.get("supabase_key")
            or sec.get("key")
            or ""
        ),
        "supabase_table": (
            os.environ.get("SUPABASE_JOURNAL_TABLE")
            or sec.get("table")
            or "signal_journal"
        ),
    }
    return cfg


def journal_backend_info():
    """Human-readable backend status for the UI."""
    cfg = get_journal_config()
    if cfg["backend"] == "supabase" and cfg["supabase_url"] and cfg["supabase_key"]:
        return {
            "backend": "supabase",
            "label": "Supabase (cloud-persistent)",
            "persistent": True,
            "detail": f"table `{cfg['supabase_table']}`",
        }
    if cfg["backend"] == "supabase":
        return {
            "backend": "local",
            "label": "Local CSV (Supabase misconfigured — falling back)",
            "persistent": False,
            "detail": "Set journal.supabase_url + journal.supabase_key in secrets",
        }
    return {
        "backend": "local",
        "label": "Local CSV (ephemeral on Streamlit Cloud)",
        "persistent": False,
        "detail": str(JOURNAL_DIR),
    }


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

def _empty_df():
    return pd.DataFrame(columns=COLUMNS)


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return _empty_df()
    out = df.copy()
    for col in COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out[COLUMNS]


def _record_row(record: dict) -> dict:
    return {c: record.get(c) for c in COLUMNS}


class LocalCSVBackend:
    name = "local"

    def load(self, user=None, path=None) -> pd.DataFrame:
        path = Path(path) if path is not None else journal_path_for(user)
        if not path.exists():
            return _empty_df()
        try:
            df = pd.read_csv(path)
        except Exception:
            return _empty_df()
        return _normalize_df(df)

    def append(self, record: dict, user=None, path=None) -> bool:
        path = Path(path) if path is not None else journal_path_for(user)
        if path.parent == JOURNAL_DIR:
            JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
        df = self.load(path=path)
        dup = (
            (df["signal_date"].astype(str) == str(record["signal_date"]))
            & (df["symbol"].astype(str) == str(record["symbol"]))
            & (df["model_type"].astype(str) == str(record["model_type"]))
        )
        if len(df) and dup.any():
            return False
        row = pd.DataFrame([_record_row(record)])
        if df.empty:
            out = row
        else:
            out = pd.concat([df, row], ignore_index=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(path, index=False)
        return True


class SupabaseBackend:
    """PostgREST client for a simple journal table (stdlib urllib only)."""

    name = "supabase"

    def __init__(self, url: str, key: str, table: str = "signal_journal"):
        self.url = url.rstrip("/")
        self.key = key
        self.table = table

    def _headers(self, prefer: str | None = None) -> dict:
        h = {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if prefer:
            h["Prefer"] = prefer
        return h

    def _request(self, method: str, path: str, body=None, prefer=None, query=None):
        q = f"?{urllib.parse.urlencode(query, doseq=True)}" if query else ""
        req = urllib.request.Request(
            f"{self.url}/rest/v1/{path}{q}",
            data=None if body is None else json.dumps(body).encode("utf-8"),
            headers=self._headers(prefer=prefer),
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                raw = resp.read().decode("utf-8")
                if not raw:
                    return []
                return json.loads(raw)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError(f"Supabase HTTP {e.code}: {detail}") from e
        except Exception as e:
            raise RuntimeError(f"Supabase request failed: {e}") from e

    def load(self, user=None, path=None) -> pd.DataFrame:
        # path ignored — remote is keyed by username
        username = safe_username(user) if user else "anonymous"
        rows = self._request(
            "GET",
            self.table,
            query={
                "username": f"eq.{username}",
                "select": ",".join(COLUMNS),
                "order": "signal_date.desc,logged_at.desc",
            },
        )
        if not rows:
            return _empty_df()
        return _normalize_df(pd.DataFrame(rows))

    def append(self, record: dict, user=None, path=None) -> bool:
        username = safe_username(user) if user else "anonymous"
        # Duplicate check
        existing = self._request(
            "GET",
            self.table,
            query={
                "username": f"eq.{username}",
                "signal_date": f"eq.{record['signal_date']}",
                "symbol": f"eq.{record['symbol']}",
                "model_type": f"eq.{record['model_type']}",
                "select": "symbol",
                "limit": "1",
            },
        )
        if existing:
            return False

        payload = {"username": username, **_record_row(record)}
        # Coerce numerics that might be numpy types
        for k, v in list(payload.items()):
            if hasattr(v, "item"):
                try:
                    payload[k] = v.item()
                except Exception:
                    payload[k] = v
            if isinstance(v, float) and (np.isnan(v) if isinstance(v, (float, np.floating)) else False):
                payload[k] = None

        self._request(
            "POST",
            self.table,
            body=payload,
            prefer="return=minimal",
        )
        return True


def get_backend():
    """Active backend instance for user-scoped operations."""
    cfg = get_journal_config()
    if (
        cfg["backend"] == "supabase"
        and cfg["supabase_url"]
        and cfg["supabase_key"]
    ):
        return SupabaseBackend(
            cfg["supabase_url"], cfg["supabase_key"], cfg["supabase_table"],
        )
    return LocalCSVBackend()


# ---------------------------------------------------------------------------
# Public API (path= forces local CSV — used by unit tests)
# ---------------------------------------------------------------------------

def load_journal(path=None, user=None):
    if path is not None:
        return LocalCSVBackend().load(path=path, user=user)
    return get_backend().load(user=user)


def append_signal(record, path=None, user=None):
    """Append one signal. Deduped on (signal_date, symbol, model_type).
    Returns False if duplicate."""
    if path is not None:
        return LocalCSVBackend().append(record, path=path, user=user)
    try:
        return get_backend().append(record, user=user)
    except RuntimeError:
        # Soft fallback to local if remote fails mid-session
        return LocalCSVBackend().append(record, user=user)


def _open_col(future, entry):
    """Session opens; without an Open column assume each day opens at the
    prior close (no gap), the first at the logged entry."""
    if "Open" in future.columns:
        return future["Open"].astype(float)
    return future["Close"].astype(float).shift(1).fillna(entry)


def resolve_entry(rec, prices, max_days=MAX_HOLD_DAYS):
    """Score one journal entry against subsequent price action, the way a
    real order would have filled.

    BUY:
      - Fill at the NEXT session's open (the signal is formed at the close,
        so that close is not available to trade). If that session is locked
        at the upper circuit, or opens already beyond the stop or target
        (the plan is void), the trade is NO FILL.
      - Each later session: an open through the stop exits at the OPEN
        (a gap down fills worse than the stop); an open through the target
        exits at the open (better). Otherwise an intraday touch exits at the
        level. Both touched on one day → STOP HIT (intraday order is
        unknown, score conservatively). A session locked at the lower
        circuit can't be exited — the position carries to the next day.
      - After `max_days` with neither → EXPIRED at that day's close.
        Not enough days yet → OPEN, with the unrealized return so far.
      - `outcome_return` is net of statutory costs and slippage;
        `gross_return` is the price move alone.

    SELL / HOLD: no trade to resolve — just the forward return from the next
    open to the close after `max_days` (CLOSED) or so far (OPEN). For SELL, a
    negative forward return means exiting was the right call."""
    signal_date = pd.Timestamp(rec["signal_date"])
    entry = float(rec["entry"])
    future = prices.loc[prices.index > signal_date].head(max_days)
    empty = {"fill_price": np.nan, "exit_price": np.nan, "gross_return": np.nan}

    if future.empty:
        return {"status": "OPEN", "days": 0, "outcome_return": np.nan,
                "exit_date": None, **empty}

    opens = _open_col(future, entry)
    fill = float(opens.iloc[0])

    if rec["signal"] != "BUY":
        last_close = float(future["Close"].iloc[-1])
        status = "CLOSED" if len(future) >= max_days else "OPEN"
        ret = last_close / fill - 1.0
        return {"status": status, "days": len(future), "outcome_return": ret,
                "exit_date": future.index[-1] if status == "CLOSED" else None,
                "fill_price": fill, "exit_price": last_close, "gross_return": ret}

    stop, target = float(rec["stop"]), float(rec["target"])
    up_locked, down_locked = locked_sessions(prices)
    up_locked = up_locked.reindex(future.index, fill_value=False)
    down_locked = down_locked.reindex(future.index, fill_value=False)

    if bool(up_locked.iloc[0]) or not (stop < fill < target):
        return {"status": "NO FILL", "days": 0, "outcome_return": np.nan,
                "exit_date": None, **empty}

    # Costs use liquidity known at the signal date (no lookahead).
    buy_costs, sell_costs = side_costs(prices, cost_profile_for(rec.get("symbol")))
    known = prices.index <= signal_date
    buy_cost = float(buy_costs[known].iloc[-1]) if known.any() else float(buy_costs.iloc[0])
    sell_cost = float(sell_costs[known].iloc[-1]) if known.any() else float(sell_costs.iloc[0])

    def done(status, days, exit_price, exit_date):
        net = exit_price * (1.0 - sell_cost) / (fill * (1.0 + buy_cost)) - 1.0
        return {"status": status, "days": days, "outcome_return": net,
                "exit_date": exit_date, "fill_price": fill,
                "exit_price": float(exit_price), "gross_return": exit_price / fill - 1.0}

    for i, (dt, row) in enumerate(future.iterrows(), start=1):
        if bool(down_locked.loc[dt]):
            continue  # frozen at the lower circuit: no buyers, can't exit today
        day_open = float(opens.loc[dt])
        if i > 1 and day_open <= stop:
            return done("STOP HIT", i, day_open, dt)
        if i > 1 and day_open >= target:
            return done("TARGET HIT", i, day_open, dt)
        if float(row["Low"]) <= stop:  # checked first: same-day double-touch → STOP
            return done("STOP HIT", i, stop, dt)
        if float(row["High"]) >= target:
            return done("TARGET HIT", i, target, dt)

    last_close = float(future["Close"].iloc[-1])
    if len(future) >= max_days:
        return done("EXPIRED", max_days, last_close, future.index[-1])
    return done("OPEN", len(future), last_close, None)  # marked to the last close


RESOLVED_COLUMNS = ["status", "days", "outcome_return", "fill_price",
                    "exit_price", "gross_return"]


def resolve_journal(journal_df, price_fetcher, max_days=MAX_HOLD_DAYS):
    """Resolve every entry. `price_fetcher(symbol)` must return an OHLC
    DataFrame. Symbols that fail to fetch are marked NO DATA."""
    if journal_df.empty:
        return journal_df.assign(**{c: [] for c in RESOLVED_COLUMNS})

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
            results.append({"status": "NO DATA", "days": 0, "outcome_return": np.nan,
                            "exit_date": None, "fill_price": np.nan,
                            "exit_price": np.nan, "gross_return": np.nan})
        else:
            results.append(resolve_entry(rec, prices, max_days))

    out = journal_df.copy().reset_index(drop=True)
    res = pd.DataFrame(results)
    out[RESOLVED_COLUMNS] = res[RESOLVED_COLUMNS]
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
        "n_no_fill": int((buys["status"] == "NO FILL").sum()),
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
