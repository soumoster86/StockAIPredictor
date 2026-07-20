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

from pathlib import Path
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

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

    last_close = float(future["Close"].iloc[-1])
    status = "CLOSED" if len(future) >= max_days else "OPEN"
    return {"status": status, "days": len(future),
            "outcome_return": last_close / entry - 1.0,
            "exit_date": future.index[-1] if status == "CLOSED" else None}


def resolve_journal(journal_df, price_fetcher, max_days=MAX_HOLD_DAYS):
    """Resolve every entry. `price_fetcher(symbol)` must return an OHLC
    DataFrame. Symbols that fail to fetch are marked NO DATA."""
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
