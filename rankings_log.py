# =============================
# rankings_log.py — Supabase log for nightly / offline rankings runs
# =============================
"""Insert one row per precompute run (stdlib urllib only — no extra deps).

Config (env preferred for GitHub Actions):

  SUPABASE_URL
  SUPABASE_KEY  or  SUPABASE_SERVICE_KEY
  SUPABASE_RANKINGS_LOG_TABLE  (default: rankings_run_log)

Optional Streamlit secrets [rankings_log] for UI history (same fields).

If URL/key are missing, log_rankings_run is a no-op and returns skipped=True.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any


DEFAULT_TABLE = "rankings_run_log"


def _secrets_section(name: str = "rankings_log") -> dict:
    try:
        import streamlit as st
        sec = st.secrets.get(name, None)
        if sec is None:
            return {}
        return dict(sec)
    except Exception:
        return {}


def get_rankings_log_config() -> dict:
    sec = _secrets_section("rankings_log")
    # Fall back to journal credentials if rankings_log not set (same project)
    jsec = _secrets_section("journal")

    def _get(key, *envs, default=""):
        for e in envs:
            if os.environ.get(e):
                return os.environ.get(e)
        if key in sec and sec[key] not in (None, ""):
            return sec[key]
        # journal fallback for url/key only
        if key in ("supabase_url", "supabase_key") and key in jsec and jsec[key]:
            return jsec[key]
        return default

    enabled_raw = _get("enabled", "RANKINGS_LOG_ENABLED", default="true")
    if isinstance(enabled_raw, bool):
        enabled = enabled_raw
    else:
        enabled = str(enabled_raw).strip().lower() in ("1", "true", "yes", "on")

    url = str(_get("supabase_url", "SUPABASE_URL") or "").strip()
    key = str(
        _get(
            "supabase_key",
            "SUPABASE_KEY",
            "SUPABASE_SERVICE_KEY",
            "SUPABASE_SERVICE_ROLE_KEY",
        )
        or ""
    ).strip()
    table = str(
        _get("table", "SUPABASE_RANKINGS_LOG_TABLE", default=DEFAULT_TABLE)
        or DEFAULT_TABLE
    ).strip()

    return {
        "enabled": enabled,
        "supabase_url": url.rstrip("/"),
        "supabase_key": key,
        "table": table,
        "configured": bool(enabled and url and key),
    }


def rankings_log_status() -> dict:
    cfg = get_rankings_log_config()
    return {
        "configured": cfg["configured"],
        "enabled": cfg["enabled"],
        "table": cfg["table"],
        "label": (
            f"Rankings log · `{cfg['table']}`"
            if cfg["configured"]
            else "Rankings log not configured"
        ),
    }


def _json_safe(v: Any) -> Any:
    if v is None:
        return None
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:
            pass
    if isinstance(v, float):
        import math
        if math.isnan(v) or math.isinf(v):
            return None
    if isinstance(v, (dict, list, str, int, float, bool)):
        return v
    return str(v)


def _postgrest_insert(url: str, key: str, table: str, row: dict) -> None:
    endpoint = f"{url.rstrip('/')}/rest/v1/{table}"
    body = json.dumps(row).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=body,
        method="POST",
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Prefer": "return=minimal",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:400]
        raise RuntimeError(f"Supabase rankings log HTTP {e.code}: {detail}") from e


def _postgrest_select(
    url: str, key: str, table: str, *, limit: int = 10,
) -> list[dict]:
    q = urllib.parse.urlencode({
        "select": "*",
        "order": "logged_at.desc",
        "limit": str(limit),
    })
    endpoint = f"{url.rstrip('/')}/rest/v1/{table}?{q}"
    req = urllib.request.Request(
        endpoint,
        method="GET",
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else []
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"Supabase rankings log HTTP {e.code}: {detail}") from e


def build_log_row(
    meta: dict | None,
    *,
    status: str = "success",
    error_message: str | None = None,
    top_symbols: list[str] | None = None,
    max_symbols: int | None = None,
) -> dict:
    """Flatten precompute meta + CI env into a table row."""
    meta = dict(meta or {})
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    top = top_symbols
    if top is None and meta.get("top_symbols"):
        top = meta.get("top_symbols")
    top_s = None
    if top:
        top_s = ",".join(str(s) for s in top[:20])

    # Keep a compact JSON snapshot (drop huge failure samples if present)
    meta_copy = {
        k: _json_safe(v)
        for k, v in meta.items()
        if k not in ("failures_sample",)
    }

    return {
        "logged_at": now,
        "generated_at": meta.get("generated_at"),
        "status": status,
        "runner": meta.get("runner") or os.environ.get("RANKINGS_RUNNER"),
        "run_id": meta.get("run_id") or os.environ.get("GITHUB_RUN_ID"),
        "run_url": meta.get("run_url"),
        "workflow": meta.get("workflow") or os.environ.get("GITHUB_WORKFLOW"),
        "watchlist": meta.get("watchlist"),
        "engine": meta.get("engine"),
        "n_requested": _json_safe(meta.get("n_requested")),
        "n_scored": _json_safe(meta.get("n_scored")),
        "n_failed": _json_safe(meta.get("n_failed")),
        "batch_size": _json_safe(meta.get("batch_size")),
        "elapsed_s": _json_safe(meta.get("elapsed_s")),
        "max_symbols": _json_safe(max_symbols if max_symbols is not None else meta.get("max_symbols")),
        "error_message": (error_message or "")[:500] or None,
        "top_symbols": top_s,
        "repo": os.environ.get("GITHUB_REPOSITORY"),
        "git_sha": os.environ.get("GITHUB_SHA"),
        "meta": meta_copy,
    }


def log_rankings_run(
    meta: dict | None = None,
    *,
    status: str = "success",
    error_message: str | None = None,
    top_symbols: list[str] | None = None,
    max_symbols: int | None = None,
    dry_run: bool = False,
) -> dict:
    """Insert one rankings run log. Safe no-op when not configured.

    Returns {ok, skipped, reason, row?}.
    """
    cfg = get_rankings_log_config()
    if not cfg["configured"]:
        return {
            "ok": True,
            "skipped": True,
            "reason": "rankings log not configured (set SUPABASE_URL + SUPABASE_KEY)",
        }

    row = build_log_row(
        meta,
        status=status,
        error_message=error_message,
        top_symbols=top_symbols,
        max_symbols=max_symbols,
    )

    if dry_run:
        return {"ok": True, "skipped": False, "reason": "dry_run", "row": row}

    try:
        _postgrest_insert(
            cfg["supabase_url"], cfg["supabase_key"], cfg["table"], row,
        )
        return {"ok": True, "skipped": False, "reason": "inserted", "row": row}
    except Exception as e:
        return {"ok": False, "skipped": False, "reason": str(e), "row": row}


def fetch_recent_rankings_logs(limit: int = 10) -> list[dict]:
    """Read recent log rows (for UI). Raises if misconfigured / network error."""
    cfg = get_rankings_log_config()
    if not cfg["configured"]:
        return []
    limit = max(1, min(int(limit), 200))
    return _postgrest_select(
        cfg["supabase_url"], cfg["supabase_key"], cfg["table"], limit=limit,
    )
