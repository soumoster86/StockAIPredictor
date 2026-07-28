# =============================
# alerts.py — Telegram / email alerts for top BUY screens
# =============================
"""Optional alerts when names enter the top Buy Score shortlist.

Config via Streamlit secrets or env:

  [alerts]
  enabled = true
  min_buy_score = 60
  min_probability = 0.55
  top_n = 10
  require_edge = true
  telegram_bot_token = "..."
  telegram_chat_id = "..."
  # optional email
  smtp_host = "smtp.gmail.com"
  smtp_port = 587
  smtp_user = "you@gmail.com"
  smtp_password = "app-password"
  email_to = "you@gmail.com"
  email_from = "you@gmail.com"

State is stored under alerts/state.json so the same names are not re-alerted
for the same rankings snapshot.
"""
from __future__ import annotations

import json
import os
import smtplib
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from email.mime.text import MIMEText
from pathlib import Path

import pandas as pd

from model import rank_buy_candidates

ROOT = Path(__file__).resolve().parent
ALERTS_DIR = ROOT / "alerts"
STATE_FILE = ALERTS_DIR / "state.json"


def _secrets_section(name: str = "alerts") -> dict:
    try:
        import streamlit as st
        sec = st.secrets.get(name, None)
        if sec is None:
            return {}
        return dict(sec)
    except Exception:
        return {}


def get_alerts_config() -> dict:
    sec = _secrets_section("alerts")

    def _get(key, env=None, default=None):
        if env and os.environ.get(env) is not None:
            return os.environ.get(env)
        if key in sec:
            return sec[key]
        return default

    enabled_raw = _get("enabled", "ALERTS_ENABLED", "false")
    if isinstance(enabled_raw, bool):
        enabled = enabled_raw
    else:
        enabled = str(enabled_raw).strip().lower() in ("1", "true", "yes", "on")

    return {
        "enabled": enabled,
        "min_buy_score": float(_get("min_buy_score", "ALERTS_MIN_BUY_SCORE", 60)),
        "min_probability": float(_get("min_probability", "ALERTS_MIN_PROB", 0.55)),
        "top_n": int(_get("top_n", "ALERTS_TOP_N", 10)),
        "require_edge": str(_get("require_edge", "ALERTS_REQUIRE_EDGE", "true")).lower()
            in ("1", "true", "yes", "on", "True"),
        "max_risk": float(_get("max_risk", "ALERTS_MAX_RISK", 8.0)),
        "telegram_bot_token": str(_get("telegram_bot_token", "TELEGRAM_BOT_TOKEN", "") or ""),
        "telegram_chat_id": str(_get("telegram_chat_id", "TELEGRAM_CHAT_ID", "") or ""),
        "smtp_host": str(_get("smtp_host", "SMTP_HOST", "") or ""),
        "smtp_port": int(_get("smtp_port", "SMTP_PORT", 587) or 587),
        "smtp_user": str(_get("smtp_user", "SMTP_USER", "") or ""),
        "smtp_password": str(_get("smtp_password", "SMTP_PASSWORD", "") or ""),
        "email_to": str(_get("email_to", "ALERTS_EMAIL_TO", "") or ""),
        "email_from": str(_get("email_from", "ALERTS_EMAIL_FROM", "") or ""),
    }


def alerts_status() -> dict:
    cfg = get_alerts_config()
    channels = []
    if cfg["telegram_bot_token"] and cfg["telegram_chat_id"]:
        channels.append("telegram")
    if cfg["smtp_host"] and cfg["email_to"] and cfg["smtp_user"]:
        channels.append("email")
    return {
        "enabled": cfg["enabled"],
        "channels": channels,
        "configured": bool(channels),
        "min_buy_score": cfg["min_buy_score"],
        "top_n": cfg["top_n"],
        "label": (
            f"Alerts on · {', '.join(channels)}" if cfg["enabled"] and channels
            else ("Channels set · disabled" if channels else "Not configured")
        ),
    }


def _load_state() -> dict:
    if not STATE_FILE.exists():
        return {"sent": {}, "last_run": None}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"sent": {}, "last_run": None}


def _save_state(state: dict) -> None:
    ALERTS_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def select_alert_candidates(
    scan_df: pd.DataFrame,
    *,
    min_buy_score: float = 60,
    min_probability: float = 0.55,
    max_risk: float = 8.0,
    require_edge: bool = True,
    top_n: int = 10,
) -> pd.DataFrame:
    """Top BUY shortlist that also clears min Buy Score."""
    if scan_df is None or scan_df.empty:
        return pd.DataFrame()
    picks = rank_buy_candidates(
        scan_df,
        min_prob=min_probability,
        max_risk=max_risk,
        require_edge=require_edge,
        top_n=top_n,
    )
    if picks.empty or "Buy Score" not in picks.columns:
        return picks
    return picks[picks["Buy Score"] >= float(min_buy_score)].reset_index(drop=True)


def _format_message(picks: pd.DataFrame, asof: str | None = None) -> str:
    lines = [
        "AI Stock Predictor — BUY screen alerts",
        f"As of: {asof or datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Candidates: {len(picks)}",
        "",
    ]
    for _, row in picks.iterrows():
        lines.append(
            f"#{int(row.get('Rank', 0))} {row.get('Symbol')}  "
            f"score={float(row.get('Buy Score', 0)):.0f}  "
            f"p={float(row.get('Probability Up', 0)) * 100:.0f}%  "
            f"risk={row.get('Risk', '—')}"
        )
        if row.get("Name"):
            lines.append(f"    {row.get('Name')}")
    lines.append("")
    lines.append("Screen only — not investment advice. Open the app for full Signal.")
    return "\n".join(lines)


def send_telegram(token: str, chat_id: str, text: str) -> None:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode({
        "chat_id": chat_id,
        "text": text[:4000],
        "disable_web_page_preview": "true",
    }).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            resp.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:200]
        raise RuntimeError(f"Telegram HTTP {e.code}: {detail}") from e


def send_email(
    *,
    host: str,
    port: int,
    user: str,
    password: str,
    to_addr: str,
    from_addr: str,
    subject: str,
    body: str,
) -> None:
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = from_addr or user
    msg["To"] = to_addr
    with smtplib.SMTP(host, port, timeout=30) as smtp:
        smtp.ehlo()
        try:
            smtp.starttls()
            smtp.ehlo()
        except smtplib.SMTPException:
            pass
        if user and password:
            smtp.login(user, password)
        smtp.sendmail(msg["From"], [to_addr], msg.as_string())


def run_alerts(
    scan_df: pd.DataFrame,
    *,
    asof: str | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict:
    """Evaluate scan results and send alerts for new top BUYs.

    Returns a result dict: {ok, sent, skipped, candidates, message, error}.
    """
    cfg = get_alerts_config()
    status = alerts_status()
    if not cfg["enabled"] and not force and not dry_run:
        return {
            "ok": False, "sent": False, "skipped": True,
            "reason": "alerts disabled", "candidates": 0, "new": 0,
            "message": "", "channels": status["channels"],
        }
    if not status["configured"] and not dry_run:
        return {
            "ok": False, "sent": False, "skipped": True,
            "reason": "no channels configured (telegram/email)",
            "candidates": 0, "new": 0, "message": "", "channels": [],
        }

    picks = select_alert_candidates(
        scan_df,
        min_buy_score=cfg["min_buy_score"],
        min_probability=cfg["min_probability"],
        max_risk=cfg["max_risk"],
        require_edge=cfg["require_edge"],
        top_n=cfg["top_n"],
    )
    if picks.empty:
        return {
            "ok": True, "sent": False, "skipped": True,
            "reason": "no candidates passed filters",
            "candidates": 0, "new": 0, "message": "", "channels": status["channels"],
        }

    snapshot_id = asof or "live"
    state = _load_state()
    sent_map = state.setdefault("sent", {})
    already = set(sent_map.get(snapshot_id, []))

    if force:
        new_picks = picks
    else:
        mask = ~picks["Symbol"].astype(str).isin(already)
        new_picks = picks.loc[mask].reset_index(drop=True)

    if new_picks.empty:
        return {
            "ok": True, "sent": False, "skipped": True,
            "reason": "all candidates already alerted for this snapshot",
            "candidates": len(picks), "new": 0, "message": "",
            "channels": status["channels"],
        }

    msg = _format_message(new_picks, asof=asof)
    if dry_run:
        return {
            "ok": True, "sent": False, "skipped": False,
            "reason": "dry_run",
            "candidates": len(picks),
            "new": len(new_picks),
            "message": msg,
            "channels": status["channels"],
            "symbols": new_picks["Symbol"].astype(str).tolist(),
            "error": None,
        }

    errors = []
    if cfg["telegram_bot_token"] and cfg["telegram_chat_id"]:
        try:
            send_telegram(cfg["telegram_bot_token"], cfg["telegram_chat_id"], msg)
        except Exception as e:
            errors.append(f"telegram: {e}")
    if cfg["smtp_host"] and cfg["email_to"] and cfg["smtp_user"]:
        try:
            send_email(
                host=cfg["smtp_host"],
                port=cfg["smtp_port"],
                user=cfg["smtp_user"],
                password=cfg["smtp_password"],
                to_addr=cfg["email_to"],
                from_addr=cfg["email_from"] or cfg["smtp_user"],
                subject=f"[StockAI] {len(new_picks)} BUY screen alert(s)",
                body=msg,
            )
        except Exception as e:
            errors.append(f"email: {e}")

    # Mark as sent only if at least one channel worked
    if not errors or len(errors) < len(status["channels"]):
        sent_map.setdefault(snapshot_id, [])
        for sym in new_picks["Symbol"].astype(str).tolist():
            if sym not in sent_map[snapshot_id]:
                sent_map[snapshot_id].append(sym)
        # Keep only recent snapshots
        if len(sent_map) > 30:
            for k in list(sent_map.keys())[:-30]:
                sent_map.pop(k, None)
        state["last_run"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        state["sent"] = sent_map
        _save_state(state)

    return {
        "ok": not errors,
        "sent": not (errors and len(errors) >= max(len(status["channels"]), 1)),
        "skipped": False,
        "reason": "; ".join(errors) if errors else "ok",
        "candidates": len(picks),
        "new": len(new_picks),
        "message": msg,
        "channels": status["channels"],
        "symbols": new_picks["Symbol"].astype(str).tolist(),
        "error": "; ".join(errors) if errors else None,
    }
