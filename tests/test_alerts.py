"""alerts.py — candidate filter, de-dupe, dry-run send path."""
import pandas as pd
import pytest

import alerts as alerts_mod
from alerts import (
    _format_message,
    alerts_status,
    get_alerts_config,
    run_alerts,
    select_alert_candidates,
)


def _scan_df():
    return pd.DataFrame([
        {
            "Symbol": "AAA.NS", "Name": "Aaa", "Screen": "BUY",
            "Buy Score": 75, "Probability Up": 0.70, "Risk": 4.0,
            "Test Acc": 0.58, "Baseline": 0.52, "Reward Risk": 2.0,
            "To Support": 0.02, "Rank": 1,
        },
        {
            "Symbol": "BBB.NS", "Name": "Bbb", "Screen": "BUY",
            "Buy Score": 55, "Probability Up": 0.60, "Risk": 5.0,
            "Test Acc": 0.55, "Baseline": 0.52, "Reward Risk": 1.8,
            "To Support": 0.03, "Rank": 2,
        },
        {
            "Symbol": "CCC.NS", "Name": "Ccc", "Screen": "SELL",
            "Buy Score": 20, "Probability Up": 0.40, "Risk": 6.0,
            "Test Acc": 0.50, "Baseline": 0.52, "Reward Risk": 1.2,
            "To Support": 0.05, "Rank": 3,
        },
        {
            "Symbol": "DDD.NS", "Name": "Ddd", "Screen": "BUY",
            "Buy Score": 80, "Probability Up": 0.68, "Risk": 9.5,
            "Test Acc": 0.60, "Baseline": 0.50, "Reward Risk": 2.5,
            "To Support": 0.01, "Rank": 4,
        },
    ])


def test_select_alert_candidates_filters():
    picks = select_alert_candidates(
        _scan_df(),
        min_buy_score=60,
        min_probability=0.55,
        max_risk=8.0,
        require_edge=True,
        top_n=10,
    )
    # AAA qualifies; BBB score too low; CCC SELL; DDD risk too high
    assert list(picks["Symbol"]) == ["AAA.NS"]


def test_select_alert_candidates_empty_input():
    assert select_alert_candidates(pd.DataFrame()).empty
    assert select_alert_candidates(None).empty


def test_format_message_contains_symbols():
    picks = select_alert_candidates(_scan_df(), min_buy_score=50, max_risk=10)
    msg = _format_message(picks, asof="2026-07-20")
    assert "BUY screen" in msg
    assert "AAA.NS" in msg
    assert "not investment advice" in msg.lower() or "Screen only" in msg


def test_run_alerts_disabled_skips(monkeypatch, tmp_path):
    monkeypatch.setattr(alerts_mod, "ALERTS_DIR", tmp_path / "alerts")
    monkeypatch.setattr(alerts_mod, "STATE_FILE", tmp_path / "alerts" / "state.json")
    monkeypatch.setattr(
        alerts_mod, "get_alerts_config",
        lambda: {
            "enabled": False,
            "min_buy_score": 60, "min_probability": 0.55, "top_n": 10,
            "require_edge": True, "max_risk": 8.0,
            "telegram_bot_token": "", "telegram_chat_id": "",
            "smtp_host": "", "smtp_port": 587, "smtp_user": "",
            "smtp_password": "", "email_to": "", "email_from": "",
        },
    )
    r = run_alerts(_scan_df(), asof="snap1")
    assert r["skipped"] is True
    assert "disabled" in r["reason"]


def test_run_alerts_dry_run_without_channels_previews_message(
    monkeypatch, tmp_path,
):
    monkeypatch.setattr(alerts_mod, "ALERTS_DIR", tmp_path / "alerts")
    monkeypatch.setattr(alerts_mod, "STATE_FILE", tmp_path / "alerts" / "state.json")

    def cfg():
        return {
            "enabled": True,
            "min_buy_score": 60, "min_probability": 0.55, "top_n": 10,
            "require_edge": True, "max_risk": 8.0,
            "telegram_bot_token": "", "telegram_chat_id": "",
            "smtp_host": "", "smtp_port": 587, "smtp_user": "",
            "smtp_password": "", "email_to": "", "email_from": "",
        }

    monkeypatch.setattr(alerts_mod, "get_alerts_config", cfg)
    r = run_alerts(_scan_df(), asof="snap1", dry_run=True)
    assert r["ok"] is True
    assert r["sent"] is False
    assert r["skipped"] is False
    assert r["reason"] == "dry_run"
    assert "AAA.NS" in r["message"]


def test_run_alerts_dry_run_with_fake_channel(monkeypatch, tmp_path):
    monkeypatch.setattr(alerts_mod, "ALERTS_DIR", tmp_path / "alerts")
    monkeypatch.setattr(alerts_mod, "STATE_FILE", tmp_path / "alerts" / "state.json")
    monkeypatch.setattr(
        alerts_mod, "get_alerts_config",
        lambda: {
            "enabled": True,
            "min_buy_score": 60, "min_probability": 0.55, "top_n": 10,
            "require_edge": True, "max_risk": 8.0,
            "telegram_bot_token": "tok", "telegram_chat_id": "1",
            "smtp_host": "", "smtp_port": 587, "smtp_user": "",
            "smtp_password": "", "email_to": "", "email_from": "",
        },
    )
    sent = []

    def fake_tg(token, chat_id, text):
        sent.append(text)

    monkeypatch.setattr(alerts_mod, "send_telegram", fake_tg)

    r = run_alerts(_scan_df(), asof="snap-dry", dry_run=True)
    assert r["ok"] is True
    assert r["sent"] is False  # dry run
    assert r["new"] >= 1
    assert "AAA.NS" in r["message"]
    assert sent == []  # nothing actually sent

    # Live send should call telegram and persist state
    r2 = run_alerts(_scan_df(), asof="snap-live", dry_run=False)
    assert r2["ok"] is True
    assert r2["sent"] is True
    assert len(sent) == 1
    assert "AAA.NS" in sent[0]

    # Second run same snapshot: de-duped
    r3 = run_alerts(_scan_df(), asof="snap-live", dry_run=False)
    assert r3["skipped"] is True
    assert r3["new"] == 0
    assert len(sent) == 1

    # Force re-alerts
    r4 = run_alerts(_scan_df(), asof="snap-live", force=True, dry_run=False)
    assert r4["sent"] is True
    assert len(sent) == 2


def test_alerts_status_shape():
    s = alerts_status()
    assert "enabled" in s
    assert "channels" in s
    assert "configured" in s
    assert "label" in s


def test_get_alerts_config_defaults(monkeypatch):
    monkeypatch.setattr(alerts_mod, "_secrets_section", lambda name="alerts": {})
    for k in list(alerts_mod.os.environ.keys()):
        if k.startswith("ALERTS_") or k.startswith("TELEGRAM_") or k.startswith("SMTP_"):
            monkeypatch.delenv(k, raising=False)
    cfg = get_alerts_config()
    assert cfg["enabled"] is False
    assert cfg["min_buy_score"] == 60
    assert cfg["top_n"] == 10
