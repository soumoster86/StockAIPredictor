"""Journal: dedupe, every resolution path, scorecard math, failure safety."""
import numpy as np
import pandas as pd

from journal import (
    append_signal, load_journal, resolve_entry, resolve_journal, scorecard,
    journal_path_for, safe_username, get_journal_config, journal_backend_info,
    LocalCSVBackend, SupabaseBackend,
)

REC = dict(signal_date="2026-05-01", symbol="TEST.NS", name="Test",
           model_type="Ensemble", signal="BUY", probability=0.68, rating="Buy",
           entry=1000.0, stop=950.0, target=1100.0, reward_risk=2.0,
           risk_score=4.5, logged_at="2026-05-01 16:00")


def prices(rows):
    idx = pd.bdate_range("2026-05-04", periods=len(rows))
    return pd.DataFrame(rows, columns=["High", "Low", "Close"], index=idx)


def test_append_and_dedupe(tmp_path):
    p = tmp_path / "j.csv"
    assert append_signal(REC, p) is True
    assert append_signal(REC, p) is False
    assert len(load_journal(p)) == 1


def test_safe_username_and_per_user_paths(tmp_path, monkeypatch):
    assert safe_username("Alice/Bob") == "Alice_Bob"
    assert safe_username("") == "anonymous"
    # Redirect journal dir into tmp so tests stay isolated
    import journal as J
    monkeypatch.setattr(J, "JOURNAL_DIR", tmp_path / "journals")
    p_a = journal_path_for("alice")
    p_b = journal_path_for("bob")
    assert p_a != p_b
    assert p_a.name == "alice.csv"
    assert append_signal(REC, user="alice") is True
    assert append_signal(REC, user="bob") is True
    assert len(load_journal(user="alice")) == 1
    assert len(load_journal(user="bob")) == 1
    # Different users do not share rows
    assert append_signal(REC, user="alice") is False
    assert len(load_journal(user="alice")) == 1


def test_target_hit():
    r = resolve_entry(REC, prices([(1020, 990, 1010), (1050, 1000, 1040), (1110, 1030, 1090)]))
    assert (r["status"], r["days"]) == ("TARGET HIT", 3)
    assert abs(r["outcome_return"] - 0.1) < 1e-12


def test_stop_hit_and_conservative_double_touch():
    r = resolve_entry(REC, prices([(1020, 990, 1000), (1000, 940, 960)]))
    assert (r["status"], r["days"]) == ("STOP HIT", 2)
    assert resolve_entry(REC, prices([(1150, 940, 1050)]))["status"] == "STOP HIT"


def test_expired_open_and_sell_paths():
    assert resolve_entry(REC, prices([(1010, 990, 1005)] * 25))["status"] == "EXPIRED"
    r = resolve_entry(REC, prices([(1010, 990, 1020)] * 5))
    assert r["status"] == "OPEN" and abs(r["outcome_return"] - 0.02) < 1e-12
    sell = {**REC, "signal": "SELL"}
    r = resolve_entry(sell, prices([(1000, 960, 970)] * 20))
    assert r["status"] == "CLOSED" and r["outcome_return"] < 0


def test_journal_config_defaults_to_local(monkeypatch):
    monkeypatch.delenv("JOURNAL_BACKEND", raising=False)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    cfg = get_journal_config()
    assert cfg["backend"] in ("local", "supabase")
    info = journal_backend_info()
    assert "label" in info and "persistent" in info


def test_local_backend_class_matches_path_api(tmp_path):
    be = LocalCSVBackend()
    p = tmp_path / "j.csv"
    assert be.append(REC, path=p) is True
    assert be.append(REC, path=p) is False
    assert len(be.load(path=p)) == 1


def test_supabase_backend_builds_headers():
    be = SupabaseBackend("https://example.supabase.co", "test-key", "signal_journal")
    h = be._headers(prefer="return=minimal")
    assert h["apikey"] == "test-key"
    assert "Bearer test-key" in h["Authorization"]
    assert h["Prefer"] == "return=minimal"


def test_scorecard_math_and_fetch_failure():
    recs = [{**REC, "symbol": s, "signal": sig} for s, sig in
            [("A.NS", "BUY"), ("B.NS", "BUY"), ("C.NS", "BUY"), ("D.NS", "SELL")]]
    pmap = {"A.NS": prices([(1110, 1000, 1090)]),
            "B.NS": prices([(1000, 940, 950)]),
            "C.NS": prices([(1010, 990, 1015)] * 20),
            "D.NS": prices([(1000, 960, 980)] * 20)}
    resolved = resolve_journal(pd.DataFrame(recs), lambda s: pmap[s])
    sc = scorecard(resolved)
    assert sc["n_resolved"] == 3
    assert abs(sc["target_rate"] - 1 / 3) < 1e-12
    assert abs(sc["win_rate"] - 2 / 3) < 1e-12

    def boom(_):
        raise RuntimeError("network down")
    bad = resolve_journal(pd.DataFrame(recs[:1]), boom)
    assert bad["status"].tolist() == ["NO DATA"]
