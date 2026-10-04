"""rankings_log — build row + skip when unconfigured."""
import rankings_log as RL


def test_build_log_row_fields():
    meta = {
        "generated_at": "2026-07-28T18:00:00Z",
        "runner": "github-actions",
        "run_id": "123",
        "run_url": "https://github.com/x/y/actions/runs/123",
        "workflow": "Nightly rankings",
        "watchlist": "stocks_universe.csv",
        "engine": "global",
        "n_requested": 100,
        "n_scored": 90,
        "n_failed": 10,
        "batch_size": 80,
        "elapsed_s": 12.5,
    }
    row = RL.build_log_row(
        meta,
        status="success",
        top_symbols=["A.NS", "B.NS"],
        max_symbols=None,
    )
    assert row["status"] == "success"
    assert row["n_scored"] == 90
    assert row["top_symbols"] == "A.NS,B.NS"
    assert row["runner"] == "github-actions"
    assert isinstance(row["meta"], dict)


def test_log_rankings_run_skips_without_config(monkeypatch):
    monkeypatch.setattr(
        RL, "get_rankings_log_config",
        lambda: {
            "enabled": True,
            "supabase_url": "",
            "supabase_key": "",
            "table": "rankings_run_log",
            "configured": False,
        },
    )
    r = RL.log_rankings_run({"n_scored": 1}, status="success")
    assert r["skipped"] is True
    assert r["ok"] is True


def test_log_rankings_run_dry_run(monkeypatch):
    monkeypatch.setattr(
        RL, "get_rankings_log_config",
        lambda: {
            "enabled": True,
            "supabase_url": "https://example.supabase.co",
            "supabase_key": "key",
            "table": "rankings_run_log",
            "configured": True,
        },
    )
    r = RL.log_rankings_run(
        {"n_scored": 5, "runner": "local"},
        status="success",
        dry_run=True,
    )
    assert r["ok"] is True
    assert r["reason"] == "dry_run"
    assert r["row"]["n_scored"] == 5


def test_log_rankings_run_insert_called(monkeypatch):
    called = {}

    def fake_insert(url, key, table, row):
        called["url"] = url
        called["table"] = table
        called["row"] = row

    monkeypatch.setattr(
        RL, "get_rankings_log_config",
        lambda: {
            "enabled": True,
            "supabase_url": "https://example.supabase.co",
            "supabase_key": "key",
            "table": "rankings_run_log",
            "configured": True,
        },
    )
    monkeypatch.setattr(RL, "_postgrest_insert", fake_insert)
    r = RL.log_rankings_run({"n_scored": 3}, status="success")
    assert r["ok"] is True
    assert called["table"] == "rankings_run_log"
    assert called["row"]["n_scored"] == 3
