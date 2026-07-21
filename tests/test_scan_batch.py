"""Pure batch-screener helpers (no Streamlit runtime required)."""

import pandas as pd

from screener import (
    SCAN_BATCH,
    filter_rankings_to_watchlist,
    load_rankings,
    merge_scan_frames,
    normalize_stock_items,
    save_rankings,
    slice_scan_batch,
)
from ui.services import cap_scan_items


def test_normalize_dedupes_and_preserves_order():
    items = [
        ("A", "A.NS"), ("B", "B.NS"), ("A2", "A.NS"), ("C", "C.NS"),
    ]
    out = normalize_stock_items(items)
    assert out == [("A", "A.NS"), ("B", "B.NS"), ("C", "C.NS")]


def test_slice_scan_batch_walks_full_list():
    items = [(f"N{i}", f"S{i}.NS") for i in range(200)]
    b0, n0, total, done0 = slice_scan_batch(items, offset=0, batch_size=80)
    assert total == 200
    assert len(b0) == 80
    assert n0 == 80
    assert done0 is False

    b1, n1, _, done1 = slice_scan_batch(items, offset=80, batch_size=80)
    assert len(b1) == 80
    assert n1 == 160
    assert done1 is False

    b2, n2, _, done2 = slice_scan_batch(items, offset=160, batch_size=80)
    assert len(b2) == 40
    assert n2 == 200
    assert done2 is True

    b3, n3, _, done3 = slice_scan_batch(items, offset=200, batch_size=80)
    assert b3 == []
    assert n3 == 200
    assert done3 is True


def test_merge_scan_frames_keeps_latest_symbol_and_resorts():
    a = pd.DataFrame([
        {"Symbol": "A.NS", "Buy Score": 40, "Probability Up": 0.5},
        {"Symbol": "B.NS", "Buy Score": 70, "Probability Up": 0.6},
    ])
    b = pd.DataFrame([
        {"Symbol": "A.NS", "Buy Score": 90, "Probability Up": 0.7},
        {"Symbol": "C.NS", "Buy Score": 50, "Probability Up": 0.55},
    ])
    m = merge_scan_frames(a, b)
    assert list(m["Symbol"]) == ["A.NS", "B.NS", "C.NS"]
    assert float(m.loc[m["Symbol"] == "A.NS", "Buy Score"].iloc[0]) == 90


def test_cap_scan_items_matches_batch_size():
    items = [(f"N{i}", f"S{i}.NS") for i in range(100)]
    assert len(cap_scan_items(items, SCAN_BATCH)) == SCAN_BATCH


def test_save_and_load_rankings_roundtrip(tmp_path):
    df = pd.DataFrame([
        {"Name": "Alpha", "Symbol": "A.NS", "Buy Score": 80,
         "Probability Up": 0.7, "Screen": "BUY"},
        {"Name": "Beta", "Symbol": "B.NS", "Buy Score": 55,
         "Probability Up": 0.56, "Screen": "BUY"},
    ])
    meta = {
        "generated_at": "2026-07-20T12:00:00Z",
        "n_scored": 2,
        "engine": "global",
    }
    save_rankings(df, [("Z.NS", "no data")], meta, directory=tmp_path)
    loaded, meta2 = load_rankings(directory=tmp_path, max_age_hours=None)
    assert len(loaded) == 2
    assert meta2["n_scored"] == 2
    assert loaded.iloc[0]["Symbol"] == "A.NS"


def test_filter_rankings_to_watchlist():
    df = pd.DataFrame([
        {"Name": "X", "Symbol": "AAA.NS", "Buy Score": 90, "Probability Up": 0.8},
        {"Name": "Y", "Symbol": "BBB.NS", "Buy Score": 70, "Probability Up": 0.6},
        {"Name": "Z", "Symbol": "CCC.NS", "Buy Score": 50, "Probability Up": 0.5},
    ])
    stocks = {"Alpha Co": "AAA.NS", "Beta Co": "BBB.NS"}
    out = filter_rankings_to_watchlist(df, stocks)
    assert list(out["Symbol"]) == ["AAA.NS", "BBB.NS"]
    assert out.loc[out["Symbol"] == "AAA.NS", "Name"].iloc[0] == "Alpha Co"
