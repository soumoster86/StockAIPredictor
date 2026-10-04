"""Screener fix: relative entry, 10-day ranking model, edge filter off."""
import pandas as pd

from model import SCREEN_HORIZON, SCREEN_TOP_PCT, load_screen_model, rank_buy_candidates


def _scan(n=40):
    # Probabilities cluster below 0.55, like the real model (base rate ~40%).
    probs = [0.30 + 0.005 * i for i in range(n)]            # 0.300 … 0.495
    return pd.DataFrame({
        "Symbol": [f"S{i}.NS" for i in range(n)],
        "Screen": "SELL", "Probability Up": probs, "Risk": 4.0,
        "Test Acc": 0.55, "Baseline": 0.60,                  # always below baseline
        "Traded Value": 5.0, "Price Band": None,
        "Buy Score": [p * 100 for p in probs],
    })


def test_defaults_match_the_tested_configuration():
    assert SCREEN_HORIZON == 10 and SCREEN_TOP_PCT == 0.05


def test_relative_entry_finds_candidates_when_absolute_cutoff_cannot():
    scan = _scan()
    assert rank_buy_candidates(scan, top_pct=None).empty     # old rule: nothing
    picks = rank_buy_candidates(scan)                         # new default
    assert picks["Symbol"].tolist() == ["S39.NS", "S38.NS"]  # top 5% of 40
    assert picks["Rank"].tolist() == [1, 2]


def test_relative_pool_is_the_liquid_universe():
    scan = _scan()
    scan.loc[scan.index[-10:], "Traded Value"] = 0.1          # top names illiquid
    picks = rank_buy_candidates(scan)
    assert set(picks["Symbol"]) <= {f"S{i}.NS" for i in range(30)}
    assert len(picks) >= 1


def test_require_edge_is_off_by_default():
    scan = _scan()
    assert len(rank_buy_candidates(scan)) == 2
    assert rank_buy_candidates(scan, require_edge=True).empty


def test_screen_model_prefers_ten_day_artifact(tmp_path, monkeypatch):
    import model as M
    loaded = []

    def fake_load(h, directory=None):
        loaded.append(h)
        return {"horizon": h} if h == 1 else None

    monkeypatch.setattr(M, "load_global_model", fake_load)
    assert load_screen_model(tmp_path) == {"horizon": 1}      # falls back to 1-day
    assert loaded == [SCREEN_HORIZON, 1]


def test_quick_scan_global_scores_against_bundle_horizon():
    from data import FEATURES
    from model import quick_scan_global, train_global_predictor
    from tests.test_tier2 import synthetic

    frames = {f"S{i}": synthetic(seed=i) for i in range(2)}
    predictor, scaler, _ = train_global_predictor(frames, "Target_10", model_type="Neural Network")
    bundle = {"predictor": predictor, "scaler": scaler, "features": list(FEATURES),
              "horizon": 10}
    scan = quick_scan_global(frames["S0"], bundle=bundle)
    assert scan["model"] == "Global 10d" and scan["source"] == "global"
