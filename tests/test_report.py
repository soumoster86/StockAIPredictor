"""report.py — CSV/PDF analysis packs."""
import pandas as pd

from report import (
    HAS_REPORTLAB,
    build_report_dict,
    filename_stem,
    report_to_csv_bytes,
    report_to_pdf_bytes,
)


def _sample_report():
    return build_report_dict(
        display_name="Test Corp",
        symbol="TEST.NS",
        signal="BUY",
        confidence=0.72,
        model_type="Ensemble (NN + RF + XGB)",
        use_global=False,
        thresholds=(0.55, 0.45),
        metrics={
            "accuracy": 0.58,
            "baseline_accuracy": 0.52,
            "precision": 0.60,
            "recall": 0.55,
            "source": "per-stock",
        },
        risk={
            "score": 4.5,
            "level": "Medium",
            "volatility_annualized": 0.28,
            "atr_pct": 0.02,
            "max_drawdown_1y": -0.22,
        },
        plan={
            "entry": 1000.0,
            "stop": 950.0,
            "target": 1100.0,
            "reward_risk": 2.0,
            "stop_basis": "1.5× ATR",
            "target_basis": "resistance",
        },
        sr={"support": 940.0, "resistance": 1100.0},
        last_close=1000.0,
        day_change=0.012,
        currency="₹",
        data_asof="2026-07-20",
    )


def test_build_report_dict_fields():
    r = _sample_report()
    assert r["symbol"] == "TEST.NS"
    assert r["signal"] == "BUY"
    assert r["probability_up"] == 0.72
    assert r["plan_entry"] == 1000.0
    assert "Educational" in r["disclaimer"]
    assert r["generated_at"]


def test_report_to_csv_bytes_utf8_and_fields():
    raw = report_to_csv_bytes(_sample_report())
    assert isinstance(raw, (bytes, bytearray))
    text = raw.decode("utf-8-sig")
    assert "field,value" in text
    assert "TEST.NS" in text
    assert "BUY" in text
    assert "disclaimer" in text


def test_filename_stem_sanitizes():
    assert filename_stem("TATA POWER.NS").startswith("stock_report_")
    assert " " not in filename_stem("A B.NS")
    assert ".NS" not in filename_stem("X.NS") or "X" in filename_stem("X.NS")


def test_report_to_pdf_bytes_or_skip():
    if not HAS_REPORTLAB:
        try:
            report_to_pdf_bytes(_sample_report())
            assert False, "expected RuntimeError without reportlab"
        except RuntimeError as e:
            assert "reportlab" in str(e).lower()
        return
    pdf = report_to_pdf_bytes(_sample_report())
    assert pdf[:4] == b"%PDF"
    assert len(pdf) > 500
