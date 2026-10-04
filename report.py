# =============================
# report.py — one-click analysis reports (CSV / PDF)
# =============================
"""Build downloadable stock analysis packs from the live app context.

No Streamlit dependency — pure bytes for st.download_button.
"""
from __future__ import annotations

import csv
import io
from datetime import datetime, timezone
from typing import Any

import pandas as pd

# Soft dependency for PDF
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, HRFlowable,
    )
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _safe(v: Any, fmt: str | None = None) -> str:
    if v is None:
        return "—"
    try:
        if pd.isna(v):
            return "—"
    except (TypeError, ValueError):
        pass
    if fmt == "pct":
        try:
            return f"{float(v) * 100:.2f}%"
        except (TypeError, ValueError):
            return str(v)
    if fmt == "pct1":
        try:
            return f"{float(v) * 100:.1f}%"
        except (TypeError, ValueError):
            return str(v)
    if fmt == "num2":
        try:
            return f"{float(v):,.2f}"
        except (TypeError, ValueError):
            return str(v)
    if fmt == "num1":
        try:
            return f"{float(v):.1f}"
        except (TypeError, ValueError):
            return str(v)
    return str(v)


def build_report_dict(
    *,
    display_name: str,
    symbol: str,
    signal: str,
    confidence: float,
    model_type: str,
    use_global: bool,
    thresholds: tuple,
    metrics: dict,
    risk: dict,
    plan: dict,
    sr: dict,
    last_close: float,
    day_change: float,
    currency: str = "",
    data_asof: str = "",
) -> dict:
    """Flatten analysis into a serializable report map."""
    entry, exit_ = thresholds if thresholds and len(thresholds) >= 2 else (None, None)
    return {
        "generated_at": _utc_now(),
        "name": display_name,
        "symbol": symbol,
        "signal": signal,
        "probability_up": confidence,
        "model_type": "Global (pooled)" if use_global else model_type,
        "entry_threshold": entry,
        "exit_threshold": exit_,
        "last_close": last_close,
        "day_change": day_change,
        "currency": currency,
        "data_asof": data_asof,
        "accuracy": metrics.get("accuracy"),
        "baseline_accuracy": metrics.get("baseline_accuracy"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "risk_score": risk.get("score"),
        "risk_level": risk.get("level"),
        "vol_ann": risk.get("volatility_annualized"),
        "atr_pct": risk.get("atr_pct"),
        "max_dd_1y": risk.get("max_drawdown_1y"),
        "plan_entry": plan.get("entry"),
        "plan_stop": plan.get("stop"),
        "plan_target": plan.get("target"),
        "plan_rr": plan.get("reward_risk"),
        "stop_basis": plan.get("stop_basis"),
        "target_basis": plan.get("target_basis"),
        "support": sr.get("support"),
        "resistance": sr.get("resistance"),
        "source": metrics.get("source", "per-stock"),
        "disclaimer": (
            "Educational decision-support only — not investment advice. "
            "Past performance does not guarantee future results."
        ),
    }


def report_to_csv_bytes(report: dict) -> bytes:
    """Single-row CSV of key fields + a notes section as extra rows."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["field", "value"])
    order = [
        "generated_at", "name", "symbol", "signal", "probability_up",
        "model_type", "source", "entry_threshold", "exit_threshold",
        "last_close", "day_change", "data_asof",
        "accuracy", "baseline_accuracy", "precision", "recall",
        "risk_score", "risk_level", "vol_ann", "atr_pct", "max_dd_1y",
        "plan_entry", "plan_stop", "plan_target", "plan_rr",
        "stop_basis", "target_basis", "support", "resistance", "disclaimer",
    ]
    for k in order:
        if k in report:
            w.writerow([k, report[k]])
    for k, v in report.items():
        if k not in order:
            w.writerow([k, v])
    return buf.getvalue().encode("utf-8-sig")


def report_to_pdf_bytes(report: dict) -> bytes:
    """Multi-section PDF summary. Requires reportlab."""
    if not HAS_REPORTLAB:
        raise RuntimeError(
            "PDF export needs reportlab. Install with: pip install reportlab"
        )

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        topMargin=0.7 * inch, bottomMargin=0.7 * inch,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title2", parent=styles["Heading1"], fontSize=18, spaceAfter=6,
    )
    h2 = ParagraphStyle(
        "H2", parent=styles["Heading2"], fontSize=12, spaceBefore=12, spaceAfter=6,
    )
    body = ParagraphStyle(
        "Body2", parent=styles["Normal"], fontSize=9, leading=12,
    )
    small = ParagraphStyle(
        "Small", parent=styles["Normal"], fontSize=8, textColor=colors.grey, leading=10,
    )

    story = []
    cur = report.get("currency") or ""
    story.append(Paragraph("AI Stock Trend Predictor — Analysis Report", title_style))
    story.append(Paragraph(
        f"{_safe(report.get('name'))} · <b>{_safe(report.get('symbol'))}</b>",
        styles["Heading2"],
    ))
    story.append(Paragraph(
        f"Generated {_safe(report.get('generated_at'))} · "
        f"Data as of {_safe(report.get('data_asof'))}",
        small,
    ))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))

    # Signal block
    story.append(Paragraph("Signal", h2))
    sig = _safe(report.get("signal"))
    prob = _safe(report.get("probability_up"), "pct1")
    rows = [
        ["Signal", sig],
        ["Probability (meaningful up, 1d)", prob],
        ["Model", _safe(report.get("model_type"))],
        ["Source", _safe(report.get("source"))],
        ["Entry threshold", _safe(report.get("entry_threshold"), "num2")],
        ["Exit threshold", _safe(report.get("exit_threshold"), "num2")],
        ["Last close", f"{cur}{_safe(report.get('last_close'), 'num2')}"],
        ["Day change", _safe(report.get("day_change"), "pct")],
    ]
    story.append(_table(rows))

    story.append(Paragraph("Risk", h2))
    story.append(_table([
        ["Risk score", f"{_safe(report.get('risk_score'), 'num1')} / 10 ({_safe(report.get('risk_level'))})"],
        ["Ann. volatility", _safe(report.get("vol_ann"), "pct")],
        ["ATR %", _safe(report.get("atr_pct"), "pct")],
        ["Max drawdown (1y)", _safe(report.get("max_dd_1y"), "pct")],
    ]))

    story.append(Paragraph("Trade plan (long reference)", h2))
    story.append(_table([
        ["Entry", f"{cur}{_safe(report.get('plan_entry'), 'num2')}"],
        ["Stop", f"{cur}{_safe(report.get('plan_stop'), 'num2')}"],
        ["Target", f"{cur}{_safe(report.get('plan_target'), 'num2')}"],
        ["Reward : risk", f"1 : {_safe(report.get('plan_rr'), 'num1')}"],
        ["Stop basis", _safe(report.get("stop_basis"))],
        ["Target basis", _safe(report.get("target_basis"))],
        ["Support", f"{cur}{_safe(report.get('support'), 'num2')}"],
        ["Resistance", f"{cur}{_safe(report.get('resistance'), 'num2')}"],
    ]))

    story.append(Paragraph("Hold-out metrics", h2))
    story.append(_table([
        ["Accuracy", _safe(report.get("accuracy"), "pct")],
        ["Baseline", _safe(report.get("baseline_accuracy"), "pct")],
        ["Precision", _safe(report.get("precision"), "pct")],
        ["Recall", _safe(report.get("recall"), "pct")],
    ]))

    story.append(Spacer(1, 14))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    story.append(Spacer(1, 6))
    story.append(Paragraph(_safe(report.get("disclaimer")), small))
    story.append(Paragraph(
        "Long-only tool. SELL means exit to cash / avoid fresh longs — not a short recommendation.",
        small,
    ))

    doc.build(story)
    return buf.getvalue()


def _table(rows: list[list[str]]) -> Table:
    t = Table(rows, colWidths=[2.2 * inch, 4.0 * inch])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#333333")),
        ("TEXTCOLOR", (1, 0), (1, -1), colors.HexColor("#111111")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f4f6f8")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fafbfc")]),
        ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#dddddd")),
    ]))
    return t


def filename_stem(symbol: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(symbol))
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"stock_report_{safe}_{day}"
