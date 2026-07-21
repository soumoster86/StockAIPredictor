"""Colors, table stylers, feature blurbs, and small HTML helpers."""
import html as html_lib

import pandas as pd

from ui.theme import ACCENT, AMBER, BLUE, RED, TEXT, TEXT_MUTED

# Public aliases used across the app
GREEN = ACCENT
# keep RED, AMBER from theme via re-export
__all__ = [
    "GREEN", "RED", "AMBER", "BLUE",
    "style_map", "color_signal", "color_status", "color_pos_neg",
    "describe_feature", "factor_row_html", "signal_class",
]


def style_map(styler, func, subset):
    """pandas renamed Styler.applymap -> Styler.map; support both."""
    fn = getattr(styler, "map", None) or styler.applymap
    return fn(func, subset=subset)


def color_signal(v):
    return {
        "BUY": f"color: {GREEN}; font-weight: 700",
        "SELL": f"color: {RED}; font-weight: 700",
        "HOLD": f"color: {AMBER}; font-weight: 600",
    }.get(v, "")


def color_status(v):
    return {
        "TARGET HIT": f"color: {GREEN}; font-weight: 700",
        "STOP HIT": f"color: {RED}; font-weight: 700",
        "EXPIRED": f"color: {TEXT_MUTED}",
        "OPEN": f"color: {BLUE}; font-weight: 600",
        "CLOSED": f"color: {TEXT}",
        "NO DATA": f"color: {TEXT_MUTED}",
    }.get(v, "")


def color_pos_neg(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return ""
    if pd.isna(v) or v == 0:
        return ""
    return f"color: {GREEN}; font-weight: 600" if v > 0 else f"color: {RED}; font-weight: 600"


def signal_class(signal: str) -> str:
    return {"BUY": "buy", "SELL": "sell", "HOLD": "hold"}.get(signal, "hold")


def factor_row_html(label: str, contribution: float) -> str:
    """Inline-styled factor row (no CSS classes)."""
    pos = contribution >= 0
    accent = GREEN if pos else RED
    bg = "rgba(54,179,126,0.12)" if pos else "rgba(240,113,120,0.12)"
    sign = "+" if pos else ""
    safe = html_lib.escape(label)
    return (
        f'<div style="display:flex;justify-content:space-between;gap:0.75rem;'
        f'padding:0.55rem 0.7rem;margin-bottom:0.35rem;border-radius:10px;'
        f'background:{bg};border:1px solid rgba(255,255,255,0.08);'
        f'border-left:3px solid {accent};">'
        f'<div style="font-size:0.88rem;color:#e8ecf1;">{safe}</div>'
        f'<div style="font-size:0.82rem;font-weight:700;color:{accent};'
        f'white-space:nowrap;">{sign}{contribution * 100:.1f}%</div>'
        f"</div>"
    )


def describe_feature(feat, value):
    if feat == 'RSI':
        zone = " (overbought)" if value > 70 else " (oversold)" if value < 30 else ""
        return f"RSI at {value:.0f}{zone}"
    if feat == 'MACD_hist':
        return "MACD bullish crossover" if value > 0 else "MACD bearish crossover"
    if feat == 'Vol_ratio':
        return f"Volume {abs(value):.0%} {'above' if value > 0 else 'below'} its 20-day average"
    if feat == 'Close_MA20':
        return f"Price {abs(value):.1%} {'above' if value > 0 else 'below'} its 20-day average"
    if feat == 'MA20_MA50':
        return f"Short-term trend {'above' if value > 0 else 'below'} long-term trend ({value:+.1%})"
    if feat == 'Return':
        return f"Yesterday's move: {value:+.1%}"
    if feat in ('Mom5', 'Mom10', 'Mom20'):
        days = feat.replace('Mom', '')
        return f"{days}-day momentum: {value:+.1%}"
    if feat == 'Vol20':
        return f"Daily volatility at {value:.2%}" + (" (elevated)" if value > 0.02 else "")
    if feat == 'ATR_pct':
        return f"Daily trading range {value:.2%} of price"
    if feat == 'Nifty_Ret':
        return f"NIFTY moved {value:+.1%} yesterday"
    if feat == 'Nifty_Mom20':
        return f"NIFTY 20-day trend: {value:+.1%}"
    if feat == 'Rel_Str5':
        side = "Outperforming" if value > 0 else "Underperforming"
        return f"{side} NIFTY by {abs(value):.1%} over 5 days"
    if feat == 'Rel_Str20':
        side = "Outperforming" if value > 0 else "Underperforming"
        return f"{side} NIFTY by {abs(value):.1%} over 20 days"
    return f"{feat}: {value:.3f}"
