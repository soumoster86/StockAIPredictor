# =============================
# app.py — thin Streamlit entrypoint
# =============================
"""AI Stock Trend Predictor.

Presentation lives under `ui/`; core logic stays in root modules
(`data`, `model`, `auth`, `journal`). Run:  streamlit run app.py
"""
from pathlib import Path

import streamlit as st

from auth import require_login
from model import (
    predict, compute_risk_score, find_support_resistance, compute_trade_plan,
    global_model_available,
)
from ui.theme import inject_global_css, page_hero, footer_bar
from ui.sidebar import render_sidebar
from ui.header import render_header
from ui.tabs import (
    render_prediction_tab, render_scanner_tab, render_plan_tab,
    render_backtest_tab, render_walkforward_tab, render_journal_tab,
    render_charts_tab,
)
from ui.services import get_data, get_trained

st.set_page_config(
    page_title="AI Stock Trend Predictor",
    page_icon=str(Path(__file__).parent / "src" / "icon_256.png"),
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_global_css()
current_user = require_login()

sidebar = render_sidebar()
symbol = sidebar["symbol"]
display_name = sidebar["display_name"]
model_type = sidebar["model_type"]
calibrate = sidebar["calibrate"]
use_global = sidebar["use_global"]
stocks = sidebar["stocks"]

chips = [f"👤 {current_user}"]
if use_global and global_model_available():
    chips.append("🌐 Global model")
else:
    chips.append(f"🧠 {model_type.split('(')[0].strip()}")
if calibrate:
    chips.append("📏 Calibrated")

page_hero(
    "Market workspace",
    "Signals, risk, and validation for decision support — not automated trading.",
    chips=chips,
)

if not symbol:
    st.info("Pick a stock from the sidebar to get started.")
    st.stop()

raw = get_data(symbol)
if raw.empty:
    st.error("Could not fetch data — check the symbol or try again later.")
    st.stop()
if len(raw) < 400:
    st.error("Not enough price history for this symbol (need roughly 2 years).")
    st.stop()

data, predictor, scaler, metrics, test_probs, thresholds, test_index = get_trained(
    symbol, model_type, calibrate, use_global,
)

currency = "₹" if symbol.endswith((".NS", ".BO")) else ""
signal, confidence = predict(predictor, scaler, data, thresholds)
risk = compute_risk_score(data)
sr = find_support_resistance(data)
plan = compute_trade_plan(data, sr['support'], sr['resistance'])

render_header(
    display_name, symbol, data, predictor, scaler, thresholds,
    signal, confidence, currency,
)

ctx = {
    "current_user": current_user,
    "stocks": stocks,
    "symbol": symbol,
    "display_name": display_name,
    "model_type": model_type,
    "calibrate": calibrate,
    "use_global": use_global,
    "data": data,
    "predictor": predictor,
    "scaler": scaler,
    "metrics": metrics,
    "test_probs": test_probs,
    "thresholds": thresholds,
    "test_index": test_index,
    "signal": signal,
    "confidence": confidence,
    "risk": risk,
    "sr": sr,
    "plan": plan,
    "currency": currency,
}

# Segmented section nav — always renders as separate controls (st.tabs was
# collapsing into one continuous label on some Streamlit builds).
_SECTIONS = [
    "Prediction",
    "Screener",
    "Trade plan",
    "Backtest",
    "Walk-forward",
    "Journal",
    "Charts",
]
# Migrate older label so saved session still opens the screener
if st.session_state.get("main_section") in ("Best buys", "Best to Buy", "Scanner"):
    st.session_state["main_section"] = "Screener"
st.markdown("**Section**")
section = st.radio(
    "Section",
    options=_SECTIONS,
    horizontal=True,
    label_visibility="collapsed",
    key="main_section",
)
st.divider()

if section == "Prediction":
    render_prediction_tab(ctx)
elif section == "Screener":
    render_scanner_tab(ctx)
elif section == "Trade plan":
    render_plan_tab(ctx)
elif section == "Backtest":
    render_backtest_tab(ctx)
elif section == "Walk-forward":
    render_walkforward_tab(ctx)
elif section == "Journal":
    render_journal_tab(ctx)
else:
    render_charts_tab(ctx)

footer_bar()
