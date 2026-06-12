# =============================
# app.py
# =============================
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data import fetch_data, add_features, FEATURES
from model import (
    train_model, predict, backtest, walk_forward,
    multi_horizon_forecast, explain_prediction,
    compute_risk_score, rating_from_prob, MODEL_TYPES, HAS_XGB,
    find_support_resistance, compute_trade_plan, position_size,
)
from journal import (
    load_journal, append_signal, resolve_journal, scorecard,
    JOURNAL_FILE, MAX_HOLD_DAYS,
)
from auth import require_login, logout_button

st.set_page_config(page_title="AI Stock Trend Predictor", page_icon="📈", layout="wide")

current_user = require_login()  # everything below runs only when authenticated

STOCKS_FILE = Path(__file__).parent / "stocks.csv"
DEFAULT_STOCKS = {"Reliance Industries": "RELIANCE.NS", "Infosys": "INFY.NS"}

# ---------------------------------------------------------------
# Plain-language explanations, used as (?) tooltips
# ---------------------------------------------------------------
HELP = {
    "strategy_return": "Total profit or loss if you had followed the model's signals over the test period, after trading costs. Example: +12% means ₹100 would have become ₹112.",
    "buy_hold": "What you'd have made by simply buying on day one and holding — no signals, no trading. The benchmark: if the strategy can't beat this, the model adds no value.",
    "sharpe": "Return earned per unit of risk taken (annualized). Was the profit worth the rollercoaster? Below 0 = losing, 0–1 = weak, above 1 = good, above 2 = excellent.",
    "max_drawdown": "The worst peak-to-bottom fall of your money. -30% means at some point you'd have been down 30% from your highest point — a gut-check for pain tolerance.",
    "win_rate": "Of all days the strategy held the stock, the percentage that ended up. Around 50% is a coin flip.",
    "exposure": "Share of days invested in the stock rather than sitting in cash. 60% = in the market 6 days out of 10.",
    "trades": "Number of times the strategy bought in. More trades = more costs, so fewer, better trades win.",
    "cost": "Brokerage, taxes and slippage charged on every position change. 0.10% per change ≈ 0.2% for a full round trip.",
    "accuracy": "How often the model's up/down guess was right on data it never saw in training. Only meaningful if it beats the baseline →",
    "baseline": "What a 'dumb' model scores by always predicting the most common outcome. If accuracy doesn't beat this, the model learned nothing.",
    "precision": "When the model said 'up', how often it was right. High precision = trustworthy BUY calls.",
    "recall": "Of all days the price actually rose, how many the model caught. High recall = it doesn't miss rallies.",
    "rsi": "Relative Strength Index — a 0–100 speedometer of buying vs selling pressure. Above 70: possibly overbought. Below 30: possibly oversold.",
    "last_close": "Most recent closing price, with change vs the previous trading day.",
    "prob_up": "The model's estimated chance that the price ends higher over this horizon. 50% = coin flip; further from 50% = more conviction.",
    "rating": "Plain-language band for the probability: Strong Buy >80%, Buy 65–80%, Neutral 45–65%, Sell <45%.",
    "risk_score": "1 (calm) to 10 (wild), blending the stock's volatility, daily trading range, and worst fall of the past year. Higher risk = consider a smaller position size.",
    "horizon_accuracy": "Each horizon has its own model, tested on recent data it never trained on. Longer horizons overlap heavily, so treat their accuracy as optimistic.",
    "model_type": "Ensemble averages a neural network, XGBoost and a Random Forest — usually the most reliable choice. LSTM/GRU read the last 20 days as a sequence instead of one day at a time; slower to train and rarely better on this little data, but worth comparing in the Walk-Forward tab.",
    "support": "A price floor where the stock repeatedly stopped falling and bounced (recent swing lows). Buyers tend to step in here; a fall through it is a warning sign.",
    "resistance": "A price ceiling where the stock repeatedly stopped rising and turned back (recent swing highs). Sellers tend to appear here; a break above it is often bullish.",
    "entry": "The price you'd pay to open the trade — here, the latest closing price.",
    "stop_loss": "The price at which you exit to cap the loss if the trade goes wrong. Placed below support or 1.5× the stock's typical daily range, so normal wiggle doesn't knock you out.",
    "target": "The price at which you'd take profit — the nearest resistance, or 3× the typical daily range if no resistance is close enough to be worth the risk.",
    "reward_risk": "How much you stand to gain per rupee risked. 1:2.5 means risking ₹1 to potentially make ₹2.50. Professionals rarely take trades below 1:1.5.",
    "capital": "Your total trading capital — the pot you size every position from.",
    "risk_per_trade": "The percentage of capital you accept losing if this one trade hits its stop loss. The classic rule is 1–2%: it takes dozens of consecutive losses to do serious damage.",
    "position_shares": "Shares to buy so that hitting the stop loses exactly your chosen risk amount: (capital × risk%) ÷ (entry − stop). Capped so the position never costs more than your capital.",
    "calibrate": "Remaps the model's probabilities so they match reality: if the raw model says '75%' but such days only rise 60% of the time, calibrated output says 60%. Makes percentages honest; doesn't make the model smarter, and on limited data can slightly blur sharp predictions.",
    "brier": "Average squared error of the probabilities (lower = better). The number to beat is the baseline — what you'd score by always predicting the historical average. Beating it means the probabilities carry real information.",
    "ece": "Expected Calibration Error: the average gap between stated probability and observed frequency. 0.02 = percentages are trustworthy; 0.10+ = when the model says 70%, don't believe 70%.",
    "journal": "Backtests look backward; the journal looks forward. Each logged signal is later scored against what the market actually did — the most honest performance measure this app produces.",
    "target_rate": "Of resolved BUY plans, the share that reached the target before hitting the stop.",
    "journal_status": "TARGET HIT / STOP HIT: which level the price touched first. EXPIRED: neither within 20 trading days — scored at that day's close. OPEN: still running.",
}

# Human-readable descriptions for the explainability panel
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
    return f"{feat}: {value:.3f}"


@st.cache_data(ttl=3600)
def load_stock_list(file_or_path):
    df = pd.read_csv(file_or_path)
    df.columns = [c.strip().lower() for c in df.columns]
    if not {"name", "symbol"}.issubset(df.columns):
        raise ValueError("CSV must have 'Name' and 'Symbol' columns.")
    df = df.dropna(subset=["name", "symbol"])
    return dict(zip(df["name"].str.strip(), df["symbol"].str.strip().str.upper()))


@st.cache_data(ttl=3600, max_entries=20, show_spinner="Downloading price data...")
def get_data(symbol):
    return fetch_data(symbol)


@st.cache_resource(ttl=3600, max_entries=4, show_spinner="Training model...")
def get_trained(symbol, model_type, calibrate):
    data = add_features(get_data(symbol))
    return (data,) + train_model(data, model_type, calibrate)


@st.cache_data(ttl=3600, max_entries=4, show_spinner="Training one model per horizon (1/3/5/10/20 days)...")
def get_horizons(symbol, model_type):
    data = add_features(get_data(symbol))
    return multi_horizon_forecast(data, model_type)


@st.cache_data(ttl=3600, max_entries=2, show_spinner="Running walk-forward validation (trains one model per fold)...")
def run_walk_forward(symbol, model_type, calibrate):
    data = add_features(get_data(symbol))
    return walk_forward(data, model_type, calibrate=calibrate)


# ---------- Sidebar ----------
with st.sidebar:
    logout_button()
    st.divider()
    st.header("⚙️ Settings")

    uploaded = st.file_uploader(
        "Upload your own stock list (optional)", type="csv",
        help="A CSV with two columns: Name, Symbol (Yahoo Finance tickers, "
             "e.g. TATAPOWER.NS). Replaces the default list for this session.",
    )

    stocks, list_source = {}, ""
    try:
        if uploaded is not None:
            stocks = load_stock_list(uploaded)
            list_source = f"your uploaded list ({len(stocks)} stocks)"
        elif STOCKS_FILE.exists():
            stocks = load_stock_list(str(STOCKS_FILE))
            list_source = f"stocks.csv ({len(stocks)} stocks)"
    except ValueError as e:
        st.error(f"Could not read stock list: {e}")

    if not stocks:
        stocks = DEFAULT_STOCKS
        list_source = "built-in fallback list"

    st.caption(f"Showing {list_source}. Edit `stocks.csv` next to app.py to change the default list.")

    choice = st.selectbox(
        "Select a stock",
        options=list(stocks.keys()) + ["Custom symbol…"],
        help="Type to search the list. Pick 'Custom symbol…' for any other ticker.",
    )

    if choice == "Custom symbol…":
        symbol = st.text_input("Yahoo Finance symbol", placeholder="e.g. TATAPOWER.NS or AAPL").strip().upper()
        display_name = symbol
    else:
        symbol = stocks[choice]
        display_name = choice

    st.caption("NSE tickers end in **.NS**. Any Yahoo Finance symbol works in custom mode.")

    model_type = st.selectbox("Model", MODEL_TYPES, index=0, help=HELP["model_type"])
    if model_type.startswith("Ensemble") and not HAS_XGB:
        st.warning("xgboost is not installed — the ensemble will use NN + Random Forest only. "
                   "Run `pip install xgboost` to enable it.")
    if model_type in ("LSTM", "GRU"):
        st.caption("Sequence models read the last 20 trading days per prediction. "
                   "Training takes a little longer.")

    calibrate = st.checkbox("Calibrate probabilities", value=False, help=HELP["calibrate"])

    with st.expander("ℹ️ How this app works"):
        st.markdown(
            "A small neural network learns patterns in price, momentum, volatility "
            "and volume, then predicts whether the price will rise over several "
            "horizons. Signals are tested on past data the model never saw, "
            "including realistic trading costs. **Educational tool — not "
            "investment advice.**"
        )

st.title("📈 AI Stock Trend Predictor")

if not symbol:
    st.info("👈 Pick a stock from the sidebar to get started.")
    st.stop()

raw = get_data(symbol)
if raw.empty:
    st.error("Could not fetch data — check the symbol or try again later.")
    st.stop()
if len(raw) < 400:
    st.error("Not enough price history for this symbol (need roughly 2 years).")
    st.stop()

data, predictor, scaler, metrics, test_probs, thresholds, test_index = get_trained(symbol, model_type, calibrate)
entry_thr, exit_thr = thresholds

# ---------- Header ----------
last_close = float(data['Close'].iloc[-1])
prev_close = float(data['Close'].iloc[-2])
day_change = (last_close / prev_close - 1) * 100
currency = "₹" if symbol.endswith((".NS", ".BO")) else ""

signal, confidence = predict(predictor, scaler, data, thresholds)
risk = compute_risk_score(data)
sr = find_support_resistance(data)
plan = compute_trade_plan(data, sr['support'], sr['resistance'])

head_l, head_r = st.columns([3, 1])
with head_l:
    st.subheader(f"{display_name}  ·  `{symbol}`")
    if signal == "BUY":
        st.success(f"**Signal: BUY** — model confidence {confidence * 100:.1f}%", icon="📈")
    elif signal == "SELL":
        st.error(f"**Signal: SELL / move to cash** — model confidence {(1 - confidence) * 100:.1f}%", icon="📉")
    else:
        st.warning(f"**Signal: HOLD** — model is uncertain ({confidence * 100:.1f}%)", icon="⏸️")
    st.caption(
        f"Long-only strategy (SELL means exit to cash, never short). Trading "
        f"thresholds tuned on validation data: enter above {entry_thr:.2f}, exit below {exit_thr:.2f}."
    )
    if hasattr(predictor, "member_probs_last"):
        votes = predictor.member_probs_last(scaler.transform(data[FEATURES].values))
        spread = max(votes.values()) - min(votes.values())
        agreement = "models broadly agree" if spread < 0.10 else "models disagree — lower conviction"
        st.caption("🗳️ Model votes: " +
                   " · ".join(f"{k} {v * 100:.0f}%" for k, v in votes.items()) +
                   f" — {agreement}.")
with head_r:
    st.metric("Last Close", f"{currency}{last_close:,.2f}", delta=f"{day_change:+.2f}%",
              help=HELP["last_close"])

# ---------- Tabs ----------
tab_pred, tab_plan, tab_back, tab_wf, tab_journal, tab_charts = st.tabs(
    ["🔮 Prediction", "🎯 Trade Plan", "📊 Backtest", "🧪 Walk-Forward", "📝 Journal", "📉 Charts"]
)

with tab_pred:
    # --- Confidence gauge + rating + risk score ---
    g_col, r_col = st.columns([2, 1])

    with g_col:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            number={'suffix': "%", 'font': {'size': 40}},
            title={'text': "Probability of an UP move (1 day)"},
            gauge={
                'axis': {'range': [0, 100], 'ticksuffix': "%"},
                'bar': {'color': "#2c3e50", 'thickness': 0.25},
                'steps': [
                    {'range': [0, 45], 'color': "#f5b7b1"},    # Sell
                    {'range': [45, 65], 'color': "#fdebd0"},   # Neutral
                    {'range': [65, 80], 'color': "#d4efdf"},   # Buy
                    {'range': [80, 100], 'color': "#7dcea0"},  # Strong Buy
                ],
            },
        ))
        gauge.update_layout(height=260, margin=dict(l=30, r=30, t=60, b=10))
        st.plotly_chart(gauge, use_container_width=True)
        st.caption("Bands: 🔴 Sell <45% · 🟠 Neutral 45–65% · 🟢 Buy 65–80% · 🟢🟢 Strong Buy >80%")

    with r_col:
        rating = rating_from_prob(confidence)
        rating_icon = {"Strong Buy": "🟢🟢", "Buy": "🟢", "Neutral": "🟠", "Sell": "🔴"}[rating]
        st.metric("Rating (1 day)", f"{rating_icon} {rating}", help=HELP["rating"])

        risk_icon = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}[risk['level']]
        st.metric("Risk Score", f"{risk_icon} {risk['score']:.1f} / 10 ({risk['level']})",
                  help=HELP["risk_score"])
        st.caption(
            f"Annualized volatility {risk['volatility_annualized']:.0%} · "
            f"daily range {risk['atr_pct']:.1%} · "
            f"worst 1-yr fall {risk['max_drawdown_1y']:.0%}. "
            f"{'Higher risk → consider smaller position sizes.' if risk['level'] != 'Low' else ''}"
        )

    st.divider()

    # --- Multi-day predictions ---
    st.subheader("Multi-Day Outlook")
    horizons_df = get_horizons(symbol, model_type)
    if horizons_df.empty:
        st.info("Not enough history for multi-horizon forecasts.")
    else:
        st.dataframe(
            horizons_df,
            use_container_width=True, hide_index=True,
            column_config={
                "Probability Up": st.column_config.ProgressColumn(
                    "Probability Up", format="percent", min_value=0, max_value=1,
                    help=HELP["prob_up"],
                ),
                "Rating": st.column_config.TextColumn("Rating", help=HELP["rating"]),
                "Test Accuracy": st.column_config.NumberColumn(
                    "Test Accuracy", format="percent", help=HELP["horizon_accuracy"]),
                "Baseline": st.column_config.NumberColumn(
                    "Baseline", format="percent", help=HELP["baseline"]),
            },
        )
        st.caption(
            "One independent model per horizon. ⚠️ Longer horizons use overlapping "
            "windows, so their accuracy figures run optimistic — read them as a "
            "directional tilt, not a promise."
        )

    st.divider()

    # --- Why this signal? ---
    st.subheader(f"Why {signal}?")
    base_prob, contribs = explain_prediction(predictor, scaler, data)
    pos = [c for c in contribs if c['contribution'] > 0.005][:4]
    neg = [c for c in contribs if c['contribution'] < -0.005][:4]

    e_col1, e_col2 = st.columns(2)
    with e_col1:
        st.markdown("**Pushing toward BUY**")
        if pos:
            for c in pos:
                st.markdown(f"🟢 {describe_feature(c['feature'], c['value'])} "
                            f"&nbsp; `+{c['contribution'] * 100:.1f}%`")
        else:
            st.markdown("_Nothing significant_")
    with e_col2:
        st.markdown("**Pushing toward SELL**")
        if neg:
            for c in neg:
                st.markdown(f"🔴 {describe_feature(c['feature'], c['value'])} "
                            f"&nbsp; `{c['contribution'] * 100:.1f}%`")
        else:
            st.markdown("_Nothing significant_")

    st.caption(
        "How to read this: each factor is compared against its 'typical' level for "
        "this stock. The percentage shows how much that factor, at today's value, "
        "moves the model's probability up or down."
    )

    st.divider()

    # --- Classification metrics ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model Accuracy", f"{metrics['accuracy'] * 100:.2f}%", help=HELP["accuracy"])
    c2.metric("Baseline (majority class)", f"{metrics['baseline_accuracy'] * 100:.2f}%", help=HELP["baseline"])
    c3.metric("Precision", f"{metrics['precision'] * 100:.1f}%", help=HELP["precision"])
    c4.metric("Recall", f"{metrics['recall'] * 100:.1f}%", help=HELP["recall"])
    st.caption("1-day model, measured on the untouched test slice. Hover the (?) icons for explanations.")

    cal = metrics.get('calibration')
    if cal:
        with st.expander("📏 Calibration — can you trust the percentages?"):
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("Brier Score", f"{cal['brier']:.4f}", help=HELP["brier"])
            cc2.metric("Baseline Brier", f"{cal['brier_baseline']:.4f}",
                       help="Score from always predicting the historical average — the bar to beat.")
            cc3.metric("Calibration Error (ECE)", f"{cal['ece']:.3f}", help=HELP["ece"])

            curve = pd.DataFrame(cal['curve'])
            cal_fig = go.Figure()
            cal_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                         name="Perfect calibration",
                                         line=dict(dash="dash", color="gray")))
            cal_fig.add_trace(go.Scatter(
                x=curve['predicted'], y=curve['actual'],
                mode="markers+lines", name="This model",
                marker=dict(size=(curve['count'] / curve['count'].max() * 25 + 8)),
                hovertemplate="Model said %{x:.0%}<br>Actually rose %{y:.0%}<extra></extra>"))
            cal_fig.update_layout(
                height=350, margin=dict(l=10, r=10, t=30, b=10),
                xaxis=dict(title="Stated probability", range=[0, 1], tickformat=".0%"),
                yaxis=dict(title="Observed frequency", range=[0, 1], tickformat=".0%"),
                legend=dict(orientation="h", y=1.08))
            st.plotly_chart(cal_fig, use_container_width=True)

            verdict = ("✅ Percentages are trustworthy." if cal['ece'] < 0.05
                       else "⚠️ Mild miscalibration — read percentages as a tendency, not a promise."
                       if cal['ece'] < 0.10
                       else "❌ Significant miscalibration — the stated percentages overstate "
                            "the model's real conviction. Try the 'Calibrate probabilities' "
                            "toggle in the sidebar.")
            note = (" Probabilities shown are calibrated." if metrics.get('calibrated')
                    else "")
            st.caption(f"Points on the dashed line = honest percentages; above = "
                       f"underconfident; below = overconfident. Bubble size = number of "
                       f"days in that bucket. {verdict}{note}")

with tab_plan:
    # --- Support & Resistance ---
    st.subheader("Support & Resistance (auto-detected)")
    s_col, p_col, r_col = st.columns(3)
    if sr['support'] is not None:
        s_dist = (sr['price'] / sr['support'] - 1) * 100
        s_col.metric("Support", f"{currency}{sr['support']:,.0f}",
                     delta=f"{s_dist:.1f}% below price", delta_color="off",
                     help=HELP["support"])
    else:
        s_col.metric("Support", "Not found", help=HELP["support"])
        s_col.caption("No swing low below the current price in the past year — "
                      "the stock may be at its lows.")
    p_col.metric("Current Price", f"{currency}{sr['price']:,.2f}")
    if sr['resistance'] is not None:
        r_dist = (sr['resistance'] / sr['price'] - 1) * 100
        r_col.metric("Resistance", f"{currency}{sr['resistance']:,.0f}",
                     delta=f"{r_dist:.1f}% above price", delta_color="off",
                     help=HELP["resistance"])
    else:
        r_col.metric("Resistance", "Not found", help=HELP["resistance"])
        r_col.caption("No swing high above the current price in the past year — "
                      "the stock may be at all-time highs.")
    st.caption("Swing highs/lows of the past year, with nearby levels merged. "
               "Both are drawn on the Charts tab.")

    st.divider()

    # --- Stop Loss & Target ---
    st.subheader("Trade Plan (long entry at current price)")
    if signal != "BUY":
        st.info(f"ℹ️ The model's current signal is **{signal}**, not BUY — the plan "
                "below is for reference, e.g. if you already hold the stock or are "
                "averaging in on your own judgement.")

    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Entry", f"{currency}{plan['entry']:,.2f}", help=HELP["entry"])
    t2.metric("Target", f"{currency}{plan['target']:,.2f}",
              delta=f"+{(plan['target'] / plan['entry'] - 1) * 100:.1f}%",
              help=HELP["target"])
    t3.metric("Stop Loss", f"{currency}{plan['stop']:,.2f}",
              delta=f"-{(1 - plan['stop'] / plan['entry']) * 100:.1f}%",
              delta_color="inverse", help=HELP["stop_loss"])
    rr = plan['reward_risk']
    rr_note = "✅" if rr >= 2 else "⚠️" if rr >= 1.5 else "❌"
    t4.metric("Reward : Risk", f"{rr_note} 1 : {rr:.1f}", help=HELP["reward_risk"])

    st.caption(
        f"Stop placed {plan['stop_basis']}; target set at {plan['target_basis']}. "
        f"ATR (typical daily range) is {currency}{plan['atr']:,.1f}. "
        + ("Reward:risk below 1:1.5 — many traders would skip this setup."
           if rr < 1.5 else "")
    )

    st.divider()

    # --- Position Sizing Calculator ---
    st.subheader("Position Sizing Calculator")
    in1, in2 = st.columns(2)
    capital = in1.number_input(
        "Capital (₹)", min_value=10_000, max_value=1_000_000_000,
        value=1_000_000, step=50_000, help=HELP["capital"])
    risk_pct = in2.slider(
        "Risk per trade (%)", min_value=0.25, max_value=3.0, value=1.0, step=0.25,
        help=HELP["risk_per_trade"])

    ps = position_size(capital, risk_pct, plan['entry'], plan['stop'])
    if ps is None or ps['shares'] == 0:
        st.warning("Stop is too close to entry (or capital too small) to size a "
                   "position — widen the stop or increase capital.")
    else:
        o1, o2, o3, o4 = st.columns(4)
        o1.metric("Position Size", f"{ps['shares']:,} shares", help=HELP["position_shares"])
        o2.metric("Position Value", f"{currency}{ps['position_value']:,.0f}")
        o3.metric("Capital Deployed", f"{ps['pct_of_capital'] * 100:.1f}%")
        o4.metric("Loss if Stop Hits", f"{currency}{ps['actual_risk']:,.0f}")
        if ps['capped_by_capital']:
            st.warning("⚠️ The risk formula suggested more shares than your capital "
                       "can buy — size was capped at what's affordable, so your "
                       "actual risk is below the chosen percentage.")
        st.caption(
            f"Formula: ({currency}{capital:,.0f} × {risk_pct:.2f}%) ÷ "
            f"({currency}{plan['entry']:,.2f} − {currency}{plan['stop']:,.2f}) "
            f"= {ps['shares']:,} shares. If the stop is hit you lose "
            f"{currency}{ps['actual_risk']:,.0f} — and no more."
        )

with tab_back:
    stats, equity, buy_hold = backtest(test_probs, data['Close'], test_index, thresholds)
    if stats['n_trades'] == 0:
        st.info("ℹ️ The model never reached its entry threshold during the test "
                "period, so the strategy stayed fully in cash — 0% return by design, "
                "not a bug. Compare against Buy & Hold to judge what that caution cost.")
    edge = (stats['total_return'] - stats['buy_hold_return']) * 100

    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    r1c1.metric("Strategy Return", f"{stats['total_return'] * 100:.2f}%",
                delta=f"{edge:+.2f}% vs Buy & Hold", help=HELP["strategy_return"])
    r1c2.metric("Buy & Hold Return", f"{stats['buy_hold_return'] * 100:.2f}%", help=HELP["buy_hold"])
    r1c3.metric("Sharpe Ratio", f"{stats['sharpe']:.2f}" if pd.notna(stats['sharpe']) else "N/A",
                help=HELP["sharpe"])
    r1c4.metric("Max Drawdown", f"{stats['max_drawdown'] * 100:.2f}%", help=HELP["max_drawdown"])

    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    r2c1.metric("Win Rate (days in market)",
                f"{stats['win_rate'] * 100:.2f}%" if pd.notna(stats['win_rate']) else "N/A",
                help=HELP["win_rate"])
    r2c2.metric("Exposure", f"{stats['exposure'] * 100:.1f}%", help=HELP["exposure"])
    r2c3.metric("Trades (entries)", f"{stats['n_trades']}", help=HELP["trades"])
    r2c4.metric("Cost per Change", "0.10%", help=HELP["cost"])

    bt_fig = go.Figure()
    bt_fig.add_trace(go.Scatter(x=equity.index, y=equity, name="Strategy", line=dict(width=2)))
    bt_fig.add_trace(go.Scatter(x=buy_hold.index, y=buy_hold, name="Buy & Hold",
                                line=dict(width=2, dash="dash")))
    bt_fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10),
                         yaxis_title="Growth of ₹1", legend=dict(orientation="h", y=1.05))
    st.plotly_chart(bt_fig, use_container_width=True)

    with st.expander("📖 Glossary — what do these terms mean?"):
        st.markdown(
            f"**Strategy Return** — {HELP['strategy_return']}\n\n"
            f"**Buy & Hold Return** — {HELP['buy_hold']}\n\n"
            f"**Sharpe Ratio** — {HELP['sharpe']}\n\n"
            f"**Max Drawdown** — {HELP['max_drawdown']}\n\n"
            f"**Win Rate** — {HELP['win_rate']}\n\n"
            f"**Exposure** — {HELP['exposure']}\n\n"
            f"**Trades** — {HELP['trades']}\n\n"
            f"**Trading Cost** — {HELP['cost']}\n\n"
            f"**Risk Score** — {HELP['risk_score']}"
        )

with tab_wf:
    st.markdown(
        "Walk-forward validation retrains the model on an expanding window and "
        "tests on the next unseen chunk — like repeatedly asking *'if I had built "
        "this model a year ago, would it have worked since?'* "
        "**Consistency across folds matters more than any single number.**"
    )
    if st.button(f"Run walk-forward validation ({model_type})", type="primary"):
        try:
            wf = run_walk_forward(symbol, model_type, calibrate)
            styled = wf.style.format({
                'Accuracy': '{:.1%}', 'Win Rate': '{:.1%}',
                'Strategy Return': '{:+.1%}', 'Buy & Hold': '{:+.1%}',
                'Sharpe': '{:.2f}', 'Max Drawdown': '{:.1%}',
                'Exposure': '{:.0%}', 'Entry Thr': '{:.2f}', 'Exit Thr': '{:.2f}',
            }, na_rep="—")
            st.dataframe(styled, use_container_width=True, hide_index=True)

            sharpes = wf['Sharpe'].dropna()
            if len(sharpes) > 0:
                st.caption(
                    f"Sharpe across folds: median {sharpes.median():.2f}, "
                    f"range {sharpes.min():.2f} to {sharpes.max():.2f}. "
                    "If the strategy only wins in some folds, the edge is likely regime luck."
                )

            st.download_button("⬇️ Download results as CSV",
                               wf.to_csv(index=False).encode(),
                               file_name=f"walk_forward_{symbol}.csv", mime="text/csv")
        except ValueError as e:
            st.warning(str(e))

with tab_journal:
    st.markdown(
        "Backtests look **backward**; this journal looks **forward**. Log today's "
        "signal, and the app scores it later against what the market actually did — "
        "the most honest performance measure here."
    )

    today = data.index[-1].strftime("%Y-%m-%d")
    if st.button(f"📝 Log today's {signal} signal for {symbol}", type="primary",
                 help=HELP["journal"]):
        record = {
            "signal_date": today, "symbol": symbol, "name": display_name,
            "model_type": model_type, "signal": signal,
            "probability": round(confidence, 4), "rating": rating_from_prob(confidence),
            "entry": round(plan['entry'], 2), "stop": round(plan['stop'], 2),
            "target": round(plan['target'], 2),
            "reward_risk": round(plan['reward_risk'], 2),
            "risk_score": risk['score'],
            "logged_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        }
        if append_signal(record):
            st.success(f"Logged: {signal} {symbol} @ {currency}{plan['entry']:,.2f} "
                       f"(stop {currency}{plan['stop']:,.2f} / target {currency}{plan['target']:,.2f})")
        else:
            st.info("Already logged today for this stock and model — one entry per day.")

    jdf = load_journal()
    if jdf.empty:
        st.info("No signals logged yet. Log a few each day, come back in some weeks, "
                "and the scorecard below will tell you whether the model's calls "
                "actually worked.")
    else:
        with st.spinner("Scoring journal entries against price history..."):
            resolved = resolve_journal(jdf, get_data)

        sc = scorecard(resolved)
        j1, j2, j3, j4, j5 = st.columns(5)
        j1.metric("Signals Logged", sc['n_signals'])
        j2.metric("BUY Plans Resolved", f"{sc['n_resolved']} ({sc['n_open']} open)")
        j3.metric("Target-Hit Rate",
                  f"{sc['target_rate'] * 100:.0f}%" if pd.notna(sc['target_rate']) else "—",
                  help=HELP["target_rate"])
        j4.metric("Win Rate",
                  f"{sc['win_rate'] * 100:.0f}%" if pd.notna(sc['win_rate']) else "—",
                  help="Resolved BUY plans that ended with any positive return.")
        j5.metric("Avg Return / Plan",
                  f"{sc['avg_return'] * 100:+.1f}%" if pd.notna(sc['avg_return']) else "—")

        show = resolved[["signal_date", "symbol", "model_type", "signal", "probability",
                         "entry", "stop", "target", "status", "days",
                         "outcome_return"]].sort_values("signal_date", ascending=False)
        st.dataframe(
            show, use_container_width=True, hide_index=True,
            column_config={
                "signal_date": "Date", "symbol": "Symbol", "model_type": "Model",
                "signal": "Signal",
                "probability": st.column_config.NumberColumn("Prob", format="percent"),
                "entry": st.column_config.NumberColumn("Entry", format="%.2f"),
                "stop": st.column_config.NumberColumn("Stop", format="%.2f"),
                "target": st.column_config.NumberColumn("Target", format="%.2f"),
                "status": st.column_config.TextColumn("Status", help=HELP["journal_status"]),
                "days": st.column_config.NumberColumn("Days", format="%d"),
                "outcome_return": st.column_config.NumberColumn("Return", format="percent"),
            })
        st.caption(
            f"BUY plans resolve when price touches stop or target, or expire after "
            f"{MAX_HOLD_DAYS} trading days. Same-day double-touches score as STOP "
            f"(conservative). Stored in `{JOURNAL_FILE.name}` next to app.py — "
            "back it up if you redeploy, and note that cloud hosts with ephemeral "
            "storage will lose it on restart."
        )
        st.download_button("⬇️ Download journal as CSV",
                           resolved.to_csv(index=False).encode(),
                           file_name="signal_journal.csv", mime="text/csv")

with tab_charts:
    months = st.radio("Range", ["3M", "6M", "1Y", "All"], index=2, horizontal=True)
    lookback = {"3M": 63, "6M": 126, "1Y": 252, "All": len(data)}[months]
    view = data.tail(lookback)

    ma20 = data['Close'].rolling(20).mean().tail(lookback)
    ma50 = data['Close'].rolling(50).mean().tail(lookback)

    price_fig = go.Figure()
    price_fig.add_trace(go.Candlestick(
        x=view.index, open=view['Open'], high=view['High'],
        low=view['Low'], close=view['Close'], name="Price"))
    price_fig.add_trace(go.Scatter(x=ma20.index, y=ma20, name="MA20", line=dict(width=1.2)))
    price_fig.add_trace(go.Scatter(x=ma50.index, y=ma50, name="MA50", line=dict(width=1.2)))
    if sr['support'] is not None:
        price_fig.add_hline(y=sr['support'], line_dash="dash", line_color="green",
                            annotation_text=f"Support {currency}{sr['support']:,.0f}",
                            annotation_position="bottom right")
    if sr['resistance'] is not None:
        price_fig.add_hline(y=sr['resistance'], line_dash="dash", line_color="red",
                            annotation_text=f"Resistance {currency}{sr['resistance']:,.0f}",
                            annotation_position="top right")
    price_fig.update_layout(height=450, margin=dict(l=10, r=10, t=30, b=10),
                            xaxis_rangeslider_visible=False,
                            legend=dict(orientation="h", y=1.05))
    st.plotly_chart(price_fig, use_container_width=True)
    st.caption("Candles show each day's open/high/low/close. MA20/MA50 are 20- and "
               "50-day average prices. Dashed lines mark auto-detected support "
               "(green) and resistance (red) from the past year's swing points.")

    rsi_fig = go.Figure()
    rsi_fig.add_trace(go.Scatter(x=view.index, y=view['RSI'], name="RSI"))
    rsi_fig.add_hline(y=70, line_dash="dot", line_color="red")
    rsi_fig.add_hline(y=30, line_dash="dot", line_color="green")
    rsi_fig.update_layout(height=220, margin=dict(l=10, r=10, t=30, b=10),
                          yaxis=dict(range=[0, 100], title="RSI (14)"))
    st.plotly_chart(rsi_fig, use_container_width=True)
    st.caption(HELP["rsi"])

st.caption("⚠️ Educational tool only — not financial advice. "
           "Past backtest performance does not predict future results.")
