"""Main analysis tabs — prediction, scanner, plan, backtest, walk-forward,
journal, charts."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from journal import (
    MAX_HOLD_DAYS,
    append_signal,
    delete_signal,
    entry_label,
    journal_backend_info,
    journal_path_for,
    load_journal,
    resolve_journal,
    scorecard,
)
from model import (
    backtest,
    explain_prediction,
    global_model_available,
    position_size,
    random_signal_benchmark,
    rank_buy_candidates,
    rating_from_prob,
)
from ui.help_text import HELP
from ui.services import (
    SCAN_BATCH,
    advance_scan_session,
    ensure_scan_session,
    get_data,
    get_horizons,
    get_index,
    get_scan_results,
    maybe_autoseed_precomputed,
    precomputed_status,
    reset_scan_session,
    run_walk_forward,
    scan_progress,
    seed_session_from_precomputed,
)
from ui.styles import (
    color_pos_neg,
    color_signal,
    color_status,
    describe_feature,
    style_map,
)
from ui.theme import (
    ACCENT,
    AMBER,
    BLUE,
    RED,
    TEXT_MUTED,
    pick_card_html,
    plotly_layout,
    section_header,
)


def render_prediction_tab(ctx):
    data = ctx["data"]
    predictor = ctx["predictor"]
    scaler = ctx["scaler"]
    metrics = ctx["metrics"]
    confidence = ctx["confidence"]
    signal = ctx["signal"]
    risk = ctx["risk"]
    model_type = ctx["model_type"]
    symbol = ctx["symbol"]
    use_global = ctx["use_global"]

    section_header("1-day conviction")
    g_col, r_col = st.columns([2, 1], gap="medium")

    with g_col:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            number={
                'suffix': "%",
                'font': {'size': 42, 'color': "#e8ecf1", 'family': "Inter"},
            },
            title={
                'text': "P(meaningful UP · 1 day)",
                'font': {'size': 13, 'color': TEXT_MUTED},
            },
            gauge={
                'axis': {
                    'range': [0, 100], 'ticksuffix': "%",
                    'tickfont': {'color': TEXT_MUTED, 'size': 11},
                },
                'bar': {'color': ACCENT, 'thickness': 0.22},
                'bgcolor': "rgba(255,255,255,0.03)",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 45], 'color': "rgba(240,113,120,0.22)"},
                    {'range': [45, 65], 'color': "rgba(230,180,80,0.18)"},
                    {'range': [65, 80], 'color': "rgba(54,179,126,0.18)"},
                    {'range': [80, 100], 'color': "rgba(54,179,126,0.32)"},
                ],
                'threshold': {
                    'line': {'color': "#e8ecf1", 'width': 2},
                    'thickness': 0.75,
                    'value': confidence * 100,
                },
            },
        ))
        gauge.update_layout(**plotly_layout(
            height=280, margin=dict(l=24, r=24, t=48, b=12),
        ))
        st.plotly_chart(gauge, use_container_width=True)
        st.caption(
            "Bands · Sell <45% · Neutral 45–65% · Buy 65–80% · Strong Buy >80%"
        )

    with r_col:
        rating = rating_from_prob(confidence)
        rating_icon = {
            "Strong Buy": "●●", "Buy": "●", "Neutral": "○", "Sell": "▼",
        }[rating]
        st.metric(
            "Probability band", f"{rating_icon}  {rating}",
            help=HELP["rating"],
        )
        st.caption("Fixed bands · signal uses tuned thresholds")

        risk_icon = {"Low": "●", "Medium": "●", "High": "●"}[risk['level']]
        st.metric(
            "Risk score",
            f"{risk_icon}  {risk['score']:.1f} / 10  ·  {risk['level']}",
            help=HELP["risk_score"],
        )
        st.caption(
            f"Vol {risk['volatility_annualized']:.0%} · "
            f"range {risk['atr_pct']:.1%} · "
            f"1y DD {risk['max_drawdown_1y']:.0%}"
        )

    st.divider()
    section_header("Multi-day outlook")
    horizons_df = get_horizons(symbol, model_type, use_global)
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
    section_header(f"Why {signal}?")
    _, contribs = explain_prediction(predictor, scaler, data)
    pos = [c for c in contribs if c['contribution'] > 0.005][:4]
    neg = [c for c in contribs if c['contribution'] < -0.005][:4]

    e_col1, e_col2 = st.columns(2, gap="medium")
    with e_col1:
        st.markdown("**Pushing BUY**")
        if pos:
            for c in pos:
                st.success(
                    f"{describe_feature(c['feature'], c['value'])}  "
                    f"(**+{c['contribution'] * 100:.1f}%**)"
                )
        else:
            st.caption("Nothing significant")
    with e_col2:
        st.markdown("**Pushing SELL**")
        if neg:
            for c in neg:
                st.error(
                    f"{describe_feature(c['feature'], c['value'])}  "
                    f"(**{c['contribution'] * 100:.1f}%**)"
                )
        else:
            st.caption("Nothing significant")

    st.caption(
        "Occlusion attribution vs each factor's typical level for this stock."
    )

    st.divider()
    section_header("Hold-out metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model Accuracy", f"{metrics['accuracy'] * 100:.2f}%", help=HELP["accuracy"])
    c2.metric(
        "Baseline (majority class)",
        f"{metrics['baseline_accuracy'] * 100:.2f}%", help=HELP["baseline"],
    )
    c3.metric("Precision", f"{metrics['precision'] * 100:.1f}%", help=HELP["precision"])
    c4.metric("Recall", f"{metrics['recall'] * 100:.1f}%", help=HELP["recall"])
    _ctx = (
        "Features include NIFTY market context (index trend + relative strength)."
        if get_index() is not None else
        "⚠️ NIFTY data unavailable this session — market-context features are "
        "neutral; predictions still work, slightly less informed."
    )
    if metrics.get("source") == "global":
        _src = (
            "🌐 **Global model** (pooled weights). Thresholds and the metrics "
            "below use **this stock's** chronology, but the weights may have "
            "seen this name during offline training — **not fully out-of-sample**. "
            "Use Walk-Forward for a stricter check."
        )
        if metrics.get("oos_note"):
            st.info(metrics["oos_note"])
    else:
        _src = f"Per-stock {model_type} model."
    st.caption(
        f"{_src} Measured on the untouched test slice. {_ctx} "
        "Hover the (?) icons for explanations."
    )

    cal = metrics.get('calibration')
    if cal:
        with st.expander("📏 Calibration — can you trust the percentages?"):
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("Brier Score", f"{cal['brier']:.4f}", help=HELP["brier"])
            cc2.metric(
                "Baseline Brier", f"{cal['brier_baseline']:.4f}",
                help="Score from always predicting the historical average — the bar to beat.",
            )
            cc3.metric("Calibration Error (ECE)", f"{cal['ece']:.3f}", help=HELP["ece"])

            curve = pd.DataFrame(cal['curve'])
            cal_fig = go.Figure()
            cal_fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                name="Perfect",
                line=dict(dash="dash", color="rgba(232,236,241,0.25)", width=1.5),
            ))
            cal_fig.add_trace(go.Scatter(
                x=curve['predicted'], y=curve['actual'],
                mode="markers+lines", name="This model",
                line=dict(color=ACCENT, width=2),
                marker=dict(
                    size=(curve['count'] / curve['count'].max() * 22 + 8),
                    color=ACCENT,
                    line=dict(width=1, color="rgba(255,255,255,0.2)"),
                ),
                hovertemplate="Said %{x:.0%}<br>Actual %{y:.0%}<extra></extra>",
            ))
            cal_fig.update_layout(**plotly_layout(
                height=340,
                xaxis=dict(
                    title="Stated probability", range=[0, 1], tickformat=".0%",
                    gridcolor="rgba(255,255,255,0.05)",
                ),
                yaxis=dict(
                    title="Observed frequency", range=[0, 1], tickformat=".0%",
                    gridcolor="rgba(255,255,255,0.05)",
                ),
            ))
            st.plotly_chart(cal_fig, use_container_width=True)

            verdict = (
                "✅ Percentages are trustworthy." if cal['ece'] < 0.05
                else "⚠️ Mild miscalibration — read percentages as a tendency, not a promise."
                if cal['ece'] < 0.10
                else "❌ Significant miscalibration — the stated percentages overstate "
                     "the model's real conviction. Try the 'Calibrate probabilities' "
                     "toggle in the sidebar."
            )
            note = (
                " Probabilities shown are calibrated."
                if metrics.get('calibrated') else ""
            )
            st.caption(
                f"Points on the dashed line = honest percentages; above = "
                f"underconfident; below = overconfident. Bubble size = number of "
                f"days in that bucket. {verdict}{note}"
            )


def _render_buy_pick_card(rank, row, key_prefix, on_jump):
    """One top-pick card with jump-to-stock button (rich inline card)."""
    name = row["Name"]
    sym = row["Symbol"]
    prob = float(row["Probability Up"]) * 100
    score = float(row.get("Buy Score", 0))
    risk = row.get("Risk")
    rr = row.get("Reward Risk")
    day = row.get("Day")
    price = row.get("Price")

    day_s = f"{float(day) * 100:+.1f}%" if day is not None and pd.notna(day) else "—"
    rr_s = f"1:{float(rr):.1f}" if rr is not None and pd.notna(rr) else "—"
    risk_s = f"{float(risk):.1f}" if risk is not None and pd.notna(risk) else "—"
    price_s = f"{float(price):,.2f}" if price is not None and pd.notna(price) else "—"

    st.markdown(
        pick_card_html(
            rank=int(rank),
            symbol=str(sym),
            name=str(name),
            prob_pct=prob,
            score=score,
            price_s=price_s,
            day_s=day_s,
            risk_s=risk_s,
            rr_s=rr_s,
        ),
        unsafe_allow_html=True,
    )
    st.button(
        f"Open analysis · {sym}",
        key=f"{key_prefix}_{rank}_{sym}",
        use_container_width=True,
        type="secondary",
        on_click=on_jump,
        args=(name,),
    )


def render_scanner_tab(ctx):
    stocks = ctx["stocks"]
    symbol = ctx["symbol"]
    signal = ctx["signal"]

    ensure_scan_session(stocks)
    # Instant path: seed from offline rankings when session is empty & file is fresh
    maybe_autoseed_precomputed(stocks)
    prog = scan_progress(stocks)
    pre = precomputed_status(stocks)

    section_header("Screener")
    st.caption(
        f"Full-universe screen · **{prog['total']}** names in list · "
        f"live batches of **{SCAN_BATCH}** · optional precomputed rankings"
    )
    _engine = (
        "global model"
        if global_model_available()
        else "fast per-stock tree"
    )
    st.markdown(
        f"Prefer **precomputed rankings** (offline job) for instant results, or "
        f"walk the list **live in batches of {SCAN_BATCH}**. Engine: **{_engine}**. "
        "This is a **screen**, not the full stock Signal — open a name before "
        "acting. **Educational only — not investment advice.**"
    )

    # ---- Precomputed rankings card ----
    section_header("Precomputed rankings")
    if pre["available"]:
        stale_note = (
            f" · ⚠️ file is **{pre.get('age_hours')}h** old (stale)"
            if pre.get("stale") else " · fresh"
        )
        st.success(
            f"Offline file ready · **{pre['n_watchlist']}** names overlap your "
            f"watchlist (file has {pre['n_file']}) · as of **{pre.get('asof') or '—'}**"
            f"{stale_note}"
        )
        pc1, pc2 = st.columns(2)
        with pc1:
            if st.button(
                "Load precomputed rankings", type="primary",
                use_container_width=True,
                help="Instant load from rankings/rankings_latest.csv (no Yahoo calls).",
            ):
                if seed_session_from_precomputed(stocks, allow_stale=True):
                    st.toast("Loaded precomputed rankings", icon="⚡")
                    st.rerun()
                else:
                    st.warning("Could not load precomputed file.")
        with pc2:
            st.caption(
                "Regenerate:  \n"
                "`python scripts/precompute_rankings.py`  \n"
                "or **Actions → Nightly rankings → Run workflow**"
            )
        meta = pre.get("meta") or {}
        if meta.get("runner") == "github-actions" and meta.get("run_url"):
            st.caption(f"Last CI run: {meta.get('run_url')}")
    else:
        st.info(
            "No precomputed rankings found (or none match this watchlist). "
            "Run offline: `python scripts/precompute_rankings.py`, or trigger "
            "**GitHub Actions → Nightly rankings**, then redeploy/restart — "
            "or use **live batch scan** below."
        )

    with st.expander("How Buy Score is calculated", expanded=False):
        st.markdown(
            """
            | Factor | Weight (approx.) | What it rewards |
            |--------|------------------|-----------------|
            | **Probability Up** | ~50% | Higher chance of a meaningful 1-day up move |
            | **Model edge** | ~20% | Test accuracy beating the majority baseline |
            | **Lower risk** | ~12% | Calmer vol / ATR / drawdown score |
            | **Reward : risk** | ~12% | ATR/structure trade plan with better R:R |
            | **Near support** | ~6% | Price sitting closer to a recent swing floor |

            Only names with a **BUY** screen call (prob above the default entry
            threshold, usually 0.55) enter the shortlist. Optional filters below
            can require edge and cap risk.
            """
        )

    # ---- Live batch controls ----
    section_header("Live batch scan")
    if prog.get("source") == "precomputed" and prog.get("asof"):
        st.info(
            f"⚡ Showing **precomputed** rankings (as of **{prog['asof']}**). "
            "Use **Rescan from start** below to switch to a live Yahoo walk."
        )

    attempted = prog["attempted"]
    total = max(prog["total"], 1)
    live_mode = prog.get("source") != "precomputed"
    st.progress(
        min(attempted / total, 1.0) if live_mode else 1.0,
        text=(
            f"Precomputed · {prog['succeeded']} names · as of {prog.get('asof') or '—'}"
            if prog.get("source") == "precomputed" else
            f"Covered {prog['attempted']} / {prog['total']} symbols · "
            f"{prog['succeeded']} scored · {prog['failed']} skipped · "
            f"{prog['remaining']} remaining"
        ),
    )

    b1, b2, b3, b4 = st.columns([1.2, 1.2, 1.1, 1.1])
    with b1:
        start_label = (
            f"Start live scan ({SCAN_BATCH})"
            if not prog["active"] and prog["attempted"] == 0
            else f"Rescan from start ({SCAN_BATCH})"
        )
        if st.button(start_label, type="primary", use_container_width=True,
                     help=HELP["scanner"]):
            reset_scan_session(stocks)
            with st.spinner(f"Scanning first batch of {SCAN_BATCH}…"):
                advance_scan_session(stocks, n_batches=1)
            st.rerun()
    with b2:
        next_n = min(SCAN_BATCH, prog["remaining"]) if prog["remaining"] else SCAN_BATCH
        next_disabled = (
            prog.get("source") == "precomputed"
            or prog["complete"]
            or prog["total"] == 0
        )
        if st.button(
            f"Scan next batch ({next_n})" if prog["remaining"] else "Scan complete",
            use_container_width=True,
            disabled=next_disabled,
            help="Continue through the full universe without losing prior results.",
        ):
            with st.spinner(f"Scanning next {SCAN_BATCH}…"):
                advance_scan_session(stocks, n_batches=1)
            st.rerun()
    with b3:
        multi = min(3, max(1, (prog["remaining"] + SCAN_BATCH - 1) // SCAN_BATCH))
        multi_disabled = (
            prog.get("source") == "precomputed"
            or prog["complete"]
            or prog["total"] == 0
        )
        if st.button(
            f"Scan +{multi} batches",
            use_container_width=True,
            disabled=multi_disabled,
            help="Run up to 3 batches in a row (faster coverage; still rate-limit aware).",
        ):
            with st.spinner(f"Scanning up to {multi} batches…"):
                advance_scan_session(stocks, n_batches=multi)
            st.rerun()
    with b4:
        if st.button("Reset scan", use_container_width=True):
            reset_scan_session(stocks)
            st.rerun()

    if prog["attempted"] == 0 and prog.get("source") != "precomputed":
        st.info(
            f"Load **precomputed rankings** above (if available), or "
            f"**Start live scan** for the first **{SCAN_BATCH}** names, then "
            f"**Scan next batch** until you cover all **{prog['total']}**."
        )
        return

    scan_df = get_scan_results()
    scan_failures = st.session_state.get("scan_failures") or []

    if scan_df.empty:
        st.warning(
            "No stocks scored yet — load precomputed rankings or run a live batch. "
            "Yahoo may be rate-limiting live scans."
        )
        if scan_failures:
            with st.expander(f"{len(scan_failures)} stock(s) skipped so far"):
                for sym, reason in scan_failures[:80]:
                    st.markdown(f"- `{sym}` — {reason}")
                if len(scan_failures) > 80:
                    st.caption(f"…and {len(scan_failures) - 80} more")
        return

    n_buy = int((scan_df["Screen"] == "BUY").sum())
    n_sell = int((scan_df["Screen"] == "SELL").sum())
    top_score = float(scan_df["Buy Score"].max()) if "Buy Score" in scan_df else 0

    # Short labels so values never clip in narrow metric cards
    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    if prog.get("source") == "precomputed":
        sc1.metric(
            "Source",
            "Offline",
            help="Precomputed rankings file (not a live Yahoo walk).",
        )
    else:
        sc1.metric(
            "Coverage",
            f"{prog['attempted']}/{prog['total']}",
            help="Symbols attempted vs full watchlist size.",
        )
    sc2.metric("Scored", f"{len(scan_df):,}")
    sc3.metric("BUY", f"{n_buy:,}", help="Screen calls = BUY")
    sc4.metric("SELL", f"{n_sell:,}", help="Screen calls = SELL")
    sc5.metric("Top score", f"{top_score:.0f}", help="Highest Buy Score in results")

    if prog.get("source") == "precomputed":
        asof = prog.get("asof") or "—"
        st.success(
            f"**Source: Precomputed (offline)** · **{len(scan_df):,}** names · "
            f"as of **{asof}**."
        )
    elif prog["complete"]:
        st.success(
            f"Full list covered ({prog['total']} symbols attempted). "
            f"**{len(scan_df)}** scored · **{prog['failed']}** skipped."
        )
    else:
        st.caption(
            f"Partial universe — rankings use the **{len(scan_df)}** names scored "
            f"so far. Keep clicking **Scan next batch** to improve coverage."
        )

    current_screen = scan_df[scan_df["Symbol"] == symbol]
    if not current_screen.empty:
        screen_call = current_screen.iloc[0]["Screen"]
        if screen_call != signal:
            st.warning(
                f"Scanner shows **{screen_call}** for {symbol}, while the full "
                f"analysis signal is **{signal}**. Prefer the full Signal for "
                "the selected stock."
            )

    # ---- Filters for "best buys" shortlist ----
    section_header("Shortlist filters")
    f1, f2, f3, f4 = st.columns(4)
    min_prob = f1.slider(
        "Min probability", 0.50, 0.80, 0.55, 0.01,
        help="Only BUY screens at or above this model probability.",
    )
    max_risk = f2.slider(
        "Max risk score", 3.0, 10.0, 8.0, 0.5,
        help="Drop names riskier than this (1 calm → 10 wild).",
    )
    top_n = f3.slider("Show top N", 3, 20, 8, 1)
    require_edge = f4.checkbox(
        "Require model edge",
        value=True,
        help="Keep only names where test accuracy ≥ majority baseline.",
    )

    picks = rank_buy_candidates(
        scan_df,
        min_prob=min_prob,
        max_risk=max_risk,
        require_edge=require_edge,
        top_n=top_n,
    )

    section_header("Top picks to open long")
    if picks.empty:
        st.info(
            "No names pass the filters right now. Try lowering min probability, "
            "raising max risk, or unticking “Require model edge”. "
            "A empty shortlist is useful information too."
        )
    else:
        def _jump_to(stock_name):
            from ui.stock_picker import set_stock_pick
            set_stock_pick(stock_name)

        # Responsive grid of pick cards
        cols = st.columns(min(3, len(picks)))
        for i, (_, row) in enumerate(picks.iterrows()):
            with cols[i % len(cols)]:
                _render_buy_pick_card(
                    row.get("Rank", i + 1), row,
                    key_prefix="pick", on_jump=_jump_to,
                )

        st.caption(
            f"**{len(picks)}** candidate(s) after filters · sorted by Buy Score. "
            "Tap **Open full analysis** before any decision."
        )

        # Compact table of the shortlist
        pick_view = picks[[
            c for c in [
                "Rank", "Symbol", "Name", "Buy Score", "Probability Up",
                "Screen", "Risk", "Reward Risk", "To Support", "Test Acc",
                "Baseline", "Day", "Price",
            ] if c in picks.columns
        ]]
        _pv = style_map(pick_view.style, color_signal, ["Screen"])
        if "Day" in pick_view.columns:
            _pv = style_map(_pv, color_pos_neg, ["Day"])
        st.dataframe(
            _pv, use_container_width=True, hide_index=True,
            column_config={
                "Buy Score": st.column_config.ProgressColumn(
                    "Buy Score", format="%.0f", min_value=0, max_value=100,
                    help=HELP.get("buy_score", "Composite long-candidate score."),
                ),
                "Probability Up": st.column_config.ProgressColumn(
                    format="percent", min_value=0, max_value=1, help=HELP["prob_up"],
                ),
                "Risk": st.column_config.NumberColumn("Risk /10", format="%.1f"),
                "Reward Risk": st.column_config.NumberColumn("R:R", format="%.2f"),
                "To Support": st.column_config.NumberColumn(format="percent"),
                "Test Acc": st.column_config.NumberColumn(format="percent"),
                "Baseline": st.column_config.NumberColumn(format="percent"),
                "Day": st.column_config.NumberColumn(format="percent"),
                "Price": st.column_config.NumberColumn(format="%.2f"),
            },
        )

    # ---- Full watchlist table ----
    st.divider()
    section_header("Full watchlist ranking")
    show_mode = st.radio(
        "Table filter",
        ["All", "BUY only", "SELL only"],
        horizontal=True,
        label_visibility="collapsed",
    )
    view_df = scan_df
    if show_mode == "BUY only":
        view_df = scan_df[scan_df["Screen"] == "BUY"]
    elif show_mode == "SELL only":
        view_df = scan_df[scan_df["Screen"] == "SELL"]

    if view_df.empty:
        st.info("No rows for this filter.")
    else:
        display_cols = [
            c for c in [
                "Symbol", "Name", "Buy Score", "Screen", "Probability Up",
                "Rating", "Risk", "Reward Risk", "To Support", "To Resistance",
                "Test Acc", "Baseline", "Day", "Price", "Model",
            ] if c in view_df.columns
        ]
        _scan_styled = style_map(view_df[display_cols].style, color_signal, ["Screen"])
        if "Day" in display_cols:
            _scan_styled = style_map(_scan_styled, color_pos_neg, ["Day"])
        st.dataframe(
            _scan_styled, use_container_width=True, hide_index=True, height=480,
            column_config={
                "Buy Score": st.column_config.ProgressColumn(
                    format="%.0f", min_value=0, max_value=100,
                    help=HELP.get("buy_score", "Composite long-candidate score."),
                ),
                "Price": st.column_config.NumberColumn(format="%.2f"),
                "Day": st.column_config.NumberColumn("Day %", format="percent"),
                "Probability Up": st.column_config.ProgressColumn(
                    format="percent", min_value=0, max_value=1, help=HELP["prob_up"],
                ),
                "Screen": st.column_config.TextColumn("Screen", help=HELP["scanner"]),
                "Test Acc": st.column_config.NumberColumn(format="percent"),
                "Baseline": st.column_config.NumberColumn(format="percent"),
                "Risk": st.column_config.NumberColumn("Risk /10", format="%.1f"),
                "Reward Risk": st.column_config.NumberColumn("R:R", format="%.2f"),
                "To Support": st.column_config.NumberColumn(
                    format="percent", help=HELP["scan_to_support"],
                ),
                "To Resistance": st.column_config.NumberColumn(
                    format="percent", help=HELP["scan_to_resistance"],
                ),
            },
        )

    src = prog.get("source") or "live"
    st.caption(
        f"Source **{src}** · coverage **{prog['attempted']}/{prog['total']}** · "
        "screen only — default thresholds. Open a stock for the full Signal. "
        "Not financial advice."
    )
    st.download_button(
        "Download scored results CSV",
        scan_df.to_csv(index=False).encode(),
        file_name="screener_results.csv", mime="text/csv",
    )

    if scan_failures:
        with st.expander(f"{len(scan_failures)} stock(s) skipped so far"):
            for sym, reason in scan_failures[:100]:
                st.markdown(f"- `{sym}` — {reason}")
            if len(scan_failures) > 100:
                st.caption(f"…and {len(scan_failures) - 100} more")


def render_plan_tab(ctx):
    sr = ctx["sr"]
    plan = ctx["plan"]
    signal = ctx["signal"]
    currency = ctx["currency"]

    section_header("Support & resistance")
    s_col, p_col, r_col = st.columns(3)
    if sr['support'] is not None:
        s_dist = (sr['price'] / sr['support'] - 1) * 100
        s_col.metric(
            "Support", f"{currency}{sr['support']:,.0f}",
            delta=f"-{s_dist:.1f}% below price", delta_color="inverse",
            help=HELP["support"],
        )
    else:
        s_col.metric("Support", "Not found", help=HELP["support"])
        s_col.caption(
            "No swing low below the current price in the past year — "
            "the stock may be at its lows."
        )
    p_col.metric("Current Price", f"{currency}{sr['price']:,.2f}")
    if sr['resistance'] is not None:
        r_dist = (sr['resistance'] / sr['price'] - 1) * 100
        r_col.metric(
            "Resistance", f"{currency}{sr['resistance']:,.0f}",
            delta=f"+{r_dist:.1f}% above price", delta_color="inverse",
            help=HELP["resistance"],
        )
    else:
        r_col.metric("Resistance", "Not found", help=HELP["resistance"])
        r_col.caption(
            "No swing high above the current price in the past year — "
            "the stock may be at all-time highs."
        )
    st.caption(
        "Swing highs/lows of the past year, with nearby levels merged. "
        "Both are drawn on the Charts tab."
    )

    st.divider()
    section_header("Trade plan · long at last close")
    if signal != "BUY":
        st.info(
            f"Current signal is **{signal}**, not BUY — plan is reference only "
            "(existing holds / discretionary adds)."
        )

    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Entry", f"{currency}{plan['entry']:,.2f}", help=HELP["entry"])
    t2.metric(
        "Target", f"{currency}{plan['target']:,.2f}",
        delta=f"+{(plan['target'] / plan['entry'] - 1) * 100:.1f}%",
        help=HELP["target"],
    )
    t3.metric(
        "Stop Loss", f"{currency}{plan['stop']:,.2f}",
        delta=f"-{(1 - plan['stop'] / plan['entry']) * 100:.1f}%",
        delta_color="inverse", help=HELP["stop_loss"],
    )
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
    section_header("Position sizing")
    in1, in2 = st.columns(2)
    capital = in1.number_input(
        "Capital (₹)", min_value=10_000, max_value=1_000_000_000,
        value=1_000_000, step=50_000, help=HELP["capital"],
    )
    risk_pct = in2.slider(
        "Risk per trade (%)", min_value=0.25, max_value=3.0, value=1.0, step=0.25,
        help=HELP["risk_per_trade"],
    )

    ps = position_size(capital, risk_pct, plan['entry'], plan['stop'])
    if ps is None or ps['shares'] == 0:
        st.warning(
            "Stop is too close to entry (or capital too small) to size a "
            "position — widen the stop or increase capital."
        )
    else:
        o1, o2, o3, o4 = st.columns(4)
        o1.metric("Position Size", f"{ps['shares']:,} shares", help=HELP["position_shares"])
        o2.metric("Position Value", f"{currency}{ps['position_value']:,.0f}")
        o3.metric("Capital Deployed", f"{ps['pct_of_capital'] * 100:.1f}%")
        o4.metric("Loss if Stop Hits", f"{currency}{ps['actual_risk']:,.0f}")
        if ps['capped_by_capital']:
            st.warning(
                "⚠️ The risk formula suggested more shares than your capital "
                "can buy — size was capped at what's affordable, so your "
                "actual risk is below the chosen percentage."
            )
        st.caption(
            f"Formula: ({currency}{capital:,.0f} × {risk_pct:.2f}%) ÷ "
            f"({currency}{plan['entry']:,.2f} − {currency}{plan['stop']:,.2f}) "
            f"= {ps['shares']:,} shares. If the stop is hit you lose "
            f"{currency}{ps['actual_risk']:,.0f} — and no more."
        )


def render_backtest_tab(ctx):
    data = ctx["data"]
    test_probs = ctx["test_probs"]
    test_index = ctx["test_index"]
    thresholds = ctx["thresholds"]

    section_header("Out-of-sample performance")
    stats, equity, buy_hold = backtest(
        test_probs, data['Close'], test_index, thresholds,
    )
    if stats['n_trades'] == 0:
        st.info(
            "The model never hit its entry threshold on the test slice — "
            "stayed in cash (0% by design). Compare vs Buy & Hold."
        )
    edge = (stats['total_return'] - stats['buy_hold_return']) * 100

    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    r1c1.metric(
        "Strategy Return", f"{stats['total_return'] * 100:.2f}%",
        delta=f"{edge:+.2f}% vs Buy & Hold", help=HELP["strategy_return"],
    )
    r1c2.metric(
        "Buy & Hold Return", f"{stats['buy_hold_return'] * 100:.2f}%",
        help=HELP["buy_hold"],
    )
    r1c3.metric(
        "Sharpe Ratio",
        f"{stats['sharpe']:.2f}" if pd.notna(stats['sharpe']) else "N/A",
        help=HELP["sharpe"],
    )
    r1c4.metric(
        "Max Drawdown", f"{stats['max_drawdown'] * 100:.2f}%",
        help=HELP["max_drawdown"],
    )

    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    r2c1.metric(
        "Win Rate (days in market)",
        f"{stats['win_rate'] * 100:.2f}%" if pd.notna(stats['win_rate']) else "N/A",
        help=HELP["win_rate"],
    )
    r2c2.metric("Exposure", f"{stats['exposure'] * 100:.1f}%", help=HELP["exposure"])
    r2c3.metric("Trades (entries)", f"{stats['n_trades']}", help=HELP["trades"])
    r2c4.metric("Cost per Change", "0.10%", help=HELP["cost"])

    section_header("Equity curve")
    bt_fig = go.Figure()
    bt_fig.add_trace(go.Scatter(
        x=equity.index, y=equity, name="Strategy",
        line=dict(width=2.4, color=ACCENT),
        fill="tozeroy", fillcolor="rgba(54,179,126,0.08)",
    ))
    bt_fig.add_trace(go.Scatter(
        x=buy_hold.index, y=buy_hold, name="Buy & Hold",
        line=dict(width=2, dash="dash", color=BLUE),
    ))
    bt_fig.update_layout(**plotly_layout(
        height=400, yaxis_title="Growth of ₹1",
    ))
    st.plotly_chart(bt_fig, use_container_width=True)

    with st.expander("Random-signal benchmark — does this beat luck?"):
        st.caption(
            "The honest gut-check: hundreds of random strategies that "
            "trade as often as the model, ranked against it. Skill should "
            "sit far right of the random crowd; luck sits in the middle."
        )
        bench = random_signal_benchmark(
            test_probs, data['Close'], test_index, thresholds,
        )
        if bench is None:
            st.info(
                "Not enough trades in the test period to benchmark — the "
                "strategy barely took positions, so there's nothing to "
                "compare against the random crowd."
            )
        else:
            b1, b2 = st.columns(2)
            b1.metric(
                "Sharpe percentile", f"{bench['sharpe_percentile']:.0f}th",
                help=HELP["benchmark"],
            )
            b2.metric(
                "Return percentile", f"{bench['return_percentile']:.0f}th",
                help=HELP["benchmark"],
            )

            pct = bench["sharpe_percentile"]
            verdict = (
                "✅ Strong: the model clearly beats random strategies of the "
                "same activity — evidence of a real edge."
                if pct >= 90 else
                "🟡 Mixed: the model is somewhat above random, but not "
                "decisively — treat the edge as unproven."
                if pct >= 70 else
                "❌ No edge: the model performs like a random strategy of "
                "the same trade frequency. The backtest return is likely "
                "luck, not skill."
            )
            st.markdown(f"**{verdict}**")

            hist = go.Figure()
            hist.add_trace(go.Histogram(
                x=bench["rand_sharpes"], nbinsx=30,
                name="Random strategies",
                marker=dict(color=BLUE, opacity=0.7, line=dict(width=0)),
            ))
            hist.add_vline(
                x=bench["real_sharpe"], line_color=ACCENT, line_width=3,
                annotation_text="This model", annotation_position="top",
                annotation_font_color=ACCENT,
            )
            hist.update_layout(**plotly_layout(
                height=300, showlegend=False,
                xaxis_title="Sharpe ratio",
                yaxis_title=f"# of {bench['n_random']} random strategies",
            ))
            st.plotly_chart(hist, use_container_width=True)
            st.caption(
                f"Each random strategy held {bench['exposure_days']} of "
                f"{bench['total_days']} test days — matching the model's "
                "exposure exactly, so only signal *quality* is being tested, "
                "not how often it trades."
            )

    with st.expander("Glossary"):
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


def render_walkforward_tab(ctx):
    symbol = ctx["symbol"]
    model_type = ctx["model_type"]
    calibrate = ctx["calibrate"]

    section_header("Expanding-window validation")
    st.markdown(
        "Retrains on an expanding history and scores the next unseen chunk — "
        "*if I had built this a year ago, would it have worked since?* "
        "**Consistency across folds beats any single number.**"
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
            styled = style_map(
                styled, color_pos_neg,
                ['Strategy Return', 'Buy & Hold', 'Sharpe'],
            )
            st.dataframe(styled, use_container_width=True, hide_index=True)

            sharpes = wf['Sharpe'].dropna()
            if len(sharpes) > 0:
                st.caption(
                    f"Sharpe across folds: median {sharpes.median():.2f}, "
                    f"range {sharpes.min():.2f} to {sharpes.max():.2f}. "
                    "If the strategy only wins in some folds, the edge is likely regime luck."
                )

            st.download_button(
                "⬇️ Download results as CSV",
                wf.to_csv(index=False).encode(),
                file_name=f"walk_forward_{symbol}.csv", mime="text/csv",
            )
        except ValueError as e:
            st.warning(str(e))


def render_journal_tab(ctx):
    data = ctx["data"]
    symbol = ctx["symbol"]
    display_name = ctx["display_name"]
    model_type = ctx["model_type"]
    signal = ctx["signal"]
    confidence = ctx["confidence"]
    plan = ctx["plan"]
    risk = ctx["risk"]
    currency = ctx["currency"]
    current_user = ctx["current_user"]

    jpath = journal_path_for(current_user)
    backend = journal_backend_info()
    section_header("Forward-test journal")
    st.markdown(
        "Backtests look **backward**; this journal looks **forward**. Log today's "
        "signal and score it later against real prices."
    )

    if backend["persistent"]:
        st.success(
            f"Storage · **{backend['label']}** · {backend['detail']} · "
            f"user `{current_user}`"
        )
    else:
        st.warning(
            f"Storage · **{backend['label']}** · {backend['detail']}. "
            "On Streamlit Cloud this is wiped on restart. "
            "Configure Supabase in secrets for persistence "
            "(see `scripts/supabase_journal.sql`)."
        )

    today = data.index[-1].strftime("%Y-%m-%d")
    if st.button(
        f"Log today's {signal} signal for {symbol}", type="primary",
        help=HELP["journal"],
    ):
        record = {
            "signal_date": today, "symbol": symbol, "name": display_name,
            "model_type": model_type, "signal": signal,
            "probability": round(confidence, 4),
            "rating": rating_from_prob(confidence),
            "entry": round(plan['entry'], 2), "stop": round(plan['stop'], 2),
            "target": round(plan['target'], 2),
            "reward_risk": round(plan['reward_risk'], 2),
            "risk_score": risk['score'],
            "logged_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        }
        try:
            ok = append_signal(record, user=current_user)
        except Exception as e:
            st.error(f"Could not write journal: {e}")
            ok = None
        if ok is True:
            st.toast(f"Logged {signal} for {symbol}", icon="📝")
            st.success(
                f"Logged: {signal} {symbol} @ {currency}{plan['entry']:,.2f} "
                f"(stop {currency}{plan['stop']:,.2f} / "
                f"target {currency}{plan['target']:,.2f})"
            )
        elif ok is False:
            st.info("Already logged today for this stock and model — one entry per day.")

    try:
        jdf = load_journal(user=current_user)
    except Exception as e:
        st.error(f"Could not load journal: {e}")
        jdf = pd.DataFrame()

    if jdf.empty:
        st.info(
            "No signals logged yet. Log a few each day, come back in some weeks, "
            "and the scorecard below will tell you whether the model's calls "
            "actually worked."
        )
    else:
        with st.spinner("Scoring journal entries against price history..."):
            resolved = resolve_journal(jdf, get_data)

        sc = scorecard(resolved)
        j1, j2, j3, j4, j5 = st.columns(5)
        j1.metric("Signals Logged", sc['n_signals'])
        j2.metric("BUY Plans Resolved", f"{sc['n_resolved']} ({sc['n_open']} open)")
        j3.metric(
            "Target-Hit Rate",
            f"{sc['target_rate'] * 100:.0f}%" if pd.notna(sc['target_rate']) else "—",
            help=HELP["target_rate"],
        )
        j4.metric(
            "Win Rate",
            f"{sc['win_rate'] * 100:.0f}%" if pd.notna(sc['win_rate']) else "—",
            help="Resolved BUY plans that ended with any positive return.",
        )
        j5.metric(
            "Avg Return / Plan",
            f"{sc['avg_return'] * 100:+.1f}%" if pd.notna(sc['avg_return']) else "—",
        )

        show = resolved[[
            "signal_date", "symbol", "model_type", "signal", "probability",
            "entry", "stop", "target", "status", "days", "outcome_return",
        ]].sort_values("signal_date", ascending=False)
        _journal_styled = style_map(show.style, color_status, ["status"])
        _journal_styled = style_map(_journal_styled, color_signal, ["signal"])
        _journal_styled = style_map(_journal_styled, color_pos_neg, ["outcome_return"])
        st.dataframe(
            _journal_styled, use_container_width=True, hide_index=True,
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
            },
        )
        store_note = (
            f"Cloud store · {backend['detail']}"
            if backend["persistent"]
            else f"Local file `{jpath}` — download often on Cloud"
        )
        st.caption(
            f"BUY plans resolve when price touches stop or target, or expire after "
            f"{MAX_HOLD_DAYS} trading days. Same-day double-touches score as STOP "
            f"(conservative). {store_note}."
        )

        # ---- Remove logged signals ----
        section_header("Remove logged signals")
        st.caption(
            "Select one or more entries to permanently delete from your journal. "
            "This cannot be undone."
        )
        # Build stable label → key map (date, symbol, model_type)
        label_to_key = {}
        labels = []
        for _, row in show.iterrows():
            lab = entry_label(row)
            # Disambiguate rare collisions
            base = lab
            n = 2
            while lab in label_to_key:
                lab = f"{base} ({n})"
                n += 1
            label_to_key[lab] = (
                str(row["signal_date"]),
                str(row["symbol"]),
                str(row["model_type"]),
            )
            labels.append(lab)

        selected = st.multiselect(
            "Entries to remove",
            options=labels,
            default=[],
            key="journal_delete_select",
            help="Pick entries by date · symbol · signal · model.",
        )
        d1, d2 = st.columns([1, 2])
        with d1:
            confirm = st.checkbox(
                "Confirm permanent delete",
                value=False,
                key="journal_delete_confirm",
            )
        with d2:
            if st.button(
                f"Delete {len(selected)} selected" if selected else "Delete selected",
                type="primary",
                disabled=not selected or not confirm,
                use_container_width=True,
            ):
                keys = [label_to_key[lab] for lab in selected if lab in label_to_key]
                try:
                    n = delete_signal(keys, user=current_user)
                except Exception as e:
                    st.error(f"Delete failed: {e}")
                    n = 0
                if n > 0:
                    st.toast(f"Removed {n} journal entr{'y' if n == 1 else 'ies'}", icon="🗑️")
                    st.success(f"Removed **{n}** journal entry(ies).")
                    st.session_state.pop("journal_delete_select", None)
                    st.session_state.pop("journal_delete_confirm", None)
                    st.rerun()
                else:
                    st.warning("No matching entries were removed.")

        st.download_button(
            "Download journal as CSV",
            resolved.to_csv(index=False).encode(),
            file_name=f"signal_journal_{current_user}.csv", mime="text/csv",
        )


def render_charts_tab(ctx):
    data = ctx["data"]
    sr = ctx["sr"]
    currency = ctx["currency"]

    section_header("Price action")
    months = st.radio(
        "Range", ["3M", "6M", "1Y", "All"], index=2, horizontal=True,
        label_visibility="collapsed",
    )
    lookback = {"3M": 63, "6M": 126, "1Y": 252, "All": len(data)}[months]
    view = data.tail(lookback)

    ma20 = data['Close'].rolling(20).mean().tail(lookback)
    ma50 = data['Close'].rolling(50).mean().tail(lookback)

    price_fig = go.Figure()
    price_fig.add_trace(go.Candlestick(
        x=view.index, open=view['Open'], high=view['High'],
        low=view['Low'], close=view['Close'], name="Price",
        increasing_line_color=ACCENT, increasing_fillcolor=ACCENT,
        decreasing_line_color=RED, decreasing_fillcolor=RED,
    ))
    price_fig.add_trace(go.Scatter(
        x=ma20.index, y=ma20, name="MA20",
        line=dict(width=1.4, color=BLUE),
    ))
    price_fig.add_trace(go.Scatter(
        x=ma50.index, y=ma50, name="MA50",
        line=dict(width=1.4, color=AMBER),
    ))
    if sr['support'] is not None:
        price_fig.add_hline(
            y=sr['support'], line_dash="dash", line_color=ACCENT,
            annotation_text=f"Support {currency}{sr['support']:,.0f}",
            annotation_position="bottom right",
            annotation_font_color=ACCENT,
        )
    if sr['resistance'] is not None:
        price_fig.add_hline(
            y=sr['resistance'], line_dash="dash", line_color=RED,
            annotation_text=f"Resistance {currency}{sr['resistance']:,.0f}",
            annotation_position="top right",
            annotation_font_color=RED,
        )
    price_fig.update_layout(**plotly_layout(
        height=460, xaxis_rangeslider_visible=False,
    ))
    st.plotly_chart(price_fig, use_container_width=True)
    st.caption(
        "OHLC candles · MA20/MA50 · dashed S/R from the past year's swing points."
    )

    section_header("RSI (14)")
    rsi_fig = go.Figure()
    rsi_fig.add_trace(go.Scatter(
        x=view.index, y=view['RSI'], name="RSI",
        line=dict(width=2, color=BLUE),
        fill="tozeroy", fillcolor="rgba(108,182,255,0.08)",
    ))
    rsi_fig.add_hrect(
        y0=70, y1=100, fillcolor="rgba(240,113,120,0.08)", line_width=0,
    )
    rsi_fig.add_hrect(
        y0=0, y1=30, fillcolor="rgba(54,179,126,0.08)", line_width=0,
    )
    rsi_fig.add_hline(y=70, line_dash="dot", line_color=RED, line_width=1)
    rsi_fig.add_hline(y=30, line_dash="dot", line_color=ACCENT, line_width=1)
    rsi_fig.update_layout(**plotly_layout(
        height=230,
        yaxis=dict(range=[0, 100], title="RSI", gridcolor="rgba(255,255,255,0.05)"),
        showlegend=False,
    ))
    st.plotly_chart(rsi_fig, use_container_width=True)
    st.caption(HELP["rsi"])


def render_footer():
    """Back-compat shim — app now uses ui.theme.footer_bar."""
    from ui.theme import footer_bar
    footer_bar()
