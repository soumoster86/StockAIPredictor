"""Main analysis tabs — prediction, scanner, plan, backtest, walk-forward,
journal, charts."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from alerts import alerts_status, get_alerts_config, run_alerts, select_alert_candidates
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
from report import (
    HAS_REPORTLAB,
    build_report_dict,
    filename_stem,
    report_to_csv_bytes,
    report_to_pdf_bytes,
)
from ui.help_text import HELP
from ui.sectors import classify_sector, enrich_with_sector
from ui.services import (
    SCAN_BATCH,
    advance_scan_session,
    batch_progress_label,
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


def _render_report_download(ctx):
    """One-click CSV / PDF analysis pack for the selected stock."""
    data = ctx["data"]
    last_close = float(data["Close"].iloc[-1])
    prev_close = float(data["Close"].iloc[-2]) if len(data) > 1 else last_close
    day_change = (last_close / prev_close - 1) if prev_close else 0.0
    data_asof = data.index[-1].strftime("%Y-%m-%d") if len(data) else ""

    report = build_report_dict(
        display_name=ctx["display_name"],
        symbol=ctx["symbol"],
        signal=ctx["signal"],
        confidence=ctx["confidence"],
        model_type=ctx["model_type"],
        use_global=ctx["use_global"],
        thresholds=ctx["thresholds"],
        metrics=ctx["metrics"],
        risk=ctx["risk"],
        plan=ctx["plan"],
        sr=ctx["sr"],
        last_close=last_close,
        day_change=day_change,
        currency=ctx.get("currency") or "",
        data_asof=data_asof,
    )
    stem = filename_stem(ctx["symbol"])

    section_header("Download report")
    st.caption(
        "One-click pack of signal, risk, trade plan, and hold-out metrics — "
        "for notes or sharing. Educational only."
    )
    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "Download CSV",
            data=report_to_csv_bytes(report),
            file_name=f"{stem}.csv",
            mime="text/csv",
            use_container_width=True,
            help=HELP.get("report_download", "Flat key/value export of this analysis."),
            key=f"report_csv_{ctx['symbol']}",
        )
    with d2:
        if HAS_REPORTLAB:
            try:
                pdf_bytes = report_to_pdf_bytes(report)
                st.download_button(
                    "Download PDF",
                    data=pdf_bytes,
                    file_name=f"{stem}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    type="primary",
                    help=HELP.get("report_download", "Printable multi-section PDF summary."),
                    key=f"report_pdf_{ctx['symbol']}",
                )
            except Exception as e:
                st.warning(f"PDF generation failed: {e}")
        else:
            st.caption("PDF needs `reportlab` (`pip install reportlab`).")


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

    st.divider()
    _render_report_download(ctx)


def _render_alerts_panel(scan_df, asof=None):
    """Telegram / email alerts for top BUY screens (secrets-driven)."""
    section_header("Alerts")
    status = alerts_status()
    cfg = get_alerts_config()

    st.caption(
        f"**{status['label']}** · min Buy Score **{cfg['min_buy_score']:.0f}** · "
        f"top **{cfg['top_n']}** · min P(up) **{cfg['min_probability']:.0%}**. "
        "Configure via Streamlit secrets `[alerts]` (see DEPLOYMENT.md). "
        "Same snapshot is not re-alerted."
    )

    preview = select_alert_candidates(
        scan_df,
        min_buy_score=cfg["min_buy_score"],
        min_probability=cfg["min_probability"],
        max_risk=cfg["max_risk"],
        require_edge=cfg["require_edge"],
        top_n=cfg["top_n"],
    )
    if preview.empty:
        st.info(
            "No alert candidates with current thresholds. Lower min Buy Score "
            "in secrets, or wait for stronger BUY screens."
        )
    else:
        st.markdown(
            f"**{len(preview)}** name(s) would alert: "
            + ", ".join(f"`{s}`" for s in preview["Symbol"].astype(str).tolist()[:12])
            + ("…" if len(preview) > 12 else "")
        )

    a1, a2, a3 = st.columns([1.2, 1.2, 1.1])
    with a1:
        dry = st.checkbox(
            "Dry run (preview only)",
            value=not status["configured"],
            help="Build the message without sending Telegram/email.",
            key="alerts_dry_run",
        )
    with a2:
        force = st.checkbox(
            "Re-alert already sent",
            value=False,
            help="Ignore de-dupe state for this snapshot (useful for testing).",
            key="alerts_force",
        )
    with a3:
        send_disabled = not status["configured"] and not dry
        if st.button(
            "Send / preview alerts",
            type="primary",
            use_container_width=True,
            disabled=send_disabled,
            help=HELP.get(
                "alerts",
                "Notify Telegram/email when top BUY screens clear your filters.",
            ),
            key="alerts_send_btn",
        ):
            with st.spinner("Running alerts…"):
                result = run_alerts(
                    scan_df,
                    asof=asof or "live",
                    force=force,
                    dry_run=dry,
                )
            if result.get("skipped"):
                st.info(
                    f"Skipped: {result.get('reason')} "
                    f"(candidates={result.get('candidates')}, new={result.get('new')})"
                )
            elif result.get("ok"):
                action = "Preview" if dry else "Sent"
                st.success(
                    f"{action}: **{result.get('new', 0)}** new · "
                    f"channels {', '.join(result.get('channels') or []) or '—'}"
                )
                if result.get("message"):
                    with st.expander("Message body", expanded=dry):
                        st.code(result["message"], language=None)
            else:
                st.error(result.get("error") or result.get("reason") or "Alert failed")
                if result.get("message"):
                    with st.expander("Message body"):
                        st.code(result["message"], language=None)

    if not status["configured"]:
        st.caption(
            "To enable: set `telegram_bot_token` + `telegram_chat_id` and/or SMTP "
            "fields under `[alerts]` in secrets, and `enabled = true`."
        )


def _run_scan_with_progress(stocks, n_batches: int = 1) -> None:
    """Live scan with Batch N of M progress UI."""
    labels = batch_progress_label(stocks, n_batches)
    run_n = max(1, labels["run_n"])
    total_batches = labels["total_batches"]
    start = labels["next_batch"]
    end = min(start + run_n - 1, total_batches)

    status = st.status(
        f"Scanning batch {start} of {total_batches}…",
        expanded=True,
    )
    bar = st.progress(0.0, text=f"Batch {start} of {total_batches}")

    def _cb(done_i, run_n_, batch_no, total_b, offset, total_sym):
        # done_i is 0..run_n (pre/post); clamp for bar
        frac = min(max(done_i / max(run_n_, 1), 0.0), 1.0)
        label = (
            f"Batch {batch_no} of {total_b} · "
            f"{offset:,}/{total_sym:,} symbols"
        )
        bar.progress(frac, text=label)
        status.update(label=f"Scanning · {label}", state="running")

    with status:
        st.write(
            f"Live Yahoo walk · batches **{start}–{end}** of **{total_batches}** "
            f"(~{SCAN_BATCH} names each)"
        )
        advance_scan_session(
            stocks, n_batches=run_n, progress_callback=_cb,
        )
        status.update(
            label=f"Done · batch {end} of {total_batches}",
            state="complete",
        )
        bar.progress(1.0, text=f"Finished through batch {end} of {total_batches}")


def _sort_scan_df(df: pd.DataFrame, sort_by: str, ascending: bool) -> pd.DataFrame:
    """Sort scored results by a user-chosen column."""
    if df is None or df.empty:
        return df
    col_map = {
        "Buy Score": "Buy Score",
        "Probability Up": "Probability Up",
        "Risk": "Risk",
        "Reward Risk": "Reward Risk",
        "Day": "Day",
        "Name": "Name",
        "Symbol": "Symbol",
        "Price": "Price",
        "Sector": "Sector",
    }
    col = col_map.get(sort_by, "Buy Score")
    if col not in df.columns:
        return df
    out = df.sort_values(col, ascending=ascending, na_position="last")
    return out.reset_index(drop=True)


def _render_sticky_filters(display_name: str, *, key_prefix: str = "scr") -> dict:
    """Always-visible sticky filter bar (session-persistent keys)."""
    my_sector = classify_sector(display_name)
    st.markdown(
        '<div class="scr-sticky-filters">'
        '<div class="scr-sticky-title">Filters · sticky</div>',
        unsafe_allow_html=True,
    )
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    min_prob = r1c1.slider(
        "Min probability", 0.50, 0.80, 0.55, 0.01,
        help="Only BUY screens at or above this model probability.",
        key=f"{key_prefix}_min_prob",
    )
    max_risk = r1c2.slider(
        "Max risk score", 3.0, 10.0, 8.0, 0.5,
        help="Drop names riskier than this (1 calm → 10 wild).",
        key=f"{key_prefix}_max_risk",
    )
    top_n = r1c3.slider(
        "Show top N", 3, 20, 8, 1, key=f"{key_prefix}_top_n",
    )
    require_edge = r1c4.checkbox(
        "Require model edge",
        value=True,
        help="Keep only names where test accuracy ≥ majority baseline.",
        key=f"{key_prefix}_require_edge",
    )

    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    only_sector = r2c1.checkbox(
        f"Only my sector ({my_sector})",
        value=False,
        help=(
            f"Keep names tagged **{my_sector}** (from the selected stock’s name). "
            "Heuristic tags — not official GICS industries."
        ),
        key=f"{key_prefix}_only_sector",
    )
    sort_by = r2c2.selectbox(
        "Sort by",
        [
            "Buy Score", "Probability Up", "Risk", "Reward Risk",
            "Day", "Price", "Name", "Symbol", "Sector",
        ],
        index=0,
        key=f"{key_prefix}_sort_by",
    )
    sort_asc = r2c3.selectbox(
        "Order",
        ["High → low", "Low → high"],
        index=0,
        key=f"{key_prefix}_sort_order",
    )
    r2c4.caption(f"Your sector · **{my_sector}**")
    st.markdown("</div>", unsafe_allow_html=True)

    return {
        "min_prob": min_prob,
        "max_risk": max_risk,
        "top_n": top_n,
        "require_edge": require_edge,
        "only_sector": only_sector,
        "my_sector": my_sector,
        "sort_by": sort_by,
        "sort_asc": sort_asc == "Low → high",
    }


def _apply_sector_filter(df: pd.DataFrame, only_sector: bool, my_sector: str) -> pd.DataFrame:
    if df is None or df.empty or not only_sector:
        return df
    if "Sector" not in df.columns:
        df = enrich_with_sector(df)
    return df[df["Sector"] == my_sector].reset_index(drop=True)


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
    sector = row.get("Sector")

    day_s = f"{float(day) * 100:+.1f}%" if day is not None and pd.notna(day) else "—"
    rr_s = f"1:{float(rr):.1f}" if rr is not None and pd.notna(rr) else "—"
    risk_s = f"{float(risk):.1f}" if risk is not None and pd.notna(risk) else "—"
    price_s = f"{float(price):,.2f}" if price is not None and pd.notna(price) else "—"
    sector_s = str(sector) if sector is not None and pd.notna(sector) else None

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
            sector=sector_s,
        ),
        unsafe_allow_html=True,
    )
    st.button(
        f"Open full Signal · {sym}",
        key=f"{key_prefix}_{rank}_{sym}",
        use_container_width=True,
        type="secondary",
        on_click=on_jump,
        args=(name,),
    )


def _scanner_summary_metrics(scan_df, prog):
    """Compact KPI strip for scored results."""
    n_buy = int((scan_df["Screen"] == "BUY").sum()) if "Screen" in scan_df else 0
    n_sell = int((scan_df["Screen"] == "SELL").sum()) if "Screen" in scan_df else 0
    top_score = float(scan_df["Buy Score"].max()) if "Buy Score" in scan_df else 0.0

    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    # Always show dynamic Source (Nightly / Manual load / Live scan / …)
    sc1.metric(
        "Source",
        prog.get("source_short") or "—",
        help=prog.get("source_help") or "How this session’s rankings were loaded.",
    )
    if prog.get("source") == "live":
        sc2.metric(
            "Coverage",
            f"{prog['attempted']}/{prog['total']}",
            help="Symbols attempted vs full watchlist size (live scan only).",
        )
    else:
        sc2.metric("Scored", f"{len(scan_df):,}")
    if prog.get("source") == "live":
        sc3.metric("Scored", f"{len(scan_df):,}")
        sc4.metric("BUY", f"{n_buy:,}", help="Screen calls = BUY")
        sc5.metric("SELL", f"{n_sell:,}", help="Screen calls = SELL")
    else:
        sc3.metric("BUY", f"{n_buy:,}", help="Screen calls = BUY")
        sc4.metric("SELL", f"{n_sell:,}", help="Screen calls = SELL")
        sc5.metric("Top score", f"{top_score:.0f}", help="Highest Buy Score in results")


def _scanner_status_line(prog, scan_df):
    """One-line status under the KPI strip — reflects Nightly / Manual / Live."""
    title = prog.get("source_title") or prog.get("source_short") or "Screener"
    asof = prog.get("asof") or "—"
    n = len(scan_df)
    origin = prog.get("origin") or prog.get("source")

    if origin in ("nightly", "manual", "local", "precomputed") or prog.get("source") == "precomputed":
        bits = [
            f"**{title}** · **{n:,}** names · as of **{asof}**",
        ]
        if prog.get("workflow"):
            bits.append(f"workflow **{prog['workflow']}**")
        elif prog.get("runner"):
            bits.append(f"runner **{prog['runner']}**")
        bits.append("screen only, not the full Signal")
        st.caption(" · ".join(bits))
        if prog.get("run_url"):
            st.caption(f"CI run: {prog['run_url']}")
    elif prog.get("complete"):
        st.caption(
            f"**Live scan** complete · **{n}** scored · "
            f"**{prog['failed']}** skipped · screen only"
        )
    else:
        st.caption(
            f"**Live scan** partial · **{n}** scored so far · "
            f"open **Data source** to cover more of the list"
        )


def _render_scanner_data_source(stocks, prog, pre, scan_failures):
    """Section: load precomputed rankings + live batch controls."""
    section_header("Data source")
    st.caption(
        "Load offline rankings for instant results, or walk the list live in batches. "
        "Screen only — open a stock for the full Signal."
    )

    # ---- Offline ----
    st.markdown("##### Offline rankings")
    if pre["available"]:
        stale_note = (
            f" · ⚠️ **{pre.get('age_hours')}h** old"
            if pre.get("stale") else " · fresh"
        )
        st.success(
            f"**Ready** · {pre['n_watchlist']} names match your watchlist "
            f"(file has {pre['n_file']}) · as of **{pre.get('asof') or '—'}**"
            f"{stale_note}"
        )
        pc1, pc2 = st.columns([1.2, 1.5])
        with pc1:
            if st.button(
                "Load precomputed rankings",
                type="primary",
                use_container_width=True,
                help="Manual load from rankings/rankings_latest.csv (no Yahoo calls). Source becomes Manual load.",
                key="scr_load_precomputed",
            ):
                if seed_session_from_precomputed(
                    stocks, allow_stale=True, load_mode="manual",
                ):
                    st.session_state["scanner_panel"] = "Top picks"
                    st.toast("Manual load · precomputed rankings", icon="⚡")
                    st.rerun()
                else:
                    st.warning("Could not load precomputed file.")
        with pc2:
            st.caption(
                "Refresh offline file: `python scripts/precompute_rankings.py` "
                "or **Actions → Nightly rankings**."
            )
            meta = pre.get("meta") or {}
            if meta.get("runner") == "github-actions" and meta.get("run_url"):
                st.caption(f"Last CI run: {meta.get('run_url')}")
    else:
        st.info(
            "No precomputed rankings for this watchlist. Use live scan below, "
            "or generate offline / trigger **Nightly rankings**."
        )

    st.divider()

    # ---- Live ----
    st.markdown("##### Live batch scan")
    if prog.get("source") == "precomputed" and prog.get("asof"):
        st.info(
            f"Currently showing **{prog.get('source_title') or 'precomputed'}** "
            f"(as of **{prog['asof']}**). "
            "Start a live scan to switch Source to **Live scan**."
        )

    attempted = prog["attempted"]
    total = max(prog["total"], 1)
    live_mode = prog.get("source") != "precomputed"
    st.progress(
        min(attempted / total, 1.0) if live_mode else 1.0,
        text=(
            f"Precomputed · {prog['succeeded']} names · as of {prog.get('asof') or '—'}"
            if prog.get("source") == "precomputed" else
            f"Covered {prog['attempted']} / {prog['total']} · "
            f"{prog['succeeded']} scored · {prog['failed']} skipped · "
            f"{prog['remaining']} left"
        ),
    )

    b1, b2, b3, b4 = st.columns(4)
    with b1:
        start_label = (
            f"Start live scan ({SCAN_BATCH})"
            if not prog["active"] and prog["attempted"] == 0
            else f"Rescan from start ({SCAN_BATCH})"
        )
        if st.button(
            start_label, type="primary", use_container_width=True,
            help=HELP["scanner"], key="scr_start_scan",
        ):
            reset_scan_session(stocks)
            _run_scan_with_progress(stocks, n_batches=1)
            st.session_state["scanner_panel"] = "Top picks"
            st.rerun()
    with b2:
        next_n = min(SCAN_BATCH, prog["remaining"]) if prog["remaining"] else SCAN_BATCH
        next_disabled = (
            prog.get("source") == "precomputed"
            or prog["complete"]
            or prog["total"] == 0
        )
        bl = batch_progress_label(stocks, 1)
        next_label = (
            f"Next · batch {bl['next_batch']}/{bl['total_batches']}"
            if prog["remaining"] else "Scan complete"
        )
        if st.button(
            next_label if prog["remaining"] else "Scan complete",
            use_container_width=True,
            disabled=next_disabled,
            help="Continue through the universe without losing prior results.",
            key="scr_next_batch",
        ):
            _run_scan_with_progress(stocks, n_batches=1)
            st.rerun()
    with b3:
        multi = min(3, max(1, (prog["remaining"] + SCAN_BATCH - 1) // SCAN_BATCH))
        multi_disabled = (
            prog.get("source") == "precomputed"
            or prog["complete"]
            or prog["total"] == 0
        )
        blm = batch_progress_label(stocks, multi)
        multi_label = (
            f"+{multi} · to batch "
            f"{min(blm['next_batch'] + multi - 1, blm['total_batches'])}/"
            f"{blm['total_batches']}"
        )
        if st.button(
            multi_label if prog["remaining"] else f"+{multi} batches",
            use_container_width=True,
            disabled=multi_disabled,
            help="Run up to 3 batches in a row with live Batch N of M progress.",
            key="scr_multi_batch",
        ):
            _run_scan_with_progress(stocks, n_batches=multi)
            st.rerun()
    with b4:
        if st.button("Reset", use_container_width=True, key="scr_reset_scan"):
            reset_scan_session(stocks)
            st.rerun()

    engine = "global model" if global_model_available() else "fast per-stock tree"
    bl_now = batch_progress_label(stocks, 1)
    if prog.get("source") == "precomputed":
        st.caption(
            f"Engine: **{engine}** · source **{prog.get('source_short') or 'Precomputed'}** · "
            f"watchlist **{prog['total']}**"
        )
    else:
        st.caption(
            f"Engine: **{engine}** · source **Live scan** · batch size **{SCAN_BATCH}** · "
            f"progress **batch {min(bl_now['next_batch'], bl_now['total_batches'])} "
            f"of {bl_now['total_batches']}** · "
            f"{prog['attempted']}/{prog['total']} symbols"
        )

    if scan_failures:
        with st.expander(f"{len(scan_failures)} stock(s) skipped", expanded=False):
            for sym, reason in scan_failures[:100]:
                st.markdown(f"- `{sym}` — {reason}")
            if len(scan_failures) > 100:
                st.caption(f"…and {len(scan_failures) - 100} more")


def _render_scanner_top_picks(scan_df, symbol, signal, display_name):
    """Section: sticky filters + pick cards (primary user focus)."""
    section_header("Top picks")
    st.caption(
        "Best long **screen** candidates. Each card is a **Screen** call — "
        "open full analysis for the real Signal. Educational only."
    )

    filters = _render_sticky_filters(display_name, key_prefix="scr")
    scan_df = enrich_with_sector(scan_df)
    pool = _apply_sector_filter(
        scan_df, filters["only_sector"], filters["my_sector"],
    )

    current_screen = (
        pool[pool["Symbol"] == symbol] if "Symbol" in pool.columns else pool.iloc[0:0]
    )
    if not current_screen.empty:
        screen_call = current_screen.iloc[0]["Screen"]
        if screen_call != signal:
            st.warning(
                f"Screen shows **{screen_call}** for `{symbol}`, full Signal is "
                f"**{signal}** — prefer the full Signal for the selected stock."
            )

    # rank_buy_candidates sorts by Buy Score; we re-sort after if user picked another col
    picks = rank_buy_candidates(
        pool,
        min_prob=filters["min_prob"],
        max_risk=filters["max_risk"],
        require_edge=filters["require_edge"],
        top_n=None if filters["sort_by"] != "Buy Score" else filters["top_n"],
    )
    if not picks.empty:
        # If sorting by non-score, re-rank from full filtered BUY pool
        if filters["sort_by"] != "Buy Score" or filters["sort_asc"]:
            picks = _sort_scan_df(picks, filters["sort_by"], filters["sort_asc"])
            picks = picks.head(int(filters["top_n"])).reset_index(drop=True)
            if "Rank" in picks.columns:
                picks = picks.drop(columns=["Rank"])
            picks.insert(0, "Rank", range(1, len(picks) + 1))
        elif filters["sort_by"] == "Buy Score" and not filters["sort_asc"]:
            picks = picks.head(int(filters["top_n"])).reset_index(drop=True)

    if picks.empty:
        msg = "No names pass these filters."
        if filters["only_sector"]:
            msg += f" Sector filter is on (**{filters['my_sector']}**)."
        msg += " Loosen probability / risk, or turn off model edge."
        st.info(msg)
    else:
        def _jump_to(stock_name):
            from ui.stock_picker import set_stock_pick
            set_stock_pick(stock_name)

        cols = st.columns(min(3, len(picks)))
        for i, (_, row) in enumerate(picks.iterrows()):
            with cols[i % len(cols)]:
                _render_buy_pick_card(
                    row.get("Rank", i + 1), row,
                    key_prefix="pick", on_jump=_jump_to,
                )
        order_note = (
            f"{filters['sort_by']} "
            f"({'↑' if filters['sort_asc'] else '↓'})"
        )
        sector_note = (
            f" · sector **{filters['my_sector']}**"
            if filters["only_sector"] else ""
        )
        st.caption(
            f"**{len(picks)}** candidate(s) · sorted by {order_note}{sector_note} · "
            "each card: **Screen ≠ full Signal**"
        )

        with st.expander("Shortlist table", expanded=False):
            pick_view = picks[[
                c for c in [
                    "Rank", "Symbol", "Name", "Sector", "Buy Score", "Probability Up",
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

    with st.expander("How Buy Score works", expanded=False):
        st.markdown(
            """
            | Factor | Weight (approx.) | What it rewards |
            |--------|------------------|-----------------|
            | **Probability Up** | ~50% | Higher chance of a meaningful 1-day up move |
            | **Model edge** | ~20% | Test accuracy beating the majority baseline |
            | **Lower risk** | ~12% | Calmer vol / ATR / drawdown score |
            | **Reward : risk** | ~12% | ATR/structure trade plan with better R:R |
            | **Near support** | ~6% | Price sitting closer to a recent swing floor |

            Only **BUY** screen calls enter the shortlist (default entry ~0.55).
            **Screen ≠ full Signal** — open the stock for tuned thresholds.
            Sector tags are name heuristics for filtering, not official industries.
            """
        )


def _render_scanner_full_table(scan_df, prog, display_name):
    """Section: full watchlist ranking + sort + CSV download."""
    section_header("Full ranking")
    st.caption(
        "All scored names in this session. Click column headers in the table "
        "or use Sort below. Screen only — not investment advice."
    )

    filters = _render_sticky_filters(display_name, key_prefix="scr")
    scan_df = enrich_with_sector(scan_df)
    pool = _apply_sector_filter(
        scan_df, filters["only_sector"], filters["my_sector"],
    )

    show_mode = st.radio(
        "Show",
        ["All", "BUY only", "SELL only"],
        horizontal=True,
        key="scr_table_filter",
    )
    view_df = pool
    if show_mode == "BUY only" and "Screen" in view_df.columns:
        view_df = view_df[view_df["Screen"] == "BUY"]
    elif show_mode == "SELL only" and "Screen" in view_df.columns:
        view_df = view_df[view_df["Screen"] == "SELL"]

    view_df = _sort_scan_df(view_df, filters["sort_by"], filters["sort_asc"])

    if view_df.empty:
        st.info("No rows for this filter.")
    else:
        display_cols = [
            c for c in [
                "Symbol", "Name", "Sector", "Buy Score", "Screen", "Probability Up",
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
        st.caption(
            f"Sorted by **{filters['sort_by']}** "
            f"({'ascending' if filters['sort_asc'] else 'descending'}) · "
            f"{len(view_df):,} rows"
            + (f" · sector **{filters['my_sector']}**" if filters["only_sector"] else "")
        )

    src = prog.get("source_short") or prog.get("source") or "—"
    dl1, dl2 = st.columns([2, 1])
    with dl1:
        st.caption(
            f"Source **{src}** · coverage **{prog['attempted']}/{prog['total']}** · "
            "default thresholds · Screen ≠ full Signal"
        )
    with dl2:
        export_df = enrich_with_sector(scan_df)
        st.download_button(
            "Download CSV",
            export_df.to_csv(index=False).encode(),
            file_name="screener_results.csv",
            mime="text/csv",
            use_container_width=True,
            key="scr_download_csv",
        )


def render_scanner_tab(ctx):
    """Screener with segregated panels: Top picks | Full ranking | Data source | Alerts."""
    stocks = ctx["stocks"]
    symbol = ctx["symbol"]
    signal = ctx["signal"]
    display_name = ctx.get("display_name") or symbol

    ensure_scan_session(stocks)
    maybe_autoseed_precomputed(stocks)
    prog = scan_progress(stocks)
    pre = precomputed_status(stocks)
    scan_df = get_scan_results()
    scan_failures = st.session_state.get("scan_failures") or []
    has_results = scan_df is not None and not scan_df.empty

    # ---- Slim page header ----
    section_header("Screener")
    _engine = (
        "global model" if global_model_available() else "fast per-stock tree"
    )
    st.caption(
        f"**{prog['total']}** names · engine **{_engine}** · "
        "screen ≠ full Signal · educational only"
    )

    # ---- Sub-navigation (one focus at a time) ----
    panels = ["Top picks", "Full ranking", "Data source", "Alerts"]
    if "scanner_panel" not in st.session_state:
        st.session_state["scanner_panel"] = (
            "Top picks" if has_results else "Data source"
        )
    # If user was on results panels but data disappeared, fall back
    if not has_results and st.session_state.get("scanner_panel") in (
        "Top picks", "Full ranking", "Alerts",
    ):
        st.session_state["scanner_panel"] = "Data source"

    panel = st.radio(
        "Screener section",
        options=panels,
        horizontal=True,
        label_visibility="collapsed",
        key="scanner_panel",
        help="Switch focus — only one section is shown at a time.",
    )
    st.divider()

    # ---- Shared KPI strip when we have data (except pure data-source empty state) ----
    if has_results and panel != "Data source":
        _scanner_summary_metrics(scan_df, prog)
        _scanner_status_line(prog, scan_df)
        st.divider()

    if panel == "Data source":
        _render_scanner_data_source(stocks, prog, pre, scan_failures)
        if not has_results:
            st.info(
                "No scores yet. Load **offline rankings** or **Start live scan**, "
                "then open **Top picks**."
            )
        return

    if not has_results:
        st.warning(
            "No scored results in this session. Switch to **Data source** to load "
            "or scan."
        )
        return

    if panel == "Top picks":
        _render_scanner_top_picks(scan_df, symbol, signal, display_name)
    elif panel == "Full ranking":
        _render_scanner_full_table(scan_df, prog, display_name)
    elif panel == "Alerts":
        _render_alerts_panel(scan_df, asof=prog.get("asof"))


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
