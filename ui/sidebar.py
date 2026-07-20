"""Sidebar: settings, stock picker, global toggle, mini screener."""
import streamlit as st

from model import MODEL_TYPES, HAS_XGB, global_model_available
from ui.help_text import HELP
from ui.services import (
    STOCKS_FILE, STOCKS_UNIVERSE_FILE, DEFAULT_STOCKS, SCAN_MAX, SCAN_BATCH,
    load_stock_list, ensure_scan_session, advance_scan_session,
    scan_progress, get_scan_results, reset_scan_session,
    maybe_autoseed_precomputed, seed_session_from_precomputed,
    precomputed_status,
)
from ui.theme import brand_strip
from ui.stock_picker import render_sidebar_stock_picker, resolve_selection


def render_sidebar():
    """Render sidebar controls. Returns a dict of session choices."""
    from auth import logout_button

    with st.sidebar:
        brand_strip("Decision support · NSE-ready")
        logout_button()

        st.markdown("**Watchlist**")
        uploaded = st.file_uploader(
            "Upload stock list (CSV)", type="csv",
            help="CSV with columns Name, Symbol (Yahoo tickers, e.g. TATAPOWER.NS).",
        )

        stocks, list_source = {}, ""
        try:
            if uploaded is not None:
                stocks = load_stock_list(uploaded)
                list_source = f"uploaded ({len(stocks)})"
            elif STOCKS_FILE.exists():
                stocks = load_stock_list(str(STOCKS_FILE))
                src_name = (
                    "stocks_universe.csv"
                    if STOCKS_FILE.resolve() == STOCKS_UNIVERSE_FILE.resolve()
                    else STOCKS_FILE.name
                )
                list_source = f"{src_name} ({len(stocks)})"
        except ValueError as e:
            st.error(f"Could not read stock list: {e}")

        if not stocks:
            stocks = DEFAULT_STOCKS
            list_source = "fallback"

        st.caption(
            f"**{list_source}** · full list in picker · "
            f"screener batches of **{SCAN_BATCH}**"
        )

        render_sidebar_stock_picker(stocks)
        symbol, display_name = resolve_selection(stocks)

        st.markdown("**Model**")
        _has_global = global_model_available()
        if _has_global:
            use_global = st.checkbox(
                "Use global model", value=True, help=HELP["use_global"],
            )
            st.caption("🌐 Pooled weights available")
        else:
            use_global = False
            st.caption("Per-stock training (no global artifact)")

        model_type = st.selectbox(
            "Architecture", MODEL_TYPES, index=0, help=HELP["model_type"],
            disabled=use_global,
        )
        if use_global:
            st.caption("Selector applies only when global is off.")
        if model_type.startswith("Ensemble") and not HAS_XGB:
            st.warning(
                "XGBoost missing — ensemble uses NN + RF (+ HistGBDT).",
                icon="⚠️",
            )
        if model_type in ("LSTM", "GRU"):
            st.caption("Sequence models use a 20-day lookback.")

        calibrate = st.checkbox(
            "Calibrate probabilities", value=False, help=HELP["calibrate"],
        )

        st.markdown("**Actions**")
        if st.button("Refresh data", use_container_width=True, help=HELP["refresh"]):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state.pop("scan_requested", None)
            reset_scan_session(stocks)
            st.toast("Caches cleared — reloading…", icon="🔄")
            st.rerun()

        def _jump_to(stock_name):
            from ui.stock_picker import set_stock_pick
            set_stock_pick(stock_name)

        with st.expander("Screener", expanded=False):
            ensure_scan_session(stocks)
            maybe_autoseed_precomputed(stocks)
            sp = scan_progress(stocks)
            pre = precomputed_status(stocks)
            if sp.get("source") == "precomputed":
                st.caption(f"⚡ Precomputed · {sp['succeeded']} names")
            else:
                st.caption(
                    f"Coverage **{sp['attempted']}/{sp['total']}** · "
                    f"{sp['succeeded']} scored"
                )
            if pre["available"] and sp.get("source") != "precomputed":
                if st.button(
                    "Load precomputed", key="sidebar_precomp",
                    use_container_width=True,
                ):
                    seed_session_from_precomputed(stocks, allow_stale=True)
                    st.rerun()
            if st.button(
                f"Live batch ({SCAN_BATCH})", key="sidebar_scan",
                use_container_width=True, type="primary", help=HELP["scanner"],
            ):
                if sp.get("source") == "precomputed" or sp["attempted"] == 0:
                    reset_scan_session(stocks)
                advance_scan_session(stocks, n_batches=1)
                st.rerun()

            if sp["attempted"] == 0 and sp.get("source") != "precomputed":
                st.caption("Open **Screener** for full controls.")
            else:
                from model import rank_buy_candidates
                side_df = get_scan_results()
                if side_df.empty:
                    st.caption("No scores yet — try another batch.")
                else:
                    picks = rank_buy_candidates(
                        side_df, min_prob=0.55, max_risk=8.0,
                        require_edge=True, top_n=5,
                    )
                    if picks.empty:
                        picks = rank_buy_candidates(
                            side_df, min_prob=0.55, max_risk=10.0,
                            require_edge=False, top_n=5,
                        )
                    if picks.empty:
                        st.caption("No BUY screens in scored set yet.")
                    else:
                        for _, row in picks.iterrows():
                            label = (
                                f"#{int(row['Rank'])} {row['Symbol']}  "
                                f"{float(row['Probability Up']) * 100:.0f}%  "
                                f"· score {float(row['Buy Score']):.0f}"
                            )
                            st.button(
                                label,
                                key=f"jump_{row['Symbol']}_{int(row['Rank'])}",
                                on_click=_jump_to, args=(row["Name"],),
                                use_container_width=True,
                            )
                        st.caption("Top picks from scored coverage so far")

        with st.expander("About", expanded=False):
            st.markdown(
                "Regularized ensemble on price, momentum, volatility and volume. "
                "Signals are stress-tested on held-out history with costs. "
                "**Educational — not investment advice.**"
            )

    return {
        "stocks": stocks,
        "symbol": symbol,
        "display_name": display_name,
        "model_type": model_type,
        "calibrate": calibrate,
        "use_global": use_global,
    }
