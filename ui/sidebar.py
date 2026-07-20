"""Sidebar: settings, stock picker, global toggle, mini screener."""
import streamlit as st

from model import MODEL_TYPES, HAS_XGB, global_model_available
from ui.help_text import HELP
from ui.services import (
    STOCKS_FILE, STOCKS_UNIVERSE_FILE, DEFAULT_STOCKS, SCAN_MAX,
    load_stock_list, run_scan,
)
from ui.theme import brand_strip


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
            f"screener processes up to **{SCAN_MAX}** names per run"
        )

        _options = list(stocks.keys()) + ["Custom symbol…"]
        if st.session_state.get("stock_choice") not in _options:
            st.session_state.pop("stock_choice", None)

        choice = st.selectbox(
            "Stock",
            options=_options,
            key="stock_choice",
            help="Type to search. Pick 'Custom symbol…' for any Yahoo ticker.",
        )

        if choice == "Custom symbol…":
            symbol = st.text_input(
                "Yahoo Finance symbol",
                placeholder="e.g. TATAPOWER.NS or AAPL",
            ).strip().upper()
            display_name = symbol
        else:
            symbol = stocks[choice]
            display_name = choice

        st.caption("NSE tickers end in **.NS**")

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
            st.toast("Caches cleared — reloading…", icon="🔄")
            st.rerun()

        def _jump_to(stock_name):
            st.session_state["stock_choice"] = stock_name

        with st.expander("Screener", expanded=False):
            if st.button(
                "Run screener", key="sidebar_scan",
                use_container_width=True, type="primary", help=HELP["scanner"],
            ):
                st.session_state["scan_requested"] = True

            if st.session_state.get("scan_requested"):
                from model import rank_buy_candidates
                side_df, _side_fails = run_scan(tuple(stocks.items()))
                if side_df.empty:
                    st.caption("No results — open the Screener section.")
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
                        st.caption("No BUY screens today — open Screener.")
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
                        st.caption("Top picks · open Screener for full table")

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
