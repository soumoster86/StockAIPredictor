"""Shared stock selection for sidebar + Market workspace (main page).

Streamlit rule: never assign to a widget's key after that widget is created
in the same run. We keep a non-widget source of truth:

  stock_pick          — canonical selected name (or Custom label)
  custom_symbol       — Yahoo ticker when Custom is selected

Widget keys (only written *before* each widget is created, or in on_change):
  stock_choice_side   — sidebar selectbox
  stock_choice_main   — main-page selectbox
"""
from __future__ import annotations

import streamlit as st

CUSTOM_LABEL = "Custom symbol…"
PICK_KEY = "stock_pick"
SIDE_KEY = "stock_choice_side"
MAIN_KEY = "stock_choice_main"
CUSTOM_KEY = "custom_symbol"


def stock_options(stocks: dict) -> list[str]:
    return list(stocks.keys()) + [CUSTOM_LABEL]


def set_stock_pick(name: str) -> None:
    """Programmatic select (screener jump, etc.). Safe anytime — not a widget key."""
    st.session_state[PICK_KEY] = name


def _init_pick(options: list[str]) -> None:
    pick = st.session_state.get(PICK_KEY)
    if pick not in options:
        st.session_state[PICK_KEY] = options[0] if options else CUSTOM_LABEL


def _on_side_change() -> None:
    st.session_state[PICK_KEY] = st.session_state[SIDE_KEY]


def _on_main_change() -> None:
    st.session_state[PICK_KEY] = st.session_state[MAIN_KEY]


def resolve_selection(stocks: dict) -> tuple[str, str]:
    """Return (symbol, display_name) from the shared pick."""
    choice = st.session_state.get(PICK_KEY)
    if choice == CUSTOM_LABEL or choice not in stocks:
        symbol = str(st.session_state.get(CUSTOM_KEY, "") or "").strip().upper()
        return symbol, symbol
    return stocks[choice], choice


def render_sidebar_stock_picker(stocks: dict) -> None:
    """Sidebar stock selectbox."""
    options = stock_options(stocks)
    _init_pick(options)

    # Seed widget value *before* the selectbox is created (required by Streamlit)
    st.session_state[SIDE_KEY] = st.session_state[PICK_KEY]

    st.selectbox(
        "Stock",
        options=options,
        key=SIDE_KEY,
        on_change=_on_side_change,
        help="Type to search. Pick 'Custom symbol…' for any Yahoo ticker.",
    )

    if st.session_state.get(PICK_KEY) == CUSTOM_LABEL:
        custom = str(st.session_state.get(CUSTOM_KEY, "") or "").strip()
        if custom:
            st.caption(f"Custom ticker · `{custom.upper()}` (edit on main page)")
        else:
            st.caption("Enter the Yahoo ticker under **Select stock** on the main page.")
    else:
        st.caption("NSE tickers end in **.NS** · same list on the main page")


def render_main_stock_picker(stocks: dict) -> tuple[str, str]:
    """Market workspace stock selectbox. Returns (symbol, display_name)."""
    options = stock_options(stocks)
    _init_pick(options)

    # Seed widget value *before* the selectbox is created
    st.session_state[MAIN_KEY] = st.session_state[PICK_KEY]

    st.markdown("**Select stock**")
    col_pick, col_meta = st.columns([2.4, 1.2], gap="medium")
    with col_pick:
        st.selectbox(
            "Select stock",
            options=options,
            key=MAIN_KEY,
            on_change=_on_main_change,
            help="Type to search the full watchlist. Synced with the sidebar.",
            label_visibility="collapsed",
        )
    with col_meta:
        choice = st.session_state.get(PICK_KEY)
        if choice and choice != CUSTOM_LABEL and choice in stocks:
            st.caption(f"Yahoo · `{stocks[choice]}`")
        else:
            st.caption("Or pick **Custom symbol…**")

    if st.session_state.get(PICK_KEY) == CUSTOM_LABEL:
        st.text_input(
            "Yahoo Finance symbol",
            key=CUSTOM_KEY,
            placeholder="e.g. TATAPOWER.NS or AAPL",
            help="NSE tickers end in .NS. Any Yahoo Finance symbol works.",
        )

    return resolve_selection(stocks)
