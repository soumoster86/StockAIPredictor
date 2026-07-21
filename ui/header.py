"""Stock header: signal banner + price metric + sparkline (Streamlit-native)."""
import plotly.graph_objects as go
import streamlit as st

from data import FEATURES
from ui.help_text import HELP
from ui.styles import RED
from ui.theme import (
    ACCENT,
    plotly_layout,
    signal_banner_html,
)


def render_header(display_name, symbol, data, predictor, scaler, thresholds,
                  signal, confidence, currency):
    last_close = float(data['Close'].iloc[-1])
    prev_close = float(data['Close'].iloc[-2])
    day_change = (last_close / prev_close - 1) * 100
    entry_thr, exit_thr = thresholds

    st.markdown(f"#### {display_name}  ·  `{symbol}`")

    if signal == "BUY":
        title = f"▲  BUY  ·  {confidence * 100:.1f}% confidence"
        body = "Model favors a long entry at the latest close."
        # Native Streamlit status for accessibility + color
        st.success(f"**BUY** — model confidence {confidence * 100:.1f}%")
    elif signal == "SELL":
        title = f"▼  SELL / cash  ·  {(1 - confidence) * 100:.1f}% cash bias"
        body = "Exit longs or stay in cash — not a short recommendation."
        st.error(
            f"**SELL / move to cash** — model confidence "
            f"{(1 - confidence) * 100:.1f}%"
        )
    else:
        title = f"◆  HOLD  ·  {confidence * 100:.1f}% conviction"
        body = "Conviction is mixed; no fresh long entry suggested."
        st.warning(f"**HOLD** — model is uncertain ({confidence * 100:.1f}%)")

    # Colored banner with INLINE styles (works even if global CSS fails)
    st.markdown(
        signal_banner_html(signal, title, body),
        unsafe_allow_html=True,
    )

    st.caption(
        f"Long-only · enter above **{entry_thr:.2f}**, exit below **{exit_thr:.2f}** "
        f"(validation-tuned thresholds)."
    )

    if hasattr(predictor, "member_probs_last"):
        votes = predictor.member_probs_last(
            scaler.transform(data[FEATURES].values)
        )
        spread = max(votes.values()) - min(votes.values())
        agreement = (
            "models broadly agree" if spread < 0.10
            else "models disagree — lower conviction"
        )
        vote_txt = " · ".join(f"**{k}** {v * 100:.0f}%" for k, v in votes.items())
        st.caption(f"Ensemble votes: {vote_txt} — {agreement}.")

    # Price row
    c1, c2 = st.columns([1, 1.4], gap="medium")
    with c1:
        st.metric(
            "Last close",
            f"{currency}{last_close:,.2f}",
            delta=f"{day_change:+.2f}%",
            help=HELP["last_close"],
        )
        st.caption(f"As of **{data.index[-1]:%d %b %Y}**")

    with c2:
        _spark = data['Close'].tail(30)
        fill = ACCENT if day_change >= 0 else RED
        fig = go.Figure(go.Scatter(
            x=_spark.index, y=_spark.values, mode="lines",
            line=dict(width=2.5, color=fill, shape="spline"),
            fill="tozeroy",
            fillcolor=(
                "rgba(54,179,126,0.14)" if day_change >= 0
                else "rgba(240,113,120,0.14)"
            ),
            hovertemplate="%{x|%d %b}<br>%{y:,.2f}<extra></extra>",
        ))
        y0 = float(_spark.min()) * 0.998
        fig.update_layout(**plotly_layout(
            height=110,
            margin=dict(l=0, r=0, t=8, b=0),
            showlegend=False,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False, range=[y0, float(_spark.max()) * 1.002]),
        ))
        st.plotly_chart(
            fig, use_container_width=True,
            config={"displayModeBar": False},
        )
        st.caption("30-day trend")
