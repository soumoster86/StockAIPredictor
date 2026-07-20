"""Visual theme helpers: tokens, optional CSS polish, Streamlit-native chrome.

Critical UI uses Streamlit widgets + **inline styles** (not CSS classes).
Many Streamlit builds strip or isolate `<style>` / class-based HTML, which
is what caused the "all plain text" look.
"""
from __future__ import annotations

import base64
import html as html_lib
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
BG = "#0b0f14"
BG_CARD = "#161d27"
BG_SOFT = "#1c2430"
BORDER = "rgba(255,255,255,0.10)"
TEXT = "#e8ecf1"
TEXT_MUTED = "rgba(232,236,241,0.65)"
TEXT_DIM = "rgba(232,236,241,0.45)"
ACCENT = "#36b37e"
ACCENT_SOFT = "rgba(54,179,126,0.16)"
RED = "#f07178"
RED_SOFT = "rgba(240,113,120,0.16)"
AMBER = "#e6b450"
AMBER_SOFT = "rgba(230,180,80,0.16)"
BLUE = "#6cb6ff"

ROOT = Path(__file__).resolve().parent.parent


def _icon_data_uri() -> str:
    for name in ("icon_256.png", "icon.png"):
        p = ROOT / "src" / name
        if p.exists():
            b64 = base64.b64encode(p.read_bytes()).decode()
            return f"data:image/png;base64,{b64}"
    return ""


def inject_global_css():
    """Chrome polish. Injected every rerun (Streamlit can drop prior HTML)."""
    # Prefer st.html (Streamlit ≥1.33) so CSS isn't sandboxed as markdown text
    css = f"""
<style>
  /* Keep enough top padding so content clears Streamlit's fixed toolbar
     (too-small padding clips the first heading — see login hero). */
  .block-container {{
    padding-top: 3.25rem !important;
    padding-bottom: 4rem !important;
    max-width: 1280px;
  }}
  /* Avoid clipping large headings inside markdown/HTML blocks */
  .block-container h1, .block-container h2, .block-container h3,
  .block-container h4, .block-container p {{
    overflow: visible !important;
  }}
  [data-testid="stVerticalBlock"] {{ overflow: visible !important; }}
  [data-testid="stMetric"] {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 0.75rem 0.9rem;
  }}
  [data-testid="stMetricLabel"] {{
    color: {TEXT_MUTED} !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }}
  /* ---- Section radio (horizontal) as clear separate pills ---- */
  div[role="radiogroup"] {{
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 0.5rem !important;
    row-gap: 0.5rem !important;
    padding: 0.35rem 0 0.15rem 0 !important;
  }}
  div[role="radiogroup"] label {{
    margin: 0 !important;
    padding: 0.45rem 0.95rem !important;
    border-radius: 999px !important;
    border: 1px solid {BORDER} !important;
    background: {BG_SOFT} !important;
    color: {TEXT_MUTED} !important;
    font-weight: 650 !important;
    font-size: 0.88rem !important;
    white-space: nowrap !important;
    cursor: pointer !important;
    min-height: 2.35rem !important;
    display: inline-flex !important;
    align-items: center !important;
  }}
  div[role="radiogroup"] label:hover {{
    border-color: rgba(54,179,126,0.45) !important;
    color: {TEXT} !important;
  }}
  /* Selected option (Base Web marks the input; style the checked label) */
  div[role="radiogroup"] label:has(input:checked) {{
    background: {ACCENT_SOFT} !important;
    border-color: rgba(54,179,126,0.7) !important;
    color: {ACCENT} !important;
    font-weight: 750 !important;
    box-shadow: 0 0 0 1px rgba(54,179,126,0.2) !important;
  }}
  div[role="radiogroup"] label p {{
    margin: 0 !important;
    font-size: 0.88rem !important;
  }}
  /* Hide radio dots — look like tabs/pills */
  div[role="radiogroup"] label > div:first-child {{
    display: none !important;
  }}
  .stButton > button {{
    border-radius: 10px !important;
    font-weight: 600 !important;
  }}
  div[data-testid="stAlert"] {{ border-radius: 12px !important; }}
  footer {{ visibility: hidden; }}
</style>
"""
    try:
        st.html(css)  # type: ignore[attr-defined]
    except Exception:
        st.markdown(css, unsafe_allow_html=True)


def plotly_layout(**kwargs):
    """Shared dark Plotly layout defaults."""
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, system-ui, sans-serif", color=TEXT, size=12),
        margin=dict(l=12, r=12, t=36, b=12),
        legend=dict(
            orientation="h", y=1.08, bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color=TEXT_MUTED),
        ),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            zerolinecolor="rgba(255,255,255,0.08)",
            tickfont=dict(color=TEXT_MUTED, size=11),
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            zerolinecolor="rgba(255,255,255,0.08)",
            tickfont=dict(color=TEXT_MUTED, size=11),
        ),
        colorway=[ACCENT, BLUE, AMBER, RED, "#a78bfa", "#22d3ee"],
    )
    base.update(kwargs)
    return base


def section_header(title: str):
    """Section title with a green accent bar (inline styles only)."""
    safe = html_lib.escape(title)
    st.markdown(
        f"""
<div style="display:flex;align-items:center;gap:0.55rem;margin:0.4rem 0 0.75rem 0;">
  <div style="width:4px;height:1.15rem;border-radius:4px;
              background:linear-gradient(180deg,{ACCENT},#2d8f65);flex:none;"></div>
  <div style="font-size:1.05rem;font-weight:700;color:{TEXT};letter-spacing:-0.02em;">
    {safe}
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def brand_strip(subtitle: str = "AI analytics"):
    icon = _icon_data_uri()
    sub = html_lib.escape(subtitle)
    if icon:
        img = (
            f'<img src="{icon}" alt="logo" width="36" height="36" '
            f'style="border-radius:10px;display:block;" />'
        )
    else:
        img = (
            f'<div style="width:36px;height:36px;border-radius:10px;'
            f'background:{ACCENT};color:#04110a;font-weight:800;'
            f'display:flex;align-items:center;justify-content:center;'
            f'font-size:0.85rem;">AI</div>'
        )
    st.markdown(
        f"""
<div style="display:flex;align-items:center;gap:0.65rem;padding:0.2rem 0 0.7rem 0;">
  {img}
  <div>
    <div style="font-size:0.95rem;font-weight:700;color:{TEXT};line-height:1.15;">
      Stock Predictor
    </div>
    <div style="font-size:0.72rem;color:{TEXT_DIM};margin-top:0.1rem;">{sub}</div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def page_hero(title: str, subtitle: str, chips: list[str] | None = None):
    """Page title + subtitle using Streamlit natives (always readable)."""
    # Small spacer so the first heading never sits under the Streamlit toolbar
    st.markdown(
        "<div style='height:0.35rem;overflow:visible;'></div>",
        unsafe_allow_html=True,
    )
    st.markdown(f"### {html_lib.escape(title)}")
    st.caption(subtitle)
    if chips:
        parts = []
        for c in chips:
            safe = html_lib.escape(c)
            parts.append(
                f'<span style="display:inline-block;margin:0 0.35rem 0.35rem 0;'
                f'padding:0.28rem 0.7rem;border-radius:999px;font-size:0.78rem;'
                f'font-weight:600;background:{BG_SOFT};border:1px solid {BORDER};'
                f'color:{TEXT_MUTED};">{safe}</span>'
            )
        st.markdown("".join(parts), unsafe_allow_html=True)
    st.divider()


def footer_bar(
    text: str = "Educational tool only — not financial advice · © 2026 Soumoster Analytics",
):
    safe = html_lib.escape(text)
    st.markdown(
        f"""
<div style="position:fixed;left:0;bottom:0;width:100%;text-align:center;
            padding:0.55rem 1rem;background:rgba(11,15,20,0.94);
            color:{TEXT_DIM};font-size:0.78rem;z-index:999;
            border-top:1px solid {BORDER};">
  {safe}
</div>
        """,
        unsafe_allow_html=True,
    )


def signal_banner_html(signal: str, title: str, body: str) -> str:
    """Self-contained colored banner (inline styles only)."""
    if signal == "BUY":
        bg, border, fg = ACCENT_SOFT, "rgba(54,179,126,0.45)", ACCENT
    elif signal == "SELL":
        bg, border, fg = RED_SOFT, "rgba(240,113,120,0.45)", RED
    else:
        bg, border, fg = AMBER_SOFT, "rgba(230,180,80,0.45)", AMBER
    t = html_lib.escape(title)
    b = html_lib.escape(body)
    return f"""
<div style="border-radius:14px;padding:1rem 1.15rem;margin:0.15rem 0 0.5rem 0;
            background:{bg};border:1px solid {border};">
  <div style="font-size:1.15rem;font-weight:800;color:{fg};letter-spacing:-0.02em;">
    {t}
  </div>
  <div style="font-size:0.9rem;color:{TEXT_MUTED};margin-top:0.35rem;line-height:1.4;">
    {b}
  </div>
</div>
"""


def card_html(title: str, lines: list[str], accent: str = ACCENT) -> str:
    """Generic info card with full inline styles."""
    t = html_lib.escape(title)
    body = "".join(
        f'<div style="font-size:0.85rem;color:{TEXT_MUTED};margin-top:0.25rem;">'
        f"{html_lib.escape(line)}</div>"
        for line in lines
    )
    return f"""
<div style="border-radius:14px;padding:0.9rem 1rem;margin-bottom:0.5rem;
            background:{BG_CARD};border:1px solid {BORDER};
            border-left:3px solid {accent};">
  <div style="font-size:0.95rem;font-weight:700;color:{TEXT};">{t}</div>
  {body}
</div>
"""
