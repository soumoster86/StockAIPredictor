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
  /* ---- Typography: Inter with tabular numerals for the finance look.
     Falls back to system fonts when offline. ---- */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
  html, body, [class*="css"], .stApp, .stMarkdown, button, input, textarea {{
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
  }}
  h1, h2, h3, h4 {{ letter-spacing: -0.02em !important; }}
  ::selection {{ background: rgba(54,179,126,0.35); }}

  /* ---- App canvas: near-black with a faint accent glow up top ---- */
  .stApp {{
    background:
      radial-gradient(1100px 480px at 12% -8%, rgba(54,179,126,0.07), transparent 60%),
      radial-gradient(900px 420px at 92% -12%, rgba(108,182,255,0.05), transparent 55%),
      {BG};
  }}
  [data-testid="stHeader"] {{ background: transparent !important; }}
  [data-testid="stAppDeployButton"], .stDeployButton {{ display: none !important; }}

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

  /* ---- Sidebar: slightly darker column with a hairline edge ---- */
  [data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0d1219 0%, {BG} 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
  }}

  /* ---- Metric cards ---- */
  [data-testid="stMetric"] {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 0.75rem 0.9rem;
    overflow: visible !important;
    min-width: 0;
    transition: border-color .15s ease, box-shadow .15s ease, transform .15s ease;
  }}
  [data-testid="stMetric"]:hover {{
    border-color: rgba(54,179,126,0.35);
    box-shadow: 0 6px 18px rgba(0,0,0,0.25);
    transform: translateY(-1px);
  }}
  [data-testid="stMetricLabel"] {{
    color: {TEXT_MUTED} !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    white-space: nowrap !important;
  }}
  [data-testid="stMetricValue"] {{
    overflow: visible !important;
    white-space: nowrap !important;
    font-size: 1.35rem !important;
    line-height: 1.25 !important;
    font-variant-numeric: tabular-nums;
  }}
  [data-testid="stMetricDelta"] {{
    overflow: visible !important;
    font-variant-numeric: tabular-nums;
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
    transition: border-color .15s ease, color .15s ease,
                background .15s ease, transform .15s ease;
  }}
  div[role="radiogroup"] label:hover {{
    border-color: rgba(54,179,126,0.45) !important;
    color: {TEXT} !important;
    transform: translateY(-1px);
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

  /* ---- Buttons ---- */
  .stButton > button, [data-testid="stFormSubmitButton"] > button {{
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: transform .14s ease, box-shadow .14s ease,
                border-color .14s ease, background .14s ease;
  }}
  .stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {{
    transform: translateY(-1px);
  }}
  /* kind is "primary" on st.button, "primaryFormSubmit" on form submit —
     prefix-match catches both */
  .stButton > button[kind^="primary"],
  [data-testid="stFormSubmitButton"] > button[kind^="primary"] {{
    background: linear-gradient(135deg, {ACCENT} 0%, #2d8f65 100%) !important;
    border: none !important;
    color: #04110a !important;
  }}
  .stButton > button[kind^="primary"]:hover,
  [data-testid="stFormSubmitButton"] > button[kind^="primary"]:hover {{
    box-shadow: 0 4px 16px rgba(54,179,126,0.35);
  }}

  /* ---- Inputs, selects, uploader, forms, expanders ---- */
  .stTextInput input, .stNumberInput input, .stTextArea textarea {{
    border-radius: 10px !important;
  }}
  [data-baseweb="select"] > div {{ border-radius: 10px !important; }}
  [data-testid="stFileUploaderDropzone"] {{
    background: {BG_SOFT} !important;
    border: 1px dashed rgba(255,255,255,0.18) !important;
    border-radius: 12px !important;
  }}
  [data-testid="stForm"] {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 1.1rem 1.2rem;
  }}
  [data-testid="stExpander"] details {{
    border: 1px solid {BORDER} !important;
    border-radius: 12px !important;
    background: rgba(255,255,255,0.02);
  }}
  [data-testid="stExpander"] summary {{ font-weight: 600; }}

  /* ---- Tables / alerts / dividers ---- */
  [data-testid="stDataFrame"] {{
    border: 1px solid {BORDER};
    border-radius: 12px;
    overflow: hidden;
  }}
  div[data-testid="stAlert"] {{ border-radius: 12px !important; }}
  hr {{ border-color: rgba(255,255,255,0.08) !important; }}

  /* ---- Custom HTML card hover (progressive enhancement on inline styles) */
  .sp-pick, .sp-card {{
    transition: transform .16s ease, box-shadow .16s ease, border-color .16s ease;
  }}
  .sp-pick:hover {{
    transform: translateY(-2px);
    border-color: rgba(54,179,126,0.4) !important;
    box-shadow: 0 10px 26px rgba(0,0,0,0.32) !important;
  }}
  .sp-card:hover {{ border-color: rgba(255,255,255,0.18) !important; }}

  /* ---- Thin dark scrollbars ---- */
  ::-webkit-scrollbar {{ width: 10px; height: 10px; }}
  ::-webkit-scrollbar-track {{ background: transparent; }}
  ::-webkit-scrollbar-thumb {{
    background: rgba(255,255,255,0.14);
    border-radius: 8px;
    border: 2px solid {BG};
  }}
  ::-webkit-scrollbar-thumb:hover {{ background: rgba(255,255,255,0.24); }}

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
            f'<div style="padding:2px;border-radius:12px;line-height:0;'
            f'background:linear-gradient(135deg,{ACCENT} 0%,#2d8f65 55%,{BLUE} 100%);">'
            f'<img src="{icon}" alt="logo" width="36" height="36" '
            f'style="border-radius:10px;display:block;" /></div>'
        )
    else:
        img = (
            f'<div style="width:36px;height:36px;border-radius:10px;'
            f'background:linear-gradient(135deg,{ACCENT},#2d8f65);color:#04110a;'
            f'font-weight:800;display:flex;align-items:center;justify-content:center;'
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
    """Gradient hero card: page title + subtitle + status chips."""
    # Small spacer so the hero never sits under the Streamlit toolbar
    st.markdown(
        "<div style='height:0.35rem;overflow:visible;'></div>",
        unsafe_allow_html=True,
    )
    chips_html = ""
    if chips:
        parts = []
        for c in chips:
            safe = html_lib.escape(c)
            parts.append(
                f'<span style="display:inline-block;margin:0 0.35rem 0.1rem 0;'
                f'padding:0.28rem 0.7rem;border-radius:999px;font-size:0.78rem;'
                f'font-weight:600;background:rgba(255,255,255,0.05);'
                f'border:1px solid {BORDER};color:{TEXT_MUTED};">{safe}</span>'
            )
        chips_html = (
            f'<div style="margin-top:0.7rem;">{"".join(parts)}</div>'
        )
    st.markdown(
        f"""
<div style="border-radius:18px;padding:1.15rem 1.35rem;margin:0 0 1rem 0;
            background:linear-gradient(135deg,rgba(54,179,126,0.13) 0%,
                        rgba(22,29,39,0.92) 45%,rgba(108,182,255,0.08) 100%),{BG_CARD};
            border:1px solid {BORDER};overflow:hidden;">
  <div style="font-size:1.45rem;font-weight:800;color:{TEXT};
              letter-spacing:-0.03em;line-height:1.2;">{html_lib.escape(title)}</div>
  <div style="font-size:0.92rem;color:{TEXT_MUTED};margin-top:0.3rem;
              line-height:1.45;">{html_lib.escape(subtitle)}</div>
  {chips_html}
</div>
        """,
        unsafe_allow_html=True,
    )


def footer_bar(
    text: str = "Educational tool only — not financial advice · © 2026 Soumoster Analytics",
):
    safe = html_lib.escape(text)
    st.markdown(
        f"""
<div style="position:fixed;left:0;bottom:0;width:100%;text-align:center;
            padding:0.55rem 1rem;background:rgba(11,15,20,0.82);
            backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);
            color:{TEXT_DIM};font-size:0.78rem;z-index:999;
            border-top:1px solid {BORDER};">
  {safe}
</div>
        """,
        unsafe_allow_html=True,
    )


def signal_banner_html(signal: str, title: str, body: str) -> str:
    """Self-contained colored banner with an icon bubble (inline styles only)."""
    if signal == "BUY":
        bg, border, fg = ACCENT_SOFT, "rgba(54,179,126,0.45)", ACCENT
        glow, icon = "rgba(54,179,126,0.12)", "▲"
    elif signal == "SELL":
        bg, border, fg = RED_SOFT, "rgba(240,113,120,0.45)", RED
        glow, icon = "rgba(240,113,120,0.12)", "▼"
    else:
        bg, border, fg = AMBER_SOFT, "rgba(230,180,80,0.45)", AMBER
        glow, icon = "rgba(230,180,80,0.10)", "◆"
    t = html_lib.escape(title)
    b = html_lib.escape(body)
    return f"""
<div style="display:flex;gap:0.85rem;align-items:flex-start;border-radius:16px;
            padding:1rem 1.15rem;margin:0.15rem 0 0.5rem 0;
            background:linear-gradient(135deg,{bg} 0%,rgba(22,29,39,0.85) 75%);
            border:1px solid {border};box-shadow:0 0 26px {glow};">
  <div style="width:2.5rem;height:2.5rem;border-radius:12px;flex:none;
              background:{bg};border:1px solid {border};color:{fg};
              display:flex;align-items:center;justify-content:center;
              font-size:1.05rem;font-weight:800;">{icon}</div>
  <div>
    <div style="font-size:1.18rem;font-weight:800;color:{fg};letter-spacing:-0.02em;
                line-height:1.25;">{t}</div>
    <div style="font-size:0.9rem;color:{TEXT_MUTED};margin-top:0.3rem;line-height:1.4;">
      {b}
    </div>
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
<div class="sp-card" style="border-radius:14px;padding:0.9rem 1rem;margin-bottom:0.5rem;
            background:{BG_CARD};border:1px solid {BORDER};
            border-left:3px solid {accent};
            box-shadow:0 3px 10px rgba(0,0,0,0.18);">
  <div style="font-size:0.95rem;font-weight:700;color:{TEXT};">{t}</div>
  {body}
</div>
"""


def pick_card_html(
    rank: int,
    symbol: str,
    name: str,
    prob_pct: float,
    score: float,
    price_s: str,
    day_s: str,
    risk_s: str,
    rr_s: str,
) -> str:
    """Rich top-pick card for the screener shortlist (inline styles only)."""
    sym = html_lib.escape(str(symbol))
    nm = html_lib.escape(str(name))
    # Soft score color ramp
    if score >= 70:
        score_bg, score_fg = "rgba(54,179,126,0.22)", ACCENT
    elif score >= 55:
        score_bg, score_fg = "rgba(230,180,80,0.20)", AMBER
    else:
        score_bg, score_fg = "rgba(108,182,255,0.16)", BLUE

    def cell(label: str, value: str) -> str:
        return (
            f'<div style="background:rgba(255,255,255,0.03);border:1px solid {BORDER};'
            f'border-radius:10px;padding:0.45rem 0.5rem;">'
            f'<div style="font-size:0.65rem;letter-spacing:0.06em;text-transform:uppercase;'
            f'color:{TEXT_DIM};">{html_lib.escape(label)}</div>'
            f'<div style="font-size:0.95rem;font-weight:700;color:{TEXT};margin-top:0.15rem;'
            f'font-variant-numeric:tabular-nums;">{html_lib.escape(value)}</div>'
            f"</div>"
        )

    return f"""
<div class="sp-pick" style="border-radius:16px;padding:1rem 1.05rem 0.95rem 1.05rem;margin-bottom:0.65rem;
            background:linear-gradient(160deg,{BG_CARD} 0%,{BG_SOFT} 100%);
            border:1px solid {BORDER};
            box-shadow:0 6px 18px rgba(0,0,0,0.22);">
  <div style="display:flex;align-items:center;justify-content:space-between;gap:0.5rem;
              margin-bottom:0.55rem;">
    <span style="display:inline-flex;align-items:center;justify-content:center;
                 min-width:2rem;height:2rem;border-radius:999px;font-weight:800;
                 font-size:0.85rem;background:{ACCENT_SOFT};color:{ACCENT};
                 border:1px solid rgba(54,179,126,0.35);">#{int(rank)}</span>
    <span style="display:inline-flex;align-items:center;gap:0.35rem;
                 padding:0.28rem 0.7rem;border-radius:999px;font-size:0.78rem;
                 font-weight:700;background:rgba(54,179,126,0.16);color:{ACCENT};
                 border:1px solid rgba(54,179,126,0.35);">BUY · {prob_pct:.0f}%</span>
    <span style="display:inline-flex;align-items:center;padding:0.28rem 0.65rem;
                 border-radius:999px;font-size:0.78rem;font-weight:800;
                 background:{score_bg};color:{score_fg};
                 border:1px solid rgba(255,255,255,0.08);
                 font-variant-numeric:tabular-nums;">Score {score:.0f}</span>
  </div>
  <div style="font-size:1.15rem;font-weight:800;color:{TEXT};letter-spacing:-0.02em;
              line-height:1.2;">{sym}</div>
  <div style="font-size:0.82rem;color:{TEXT_MUTED};margin-top:0.2rem;
              white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{nm}</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.4rem;margin-top:0.75rem;">
    {cell("Price", price_s)}
    {cell("Day", day_s)}
    {cell("Risk", f"{risk_s}/10")}
    {cell("R:R", rr_s)}
  </div>
</div>
"""
