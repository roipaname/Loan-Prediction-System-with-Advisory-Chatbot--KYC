"""
styles/theme.py
Central CSS theme for LAPAS Streamlit frontend.
Colour palette derived from the LAPAS shield logo:
  - Deep black  #0E0D0B  (logo background)
  - Warm charcoal  #1A1714 / #221F1A  (card surfaces)
  - Champagne gold  #C4A87A / #D4BC96  (shield accent)
  - Cool silver  #8C9199  (shield lower half)
  - Cream  #F0EBE3  (primary text)
  - Warm grey  #A0978E  (secondary text)
  - Sage green  #5C8C6A  (approved / positive)
  - Muted rose  #8C5E62  (rejected / negative)
"""

import streamlit as st
import os as _os

# ── Colour constants (also used in Python chart code) ────────────────────────
GOLD       = "#C4A87A"
GOLD_LT    = "#D4BC96"
GOLD_DK    = "#A08858"
SILVER     = "#8C9199"
BG         = "#0E0D0B"
BG2        = "#141210"
CARD       = "#1A1714"
CARD2      = "#221F1A"
CARD3      = "#2A261F"
TEXT       = "#F0EBE3"
TEXT2      = "#A0978E"
TEXT3      = "#665E58"
SUCCESS    = "#5C8C6A"
SUCCESS_LT = "#7AB08A"
DANGER     = "#8C5E62"
DANGER_LT  = "#AA7A7E"
BORDER     = "#2C2822"
BORDER2    = "#3A342C"


_CSS = f"""
<style>
/* ── Google Fonts ─────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Root ─────────────────────────────────────────────────────────── */
html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif !important;
}}

/* ── App background ───────────────────────────────────────────────── */
.stApp {{
    background-color: {BG};
    color: {TEXT};
}}

.main .block-container {{
    padding-top: 1.2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}}

/* ── Sidebar ──────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #111008 0%, {CARD} 60%, #13110E 100%) !important;
    border-right: 1px solid {BORDER} !important;
}}
[data-testid="stSidebar"] > div:first-child {{
    padding-top: 1rem;
}}
[data-testid="stSidebarContent"] {{
    background: transparent !important;
}}

/* Sidebar nav links */
[data-testid="stSidebarNav"] a {{
    color: {TEXT2} !important;
    font-size: 0.92rem;
    border-radius: 8px;
    margin: 2px 0;
    padding: 6px 12px;
    transition: all 0.2s;
}}
[data-testid="stSidebarNav"] a:hover {{
    background: rgba(196,168,122,0.10) !important;
    color: {GOLD_LT} !important;
}}
[data-testid="stSidebarNav"] a[aria-selected="true"] {{
    background: linear-gradient(90deg, rgba(196,168,122,0.18), rgba(196,168,122,0.04)) !important;
    color: {GOLD_LT} !important;
    border-left: 3px solid {GOLD};
}}

/* ── Metric cards ─────────────────────────────────────────────────── */
[data-testid="metric-container"] {{
    background: linear-gradient(135deg, {CARD} 0%, {CARD2} 100%);
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 1.1rem 1.3rem !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.35);
    transition: box-shadow 0.3s;
}}
[data-testid="metric-container"]:hover {{
    box-shadow: 0 6px 32px rgba(196,168,122,0.12);
}}
[data-testid="metric-container"] label {{
    color: {TEXT2} !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    color: {TEXT} !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
}}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {{
    font-size: 0.78rem !important;
}}

/* ── Buttons ──────────────────────────────────────────────────────── */
.stButton > button {{
    background: linear-gradient(135deg, {GOLD} 0%, {GOLD_DK} 100%);
    color: #0A0906;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    font-size: 0.88rem;
    letter-spacing: 0.02em;
    padding: 0.55rem 1.4rem;
    transition: all 0.25s ease;
    box-shadow: 0 4px 16px rgba(196,168,122,0.22);
}}
.stButton > button:hover {{
    background: linear-gradient(135deg, {GOLD_LT} 0%, {GOLD} 100%);
    box-shadow: 0 6px 24px rgba(196,168,122,0.38);
    transform: translateY(-1px);
    color: #0A0906;
}}
.stButton > button:active {{
    transform: translateY(0px);
}}

/* Secondary button override via class */
.btn-secondary > button {{
    background: linear-gradient(135deg, {CARD2} 0%, {CARD3} 100%) !important;
    color: {TEXT2} !important;
    border: 1px solid {BORDER2} !important;
    box-shadow: none !important;
}}
.btn-secondary > button:hover {{
    color: {GOLD_LT} !important;
    border-color: {GOLD_DK} !important;
    background: linear-gradient(135deg, {CARD3} 0%, #302820 100%) !important;
}}

/* ── Inputs ───────────────────────────────────────────────────────── */
.stTextInput input,
.stNumberInput input,
.stTextArea textarea {{
    background-color: {CARD2} !important;
    color: {TEXT} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
    font-size: 0.9rem !important;
}}
.stTextInput input:focus,
.stNumberInput input:focus,
.stTextArea textarea:focus {{
    border-color: {GOLD_DK} !important;
    box-shadow: 0 0 0 2px rgba(196,168,122,0.15) !important;
}}

/* Selectbox */
[data-baseweb="select"] > div:first-child {{
    background-color: {CARD2} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
    color: {TEXT} !important;
}}
[data-baseweb="popover"] ul {{
    background-color: {CARD2} !important;
    border: 1px solid {BORDER2} !important;
}}
[data-baseweb="option"]:hover {{
    background-color: {CARD3} !important;
}}

/* Multiselect tags */
[data-baseweb="tag"] {{
    background: rgba(196,168,122,0.18) !important;
    color: {GOLD_LT} !important;
}}

/* Slider */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {{
    background-color: {GOLD} !important;
    border-color: {GOLD_LT} !important;
}}
[data-testid="stSlider"] div[data-testid="stThumbValue"] {{
    color: {GOLD_LT} !important;
}}

/* ── Tabs ─────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    background-color: {CARD};
    border-radius: 12px;
    border: 1px solid {BORDER};
    padding: 4px;
    gap: 2px;
}}
.stTabs [data-baseweb="tab"] {{
    background-color: transparent;
    color: {TEXT2};
    border-radius: 9px;
    font-size: 0.88rem;
    font-weight: 500;
    padding: 0.5rem 1.1rem;
    transition: all 0.2s;
}}
.stTabs [data-baseweb="tab"]:hover {{
    color: {GOLD_LT};
    background: rgba(196,168,122,0.08);
}}
.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, {GOLD} 0%, {GOLD_DK} 100%) !important;
    color: #0A0906 !important;
    font-weight: 600 !important;
}}
.stTabs [data-baseweb="tab-border"] {{
    display: none !important;
}}

/* ── Expander ─────────────────────────────────────────────────────── */
[data-testid="stExpander"] {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 12px;
}}
[data-testid="stExpander"] summary {{
    color: {TEXT2};
    font-size: 0.9rem;
}}

/* ── Dataframe ────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER};
    border-radius: 12px;
    overflow: hidden;
}}

/* ── Divider ──────────────────────────────────────────────────────── */
hr {{
    border: none;
    border-top: 1px solid {BORDER};
    margin: 1.2rem 0;
}}

/* ── Scrollbar ────────────────────────────────────────────────────── */
::-webkit-scrollbar       {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {BG2}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER2}; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: {GOLD_DK}; }}

/* ── Custom shared HTML components ───────────────────────────────────*/

/* Section heading */
.section-header {{
    font-size: 1.05rem;
    font-weight: 700;
    color: {TEXT};
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid {BORDER};
}}

/* Approved / Rejected badges */
.badge-approved {{
    background: rgba(92,140,106,0.18);
    color: {SUCCESS_LT};
    border: 1px solid rgba(92,140,106,0.35);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    display: inline-block;
}}
.badge-rejected {{
    background: rgba(140,94,98,0.18);
    color: {DANGER_LT};
    border: 1px solid rgba(140,94,98,0.35);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    display: inline-block;
}}

/* Risk tier pill */
.risk-low    {{ color: {SUCCESS_LT}; font-weight: 600; }}
.risk-medium {{ color: {GOLD_LT};   font-weight: 600; }}
.risk-high   {{ color: {DANGER_LT}; font-weight: 600; }}

/* Customer card */
.cust-card {{
    background: linear-gradient(145deg, {CARD} 0%, {CARD2} 100%);
    border: 1px solid {BORDER};
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(0,0,0,0.30);
    transition: box-shadow 0.25s, transform 0.2s;
    margin-bottom: 1rem;
}}
.cust-card:hover {{
    box-shadow: 0 8px 32px rgba(196,168,122,0.14);
    transform: translateY(-2px);
}}
.cust-card-header-approved {{
    background: linear-gradient(135deg, #1E3228 0%, #162A20 100%);
    border-bottom: 1px solid rgba(92,140,106,0.25);
    padding: 0.85rem 1.1rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}}
.cust-card-header-rejected {{
    background: linear-gradient(135deg, #2A1A1C 0%, #221518 100%);
    border-bottom: 1px solid rgba(140,94,98,0.25);
    padding: 0.85rem 1.1rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}}
.cust-card-id {{
    font-size: 0.82rem;
    font-weight: 700;
    color: {TEXT};
    font-family: 'Courier New', monospace;
    letter-spacing: 0.08em;
}}
.cust-card-body {{
    padding: 1rem 1.1rem 0.8rem;
}}
.cust-metric-row {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.6rem;
    margin-bottom: 0.7rem;
}}
.cust-metric {{
    background: rgba(255,255,255,0.025);
    border-radius: 8px;
    padding: 0.5rem 0.7rem;
}}
.cust-metric-label {{
    font-size: 0.67rem;
    color: {TEXT3};
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 2px;
}}
.cust-metric-value {{
    font-size: 0.92rem;
    font-weight: 600;
    color: {TEXT};
}}

/* Hero section */
.hero-container {{
    background: linear-gradient(135deg, {CARD} 0%, {CARD2} 50%, #1E1A12 100%);
    border: 1px solid {BORDER};
    border-radius: 20px;
    padding: 2.5rem 2.5rem 2rem;
    position: relative;
    overflow: hidden;
    margin-bottom: 1.5rem;
}}
.hero-container::before {{
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(196,168,122,0.10), transparent 70%);
    border-radius: 50%;
}}
.hero-title {{
    font-size: 2.4rem;
    font-weight: 800;
    color: {TEXT};
    letter-spacing: -0.02em;
    margin: 0 0 0.3rem 0;
    line-height: 1.15;
}}
.hero-title span {{
    background: linear-gradient(90deg, {GOLD} 0%, {GOLD_LT} 60%, {SILVER} 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}
.hero-subtitle {{
    font-size: 1.0rem;
    color: {TEXT2};
    margin: 0;
    font-weight: 400;
}}

/* Info panel */
.info-panel {{
    background: linear-gradient(135deg, {CARD} 0%, {CARD2} 100%);
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
}}
.info-panel-title {{
    font-size: 0.78rem;
    color: {TEXT2};
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.9rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}}

/* Gold divider */
.gold-divider {{
    height: 2px;
    background: linear-gradient(90deg, transparent, {GOLD_DK}, transparent);
    border: none;
    margin: 1.5rem 0;
}}

/* Explanation text area */
.explanation-box {{
    background: linear-gradient(145deg, {CARD} 0%, {CARD2} 100%);
    border: 1px solid {BORDER};
    border-left: 3px solid {GOLD};
    border-radius: 12px;
    padding: 1.5rem;
    color: {TEXT};
    line-height: 1.7;
    font-size: 0.9rem;
}}

/* Step card */
.step-card {{
    background: linear-gradient(135deg, {CARD2} 0%, {CARD3} 100%);
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.7rem;
    display: flex;
    gap: 1rem;
    align-items: flex-start;
}}
.step-num {{
    background: linear-gradient(135deg, {GOLD} 0%, {GOLD_DK} 100%);
    color: #0A0906;
    font-weight: 800;
    font-size: 0.8rem;
    width: 26px;
    height: 26px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}}

/* Form section header */
.form-section {{
    background: linear-gradient(90deg, rgba(196,168,122,0.08), transparent);
    border-left: 3px solid {GOLD_DK};
    border-radius: 0 8px 8px 0;
    padding: 0.5rem 1rem;
    margin: 1.2rem 0 0.8rem 0;
    font-size: 0.85rem;
    font-weight: 600;
    color: {GOLD_LT};
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}

/* Success submit card */
.submit-success {{
    background: linear-gradient(135deg, #1A2B20 0%, #14221A 100%);
    border: 1px solid rgba(92,140,106,0.35);
    border-radius: 16px;
    padding: 1.8rem;
    text-align: center;
}}

/* Stat pill */
.stat-pill {{
    background: rgba(196,168,122,0.10);
    border: 1px solid rgba(196,168,122,0.20);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    color: {GOLD_LT};
    display: inline-block;
}}

/* Dashboard chart card wrapper */
.chart-card {{
    background: linear-gradient(145deg, {CARD} 0%, {CARD2} 100%);
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 1.2rem 1.2rem 0.8rem;
    margin-bottom: 0;
}}
.chart-title {{
    font-size: 0.82rem;
    font-weight: 600;
    color: {TEXT2};
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 0.5rem;
}}

/* Activity feed row */
.activity-row {{
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.65rem 0;
    border-bottom: 1px solid {BORDER};
}}
.activity-dot-approved {{ width:9px; height:9px; border-radius:50%; background:{SUCCESS}; flex-shrink:0; }}
.activity-dot-rejected {{ width:9px; height:9px; border-radius:50%; background:{DANGER};  flex-shrink:0; }}
</style>
"""


def inject():
    """Inject the full LAPAS CSS theme into the current Streamlit page."""
    st.markdown(_CSS, unsafe_allow_html=True)


def sidebar_logo():
    """Render the LAPAS logo + wordmark in the sidebar."""
    try:
        st.sidebar.image("assets/logo.png", width=90)
    except Exception:
        pass
    st.sidebar.markdown(
        f"""<div style="text-align:center; margin-top:-8px; margin-bottom:12px;">
        <span style="font-size:1.1rem;font-weight:800;
        background:linear-gradient(90deg,{GOLD},{GOLD_LT});
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        letter-spacing:0.12em;">LAPAS</span>
        <div style="font-size:0.65rem;color:{TEXT3};letter-spacing:0.06em;margin-top:1px;">
        Loan Approval Prediction & Advisory System</div></div>""",
        unsafe_allow_html=True
    )
    st.sidebar.markdown(f'<hr style="border-color:{BORDER};margin:4px 0 10px;">', unsafe_allow_html=True)


# ── Plotly chart base layout (shared across all pages) ────────────────────────
CHART_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(22,19,16,0.55)',
    font=dict(color=TEXT2, family='Inter, sans-serif', size=11),
    title_font=dict(color=TEXT, size=14, family='Inter, sans-serif'),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT2, size=10)),
    xaxis=dict(gridcolor=BORDER, tickcolor=TEXT3, linecolor=BORDER, color=TEXT2, showgrid=True),
    yaxis=dict(gridcolor=BORDER, tickcolor=TEXT3, linecolor=BORDER, color=TEXT2, showgrid=True),
    margin=dict(l=10, r=10, t=40, b=10),
    colorway=[GOLD, SILVER, SUCCESS, DANGER, '#7A8FBB', '#B87A9A', '#7ABB9A'],
    hoverlabel=dict(bgcolor=CARD2, font_color=TEXT, bordercolor=BORDER),
)


def apply_chart_layout(fig, title: str = '', height: int = 320):
    """Apply the LAPAS dark theme to any Plotly figure."""
    import plotly.graph_objects as go
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text=title, font=dict(color=TEXT, size=13), x=0.01, xanchor='left'),
        height=height,
    )
    return fig
