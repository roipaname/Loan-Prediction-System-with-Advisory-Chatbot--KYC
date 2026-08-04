"""
styles/theme.py
Central CSS theme for LAPAS Streamlit frontend.

Colour palette derived from the LAPAS logo:
  - Warm charcoal  #1B1915 / #221F1A  (card surfaces)
  - Bronze gold    #C9A96B → #8E6B35  (logo A-sweep gradient)
  - Silver-grey    #8C9199            (secondary elements)
  - Cream          #EDE3D5            (logo background – accent)
  - Off-white      #F0EBE3            (primary text)
  - Sage green     #5C8C6A            (approved)
  - Muted rose     #8C5E62            (rejected)

Typography:
  - Playfair Display – display headings (mirrors logo serif)
  - Inter           – UI body copy
"""

import streamlit as st
import os as _os

# ── Absolute path to logo (works from any page) ───────────────────────────────
_STYLES_DIR = _os.path.dirname(_os.path.abspath(__file__))
_LOGO_PATH  = _os.path.join(_STYLES_DIR, '..', 'assets', 'logo.png')


def get_logo_image():
    """Return a PIL Image of the LAPAS logo (for page_icon)."""
    try:
        from PIL import Image
        return Image.open(_LOGO_PATH)
    except Exception:
        return None


# ── Colour constants (also imported by page scripts for Plotly charts) ────────
GOLD       = "#C9A96B"   # bronze-gold (logo midpoint)
GOLD_LT    = "#D9BC82"   # lighter highlight gold
GOLD_DK    = "#8E6B35"   # dark bronze shadow
SILVER     = "#8C9199"
CREAM      = "#EDE3D5"   # logo background cream – used as accent
BG         = "#0F0E0B"   # very dark warm charcoal
BG2        = "#141210"
CARD       = "#1B1815"
CARD2      = "#231F1A"
CARD3      = "#2B2620"
TEXT       = "#F0EBE3"   # primary text – warm off-white
TEXT2      = "#A09588"   # secondary text
TEXT3      = "#665E58"   # tertiary / disabled
SUCCESS    = "#5C8C6A"
SUCCESS_LT = "#7AB08A"
DANGER     = "#8C5E62"
DANGER_LT  = "#AA7A7E"
BORDER     = "#2E2A24"
BORDER2    = "#3C362E"


_CSS = f"""
<style>
/* ── Google Fonts: Playfair Display (headings) + Inter (body) ────────── */
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Root ─────────────────────────────────────────────────────────────── */
html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif !important;
}}

/* ── App background ───────────────────────────────────────────────────── */
.stApp {{
    background-color: {BG};
    color: {TEXT};
}}
.main .block-container {{
    padding-top: 1.25rem;
    padding-bottom: 3rem;
    max-width: 1420px;
}}

/* ── Sidebar ──────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0E0C09 0%, {CARD} 55%, #131109 100%) !important;
    border-right: 1px solid {BORDER} !important;
}}
[data-testid="stSidebar"] > div:first-child {{
    padding-top: 0.5rem;
}}
[data-testid="stSidebarContent"] {{
    background: transparent !important;
}}

/* Sidebar nav links */
[data-testid="stSidebarNav"] a {{
    color: {TEXT2} !important;
    font-size: 0.9rem;
    font-weight: 400;
    border-radius: 8px;
    margin: 2px 0;
    padding: 7px 14px;
    transition: all 0.2s;
    letter-spacing: 0.01em;
}}
[data-testid="stSidebarNav"] a:hover {{
    background: rgba(201,169,107,0.10) !important;
    color: {GOLD_LT} !important;
}}
[data-testid="stSidebarNav"] a[aria-selected="true"] {{
    background: linear-gradient(90deg, rgba(201,169,107,0.16), rgba(201,169,107,0.03)) !important;
    color: {GOLD_LT} !important;
    border-left: 3px solid {GOLD};
    font-weight: 500;
}}

/* ── Metric cards ─────────────────────────────────────────────────────── */
[data-testid="metric-container"] {{
    background: linear-gradient(145deg, {CARD} 0%, {CARD2} 100%);
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 1.1rem 1.3rem !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.32);
    transition: box-shadow 0.3s, transform 0.2s;
    position: relative;
    overflow: hidden;
}}
[data-testid="metric-container"]::after {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, {GOLD_DK}, transparent);
    opacity: 0;
    transition: opacity 0.3s;
}}
[data-testid="metric-container"]:hover {{
    box-shadow: 0 6px 28px rgba(201,169,107,0.13);
    transform: translateY(-1px);
}}
[data-testid="metric-container"]:hover::after {{
    opacity: 1;
}}
[data-testid="metric-container"] label {{
    color: {TEXT2} !important;
    font-size: 0.73rem !important;
    letter-spacing: 0.10em !important;
    text-transform: uppercase !important;
    font-weight: 500 !important;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    color: {TEXT} !important;
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em !important;
    font-family: 'Inter', sans-serif !important;
}}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {{
    font-size: 0.76rem !important;
}}

/* ── Buttons ──────────────────────────────────────────────────────────── */
.stButton > button {{
    background: linear-gradient(135deg, {GOLD} 0%, {GOLD_DK} 100%);
    color: #100E0A;
    border: none;
    border-radius: 9px;
    font-weight: 600;
    font-size: 0.87rem;
    letter-spacing: 0.025em;
    padding: 0.55rem 1.5rem;
    transition: all 0.22s ease;
    box-shadow: 0 3px 14px rgba(201,169,107,0.25);
    font-family: 'Inter', sans-serif !important;
}}
.stButton > button:hover {{
    background: linear-gradient(135deg, {GOLD_LT} 0%, {GOLD} 100%);
    box-shadow: 0 6px 22px rgba(201,169,107,0.40);
    transform: translateY(-1px);
    color: #100E0A;
}}
.stButton > button:active {{
    transform: translateY(0);
    box-shadow: 0 2px 8px rgba(201,169,107,0.20);
}}

/* Secondary button */
.btn-secondary > button {{
    background: linear-gradient(135deg, {CARD2} 0%, {CARD3} 100%) !important;
    color: {TEXT2} !important;
    border: 1px solid {BORDER2} !important;
    box-shadow: none !important;
}}
.btn-secondary > button:hover {{
    color: {GOLD_LT} !important;
    border-color: {GOLD_DK} !important;
    background: linear-gradient(135deg, {CARD3} 0%, #302A22 100%) !important;
}}

/* ── Inputs ───────────────────────────────────────────────────────────── */
.stTextInput input,
.stNumberInput input,
.stTextArea textarea {{
    background-color: {CARD2} !important;
    color: {TEXT} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 9px !important;
    font-size: 0.88rem !important;
}}
.stTextInput input:focus,
.stNumberInput input:focus,
.stTextArea textarea:focus {{
    border-color: {GOLD_DK} !important;
    box-shadow: 0 0 0 2px rgba(201,169,107,0.14) !important;
}}

/* Selectbox */
[data-baseweb="select"] > div:first-child {{
    background-color: {CARD2} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 9px !important;
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
    background: rgba(201,169,107,0.16) !important;
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

/* ── Tabs ─────────────────────────────────────────────────────────────── */
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
    font-size: 0.87rem;
    font-weight: 500;
    padding: 0.5rem 1.2rem;
    transition: all 0.2s;
    letter-spacing: 0.01em;
}}
.stTabs [data-baseweb="tab"]:hover {{
    color: {GOLD_LT};
    background: rgba(201,169,107,0.08);
}}
.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, {GOLD} 0%, {GOLD_DK} 100%) !important;
    color: #0E0C09 !important;
    font-weight: 600 !important;
}}
.stTabs [data-baseweb="tab-border"] {{
    display: none !important;
}}

/* ── Expander ─────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 12px;
}}
[data-testid="stExpander"] summary {{
    color: {TEXT2};
    font-size: 0.9rem;
}}

/* ── Dataframe ────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER};
    border-radius: 12px;
    overflow: hidden;
}}

/* ── Divider ──────────────────────────────────────────────────────────── */
hr {{
    border: none;
    border-top: 1px solid {BORDER};
    margin: 1.2rem 0;
}}

/* ── Scrollbar ────────────────────────────────────────────────────────── */
::-webkit-scrollbar       {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {BG2}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER2}; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: {GOLD_DK}; }}

/* ── Shared HTML components ───────────────────────────────────────────── */

/* Section heading — uses small caps feel */
.section-header {{
    font-size: 0.72rem;
    font-weight: 600;
    color: {TEXT2};
    letter-spacing: 0.13em;
    text-transform: uppercase;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid {BORDER};
    display: flex;
    align-items: center;
    gap: 0.45rem;
}}

/* Page title — Playfair Display */
.page-title {{
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 1.7rem;
    font-weight: 700;
    color: {TEXT};
    letter-spacing: -0.01em;
    line-height: 1.2;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.15rem;
}}
.page-subtitle {{
    font-size: 0.86rem;
    color: {TEXT2};
    margin-top: 3px;
    font-weight: 400;
    letter-spacing: 0.01em;
}}

/* Approved / Rejected badges */
.badge-approved {{
    background: rgba(92,140,106,0.16);
    color: {SUCCESS_LT};
    border: 1px solid rgba(92,140,106,0.32);
    border-radius: 20px;
    padding: 3px 13px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    display: inline-block;
}}
.badge-rejected {{
    background: rgba(140,94,98,0.16);
    color: {DANGER_LT};
    border: 1px solid rgba(140,94,98,0.32);
    border-radius: 20px;
    padding: 3px 13px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    display: inline-block;
}}

/* Risk tier */
.risk-low    {{ color: {SUCCESS_LT}; font-weight: 600; }}
.risk-medium {{ color: {GOLD_LT};   font-weight: 600; }}
.risk-high   {{ color: {DANGER_LT}; font-weight: 600; }}

/* Customer card */
.cust-card {{
    background: linear-gradient(145deg, {CARD} 0%, {CARD2} 100%);
    border: 1px solid {BORDER};
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(0,0,0,0.28);
    transition: box-shadow 0.25s, transform 0.2s;
    margin-bottom: 1rem;
}}
.cust-card:hover {{
    box-shadow: 0 8px 30px rgba(201,169,107,0.15);
    transform: translateY(-2px);
}}
.cust-card-header-approved {{
    background: linear-gradient(135deg, #1D3326 0%, #152820 100%);
    border-bottom: 1px solid rgba(92,140,106,0.22);
    padding: 0.85rem 1.1rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}}
.cust-card-header-rejected {{
    background: linear-gradient(135deg, #2A1A1C 0%, #211416 100%);
    border-bottom: 1px solid rgba(140,94,98,0.22);
    padding: 0.85rem 1.1rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}}
.cust-card-id {{
    font-size: 0.8rem;
    font-weight: 700;
    color: {TEXT};
    font-family: 'Courier New', monospace;
    letter-spacing: 0.10em;
}}
.cust-card-body {{
    padding: 0.95rem 1.1rem 0.8rem;
}}
.cust-metric-row {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.55rem;
    margin-bottom: 0.7rem;
}}
.cust-metric {{
    background: rgba(255,255,255,0.022);
    border-radius: 8px;
    padding: 0.5rem 0.65rem;
}}
.cust-metric-label {{
    font-size: 0.64rem;
    color: {TEXT3};
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 2px;
}}
.cust-metric-value {{
    font-size: 0.9rem;
    font-weight: 600;
    color: {TEXT};
}}

/* Hero section */
.hero-container {{
    background: linear-gradient(135deg, {CARD} 0%, {CARD2} 45%, #1F1A10 100%);
    border: 1px solid {BORDER};
    border-radius: 20px;
    padding: 2.4rem 2.8rem 2rem;
    position: relative;
    overflow: hidden;
    margin-bottom: 1.6rem;
}}
.hero-container::before {{
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(201,169,107,0.08), transparent 68%);
    border-radius: 50%;
    pointer-events: none;
}}
.hero-container::after {{
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, {GOLD_DK} 40%, {GOLD} 60%, transparent);
    opacity: 0.5;
}}
/* Playfair Display title */
.hero-title {{
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 2.6rem;
    font-weight: 700;
    color: {TEXT};
    letter-spacing: -0.015em;
    margin: 0 0 0.25rem 0;
    line-height: 1.12;
}}
.hero-title span {{
    background: linear-gradient(100deg, {GOLD_DK} 0%, {GOLD} 40%, {GOLD_LT} 70%, {CREAM} 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}
.hero-subtitle {{
    font-size: 0.95rem;
    color: {TEXT2};
    margin: 0;
    font-weight: 400;
    letter-spacing: 0.01em;
}}
/* Logo tagline strip in hero */
.hero-tagline {{
    font-size: 0.68rem;
    letter-spacing: 0.22em;
    color: {GOLD_DK};
    text-transform: uppercase;
    margin-bottom: 0.7rem;
    font-weight: 600;
}}

/* Info panel */
.info-panel {{
    background: linear-gradient(145deg, {CARD} 0%, {CARD2} 100%);
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
}}
.info-panel-title {{
    font-size: 0.72rem;
    color: {TEXT2};
    text-transform: uppercase;
    letter-spacing: 0.10em;
    margin-bottom: 0.9rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-weight: 600;
}}

/* Gold divider */
.gold-divider {{
    height: 1px;
    background: linear-gradient(90deg, transparent, {GOLD_DK} 35%, {GOLD} 50%, {GOLD_DK} 65%, transparent);
    border: none;
    margin: 1.6rem 0;
    opacity: 0.6;
}}

/* Explanation text area */
.explanation-box {{
    background: linear-gradient(145deg, {CARD} 0%, {CARD2} 100%);
    border: 1px solid {BORDER};
    border-left: 3px solid {GOLD};
    border-radius: 0 12px 12px 0;
    padding: 1.5rem;
    color: {TEXT};
    line-height: 1.75;
    font-size: 0.89rem;
}}
.explanation-box h1, .explanation-box h2, .explanation-box h3 {{
    color: {GOLD_LT};
    font-weight: 700;
    margin: 1.1rem 0 0.6rem 0;
    line-height: 1.4;
}}
.explanation-box h1 {{ font-size: 1.25rem; }}
.explanation-box h2 {{ font-size: 1.08rem; border-bottom: 1px solid {BORDER}; padding-bottom: 0.35rem; }}
.explanation-box h3 {{ font-size: 0.98rem; }}
.explanation-box h1:first-child, .explanation-box h2:first-child, .explanation-box h3:first-child {{
    margin-top: 0;
}}
.explanation-box p {{ margin: 0 0 0.9rem 0; }}
.explanation-box strong {{ color: {TEXT}; font-weight: 700; }}
.explanation-box em {{ color: {TEXT2}; }}
.explanation-box ul, .explanation-box ol {{
    margin: 0 0 0.9rem 0;
    padding-left: 1.4rem;
}}
.explanation-box li {{ margin-bottom: 0.35rem; color: {TEXT}; }}
.explanation-box hr {{
    border: none;
    border-top: 1px solid {BORDER};
    margin: 1rem 0;
}}
.explanation-box blockquote {{
    margin: 0 0 0.9rem 0;
    padding: 0.5rem 1rem;
    border-left: 3px solid {GOLD_DK};
    background: rgba(196,168,122,0.06);
    color: {TEXT2};
    font-style: italic;
    border-radius: 0 8px 8px 0;
}}
.explanation-box table {{
    width: 100%;
    border-collapse: collapse;
    margin: 0 0 1rem 0;
    font-size: 0.85rem;
}}
.explanation-box th {{
    text-align: left;
    color: {GOLD_LT};
    background: rgba(196,168,122,0.08);
    padding: 0.5rem 0.7rem;
    border: 1px solid {BORDER};
    font-weight: 600;
}}
.explanation-box td {{
    padding: 0.5rem 0.7rem;
    border: 1px solid {BORDER};
    color: {TEXT2};
}}
.explanation-box code {{
    background: rgba(196,168,122,0.10);
    color: {GOLD_LT};
    padding: 0.1rem 0.35rem;
    border-radius: 4px;
    font-size: 0.85em;
}}

/* Step card */
.step-card {{
    background: linear-gradient(135deg, {CARD2} 0%, {CARD3} 100%);
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.65rem;
    display: flex;
    gap: 1rem;
    align-items: flex-start;
    transition: border-color 0.2s;
}}
.step-card:hover {{
    border-color: {BORDER2};
}}
.step-num {{
    background: linear-gradient(135deg, {GOLD} 0%, {GOLD_DK} 100%);
    color: #0C0A07;
    font-weight: 800;
    font-size: 0.78rem;
    width: 26px;
    height: 26px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    font-family: 'Inter', sans-serif;
}}

/* Form section header */
.form-section {{
    background: linear-gradient(90deg, rgba(201,169,107,0.07), transparent);
    border-left: 3px solid {GOLD_DK};
    border-radius: 0 8px 8px 0;
    padding: 0.5rem 1rem;
    margin: 1.2rem 0 0.85rem 0;
    font-size: 0.72rem;
    font-weight: 600;
    color: {GOLD_LT};
    text-transform: uppercase;
    letter-spacing: 0.10em;
    display: flex;
    align-items: center;
    gap: 0.45rem;
}}

/* Success submit card */
.submit-success {{
    background: linear-gradient(135deg, #192A1F 0%, #122018 100%);
    border: 1px solid rgba(92,140,106,0.32);
    border-radius: 18px;
    padding: 2rem;
    text-align: center;
}}

/* Stat pill */
.stat-pill {{
    background: rgba(201,169,107,0.09);
    border: 1px solid rgba(201,169,107,0.22);
    border-radius: 20px;
    padding: 5px 15px;
    font-size: 0.76rem;
    color: {GOLD_LT};
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    letter-spacing: 0.02em;
}}

/* Dashboard chart card */
.chart-card {{
    background: linear-gradient(145deg, {CARD} 0%, {CARD2} 100%);
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 1.2rem 1.2rem 0.8rem;
}}
.chart-title {{
    font-size: 0.76rem;
    font-weight: 600;
    color: {TEXT2};
    text-transform: uppercase;
    letter-spacing: 0.09em;
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
.activity-dot-approved {{ width:8px; height:8px; border-radius:50%; background:{SUCCESS}; flex-shrink:0; }}
.activity-dot-rejected {{ width:8px; height:8px; border-radius:50%; background:{DANGER};  flex-shrink:0; }}

/* Cream accent chip */
.cream-chip {{
    display: inline-block;
    background: rgba(237,227,213,0.08);
    border: 1px solid rgba(237,227,213,0.18);
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.72rem;
    color: {CREAM};
    letter-spacing: 0.04em;
}}
</style>
"""


def inject():
    """Inject the full LAPAS CSS theme into the current Streamlit page."""
    st.markdown(_CSS, unsafe_allow_html=True)


def sidebar_logo():
    """Render the LAPAS logo + wordmark + tagline in the sidebar, centered."""
    import base64 as _b64
    # Encode logo as base64 so we can center it with CSS text-align
    _logo_b64 = ""
    try:
        with open(_LOGO_PATH, 'rb') as _f:
            _logo_b64 = _b64.b64encode(_f.read()).decode()
    except Exception:
        pass

    if _logo_b64:
        st.sidebar.markdown(
            f'<div style="text-align:center;padding:0.8rem 0 0.2rem;">'
            f'<img src="data:image/png;base64,{_logo_b64}" width="130" '
            f'style="border-radius:14px;'
            f'filter:drop-shadow(0 4px 14px rgba(201,169,107,0.18));">'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            f'<div style="text-align:center;padding:1rem 0 0.4rem;">'
            f'<span style="font-family:\'Playfair Display\',Georgia,serif;'
            f'font-size:2.2rem;font-weight:700;color:{GOLD};">LA</span></div>',
            unsafe_allow_html=True,
        )

    # Wordmark + tagline
    st.sidebar.markdown(
        f'<div style="text-align:center;margin-top:4px;margin-bottom:4px;">'
        f'<div style="font-family:\'Playfair Display\',Georgia,serif;'
        f'font-size:1.25rem;font-weight:700;'
        f'background:linear-gradient(100deg,{GOLD_DK},{GOLD},{GOLD_LT});'
        f'-webkit-background-clip:text;-webkit-text-fill-color:transparent;'
        f'background-clip:text;letter-spacing:0.16em;margin-bottom:3px;">LAPAS</div>'
        f'<div style="font-size:0.56rem;color:{GOLD_DK};letter-spacing:0.20em;'
        f'text-transform:uppercase;font-weight:600;">'
        f'Predict &middot; Advise &middot; Approve</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        f'<hr style="border:none;border-top:1px solid {BORDER};margin:10px 0 12px;">',
        unsafe_allow_html=True,
    )


# ── Plotly chart base layout (shared across all pages) ────────────────────────
CHART_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(20,18,14,0.50)',
    font=dict(color=TEXT2, family='Inter, sans-serif', size=11),
    title_font=dict(color=TEXT, size=13, family='Inter, sans-serif'),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT2, size=10)),
    xaxis=dict(gridcolor=BORDER, tickcolor=TEXT3, linecolor=BORDER, color=TEXT2, showgrid=True),
    yaxis=dict(gridcolor=BORDER, tickcolor=TEXT3, linecolor=BORDER, color=TEXT2, showgrid=True),
    margin=dict(l=10, r=10, t=40, b=10),
    colorway=[GOLD, SILVER, SUCCESS, DANGER, '#7A8FBB', '#B87A9A', '#7ABB9A'],
    hoverlabel=dict(bgcolor=CARD2, font_color=TEXT, bordercolor=BORDER2),
)


def apply_chart_layout(fig, title: str = '', height: int = 320):
    """Apply the LAPAS dark theme to any Plotly figure."""
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text=title, font=dict(color=TEXT, size=13), x=0.01, xanchor='left'),
        height=height,
    )
    return fig
