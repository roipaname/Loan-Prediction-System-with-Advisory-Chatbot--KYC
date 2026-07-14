"""
pages/2_Customers.py
Filterable customer card browser. Each application renders as a styled card
showing outcome, credit score, risk tier, loan amount and purpose.
"""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from styles.theme import inject, sidebar_logo, get_logo_image, GOLD, GOLD_LT, GOLD_DK
from styles.theme import TEXT, TEXT2, TEXT3, CARD, CARD2, BORDER, SUCCESS, SUCCESS_LT, DANGER, DANGER_LT
from styles.icons import icon as _icon
from utils import api_client
from utils.mock_data import intent_label, intent_icon, risk_color

st.set_page_config(page_title="LAPAS – Customers", page_icon=get_logo_image() or "L", layout="wide")
inject()
sidebar_logo()

df, using_mock = api_client.get_applicants_safe()
if using_mock:
    st.warning("Backend unavailable — showing sample data.")

# ── Sidebar Filters ───────────────────────────────────────────────────────────
st.sidebar.markdown(f'<div class="section-header" style="font-size:0.78rem;">{_icon("funnel",13,GOLD_LT)} Filters</div>',
                    unsafe_allow_html=True)

search = st.sidebar.text_input("Search by ID", placeholder="e.g. AB12CD34").upper().strip()

outcome_filter = st.sidebar.selectbox("Outcome", ["All", "Approved", "Rejected"])

risk_filter = st.sidebar.multiselect("Risk Tier",
    options=["Low", "Medium", "High"], default=["Low", "Medium", "High"])

grade_filter = st.sidebar.multiselect("Loan Grade",
    options=["A","B","C","D","E","F","G"], default=["A","B","C","D","E","F","G"])

score_min, score_max = st.sidebar.slider(
    "Credit Score Range", 300, 850, (300, 850), step=10)

loan_min, loan_max = st.sidebar.slider(
    "Loan Amount (R'000)", 0.5, 50.0, (0.5, 50.0), step=0.5)

intent_filter = st.sidebar.multiselect("Loan Purpose",
    options=sorted(df['loan_intent'].unique()),
    format_func=intent_label,
    default=list(df['loan_intent'].unique()))

sort_by = st.sidebar.selectbox("Sort By",
    ["Date (Newest)", "Credit Score (High-Low)", "Loan Amount (High-Low)",
     "Risk Tier", "Approval Probability (High-Low)"])

st.sidebar.markdown("<br>", unsafe_allow_html=True)
if st.sidebar.button("Reset Filters", use_container_width=True):
    st.rerun()

# ── Apply filters ─────────────────────────────────────────────────────────────
fdf = df.copy()
if search:
    fdf = fdf[fdf['id'].str.contains(search, case=False, na=False)]
if outcome_filter != "All":
    fdf = fdf[fdf['predicted_outcome'] == outcome_filter.lower()]
if risk_filter:
    fdf = fdf[fdf['risk_tier'].isin(risk_filter)]
if grade_filter:
    fdf = fdf[fdf['loan_grade'].isin(grade_filter)]
fdf = fdf[(fdf['credit_score'] >= score_min) & (fdf['credit_score'] <= score_max)]
fdf = fdf[(fdf['loan_amnt'] >= loan_min * 1000) & (fdf['loan_amnt'] <= loan_max * 1000)]
if intent_filter:
    fdf = fdf[fdf['loan_intent'].isin(intent_filter)]

sort_map = {
    "Date (Newest)":                     ("created_at", False),
    "Credit Score (High-Low)":           ("credit_score", False),
    "Loan Amount (High-Low)":            ("loan_amnt", False),
    "Risk Tier":                         ("risk_tier", True),
    "Approval Probability (High-Low)":   ("approval_probability", False),
}
scol, sasc = sort_map[sort_by]
fdf = fdf.sort_values(scol, ascending=sasc)

# ── Page header ───────────────────────────────────────────────────────────────
approved_n  = (fdf['predicted_outcome'] == 'approved').sum()
rejected_n  = (fdf['predicted_outcome'] == 'rejected').sum()

st.markdown(f"""
<div style="display:flex; align-items:baseline; justify-content:space-between;
            margin-bottom:1.2rem; flex-wrap:wrap; gap:0.5rem;">
  <div style="display:flex;align-items:center;gap:0.6rem;">
    {_icon('user-group',24,GOLD)}
    <span style="font-size:1.5rem;font-weight:800;color:{TEXT};">Customers</span>
    <span style="font-size:0.85rem;color:{TEXT3};margin-left:0.7rem;">
      {len(fdf):,} of {len(df):,} applications</span>
  </div>
  <div style="display:flex;gap:0.6rem;">
    <span class="badge-approved">{approved_n} Approved</span>
    <span class="badge-rejected">{rejected_n} Rejected</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Cards grid ────────────────────────────────────────────────────────────────
if fdf.empty:
    st.markdown(f"""
    <div style="text-align:center; padding:3rem; color:{TEXT3};">
      <div style="margin-bottom:0.8rem;">{_icon('magnifying-glass',48,TEXT3)}</div>
      <div style="font-size:1rem;font-weight:600;color:{TEXT2};">No applications match the current filters.</div>
      <div style="font-size:0.84rem;margin-top:0.3rem;">Try adjusting the filters in the sidebar.</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

PAGE_SIZE = 12
total_pages = max(1, -(-len(fdf) // PAGE_SIZE))

if 'cust_page' not in st.session_state:
    st.session_state['cust_page'] = 0
if st.session_state['cust_page'] >= total_pages:
    st.session_state['cust_page'] = 0

page_df = fdf.iloc[
    st.session_state['cust_page'] * PAGE_SIZE :
    (st.session_state['cust_page'] + 1) * PAGE_SIZE
]

# ── Render cards in 3 columns ─────────────────────────────────────────────────
def _score_bar(score, max=850):
    pct = int(score / max * 100)
    colour = SUCCESS if score >= 670 else (GOLD if score >= 580 else DANGER)
    return f"""
    <div style="height:4px;background:{BORDER};border-radius:2px;margin-top:4px;">
      <div style="width:{pct}%;height:4px;background:{colour};border-radius:2px;
                  transition:width 0.4s;"></div>
    </div>"""

def _render_card(row):
    approved = row['predicted_outcome'] == 'approved'
    hdr_bg   = "cust-card-header-approved" if approved else "cust-card-header-rejected"
    badge    = f'<span class="badge-approved">Approved</span>' if approved else f'<span class="badge-rejected">Rejected</span>'
    risk_col = {
        'Low':    SUCCESS_LT,
        'Medium': GOLD_LT,
        'High':   DANGER_LT,
    }.get(row['risk_tier'], TEXT2)
    icon  = intent_icon(row['loan_intent'])
    ilbl  = intent_label(row['loan_intent'])
    prob  = f"{row['approval_probability']*100:.0f}%"
    score = int(row['credit_score'])

    score_col = SUCCESS if score >= 670 else (GOLD if score >= 580 else DANGER)

    return f"""
    <div class="cust-card">
      <div class="{hdr_bg}">
        <span class="cust-card-id">APP-{row['id']}</span>
        {badge}
      </div>
      <div class="cust-card-body">
        <div class="cust-metric-row">
          <div class="cust-metric">
            <div class="cust-metric-label">Credit Score</div>
            <div class="cust-metric-value" style="color:{score_col};">{score}</div>
            {_score_bar(score)}
          </div>
          <div class="cust-metric">
            <div class="cust-metric-label">Risk Tier</div>
            <div class="cust-metric-value" style="color:{risk_col};">
              {row['risk_tier']}</div>
          </div>
          <div class="cust-metric">
            <div class="cust-metric-label">Loan Amount</div>
            <div class="cust-metric-value">R{row['loan_amnt']:,.0f}</div>
          </div>
          <div class="cust-metric">
            <div class="cust-metric-label">Probability</div>
            <div class="cust-metric-value" style="color:{GOLD_LT};">{prob}</div>
          </div>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center;
                    margin-top:0.4rem;font-size:0.74rem;color:{TEXT3};">
          <span>{icon} {ilbl}</span>
          <span>Grade {row['loan_grade']} · {row['created_at'].strftime('%d %b %y')}</span>
        </div>
      </div>
    </div>"""

rows_iter = list(page_df.iterrows())
n_cols = 3
for i in range(0, len(rows_iter), n_cols):
    cols = st.columns(n_cols, gap="small")
    for j, col in enumerate(cols):
        if i + j < len(rows_iter):
            _, row = rows_iter[i + j]
            with col:
                st.markdown(_render_card(row), unsafe_allow_html=True)
                btn_col, _ = st.columns([1, 1])
                with btn_col:
                    if st.button(f"Advisory", key=f"adv_{row['id']}",
                                 use_container_width=True):
                        st.session_state['selected_customer'] = row['id']
                        st.switch_page("pages/4_AI_Advisory.py")

# ── Pagination ────────────────────────────────────────────────────────────────
if total_pages > 1:
    st.markdown("<br>", unsafe_allow_html=True)
    p_cols = st.columns([1, 2, 1])
    with p_cols[0]:
        if st.button("Previous", disabled=st.session_state['cust_page'] == 0,
                     use_container_width=True):
            st.session_state['cust_page'] -= 1
            st.rerun()
    with p_cols[1]:
        st.markdown(
            f'<div style="text-align:center;color:{TEXT2};font-size:0.84rem;padding-top:0.5rem;">'
            f'Page {st.session_state["cust_page"]+1} of {total_pages}</div>',
            unsafe_allow_html=True)
    with p_cols[2]:
        if st.button("Next", disabled=st.session_state['cust_page'] >= total_pages - 1,
                     use_container_width=True):
            st.session_state['cust_page'] += 1
            st.rerun()
