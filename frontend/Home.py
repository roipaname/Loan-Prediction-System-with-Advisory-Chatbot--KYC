"""
Home.py  –  LAPAS landing page
Hero · KPI strip · recent activity feed · quick-navigation cards
"""

import streamlit as st
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from styles.theme import (inject, sidebar_logo, get_logo_image,
                           GOLD, GOLD_LT, CREAM,
                           TEXT, TEXT2, TEXT3,
                           BORDER, SUCCESS, DANGER, SILVER, apply_chart_layout)
from styles.icons import icon as _icon
from utils import api_client
from utils.mock_data import intent_icon, intent_label

st.set_page_config(
    page_title="LAPAS – Home",
    page_icon=get_logo_image() or "L",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject()
sidebar_logo()

df, using_mock = api_client.get_applicants_safe()
if using_mock:
    st.warning("Backend unavailable — showing sample data.")

# ── KPIs ──────────────────────────────────────────────────────────────────────
total        = len(df)
approved     = (df['predicted_outcome'] == 'approved').sum()
approval_pct = approved / total * 100
avg_score    = int(df['credit_score'].mean())
avg_loan     = df['loan_amnt'].mean()
high_risk_n  = (df['risk_tier'] == 'High').sum()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-container">
  <div class="hero-tagline">Predict &middot; Advise &middot; Approve</div>
  <div class="hero-title"><span>LAPAS</span></div>
  <div class="hero-subtitle" style="margin-bottom:1.6rem;">
    Loan Approval Prediction &amp; Advisory System</div>
  <div style="display:flex; gap:0.65rem; flex-wrap:wrap;">
    <span class="stat-pill">{_icon('cpu-chip',13,GOLD_LT)} ML Ensemble Active</span>
    <span class="stat-pill">{_icon('document-text',13,GOLD_LT)} RAG Explanations</span>
    <span class="stat-pill">{_icon('scale',13,GOLD_LT)} Fairness Monitoring</span>
    <span class="stat-pill">{_icon('server-stack',13,GOLD_LT)} {total:,} Applications</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPI Strip ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Applications",  f"{total:,}")
c2.metric("Approval Rate",       f"{approval_pct:.1f}%",  delta="+2.3% vs last month")
c3.metric("Avg Credit Score",    f"{avg_score}",           delta="+12 pts")
c4.metric("Avg Loan Amount",     f"R{avg_loan:,.0f}")
c5.metric("High Risk Flagged",   f"{high_risk_n}",         delta="-3 this week", delta_color="inverse")

st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

# ── Two-column layout: recent activity + mini charts ──────────────────────────
left, right = st.columns([1.4, 1], gap="large")

with left:
    st.markdown(
        f'<div class="section-header">'
        f'{_icon("clipboard-list",14,GOLD_LT)} Recent Applications</div>',
        unsafe_allow_html=True,
    )
    recent = df.sort_values('created_at', ascending=False).head(10)
    for _, row in recent.iterrows():
        approved_flag = row['predicted_outcome'] == 'approved'
        dot_cls = "activity-dot-approved" if approved_flag else "activity-dot-rejected"
        badge = (
            f'<span class="badge-approved">Approved</span>'
            if approved_flag else
            f'<span class="badge-rejected">Rejected</span>'
        )
        ilabel = intent_label(row['loan_intent'])
        iicon  = intent_icon(row['loan_intent'])
        st.markdown(
            f'<div class="activity-row">'
            f'<div class="{dot_cls}"></div>'
            f'<div style="flex:1;min-width:0;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;">'
            f'<span style="font-size:0.82rem;font-weight:600;color:{TEXT};'
            f'font-family:monospace;letter-spacing:0.06em;">APP-{row["id"]}</span>'
            f'{badge}</div>'
            f'<div style="font-size:0.73rem;color:{TEXT3};margin-top:2px;">'
            f'{iicon} {ilabel} &nbsp;&middot;&nbsp; R{row["loan_amnt"]:,.0f}'
            f' &nbsp;&middot;&nbsp; Score {row["credit_score"]}'
            f' &nbsp;&middot;&nbsp; {row["created_at"].strftime("%d %b %Y")}</div>'
            f'</div>'
            f'<div style="font-size:0.76rem;color:{GOLD_LT};font-weight:600;'
            f'white-space:nowrap;">{row["risk_tier"]} Risk</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

with right:
    st.markdown(
        f'<div class="section-header">'
        f'{_icon("chart-pie",14,GOLD_LT)} Quick Insights</div>',
        unsafe_allow_html=True,
    )

    # Approval donut
    fig_donut = go.Figure(go.Pie(
        labels=['Approved', 'Rejected'],
        values=[approved, total - approved],
        hole=0.72,
        marker_colors=[SUCCESS, DANGER],
        textinfo='none',
        hovertemplate="%{label}: %{value}<br>%{percent}<extra></extra>",
    ))
    fig_donut.add_annotation(
        text=f"<b>{approval_pct:.0f}%</b><br><span style='font-size:10px'>Approved</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=18, color=TEXT, family='Inter'),
    )
    apply_chart_layout(fig_donut, "Approval Overview", height=240)
    fig_donut.update_layout(
        showlegend=True,
        legend=dict(orientation='h', y=-0.05, x=0.5, xanchor='center'),
    )
    st.plotly_chart(fig_donut, use_container_width=True, config={'displayModeBar': False})

    # Risk tier bars
    risk_counts = df['risk_tier'].value_counts().reindex(['Low', 'Medium', 'High'], fill_value=0)
    fig_risk = go.Figure(go.Bar(
        x=list(risk_counts.index),
        y=list(risk_counts.values),
        marker_color=[SUCCESS, GOLD, DANGER],
        marker_line_width=0,
        text=list(risk_counts.values),
        textposition='outside',
        textfont=dict(color=TEXT2, size=11),
    ))
    apply_chart_layout(fig_risk, "Risk Tier Distribution", height=220)
    fig_risk.update_layout(showlegend=False, xaxis_title=None, yaxis_title=None)
    st.plotly_chart(fig_risk, use_container_width=True, config={'displayModeBar': False})

st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

# ── Quick Navigation Cards ────────────────────────────────────────────────────
st.markdown(
    f'<div class="section-header">{_icon("map",14,GOLD_LT)} Navigate</div>',
    unsafe_allow_html=True,
)

nav_items = [
    (_icon("document-plus", 30, GOLD), "Customer Application",
     "Submit a new loan application with full demographic and financial details."),
    (_icon("user-group",    30, GOLD), "Customers",
     "Browse all applications as cards. Filter by score, risk tier, grade and more."),
    (_icon("chart-bar-square", 30, GOLD), "Dashboard",
     "12+ KPI cards and 18+ interactive charts covering approvals, risk, and fairness."),
    (_icon("cpu-chip",      30, GOLD), "AI Advisory",
     "Select any applicant to view SHAP attribution, policy explanation, and export PDF."),
]

cols = st.columns(4, gap="small")
for col, (nav_icon, title, desc) in zip(cols, nav_items):
    with col:
        st.markdown(
            f'<div class="info-panel" style="min-height:155px;">'
            f'<div style="margin-bottom:0.75rem;">{nav_icon}</div>'
            f'<div style="font-family:\'Playfair Display\',Georgia,serif;'
            f'font-size:0.97rem;font-weight:600;color:{TEXT};margin-bottom:0.4rem;">'
            f'{title}</div>'
            f'<div style="font-size:0.77rem;color:{TEXT2};line-height:1.55;">{desc}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    f'<div style="text-align:center;margin-top:2.5rem;padding-top:1rem;'
    f'border-top:1px solid {BORDER};color:{TEXT3};font-size:0.72rem;">'
    f'LAPAS &nbsp;&middot;&nbsp; Loan Approval Prediction &amp; Advisory System'
    f' &nbsp;&middot;&nbsp; University of Johannesburg &nbsp;&middot;&nbsp; Honours Project 2026'
    f'</div>',
    unsafe_allow_html=True,
)
