"""
Home.py  –  LAPAS landing page
Hero section · KPI strip · recent activity feed · quick-navigation cards
"""

import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from styles.theme import inject, sidebar_logo, GOLD, GOLD_LT, GOLD_DK, TEXT, TEXT2, TEXT3
from styles.theme import CARD, CARD2, BORDER, SUCCESS, DANGER, SILVER, apply_chart_layout
from utils.mock_data import get_data, intent_icon, intent_label

st.set_page_config(
    page_title="LAPAS – Home",
    page_icon="./frontend/assets/logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject()
sidebar_logo()

df = get_data()

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
  <div style="display:flex; align-items:center; gap:1.2rem; margin-bottom:0.5rem;">
    <div>
      <div class="hero-title"><span>LAPAS</span></div>
      <div class="hero-subtitle">Loan Approval Prediction &amp; Advisory System</div>
    </div>
  </div>
  <div style="margin-top:1.2rem; display:flex; gap:0.8rem; flex-wrap:wrap;">
    <span class="stat-pill">🤖 ML Ensemble Active</span>
    <span class="stat-pill">📄 RAG Explanation Ready</span>
    <span class="stat-pill">⚖️ Fairness Monitoring On</span>
    <span class="stat-pill">🗄️ {total} Applications Loaded</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPI Strip ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Applications",  f"{total:,}")
c2.metric("Approval Rate",       f"{approval_pct:.1f}%",   delta="+2.3% vs last month")
c3.metric("Avg Credit Score",    f"{avg_score}",            delta="+12 pts")
c4.metric("Avg Loan Amount",     f"${avg_loan:,.0f}")
c5.metric("High Risk Flagged",   f"{high_risk_n}",          delta=f"-{3} this week", delta_color="inverse")

st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

# ── Two-column layout: recent activity + mini charts ──────────────────────────
left, right = st.columns([1.4, 1], gap="large")

with left:
    st.markdown('<div class="section-header">📋 Recent Applications</div>', unsafe_allow_html=True)
    recent = df.sort_values('created_at', ascending=False).head(10)
    for _, row in recent.iterrows():
        approved_flag = row['predicted_outcome'] == 'approved'
        dot_class = "activity-dot-approved" if approved_flag else "activity-dot-rejected"
        badge = (f'<span class="badge-approved">Approved</span>'
                 if approved_flag else
                 f'<span class="badge-rejected">Rejected</span>')
        icon = intent_icon(row['loan_intent'])
        label_text = intent_label(row['loan_intent'])
        st.markdown(f"""
        <div class="activity-row">
          <div class="{dot_class}"></div>
          <div style="flex:1; min-width:0;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
              <span style="font-size:0.82rem;font-weight:600;color:{TEXT};font-family:monospace;">
                APP-{row['id']}</span>
              {badge}
            </div>
            <div style="font-size:0.74rem;color:{TEXT3};margin-top:2px;">
              {icon} {label_text} &nbsp;·&nbsp; ${row['loan_amnt']:,.0f}
              &nbsp;·&nbsp; Score {row['credit_score']}
              &nbsp;·&nbsp; {row['created_at'].strftime('%d %b %Y')}
            </div>
          </div>
          <div style="font-size:0.78rem;color:{GOLD_LT};font-weight:600;white-space:nowrap;">
            {row['risk_tier']} Risk
          </div>
        </div>
        """, unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-header">📊 Quick Insights</div>', unsafe_allow_html=True)

    # Approval donut
    fig_donut = go.Figure(go.Pie(
        labels=['Approved', 'Rejected'],
        values=[approved, total - approved],
        hole=0.72,
        marker_colors=[SUCCESS, DANGER],
        textfont=dict(size=10, color=TEXT2),
        hovertemplate="%{label}: %{value}<br>%{percent}<extra></extra>",
    ))
    fig_donut.add_annotation(
        text=f"<b>{approval_pct:.0f}%</b><br><span style='font-size:10px'>Approved</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=18, color=TEXT, family='Inter'),
    )
    apply_chart_layout(fig_donut, "Approval Overview", height=240)
    fig_donut.update_layout(showlegend=True, legend=dict(
        orientation='h', y=-0.05, x=0.5, xanchor='center',
    ))
    fig_donut.update_traces(textinfo='none')
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
st.markdown('<div class="section-header">🧭 Navigate</div>', unsafe_allow_html=True)

nav_items = [
    ("📝", "Customer Application",
     "Submit a new loan application with full demographic and financial details.",
     "pages/1_Customer_Application.py"),
    ("👥", "Customers",
     "Browse all applications as cards. Filter by score, risk tier, grade and more.",
     "pages/2_Customers.py"),
    ("📊", "Dashboard",
     "18+ interactive charts covering approvals, credit risk, income, and fairness.",
     "pages/3_Dashboard.py"),
    ("🤖", "AI Advisory",
     "Select any applicant to view SHAP attribution, policy-grounded explanation, and export a PDF report.",
     "pages/4_AI_Advisory.py"),
]

cols = st.columns(4, gap="small")
for col, (icon, title, desc, _) in zip(cols, nav_items):
    with col:
        st.markdown(f"""
        <div class="info-panel" style="height:160px; position:relative;">
          <div style="font-size:1.8rem; margin-bottom:0.5rem;">{icon}</div>
          <div style="font-size:0.95rem; font-weight:700; color:{TEXT}; margin-bottom:0.4rem;">
            {title}</div>
          <div style="font-size:0.78rem; color:{TEXT2}; line-height:1.5;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center; margin-top:2.5rem; padding-top:1rem;
            border-top:1px solid {BORDER}; color:{TEXT3}; font-size:0.74rem;">
  LAPAS &nbsp;·&nbsp; Loan Approval Prediction &amp; Advisory System
  &nbsp;·&nbsp; University of Johannesburg &nbsp;·&nbsp; Honours Year Project 2026
</div>
""", unsafe_allow_html=True)
