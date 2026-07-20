"""
pages/3_Dashboard.py
Full analytics dashboard – 18+ interactive Plotly charts organised across
four tabs: Overview · Demographics · Financial Risk · Model Intelligence
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from styles.theme import (inject, sidebar_logo, get_logo_image, apply_chart_layout,
                           GOLD, GOLD_LT, GOLD_DK, SILVER, TEXT, TEXT2, TEXT3,
                           CARD, CARD2, BORDER, SUCCESS, SUCCESS_LT, DANGER, DANGER_LT)
CARD3 = CARD2
from styles.icons import icon as _icon
from utils import api_client
from utils.mock_data import intent_label

st.set_page_config(page_title="LAPAS – Dashboard", page_icon=get_logo_image() or "L", layout="wide")
inject()
sidebar_logo()

df, using_mock = api_client.get_applicants_safe()
if using_mock:
    st.warning("Backend unavailable — showing sample data.")

# ── Pre-compute ────────────────────────────────────────────────────────────────
total           = len(df)
approved        = (df['predicted_outcome'] == 'approved').sum()
rejected        = total - approved
apr_pct         = approved / total * 100
rej_pct         = rejected / total * 100
avg_score       = df['credit_score'].mean()
avg_loan        = df['loan_amnt'].mean()
avg_rate        = df['loan_int_rate'].mean()
high_risk       = (df['risk_tier'] == 'High').sum()
total_portfolio = df['loan_amnt'].sum()
avg_prob        = df['approval_probability'].mean() * 100
prior_defaults  = df['previous_loan_defaults_on_file'].sum()
thin_credit     = df['thin_credit_file'].sum()
avg_emp_yrs     = df['person_emp_exp'].mean()

df['month']        = df['created_at'].dt.to_period('M').astype(str)
df['approved_flag'] = (df['predicted_outcome'] == 'approved').astype(int)

monthly = (df.groupby('month')
             .agg(applications=('id','count'), approved=('approved_flag','sum'))
             .reset_index())
monthly['apr_rate'] = monthly['approved'] / monthly['applications'] * 100

COLORS_OUTCOME = {'approved': SUCCESS, 'rejected': DANGER}

# ── Error helper ───────────────────────────────────────────────────────────────
def _err(tab_name: str, exc: Exception):
    st.markdown(
        f'<div style="background:rgba(190,60,50,0.10);border:1px solid rgba(190,60,50,0.30);'
        f'border-radius:10px;padding:1rem 1.2rem;margin-top:0.8rem;">'
        f'<div style="font-weight:600;color:#e07070;margin-bottom:0.3rem;">'
        f'{_icon("x-circle",15,"#e07070")} Error rendering {tab_name}</div>'
        f'<div style="font-size:0.80rem;color:{TEXT2};font-family:monospace;'
        f'white-space:pre-wrap;">{type(exc).__name__}: {exc}</div></div>',
        unsafe_allow_html=True,
    )

# ── Page header ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="margin-bottom:1.2rem;display:flex;align-items:center;gap:0.7rem;">
  {_icon('chart-bar-square',26,GOLD)}
  <span style="font-size:1.5rem;font-weight:800;color:{TEXT};">Analytics Dashboard</span>
  <span style="font-size:0.84rem;color:{TEXT3};margin-left:0.4rem;">{total:,} applications</span>
</div>
""", unsafe_allow_html=True)

# ── Top KPI strip – row 1 ──────────────────────────────────────────────────────
k1,k2,k3,k4,k5,k6 = st.columns(6)
k1.metric("Total Applications",  f"{total:,}")
k2.metric("Approval Rate",       f"{apr_pct:.1f}%",   delta="+2.3%")
k3.metric("Avg Credit Score",    f"{avg_score:.0f}",  delta="+12")
k4.metric("Avg Loan Amount",     f"R{avg_loan:,.0f}")
k5.metric("Avg Interest Rate",   f"{avg_rate:.1f}%")
k6.metric("High Risk Cases",     f"{high_risk}",      delta="-3", delta_color="inverse")

# ── Top KPI strip – row 2 ──────────────────────────────────────────────────────
k7,k8,k9,k10,k11,k12 = st.columns(6)
k7.metric("Total Portfolio",        f"R{total_portfolio/1_000_000:.1f}M")
k8.metric("Rejection Rate",         f"{rej_pct:.1f}%",      delta="-2.3%", delta_color="inverse")
k9.metric("Avg Approval Prob.",     f"{avg_prob:.1f}%")
k10.metric("Prior Defaults",        f"{int(prior_defaults)}",  delta=f"+{int(prior_defaults*0.05):.0f}", delta_color="inverse")
k11.metric("Thin Credit Files",     f"{int(thin_credit)}")
k12.metric("Avg Employment Yrs",    f"{avg_emp_yrs:.1f} yrs")

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Overview & Approval",
    "Demographics",
    "Financial & Risk",
    "Model Intelligence",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – Overview & Approval
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    try:
        r1c1, r1c2, r1c3 = st.columns([1, 1.6, 1], gap="small")

        with r1c1:
            fig = go.Figure(go.Pie(
                labels=['Approved','Rejected'],
                values=[approved, total-approved],
                hole=0.70,
                marker_colors=[SUCCESS, DANGER],
                hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
                textinfo='none',
            ))
            fig.add_annotation(
                text=f"<b>{apr_pct:.0f}%</b><br><span style='font-size:11px'>Approved</span>",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20, color=TEXT, family='Inter'),
            )
            fig.update_layout(showlegend=True, legend=dict(
                orientation='h', y=-0.08, x=0.5, xanchor='center',
                font=dict(size=10, color=TEXT2)))
            apply_chart_layout(fig, "Approval Split", 280)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        with r1c2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=monthly['month'], y=monthly['applications'],
                name='Applications', fill='tozeroy',
                line=dict(color=GOLD, width=2),
                fillcolor='rgba(196,168,122,0.12)',
            ))
            fig.add_trace(go.Scatter(
                x=monthly['month'], y=monthly['approved'],
                name='Approved', line=dict(color=SUCCESS, width=2, dash='dot'),
            ))
            apply_chart_layout(fig, "Monthly Application Trend", 280)
            fig.update_layout(legend=dict(orientation='h', y=-0.15, x=0.5, xanchor='center'))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        with r1c3:
            fig = go.Figure(go.Scatter(
                x=monthly['month'], y=monthly['apr_rate'],
                fill='tozeroy', line=dict(color=GOLD_LT, width=2),
                fillcolor='rgba(196,168,122,0.10)',
                hovertemplate="%{x}<br>Approval Rate: %{y:.1f}%<extra></extra>",
            ))
            apply_chart_layout(fig, "Monthly Approval Rate (%)", 280)
            fig.add_hline(y=apr_pct, line_dash='dash', line_color=SILVER,
                          annotation_text=f"Avg {apr_pct:.0f}%",
                          annotation_font_color=TEXT2, annotation_font_size=10)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        st.markdown("<hr>", unsafe_allow_html=True)
        r2c1, r2c2 = st.columns(2, gap="small")

        with r2c1:
            grade_df = df.groupby('loan_grade').agg(
                total=('id','count'), approved=('approved_flag','sum')).reset_index()
            grade_df['rate'] = grade_df['approved'] / grade_df['total'] * 100
            grade_df = grade_df.sort_values('loan_grade')
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=grade_df['loan_grade'], y=grade_df['total'],
                name='Total', marker_color='rgba(140,145,153,0.35)',
                marker_line_width=0,
            ))
            fig.add_trace(go.Bar(
                x=grade_df['loan_grade'], y=grade_df['approved'],
                name='Approved', marker_color=SUCCESS, marker_line_width=0,
            ))
            apply_chart_layout(fig, "Applications & Approvals by Loan Grade", 300)
            fig.update_layout(barmode='overlay',
                              legend=dict(orientation='h', y=-0.12, x=0.5, xanchor='center'))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        with r2c2:
            grade_df = grade_df.sort_values('rate')
            colors   = [SUCCESS if r >= 60 else (GOLD if r >= 40 else DANGER)
                        for r in grade_df['rate']]
            fig = go.Figure(go.Bar(
                y=grade_df['loan_grade'], x=grade_df['rate'],
                orientation='h',
                marker_color=colors, marker_line_width=0,
                text=[f"{r:.0f}%" for r in grade_df['rate']],
                textposition='outside', textfont=dict(color=TEXT2, size=11),
            ))
            apply_chart_layout(fig, "Approval Rate % by Loan Grade", 300)
            fig.update_layout(showlegend=False, xaxis=dict(range=[0, 110], showgrid=False))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        intent_df = df.groupby('loan_intent').agg(
            total=('id','count'), approved=('approved_flag','sum')).reset_index()
        intent_df['rate']  = intent_df['approved'] / intent_df['total'] * 100
        intent_df['label'] = intent_df['loan_intent'].apply(intent_label)
        intent_df = intent_df.sort_values('total', ascending=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(y=intent_df['label'], x=intent_df['total'],
                             name='Total', orientation='h',
                             marker_color='rgba(140,145,153,0.35)', marker_line_width=0))
        fig.add_trace(go.Bar(y=intent_df['label'], x=intent_df['approved'],
                             name='Approved', orientation='h',
                             marker_color=SUCCESS, marker_line_width=0))
        apply_chart_layout(fig, "Applications by Loan Purpose", 300)
        fig.update_layout(barmode='overlay',
                          legend=dict(orientation='h', y=-0.14, x=0.5, xanchor='center'))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    except Exception as _e:
        _err("Overview & Approval", _e)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – Demographics
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    try:
        d1, d2 = st.columns(2, gap="small")

        with d1:
            edu_order = ['High School','Diploma','Associate','Bachelor','Master','Doctor']
            edu_df = df.groupby('person_education').agg(
                total=('id','count'), approved=('approved_flag','sum')).reset_index()
            edu_df['rate'] = edu_df['approved'] / edu_df['total'] * 100
            edu_df = edu_df.set_index('person_education').reindex(edu_order).reset_index()
            fig = go.Figure(go.Bar(
                y=edu_df['person_education'], x=edu_df['rate'],
                orientation='h',
                marker=dict(
                    color=edu_df['rate'],
                    colorscale=[[0, DANGER], [0.5, GOLD], [1, SUCCESS]],
                    showscale=False,
                ),
                marker_line_width=0,
                text=[f"{v:.0f}%" for v in edu_df['rate']],
                textposition='outside', textfont=dict(color=TEXT2, size=10),
            ))
            apply_chart_layout(fig, "Approval Rate by Education Level", 300)
            fig.update_layout(showlegend=False, xaxis=dict(range=[0, 105], showgrid=False))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        with d2:
            g_df = df.groupby('person_gender').agg(
                total=('id','count'), approved=('approved_flag','sum')).reset_index()
            g_df['rate'] = g_df['approved'] / g_df['total'] * 100
            fig = go.Figure(go.Pie(
                labels=g_df['person_gender'].str.capitalize(),
                values=g_df['total'], hole=0.60,
                marker_colors=[GOLD, SILVER, GOLD_DK],
                textinfo='none',
                hovertemplate="%{label}: %{value}<br>Approval rate: %{customdata:.0f}%<extra></extra>",
                customdata=g_df['rate'],
            ))
            fig.add_annotation(text="<b>Gender</b>", x=0.5, y=0.5, showarrow=False,
                                font=dict(size=13, color=TEXT2, family='Inter'))
            apply_chart_layout(fig, "Applications by Gender", 300)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        d3, d4 = st.columns(2, gap="small")

        with d3:
            df['age_group'] = pd.cut(df['person_age'],
                bins=[18,25,35,45,55,70], labels=['18-24','25-34','35-44','45-54','55+'])
            age_df = df.groupby('age_group', observed=True).agg(
                total=('id','count'), approved=('approved_flag','sum')).reset_index()
            age_df['rate'] = age_df['approved'] / age_df['total'] * 100
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=age_df['age_group'].astype(str), y=age_df['rate'],
                mode='lines+markers',
                line=dict(color=GOLD, width=2.5),
                marker=dict(color=GOLD, size=9, line=dict(color=GOLD_LT, width=2)),
                fill='tozeroy', fillcolor='rgba(196,168,122,0.09)',
                hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
            ))
            apply_chart_layout(fig, "Approval Rate by Age Group (%)", 280)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        with d4:
            own_df = df.groupby('person_home_ownership').agg(
                total=('id','count'), approved=('approved_flag','sum')).reset_index()
            own_df['rate'] = own_df['approved'] / own_df['total'] * 100
            own_labels = {'MORTGAGE':'Mortgage','OWN':'Own','RENT':'Rent','OTHER':'Other'}
            own_df['label'] = own_df['person_home_ownership'].map(own_labels)
            fig = go.Figure(go.Bar(
                x=own_df['label'], y=own_df['rate'],
                marker=dict(color=[GOLD, SUCCESS, SILVER, DANGER], line_width=0),
                text=[f"{v:.0f}%" for v in own_df['rate']],
                textposition='outside', textfont=dict(color=TEXT2, size=11),
            ))
            apply_chart_layout(fig, "Approval Rate by Home Ownership (%)", 280)
            fig.update_layout(showlegend=False, yaxis=dict(range=[0, 110]))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        fig = px.scatter(df, x='person_emp_exp', y='person_income',
                         color='predicted_outcome',
                         color_discrete_map={'approved': SUCCESS, 'rejected': DANGER},
                         opacity=0.65, size_max=8,
                         hover_data={'credit_score': True, 'loan_amnt': ':,.0f',
                                     'risk_tier': True},
                         labels={'person_emp_exp': 'Employment Experience (years)',
                                 'person_income': 'Annual Income (R)',
                                 'predicted_outcome': 'Outcome'})
        apply_chart_layout(fig, "Employment Experience vs Annual Income (coloured by Outcome)", 340)
        fig.update_traces(marker=dict(size=7, line=dict(width=0)))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    except Exception as _e:
        _err("Demographics", _e)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 – Financial & Risk
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    try:
        f1, f2 = st.columns(2, gap="small")

        with f1:
            fig = go.Figure()
            for outcome, color in [('approved', SUCCESS), ('rejected', DANGER)]:
                subset = df[df['predicted_outcome'] == outcome]['credit_score']
                fig.add_trace(go.Histogram(
                    x=subset, name=outcome.capitalize(), nbinsx=25,
                    marker_color=color, opacity=0.72,
                    marker_line_color=CARD, marker_line_width=0.5,
                    hovertemplate=f"{outcome.capitalize()}: %{{x}}<br>Count: %{{y}}<extra></extra>",
                ))
            apply_chart_layout(fig, "Credit Score Distribution by Outcome", 300)
            fig.update_layout(barmode='overlay',
                              legend=dict(orientation='h', y=-0.12, x=0.5, xanchor='center'))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        with f2:
            fig = go.Figure()
            for outcome, color in [('approved', SUCCESS), ('rejected', DANGER)]:
                subset = df[df['predicted_outcome'] == outcome]['loan_int_rate']
                fig.add_trace(go.Box(
                    y=subset, name=outcome.capitalize(),
                    marker_color=color, line_color=color,
                    fillcolor=f'rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.18)',
                    boxmean='sd', whiskerwidth=0.5,
                ))
            apply_chart_layout(fig, "Interest Rate Distribution by Outcome (%)", 300)
            fig.update_layout(showlegend=True,
                              legend=dict(orientation='h', y=-0.12, x=0.5, xanchor='center'))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        f3, f4 = st.columns(2, gap="small")

        with f3:
            fig = go.Figure()
            for outcome, color in [('approved', SUCCESS), ('rejected', DANGER)]:
                subset = df[df['predicted_outcome'] == outcome]['debt_to_income_ratio'].clip(0, 1.5)
                fig.add_trace(go.Violin(
                    y=subset, name=outcome.capitalize(),
                    box_visible=True, meanline_visible=True,
                    fillcolor=f'rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.20)',
                    line_color=color,
                ))
            apply_chart_layout(fig, "Debt-to-Income Ratio by Outcome", 300)
            fig.update_layout(violinmode='overlay',
                              legend=dict(orientation='h', y=-0.12, x=0.5, xanchor='center'))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        with f4:
            fig = px.scatter(df, x='credit_score', y='loan_amnt',
                             size='approval_probability',
                             color='risk_tier',
                             color_discrete_map={'Low': SUCCESS, 'Medium': GOLD, 'High': DANGER},
                             opacity=0.65, size_max=16,
                             labels={'credit_score': 'Credit Score',
                                     'loan_amnt': 'Loan Amount (R)',
                                     'risk_tier': 'Risk Tier'})
            apply_chart_layout(fig, "Credit Score vs Loan Amount (bubble = approval prob.)", 300)
            fig.update_traces(marker=dict(line=dict(width=0)))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        f5, f6 = st.columns(2, gap="small")

        with f5:
            risk_counts = df['risk_tier'].value_counts().reindex(['Low','Medium','High'])
            fig = go.Figure(go.Pie(
                labels=list(risk_counts.index), values=list(risk_counts.values),
                hole=0.55, textinfo='none',
                marker_colors=[SUCCESS, GOLD, DANGER],
                hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
            ))
            apply_chart_layout(fig, "Risk Tier Distribution", 280)
            fig.update_layout(showlegend=True,
                              legend=dict(orientation='h', y=-0.06, x=0.5, xanchor='center'))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        with f6:
            pd_df = df.groupby('previous_loan_defaults_on_file').agg(
                total=('id','count'), approved=('approved_flag','sum')).reset_index()
            pd_df['rate']  = pd_df['approved'] / pd_df['total'] * 100
            pd_df['label'] = pd_df['previous_loan_defaults_on_file'].map(
                {True: 'Prior Default: Yes', False: 'Prior Default: No'})
            fig = go.Figure(go.Bar(
                x=pd_df['label'], y=pd_df['rate'],
                marker_color=[DANGER, SUCCESS], marker_line_width=0,
                text=[f"{v:.0f}%" for v in pd_df['rate']],
                textposition='outside', textfont=dict(color=TEXT2, size=13),
            ))
            apply_chart_layout(fig, "Approval Rate: Prior Default vs None", 280)
            fig.update_layout(showlegend=False, yaxis=dict(range=[0, 110]))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        inc_loan = df.groupby(['income_bucket','loan_grade'])['loan_amnt'].mean().reset_index()
        pivot = inc_loan.pivot(index='income_bucket', columns='loan_grade', values='loan_amnt')
        inc_order = ['low','mid_low','medium','high']
        pivot = pivot.reindex([r for r in inc_order if r in pivot.index])
        fig = go.Figure(go.Heatmap(
            z=pivot.values, x=list(pivot.columns), y=list(pivot.index),
            colorscale=[[0, CARD2], [0.5, GOLD_DK], [1, GOLD_LT]],
            hovertemplate="Grade %{x} / Income: %{y}<br>Avg Loan: R%{z:,.0f}<extra></extra>",
            colorbar=dict(
                title=dict(text="Avg Loan (R)", font=dict(color=TEXT2)),
                tickfont=dict(color=TEXT2),
            ),
        ))
        apply_chart_layout(fig, "Avg Loan Amount Heatmap: Income Bucket × Loan Grade", 300)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    except Exception as _e:
        _err("Financial & Risk", _e)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 – Model Intelligence
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    try:
        m1, m2 = st.columns(2, gap="small")

        with m1:
            tier_order = ['Poor','Fair','Good','Very Good','Exceptional']
            tier_df = df.groupby('credit_score_tier').agg(
                total=('id','count'), approved=('approved_flag','sum')).reset_index()
            tier_df['rate'] = tier_df['approved'] / tier_df['total'] * 100
            tier_df = tier_df.set_index('credit_score_tier').reindex(
                [t for t in tier_order if t in tier_df.index]).reset_index()
            fig = go.Figure(go.Funnel(
                y=tier_df['credit_score_tier'], x=tier_df['total'],
                textposition='inside', textinfo='value+percent initial',
                marker_color=[DANGER, DANGER_LT, GOLD, SUCCESS_LT, SUCCESS],
                connector=dict(line=dict(color=BORDER, width=1)),
            ))
            apply_chart_layout(fig, "Credit Score Tier Funnel (All Applications)", 300)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        with m2:
            from src.ai_advisor.loan_context_builder import _readable_name
            all_shap = pd.DataFrame(df['shap_values'].tolist())
            mean_abs = all_shap.abs().mean().sort_values(ascending=True).tail(12)
            y_labels = [_readable_name(f) for f in mean_abs.index]
            fig = go.Figure(go.Bar(
                y=y_labels, x=mean_abs.values, orientation='h',
                marker=dict(
                    color=mean_abs.values,
                    colorscale=[[0, CARD2], [0.5, GOLD_DK], [1, GOLD_LT]],
                    showscale=False,
                ),
                marker_line_width=0,
                text=[f"{v:.4f}" for v in mean_abs.values],
                textposition='outside', textfont=dict(color=TEXT2, size=10),
            ))
            apply_chart_layout(fig, "Global Feature Importance (mean |LIME weight|)", 300)
            fig.update_layout(showlegend=False, xaxis=dict(showgrid=False))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        m3, m4 = st.columns(2, gap="small")

        with m3:
            cmp_df = api_client.get_model_comparison()
            algo_labels = {
                'random_forest': 'Random Forest', 'xgboost': 'XGBoost',
                'logistic_regression': 'Logistic Reg.', 'naive_bayes': 'Naive Bayes',
            }
            models = [algo_labels.get(a, a) for a in cmp_df['algorithm']]
            fig = go.Figure()
            for col, name, color in [
                ('roc_auc', 'AUC-ROC', GOLD), ('f1_weighted', 'F1', SUCCESS),
                ('precision', 'Precision', SILVER), ('recall', 'Recall', DANGER_LT)]:
                fig.add_trace(go.Bar(
                    x=models, y=cmp_df[col], name=name,
                    marker_color=color, marker_line_width=0, opacity=0.88,
                ))
            apply_chart_layout(fig, "Classifier Performance Comparison", 320)
            fig.update_layout(barmode='group',
                              legend=dict(orientation='h', y=-0.18, x=0.5, xanchor='center'))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        with m4:
            fig = go.Figure()
            for outcome, color in [('approved', SUCCESS), ('rejected', DANGER)]:
                probs = df[df['predicted_outcome'] == outcome]['approval_probability']
                fig.add_trace(go.Histogram(
                    x=probs, name=outcome.capitalize(), nbinsx=20,
                    marker_color=color, opacity=0.72,
                    marker_line_color=CARD, marker_line_width=0.5,
                ))
            fig.add_vline(x=0.52, line_dash='dash', line_color=GOLD_LT,
                          annotation_text="Decision Threshold 52%",
                          annotation_font_color=GOLD_LT, annotation_font_size=10)
            apply_chart_layout(fig, "Approval Probability Distribution", 320)
            fig.update_layout(barmode='overlay',
                              legend=dict(orientation='h', y=-0.14, x=0.5, xanchor='center'))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        fig = go.Figure()
        colors_tier = {'Low': SUCCESS, 'Medium': GOLD, 'High': DANGER}
        for tier in ['Low', 'Medium', 'High']:
            sub   = df[df['risk_tier'] == tier]['composite_risk_score']
            color = colors_tier[tier]
            fig.add_trace(go.Violin(
                y=sub, name=tier,
                fillcolor=f'rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.22)',
                line_color=color, box_visible=True, meanline_visible=True,
            ))
        apply_chart_layout(fig, "Composite Risk Score Distribution by Tier", 300)
        fig.update_layout(violinmode='group',
                          legend=dict(orientation='h', y=-0.10, x=0.5, xanchor='center'))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    except Exception as _e:
        _err("Model Intelligence", _e)
