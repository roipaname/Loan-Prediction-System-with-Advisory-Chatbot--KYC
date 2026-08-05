"""
pages/5_Findings.py
Summary of the project's two benchmarking experiments:
  1. TF-IDF vs Dense Embedding (Chroma) retrieval quality/speed
  2. Classifier performance comparison (random_forest, xgboost,
     logistic_regression, naive_bayes)
"""

import streamlit as st
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from styles.theme import (inject, sidebar_logo, get_logo_image, apply_chart_layout,
                           GOLD, GOLD_LT, GOLD_DK, SILVER, TEXT, TEXT2, TEXT3,
                           CARD, CARD2, BORDER, SUCCESS, SUCCESS_LT, DANGER, DANGER_LT)
from styles.icons import icon as _icon
from utils import api_client

st.set_page_config(page_title="LAPAS – Findings", page_icon=get_logo_image() or "L", layout="wide")
inject()
sidebar_logo()

st.markdown(f"""
<div style="margin-bottom:1.2rem;display:flex;align-items:center;gap:0.7rem;">
  {_icon('scale',26,GOLD)}
  <span style="font-size:1.5rem;font-weight:800;color:{TEXT};">Findings &amp; Comparisons</span>
  <span style="font-size:0.84rem;color:{TEXT3};margin-left:0.4rem;">
    Retrieval benchmark · Classifier comparison</span>
</div>
""", unsafe_allow_html=True)


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


tab1, tab2 = st.tabs(["Retrieval: TF-IDF vs Dense Embeddings", "Classifier Comparison"])

# tab 1 – retrieval benchmark
with tab1:
    try:
        rdf = api_client.get_retrieval_comparison()

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Avg TF-IDF Latency", f"{rdf['tfidf_latency_ms'].mean():.1f} ms")
        k2.metric("Avg Dense Latency", f"{rdf['chroma_latency_ms'].mean():.1f} ms")
        k3.metric("Avg TF-IDF Top-1 Score", f"{rdf['tfidf_top1_score'].mean():.3f}")
        k4.metric("Avg Dense Top-1 Score", f"{rdf['chroma_top1_score'].mean():.3f}")

        st.markdown(
            f'<div style="font-size:0.8rem;color:{TEXT3};margin:0.6rem 0 1rem;">'
            f'{len(rdf)} domain-specific loan queries benchmarked against both retrievers. '
            f'Full methodology in <code>scripts/tfidf_chroma.py</code>.</div>',
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2, gap="small")

        with c1:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=rdf['query'].str.slice(0, 20) + "…", y=rdf['tfidf_latency_ms'],
                name='TF-IDF', marker_color=GOLD, marker_line_width=0,
            ))
            fig.add_trace(go.Bar(
                x=rdf['query'].str.slice(0, 20) + "…", y=rdf['chroma_latency_ms'],
                name='Dense (Chroma)', marker_color=SILVER, marker_line_width=0,
            ))
            apply_chart_layout(fig, "Latency per Query (ms, log scale)", 340)
            fig.update_layout(barmode='group', yaxis_type='log',
                              legend=dict(orientation='h', y=-0.35, x=0.5, xanchor='center'),
                              xaxis=dict(tickangle=-35))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        with c2:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=rdf['query'].str.slice(0, 20) + "…", y=rdf['tfidf_top1_score'],
                name='TF-IDF', marker_color=GOLD, marker_line_width=0,
            ))
            fig.add_trace(go.Bar(
                x=rdf['query'].str.slice(0, 20) + "…", y=rdf['chroma_top1_score'],
                name='Dense (Chroma)', marker_color=SILVER, marker_line_width=0,
            ))
            apply_chart_layout(fig, "Top-1 Similarity Score per Query", 340)
            fig.update_layout(barmode='group',
                              legend=dict(orientation='h', y=-0.35, x=0.5, xanchor='center'),
                              xaxis=dict(tickangle=-35))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        c3, c4 = st.columns(2, gap="small")

        with c3:
            fig = go.Figure(go.Bar(
                x=rdf['query'].str.slice(0, 25) + "…", y=rdf['jaccard_overlap'],
                marker_color=GOLD_LT, marker_line_width=0,
            ))
            apply_chart_layout(fig, "Result-Set Jaccard Overlap (TF-IDF ∩ Dense)", 300)
            fig.update_layout(showlegend=False, xaxis=dict(tickangle=-35),
                              yaxis=dict(range=[0, 1]))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        with c4:
            colors = [SUCCESS if v >= 0 else DANGER for v in rdf['spearman_rho']]
            fig = go.Figure(go.Bar(
                x=rdf['query'].str.slice(0, 25) + "…", y=rdf['spearman_rho'],
                marker_color=colors, marker_line_width=0,
            ))
            apply_chart_layout(fig, "Rank Correlation (Spearman ρ, TF-IDF vs Dense)", 300)
            fig.add_hline(y=0, line_color=BORDER, line_width=1)
            fig.update_layout(showlegend=False, xaxis=dict(tickangle=-35))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        st.markdown(
            f'<div class="section-header" style="margin-top:0.6rem;">'
            f'{_icon("clipboard-list",14,GOLD_LT)} Raw Metrics</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(rdf, use_container_width=True, hide_index=True)

    except Exception as _e:
        _err("Retrieval Comparison", _e)

# tab 2 – classifier comparison
with tab2:
    try:
        cdf = api_client.get_model_comparison()
        algo_labels = {
            'random_forest': 'Random Forest', 'xgboost': 'XGBoost',
            'logistic_regression': 'Logistic Regression', 'naive_bayes': 'Naive Bayes',
        }
        cdf = cdf.sort_values('composite_rank').reset_index(drop=True)
        champion_row = cdf[cdf['is_champion']].iloc[0] if cdf['is_champion'].any() else cdf.iloc[0]

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Champion Model", algo_labels.get(champion_row['algorithm'], champion_row['algorithm']))
        k2.metric("ROC-AUC", f"{champion_row['roc_auc']:.4f}")
        k3.metric("F1 (weighted)", f"{champion_row['f1_weighted']:.4f}")
        k4.metric("MCC", f"{champion_row['mcc']:.4f}")

        st.markdown(
            f'<div style="font-size:0.8rem;color:{TEXT3};margin:0.6rem 0 1rem;">'
            f'4 algorithms evaluated on a held-out test split: random_forest, xgboost, '
            f'logistic_regression (from-scratch implementation), naive_bayes. '
            f'Champion selected by <code>MODEL_SELECTION_METRIC</code> '
            f'(default roc_auc). Full methodology in <code>scripts/train_model.py</code>.</div>',
            unsafe_allow_html=True,
        )

        models = [algo_labels.get(a, a) for a in cdf['algorithm']]

        fig = go.Figure()
        for col, name, color in [
            ('accuracy', 'Accuracy', SILVER), ('f1_weighted', 'F1 (weighted)', GOLD),
            ('roc_auc', 'ROC-AUC', SUCCESS), ('mcc', 'MCC', DANGER_LT),
        ]:
            fig.add_trace(go.Bar(x=models, y=cdf[col], name=name, marker_line_width=0, opacity=0.9,
                                  marker_color=color))
        apply_chart_layout(fig, "Classifier Performance Comparison", 360)
        fig.update_layout(barmode='group',
                          legend=dict(orientation='h', y=-0.16, x=0.5, xanchor='center'))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        c1, c2 = st.columns(2, gap="small")
        with c1:
            fig = go.Figure(go.Bar(
                x=models, y=cdf['brier_score'],
                marker_color=[SUCCESS if v == cdf['brier_score'].min() else GOLD_DK for v in cdf['brier_score']],
                marker_line_width=0,
                text=[f"{v:.4f}" for v in cdf['brier_score']],
                textposition='outside', textfont=dict(color=TEXT2, size=11),
            ))
            apply_chart_layout(fig, "Brier Score (lower = better calibrated)", 300)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        with c2:
            fig = go.Figure(go.Bar(
                x=models, y=cdf['composite_rank'],
                marker_color=[SUCCESS if v == cdf['composite_rank'].min() else GOLD_DK for v in cdf['composite_rank']],
                marker_line_width=0,
                text=[f"#{r}" for r in cdf['rank']],
                textposition='outside', textfont=dict(color=TEXT2, size=11),
            ))
            apply_chart_layout(fig, "Composite Rank (lower = better overall)", 300)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        st.markdown(
            f'<div class="section-header" style="margin-top:0.6rem;">'
            f'{_icon("clipboard-list",14,GOLD_LT)} Full Metrics Table</div>',
            unsafe_allow_html=True,
        )
        display_cols = ['rank', 'model', 'accuracy', 'precision', 'recall', 'f1_macro',
                         'f1_weighted', 'roc_auc', 'avg_precision', 'brier_score', 'mcc',
                         'composite_rank', 'is_champion']
        st.dataframe(cdf[display_cols], use_container_width=True, hide_index=True)

    except Exception as _e:
        _err("Classifier Comparison", _e)
