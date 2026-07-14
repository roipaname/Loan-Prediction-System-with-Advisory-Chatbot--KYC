"""
pages/4_AI_Advisory.py
Select any applicant, view their SHAP attribution waterfall, read the AI-generated
policy-grounded explanation, review next steps, and export a PDF report.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from styles.theme import (inject, sidebar_logo, get_logo_image, apply_chart_layout,
                           GOLD, GOLD_LT, GOLD_DK, SILVER,
                           TEXT, TEXT2, TEXT3, CARD, CARD2,
                           BORDER, SUCCESS, SUCCESS_LT, DANGER, DANGER_LT)
CARD3 = CARD2
from styles.icons import icon as _icon
from utils import api_client
from utils.mock_data import intent_label, intent_icon, risk_color

st.set_page_config(page_title="LAPAS – AI Advisory", page_icon=get_logo_image() or "L", layout="wide")
inject()
sidebar_logo()

df, using_mock = api_client.get_applicants_safe()
if using_mock:
    st.warning("Backend unavailable — showing sample data.")

# ── Page header ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="margin-bottom:1rem;display:flex;align-items:center;gap:0.6rem;">
  {_icon('cpu-chip',26,GOLD)}
  <span style="font-size:1.5rem;font-weight:800;color:{TEXT};">AI Advisory</span>
  <span style="font-size:0.84rem;color:{TEXT3};margin-left:0.8rem;">
    SHAP attribution · Policy-grounded explanation · Export report</span>
</div>
""", unsafe_allow_html=True)

# ── Customer selector ─────────────────────────────────────────────────────────
all_ids = ["— Select an applicant —"] + [f"APP-{i}" for i in df['id'].tolist()]
preselect = 0
if 'selected_customer' in st.session_state:
    pre_id = f"APP-{st.session_state['selected_customer']}"
    if pre_id in all_ids:
        preselect = all_ids.index(pre_id)

selected = st.selectbox("Select Applicant", all_ids, index=preselect,
                         label_visibility="collapsed")

if selected == "— Select an applicant —":
    st.markdown(f"""
    <div style="text-align:center;padding:4rem;color:{TEXT3};">
      <div style="margin-bottom:0.8rem;">{_icon('magnifying-glass',52,TEXT3)}</div>
      <div style="font-size:1rem;font-weight:600;color:{TEXT2};">
        Select an applicant above to generate their advisory report.</div>
      <div style="font-size:0.82rem;margin-top:0.3rem;">
        You can also click "Advisory" on any card in the Customers page.</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Load applicant ────────────────────────────────────────────────────────────
app_id = selected.replace("APP-", "")
row = df[df['id'] == app_id].iloc[0]

approved  = row['predicted_outcome'] == 'approved'
risk_col  = {
    'Low':    SUCCESS_LT,
    'Medium': GOLD_LT,
    'High':   DANGER_LT,
}.get(row['risk_tier'], TEXT2)
badge     = (f'<span class="badge-approved">Approved</span>'
             if approved else f'<span class="badge-rejected">Rejected</span>')

# ── Profile card ──────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:linear-gradient(135deg,{CARD} 0%,{CARD2} 100%);
            border:1px solid {BORDER};border-radius:18px;overflow:hidden;
            margin-bottom:1.2rem;">
  <div style="background:{'linear-gradient(135deg,#1E3228,#162A20)' if approved
              else 'linear-gradient(135deg,#2A1A1C,#221518)'};
              padding:1rem 1.4rem;
              border-bottom:1px solid {'rgba(92,140,106,0.25)' if approved
              else 'rgba(140,94,98,0.25)'};
              display:flex;justify-content:space-between;align-items:center;">
    <div>
      <div style="font-size:0.72rem;color:{TEXT3};text-transform:uppercase;
      letter-spacing:0.08em;margin-bottom:3px;">Applicant ID</div>
      <div style="font-size:1.1rem;font-weight:800;color:{TEXT};
      font-family:monospace;letter-spacing:0.06em;">APP-{row['id']}</div>
    </div>
    <div style="text-align:right;">
      {badge}
      <div style="font-size:0.8rem;color:{risk_col};font-weight:600;margin-top:4px;">
        {row['risk_tier']} Risk · {row['approval_probability']*100:.1f}% probability</div>
    </div>
  </div>
  <div style="display:grid;grid-template-columns:repeat(6,1fr);gap:0;padding:1rem 1.4rem;">
    {''.join([
      f'''<div style="padding:0.5rem 0;">
        <div style="font-size:0.66rem;color:{TEXT3};text-transform:uppercase;
        letter-spacing:0.06em;margin-bottom:3px;">{lbl}</div>
        <div style="font-size:0.9rem;font-weight:600;color:{TEXT};">{val}</div>
      </div>'''
      for lbl, val in [
        ("Age",          f"{int(row['person_age'])} yrs"),
        ("Income",       f"R{row['person_income']:,.0f}"),
        ("Credit Score", f"{int(row['credit_score'])} ({row['credit_score_tier']})"),
        ("Loan Amount",  f"R{row['loan_amnt']:,.0f}"),
        ("Purpose",      f"{intent_icon(row['loan_intent'])} {intent_label(row['loan_intent'])}"),
        ("Grade",        f"{row['loan_grade']} · {row['loan_int_rate']:.1f}%"),
      ]
    ])}
  </div>
</div>
""", unsafe_allow_html=True)

# ── Three tabs ────────────────────────────────────────────────────────────────
t1, t2, t3 = st.tabs(["Feature Attribution (LIME)", "AI Explanation", "Next Steps"])

# ─── Tab 1: Feature Attribution ───────────────────────────────────────────────
with t1:
    from src.ai_advisor.loan_context_builder import _readable_name as nice_name
    shap_dict = row['shap_values']

    # Sort by absolute value
    sorted_shap = dict(sorted(shap_dict.items(), key=lambda x: x[1]))
    nice_names = {k: nice_name(k) for k in shap_dict}
    labels = [nice_names.get(k, k) for k in sorted_shap.keys()]
    values = list(sorted_shap.values())
    colors = [SUCCESS if v > 0 else DANGER for v in values]

    sa, sb = st.columns([2, 1], gap="large")
    with sa:
        fig = go.Figure(go.Bar(
            y=labels, x=values, orientation='h',
            marker_color=colors, marker_line_width=0,
            text=[f"{'+'if v>0 else ''}{v:.4f}" for v in values],
            textposition='outside',
            textfont=dict(color=TEXT2, size=10, family='Courier New'),
            hovertemplate="%{y}<br>LIME weight: %{x:.4f}<extra></extra>",
        ))
        fig.add_vline(x=0, line_color=BORDER, line_width=1.5)
        apply_chart_layout(fig,
            f"LIME Feature Attribution – APP-{row['id']} ({row['predicted_outcome'].capitalize()})",
            height=360)
        fig.update_layout(showlegend=False,
                          xaxis=dict(zeroline=False, showgrid=True))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with sb:
        top5 = list(sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True))[:5]
        _driver_rows = ""
        for k, v in top5:
            col = SUCCESS_LT if v > 0 else DANGER_LT
            sign = "+" if v > 0 else ""
            _driver_rows += (
                f'<div style="display:flex;justify-content:space-between;'
                f'padding:0.4rem 0;border-bottom:1px solid {BORDER};font-size:0.82rem;">'
                f'<span style="color:{TEXT2};">{nice_names.get(k, k)}</span>'
                f'<span style="color:{col};font-weight:600;font-family:monospace;">'
                f'{sign}{v:.4f}</span></div>'
            )
        st.markdown(
            f'<div class="info-panel">'
            f'<div class="info-panel-title">{_icon("information-circle",14,GOLD_LT)} How to Read This Chart</div>'
            f'<div style="font-size:0.82rem;color:{TEXT2};line-height:1.65;">'
            f'Each bar shows how much a feature pushed the prediction toward '
            f'<b style="color:{SUCCESS_LT};">approval</b> (positive, right) '
            f'or <b style="color:{DANGER_LT};">rejection</b> (negative, left).<br><br>'
            f'<b style="color:{GOLD_LT};">LIME</b> (Local Interpretable Model-agnostic '
            f'Explanations) fits a simple, local surrogate model around this one '
            f'applicant to estimate how much each feature moved the prediction — '
            f'this project uses LIME rather than SHAP (see requirements.txt) because '
            f'SHAP\'s numba/llvmlite dependency is unstable on this platform.</div></div><br>'
            f'<div class="info-panel">'
            f'<div class="info-panel-title">{_icon("trophy",14,GOLD_LT)} Top Drivers</div>'
            f'{_driver_rows}</div>',
            unsafe_allow_html=True,
        )

# ─── Tab 2: AI Explanation ────────────────────────────────────────────────────
with t2:
    e1, e2 = st.columns([2, 1], gap="large")

    with e1:
        if 'generated' not in st.session_state:
            st.session_state['generated'] = {}

        retriever_choice = st.selectbox(
            "Retriever", ["tfidf", "vector"], key=f"retriever_{app_id}",
            format_func=lambda r: "TF-IDF" if r == "tfidf" else "Dense Embedding (Chroma)",
        )
        gen_key = f"gen_{app_id}_{retriever_choice}"

        if gen_key not in st.session_state['generated']:
            st.markdown(f"""
            <div style="text-align:center;padding:2rem;color:{TEXT2};">
              <div style="margin-bottom:0.7rem;">{_icon('cpu-chip',40,TEXT3)}</div>
              <div style="font-size:0.9rem;">Click the button to generate the AI advisory explanation.</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Generate Advisory Explanation", use_container_width=False):
                try:
                    with st.spinner("Generating policy-grounded explanation…"):
                        result = api_client.generate_advisory(app_id, retriever=retriever_choice)
                    st.session_state['generated'][gen_key] = result['report']
                    st.rerun()
                except api_client.BackendUnavailable as exc:
                    st.error(f"Could not reach the advisory backend: {exc}")
        else:
            explanation = st.session_state['generated'][gen_key]
            st.markdown(
                f'<div class="explanation-box">{explanation}</div>',
                unsafe_allow_html=True)

    with e2:
        _policy_sources = [
            (_icon('clipboard-list', 16, GOLD_LT), 'FATF KYC Guidelines 2023', 'Section 4.2 – Customer Due Diligence'),
            (_icon('building-library', 16, GOLD_LT), 'Basel III Framework 2017', 'Article 147 – Exposure Thresholds'),
            (_icon('shield-check', 16, GOLD_LT), 'POPIA Act 2020', 'Section 5.1 – Data Minimisation'),
        ]
        _src_rows = "".join(
            f'<div style="display:flex;gap:0.6rem;align-items:flex-start;'
            f'padding:0.5rem 0;border-bottom:1px solid {BORDER};">'
            f'<div style="flex-shrink:0;margin-top:1px;">{si}</div>'
            f'<div><div style="font-size:0.8rem;font-weight:600;color:{TEXT};">{s}</div>'
            f'<div style="font-size:0.72rem;color:{TEXT3};">{sec}</div></div></div>'
            for si, s, sec in _policy_sources
        )
        _algo = str(row.get('model_algorithm') or 'unknown').replace('_', ' ').title()
        _retriever_label = "TF-IDF" if retriever_choice == "tfidf" else "Dense Embedding (Chroma)"
        st.markdown(
            f'<div class="info-panel">'
            f'<div class="info-panel-title">{_icon("book-open",14,GOLD_LT)} Policy Sources Used</div>'
            f'{_src_rows}</div><br>'
            f'<div class="info-panel">'
            f'<div class="info-panel-title">{_icon("cog-6-tooth",14,GOLD_LT)} Model Details</div>'
            f'<div style="font-size:0.8rem;color:{TEXT2};line-height:1.7;">'
            f'<b>Classifier:</b> {_algo} (champion)<br>'
            f'<b>LLM:</b> Mistral-7B-Instruct v0.2<br>'
            f'<b>Retriever:</b> {_retriever_label}<br>'
            f'<b>Chunks retrieved:</b> Top-5</div></div>',
            unsafe_allow_html=True,
        )

# ─── Tab 3: Next Steps ────────────────────────────────────────────────────────
with t3:
    if approved:
        steps = [
            ("Maintain your credit score",
             f"Your current score of {int(row['credit_score'])} is strong. Keep credit utilisation below 30% and avoid new credit applications in the next 6 months."),
            ("Confirm loan drawdown terms",
             "Contact your loan officer to confirm disbursement schedule, repayment date, and any prepayment conditions."),
            ("Set up automatic repayment",
             "Enrol in automatic debit to avoid missed payments. A single missed payment can lower credit score by 50-100 points."),
            ("Build your credit history",
             "Ensure any new financial product is reported to all major credit bureaus to extend and diversify your credit record."),
            ("Review interest rate options",
             f"Your current rate of {row['loan_int_rate']:.1f}% may be refinanceable after 12 months of clean repayment history."),
        ]
        header_colour = SUCCESS
        header_icon   = _icon('check-circle', 22, SUCCESS_LT)
        intro = "Your application has been approved. Follow these steps to make the most of your loan."
    else:
        steps = [
            ("Improve your credit score",
             f"Target a score above 670 (currently {int(row['credit_score'])}). Pay all bills on time and reduce outstanding balances. Allow 6-12 months for improvements to register."),
            ("Reduce your loan amount",
             f"Consider re-applying for R{max(500, row['loan_amnt'] * 0.65):,.0f} (35% reduction). This improves your loan-to-income ratio from {row['loan_percent_income']*100:.1f}% toward the 20% target."),
            ("Increase documented income",
             "Additional income streams, a salary increase letter, or a co-applicant strengthens the income profile used in credit assessment."),
            ("Resolve any prior defaults",
             "Obtain a settlement letter or proof of resolved obligation for any default on record. Provide this with your next application."),
            ("Re-apply in 6-12 months",
             "Allow time for credit score improvements and for any defaults to reduce in recency. A new application after genuine improvement will be assessed on the updated profile."),
        ]
        header_colour = DANGER
        header_icon   = _icon('arrow-trending-up', 22, DANGER_LT)
        intro = "Your application was not approved at this time. The following steps can improve your eligibility."

    _step_rows = "".join(
        f'<div class="step-card">'
        f'<div class="step-num">{i+1}</div>'
        f'<div>'
        f'<div style="font-size:0.88rem;font-weight:600;color:{TEXT};margin-bottom:3px;">{title}</div>'
        f'<div style="font-size:0.81rem;color:{TEXT2};line-height:1.6;">{detail}</div>'
        f'</div></div>'
        for i, (title, detail) in enumerate(steps)
    )
    _hdr_bg = "linear-gradient(135deg,#1E3228,#162A20)" if approved else "linear-gradient(135deg,#2A1A1C,#221518)"
    st.markdown(
        f'<div style="background:linear-gradient(135deg,{CARD} 0%,{CARD2} 100%);'
        f'border:1px solid {BORDER};border-radius:16px;overflow:hidden;margin-bottom:1.2rem;">'
        f'<div style="background:{_hdr_bg};padding:1rem 1.4rem;">'
        f'<div style="font-size:1rem;font-weight:700;color:{TEXT};display:flex;align-items:center;gap:0.5rem;">'
        f'{header_icon} Recommended Next Steps</div>'
        f'<div style="font-size:0.82rem;color:{TEXT2};margin-top:4px;">{intro}</div>'
        f'</div><div style="padding:1.2rem 1.4rem;">{_step_rows}</div></div>',
        unsafe_allow_html=True,
    )

    # ── PDF Export ────────────────────────────────────────────────────────────
    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:0.82rem;color:{TEXT2};margin-bottom:0.6rem;display:flex;align-items:center;gap:0.4rem;">'
                f'{_icon("document-arrow-down",15,TEXT2)} Export this advisory report as a PDF or text file.</div>',
                unsafe_allow_html=True)

    pdf_col, txt_col, _ = st.columns([1, 1, 3])

    # The AI explanation may not have been generated yet if the user jumped
    # straight to this tab — generate it now (tfidf default) so export always works.
    _gen_keys = [k for k in st.session_state.get('generated', {}) if k.startswith(f"gen_{app_id}_")]
    if _gen_keys:
        explanation_text = st.session_state['generated'][_gen_keys[0]]
    else:
        try:
            with st.spinner("Generating advisory explanation for export…"):
                _result = api_client.generate_advisory(app_id, retriever="tfidf")
            explanation_text = _result['report']
            st.session_state.setdefault('generated', {})[f"gen_{app_id}_tfidf"] = explanation_text
        except api_client.BackendUnavailable:
            explanation_text = "_AI advisory explanation unavailable (backend unreachable)._"

    # Build text report content
    report_lines = [
        "=" * 60,
        "LAPAS – LOAN ADVISORY REPORT",
        "=" * 60,
        f"Applicant ID  : APP-{row['id']}",
        f"Report Date   : {pd.Timestamp.now().strftime('%d %B %Y')}",
        f"Outcome       : {row['predicted_outcome'].upper()}",
        f"Risk Tier     : {row['risk_tier']}",
        f"Approval Prob : {row['approval_probability']*100:.1f}%",
        "",
        "APPLICANT PROFILE",
        "-" * 40,
        f"Age           : {int(row['person_age'])}",
        f"Income        : R{row['person_income']:,.0f}",
        f"Credit Score  : {int(row['credit_score'])} ({row['credit_score_tier']})",
        f"Loan Amount   : R{row['loan_amnt']:,.0f}",
        f"Loan Purpose  : {intent_label(row['loan_intent'])}",
        f"Loan Grade    : {row['loan_grade']}",
        f"Interest Rate : {row['loan_int_rate']:.1f}%",
        "",
        "TOP FEATURE ATTRIBUTIONS (LIME)",
        "-" * 40,
    ]
    from src.ai_advisor.loan_context_builder import _readable_name as _nice_name
    for feat, val in sorted(row['shap_values'].items(), key=lambda x: abs(x[1]), reverse=True):
        direction = "positive" if val > 0 else "negative"
        report_lines.append(f"  {_nice_name(feat):<30} {'+' if val>0 else ''}{val:.4f} ({direction})")

    report_lines += [
        "",
        "AI ADVISORY EXPLANATION",
        "-" * 40,
        explanation_text,
        "",
        "RECOMMENDED NEXT STEPS",
        "-" * 40,
    ]
    for i, (title, detail) in enumerate(steps, 1):
        report_lines.append(f"{i}. {title}")
        report_lines.append(f"   {detail}")
        report_lines.append("")

    report_lines += [
        "=" * 60,
        "Policy References: FATF KYC Guidelines 2023 s.4.2 | Basel III Art.147 | POPIA 2020 s.5.1",
        "Generated by LAPAS – University of Johannesburg | Honours Year Project 2026",
        "=" * 60,
    ]
    report_text = "\n".join(report_lines)

    with txt_col:
        st.download_button(
            label="Download Report (.txt)",
            data=report_text,
            file_name=f"LAPAS_Advisory_APP-{row['id']}.txt",
            mime="text/plain",
            use_container_width=True,
        )

    # PDF via fpdf2 (graceful fallback)
    with pdf_col:
        try:
            from fpdf import FPDF
            import io

            def _pdf_safe(text: str) -> str:
                # Helvetica (core font) is latin-1 only; strip characters
                # outside that range (em-dashes, curly quotes, middots) that
                # can appear in LLM-generated or policy-doc text.
                for old, new in {
                    "–": "-", "—": "-", "‘": "'", "’": "'",
                    "“": '"', "”": '"', "·": "*", "…": "...",
                }.items():
                    text = text.replace(old, new)
                return text.encode("latin-1", errors="replace").decode("latin-1")

            def build_pdf():
                pdf = FPDF()
                pdf.add_page()
                pdf.set_margins(18, 18, 18)
                pdf.set_auto_page_break(auto=True, margin=18)

                # Header
                pdf.set_font("Helvetica", "B", 16)
                pdf.set_text_color(30, 27, 24)
                pdf.cell(0, 10, "LAPAS LOAN ADVISORY REPORT", ln=True, align='C')
                pdf.set_font("Helvetica", "", 9)
                pdf.set_text_color(100, 94, 88)
                pdf.cell(0, 6, f"APP-{row['id']}  |  {pd.Timestamp.now().strftime('%d %B %Y')}",
                         ln=True, align='C')
                pdf.ln(4)
                pdf.set_draw_color(196, 168, 122)
                pdf.set_line_width(0.5)
                pdf.line(18, pdf.get_y(), 192, pdf.get_y())
                pdf.ln(4)

                # Outcome badge
                color = (92, 140, 106) if approved else (140, 94, 98)
                pdf.set_fill_color(*color)
                pdf.set_font("Helvetica", "B", 10)
                pdf.set_text_color(255, 255, 255)
                label_text = "APPROVED" if approved else "REJECTED"
                pdf.cell(40, 8, f" {label_text} ", fill=True, border=1, ln=False, align='C')
                pdf.set_font("Helvetica", "", 10)
                pdf.set_text_color(100, 94, 88)
                pdf.cell(0, 8, f"   Risk: {row['risk_tier']}  |  "
                               f"Probability: {row['approval_probability']*100:.1f}%", ln=True)
                pdf.ln(4)

                # Sections
                sections = [
                    ("Applicant Profile", [
                        ("Age", f"{int(row['person_age'])} yrs"),
                        ("Income", f"R{row['person_income']:,.0f}"),
                        ("Credit Score", f"{int(row['credit_score'])} ({row['credit_score_tier']})"),
                        ("Loan Amount", f"R{row['loan_amnt']:,.0f}"),
                        ("Purpose", intent_label(row['loan_intent'])),
                        ("Loan Grade", row['loan_grade']),
                    ]),
                ]
                for sec_title, items in sections:
                    pdf.set_font("Helvetica", "B", 11)
                    pdf.set_text_color(196, 168, 122)
                    pdf.cell(0, 8, sec_title, ln=True)
                    pdf.set_draw_color(44, 40, 34)
                    pdf.line(18, pdf.get_y(), 192, pdf.get_y())
                    pdf.ln(2)
                    pdf.set_font("Helvetica", "", 9)
                    for label_t, val_t in items:
                        pdf.set_text_color(100, 94, 88)
                        pdf.cell(55, 6, f"{label_t}:", ln=False)
                        pdf.set_text_color(240, 235, 227)
                        pdf.set_text_color(30, 27, 24)
                        pdf.cell(0, 6, str(val_t), ln=True)

                pdf.ln(2)
                pdf.set_font("Helvetica", "B", 11)
                pdf.set_text_color(196, 168, 122)
                pdf.cell(0, 8, "AI Advisory Explanation", ln=True)
                pdf.set_draw_color(44, 40, 34)
                pdf.line(18, pdf.get_y(), 192, pdf.get_y())
                pdf.ln(2)
                pdf.set_font("Helvetica", "", 8.5)
                pdf.set_text_color(30, 27, 24)
                # Strip markdown
                clean_exp = explanation_text.replace('**','').replace('*','').replace('#','')
                pdf.multi_cell(0, 5.5, _pdf_safe(clean_exp[:2500]))

                # Footer
                pdf.set_y(-25)
                pdf.set_font("Helvetica", "I", 8)
                pdf.set_text_color(100, 94, 88)
                pdf.cell(0, 5, _pdf_safe("LAPAS – University of Johannesburg | Honours Year Project 2026"),
                         align='C', ln=True)
                pdf.cell(0, 5, _pdf_safe("Policy: FATF KYC 2023 · Basel III Art.147 · POPIA 2020"),
                         align='C', ln=True)
                return bytes(pdf.output())

            pdf_bytes = build_pdf()
            st.download_button(
                label="Export as PDF",
                data=pdf_bytes,
                file_name=f"LAPAS_Advisory_APP-{row['id']}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except ImportError:
            st.markdown(
                f'<div style="font-size:0.76rem;color:{TEXT3};">'
                f'Install <code>fpdf2</code> for PDF export.</div>',
                unsafe_allow_html=True)
