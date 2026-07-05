"""
pages/1_Customer_Application.py
Full loan application submission form matching the LoanApplicant SQLAlchemy model.
Organised in three sections: Demographic · Financial · Loan Details
"""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from styles.theme import inject, sidebar_logo, GOLD, GOLD_LT, GOLD_DK
from styles.theme import TEXT, TEXT2, TEXT3, CARD, CARD2, BORDER, SUCCESS, DANGER
from styles.icons import icon as _icon
import uuid
from datetime import datetime

st.set_page_config(page_title="LAPAS – Application", page_icon="L", layout="wide")
inject()
sidebar_logo()

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="margin-bottom:1.4rem;">
  <div style="display:flex;align-items:center;gap:0.6rem;font-size:1.6rem;font-weight:800;color:{TEXT};">
    {_icon('document-plus',28,GOLD)} Loan Application</div>
  <div style="font-size:0.88rem;color:{TEXT2};margin-top:3px;">
    Submit a new application for ML-based credit assessment and AI advisory generation.</div>
</div>
""", unsafe_allow_html=True)

# ── If already submitted show success card ────────────────────────────────────
if st.session_state.get("last_submitted_id"):
    sid = st.session_state["last_submitted_id"]
    outcome = st.session_state.get("last_outcome", "approved")
    prob    = st.session_state.get("last_prob", 0.0)
    badge   = (f'<span class="badge-approved">✓ Approved</span>'
               if outcome == 'approved' else
               f'<span class="badge-rejected">✗ Rejected</span>')

    st.markdown(f"""
    <div class="submit-success">
      <div style="margin-bottom:0.7rem;">
        {f"{_icon('check-circle',40,SUCCESS)}" if outcome == 'approved' else f"{_icon('x-circle',40,DANGER)}"}</div>
      <div style="font-size:1.1rem;font-weight:700;color:{TEXT};margin-bottom:0.3rem;">
        Application Submitted Successfully</div>
      <div style="margin-bottom:0.8rem;">{badge}</div>
      <div style="font-size:0.84rem;color:{TEXT2};margin-bottom:1rem;">
        Application ID: <code style="color:{GOLD_LT};background:rgba(196,168,122,0.10);
        padding:2px 8px;border-radius:5px;">APP-{sid}</code>
        &nbsp;·&nbsp; Approval Probability: <b style="color:{GOLD_LT}">{prob*100:.1f}%</b>
      </div>
      <div style="font-size:0.8rem;color:{TEXT3};">
        Navigate to AI Advisory to generate a full explanation report.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns([1, 3])
    with col_a:
        if st.button("New Application", use_container_width=True):
            st.session_state.pop("last_submitted_id", None)
            st.rerun()
    st.stop()

# ── Form ──────────────────────────────────────────────────────────────────────
with st.form("loan_form", clear_on_submit=False):

    # ── Section 1: Demographics ────────────────────────────────────────────
    st.markdown(f'<div class="form-section">{_icon("user",14,GOLD_LT)} Demographic Profile</div>', unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    with d1:
        age = st.number_input("Age", min_value=18, max_value=80, value=35, step=1,
                              help="Applicant's current age in years.")
    with d2:
        gender = st.selectbox("Gender", ["male", "female", "other"],
                              format_func=str.capitalize)
    with d3:
        education = st.selectbox("Education Level",
            ["High School", "Diploma", "Associate", "Bachelor", "Master", "Doctor"])

    # ── Section 2: Financial Profile ───────────────────────────────────────
    st.markdown(f'<div class="form-section">{_icon("banknotes",14,GOLD_LT)} Financial Profile</div>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    with f1:
        income = st.number_input("Annual Income (R)", min_value=5_000, max_value=500_000,
                                 value=55_000, step=1_000, format="%d",
                                 help="Total gross annual income in ZAR.")
    with f2:
        emp_exp = st.number_input("Employment Experience (years)", min_value=0, max_value=50,
                                  value=5, step=1)
    with f3:
        home_ownership = st.selectbox("Home Ownership",
            ["RENT", "MORTGAGE", "OWN", "OTHER"],
            format_func=lambda x: {"RENT":"Renting","MORTGAGE":"Mortgage","OWN":"Own","OTHER":"Other"}[x])

    # ── Section 3: Loan Details ────────────────────────────────────────────
    st.markdown(f'<div class="form-section">{_icon("building-library",14,GOLD_LT)} Loan Details</div>', unsafe_allow_html=True)
    l1, l2, l3 = st.columns(3)
    with l1:
        loan_amnt = st.number_input("Loan Amount (R)", min_value=500, max_value=100_000,
                                    value=12_000, step=500, format="%d")
    with l2:
        loan_intent = st.selectbox("Loan Purpose",
            ["PERSONAL","EDUCATION","MEDICAL","VENTURE","HOME_IMPROVEMENT","DEBTCONSOLIDATION"],
            format_func=lambda x: {
                "PERSONAL":"Personal","EDUCATION":"Education","MEDICAL":"Medical",
                "VENTURE":"Business Venture","HOME_IMPROVEMENT":"Home Improvement",
                "DEBTCONSOLIDATION":"Debt Consolidation"}[x])
    with l3:
        loan_grade = st.selectbox("Loan Grade", ["A","B","C","D","E","F","G"],
                                  index=2, help="Credit grade assigned by lender (A=best).")

    l4, l5, l6 = st.columns(3)
    with l4:
        loan_int_rate = st.slider("Interest Rate (%)", min_value=4.0, max_value=28.0,
                                  value=12.5, step=0.25, format="%.2f%%")
    with l5:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850,
                                       value=650, step=1,
                                       help="Bureau credit score (300=Poor, 850=Exceptional).")
    with l6:
        cred_hist = st.number_input("Credit History Length (years)", min_value=0.0,
                                    max_value=40.0, value=5.0, step=0.5)

    l7, l8 = st.columns(2)
    with l7:
        prev_defaults = st.radio("Previous Loan Default on Record?", ["No", "Yes"],
                                 horizontal=True)
    with l8:
        st.markdown("<br>", unsafe_allow_html=True)
        loan_pct = round(loan_amnt / max(income, 1), 4)
        st.markdown(f"""
        <div style="background:rgba(196,168,122,0.07);border:1px solid {BORDER};
        border-radius:10px;padding:0.7rem 1rem;">
          <div style="font-size:0.72rem;color:{TEXT3};text-transform:uppercase;
          letter-spacing:0.06em;">Calculated Loan-to-Income Ratio</div>
          <div style="font-size:1.4rem;font-weight:700;color:{GOLD_LT};margin-top:2px;">
            {loan_pct*100:.1f}%</div>
          <div style="font-size:0.72rem;color:{TEXT3};">
            {'Above 30% threshold' if loan_pct > 0.30 else 'Within acceptable range'}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("Submit Application for Assessment",
                                      use_container_width=False)

# ── Process submission ────────────────────────────────────────────────────────
if submitted:
    from utils.mock_data import get_data
    import numpy as np

    new_id = str(uuid.uuid4())[:8].upper()
    defaults_bool = prev_defaults == "Yes"

    # Mock scoring
    grade_enc = {'A':6,'B':5,'C':4,'D':3,'E':2,'F':1,'G':0}
    score = (
        (credit_score - 300) / 550 * 0.38 +
        (1 - loan_pct) * 0.22 +
        (not defaults_bool) * 0.22 +
        (grade_enc.get(loan_grade, 3) / 6) * 0.10 +
        min(emp_exp, 20) / 20 * 0.08
    ) + np.random.normal(0, 0.04)
    prob = float(np.clip(score, 0.02, 0.98))
    outcome = "approved" if prob > 0.52 else "rejected"

    # Build record for session state
    new_row = {
        'id': new_id, 'person_age': float(age), 'person_gender': gender,
        'person_education': education, 'person_income': float(income),
        'person_emp_exp': int(emp_exp), 'person_home_ownership': home_ownership,
        'loan_amnt': float(loan_amnt), 'loan_intent': loan_intent,
        'loan_grade': loan_grade, 'loan_int_rate': float(loan_int_rate),
        'loan_percent_income': loan_pct, 'cb_person_cred_hist_length': float(cred_hist),
        'credit_score': int(credit_score), 'previous_loan_defaults_on_file': defaults_bool,
        'created_at': datetime.now(), 'predicted_outcome': outcome,
        'approval_probability': round(prob, 4),
        'risk_tier': 'Low' if prob >= 0.70 else ('Medium' if prob >= 0.45 else 'High'),
    }
    if 'new_applications' not in st.session_state:
        st.session_state['new_applications'] = []
    st.session_state['new_applications'].append(new_row)
    st.session_state['last_submitted_id'] = new_id
    st.session_state['last_outcome']      = outcome
    st.session_state['last_prob']         = prob
    st.session_state['selected_customer'] = new_id
    st.rerun()

# ── Sidebar tip ───────────────────────────────────────────────────────────────
st.sidebar.markdown(f"""
<div style="background:rgba(196,168,122,0.07);border:1px solid rgba(196,168,122,0.15);
border-radius:10px;padding:0.9rem;margin-top:1rem;">
  <div style="font-size:0.75rem;color:{GOLD_LT};font-weight:600;margin-bottom:0.4rem;
    display:flex;align-items:center;gap:0.4rem;">
    {_icon('light-bulb',13,GOLD_LT)} Application Tips</div>
  <div style="font-size:0.73rem;color:{TEXT2};line-height:1.6;">
    • Credit score &gt; 670 significantly improves approval odds<br>
    • Keep loan-to-income ratio below 30%<br>
    • Longer credit history is a positive signal<br>
    • Stable employment (3+ years) reduces risk tier
  </div>
</div>
""", unsafe_allow_html=True)
