"""
pages/1_Customer_Application.py
Full loan application submission form matching the LoanApplicant SQLAlchemy model.
Organised in three sections: Demographic · Financial · Loan Details
"""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from styles.theme import inject, sidebar_logo, get_logo_image, GOLD, GOLD_LT, GOLD_DK
from styles.theme import TEXT, TEXT2, TEXT3, CARD, CARD2, BORDER, SUCCESS, SUCCESS_LT, DANGER, DANGER_LT
from styles.icons import icon as _icon
from utils import api_client
from utils.mock_data import intent_label, intent_icon
from utils.markdown_render import advisory_to_html

st.set_page_config(page_title="LAPAS – Application", page_icon=get_logo_image() or "L", layout="wide")
inject()
sidebar_logo()

# page header
st.markdown(f"""
<div style="margin-bottom:1.4rem;">
  <div style="display:flex;align-items:center;gap:0.6rem;font-size:1.6rem;font-weight:800;color:{TEXT};">
    {_icon('document-plus',28,GOLD)} Loan Application</div>
  <div style="font-size:0.88rem;color:{TEXT2};margin-top:3px;">
    Submit a new application for ML-based credit assessment and AI advisory generation.</div>
</div>
""", unsafe_allow_html=True)

# if already submitted show full outcome + ai advisory directly
if st.session_state.get("last_submitted_id"):
    sid     = st.session_state["last_submitted_id"]
    outcome = st.session_state.get("last_outcome", "approved")
    prob    = st.session_state.get("last_prob", 0.0)
    risk_tier = st.session_state.get("last_risk_tier", "N/A")
    approved  = outcome == "approved"
    badge     = (f'<span class="badge-approved">Approved</span>'
                 if approved else f'<span class="badge-rejected">Rejected</span>')
    risk_col  = {"Low": SUCCESS_LT, "Medium": GOLD_LT, "High": DANGER_LT}.get(risk_tier, TEXT2)

    st.markdown(f"""
    <div style="margin-bottom:1rem;display:flex;align-items:center;gap:0.6rem;">
      {_icon('check-circle',26,GOLD) if approved else _icon('x-circle',26,GOLD)}
      <span style="font-size:1.5rem;font-weight:800;color:{TEXT};">Application Outcome</span>
      <span style="font-size:0.84rem;color:{TEXT3};margin-left:0.8rem;">
        Application ID: APP-{sid}</span>
    </div>
    """, unsafe_allow_html=True)

    # outcome + profile card (mirrors ai advisory page's profile card)
    row = api_client.get_applicant_row(sid)
    _hdr_bg = "linear-gradient(135deg,#1E3228,#162A20)" if approved else "linear-gradient(135deg,#2A1A1C,#221518)"
    _hdr_border = "rgba(92,140,106,0.25)" if approved else "rgba(140,94,98,0.25)"

    if row is not None:
        profile_fields = [
            ("Age",          f"{int(row['person_age'])} yrs"),
            ("Income",       f"R{row['person_income']:,.0f}"),
            ("Credit Score", f"{int(row['credit_score'])} ({row['credit_score_tier']})"),
            ("Loan Amount",  f"R{row['loan_amnt']:,.0f}"),
            ("Purpose",      f"{intent_icon(row['loan_intent'])} {intent_label(row['loan_intent'])}"),
            ("Grade",        f"{row['loan_grade']} · {row['loan_int_rate']:.1f}%"),
        ]
    else:
        profile_fields = []

    _profile_cells = "".join(
        f'<div style="padding:0.5rem 0;">'
        f'<div style="font-size:0.66rem;color:{TEXT3};text-transform:uppercase;'
        f'letter-spacing:0.06em;margin-bottom:3px;">{lbl}</div>'
        f'<div style="font-size:0.9rem;font-weight:600;color:{TEXT};">{val}</div>'
        f'</div>'
        for lbl, val in profile_fields
    )
    _profile_grid = (
        f'<div style="display:grid;grid-template-columns:repeat(6,1fr);gap:0;'
        f'padding:1rem 1.4rem;">{_profile_cells}</div>'
        if profile_fields else ''
    )

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{CARD} 0%,{CARD2} 100%);
                border:1px solid {BORDER};border-radius:18px;overflow:hidden;
                margin-bottom:1.2rem;">
      <div style="background:{_hdr_bg};padding:1rem 1.4rem;border-bottom:1px solid {_hdr_border};
                  display:flex;justify-content:space-between;align-items:center;">
        <div>{badge}</div>
        <div style="text-align:right;">
          <div style="font-size:0.8rem;color:{risk_col};font-weight:600;">
            {risk_tier} Risk · {prob*100:.1f}% approval probability</div>
        </div>
      </div>
      {_profile_grid}
    </div>
    """, unsafe_allow_html=True)

    # top lime drivers (compact)
    if row is not None and row.get('shap_values'):
        from src.ai_advisor.loan_context_builder import _readable_name as _nice_name
        top5 = sorted(row['shap_values'].items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        _driver_rows = ""
        for k, v in top5:
            col = SUCCESS_LT if v > 0 else DANGER_LT
            sign = "+" if v > 0 else ""
            _driver_rows += (
                f'<div style="display:flex;justify-content:space-between;'
                f'padding:0.4rem 0;border-bottom:1px solid {BORDER};font-size:0.82rem;">'
                f'<span style="color:{TEXT2};">{_nice_name(k)}</span>'
                f'<span style="color:{col};font-weight:600;font-family:monospace;">'
                f'{sign}{v:.4f}</span></div>'
            )
        st.markdown(
            f'<div class="info-panel" style="margin-bottom:1.2rem;">'
            f'<div class="info-panel-title">{_icon("trophy",14,GOLD_LT)} Top Decision Drivers (LIME)</div>'
            f'{_driver_rows}</div>',
            unsafe_allow_html=True,
        )

    # ai advisory report, rendered (not raw markdown)
    st.markdown(f"""
    <div style="margin:1.2rem 0 0.6rem 0;display:flex;align-items:center;gap:0.5rem;">
      {_icon('cpu-chip',18,GOLD)}
      <span style="font-size:1.05rem;font-weight:700;color:{TEXT};">AI Advisory Explanation</span>
    </div>
    """, unsafe_allow_html=True)

    if "last_advisory_report" not in st.session_state:
        try:
            with st.spinner("Generating policy-grounded advisory report…"):
                _result = api_client.generate_advisory(sid, retriever="tfidf")
            st.session_state["last_advisory_report"] = _result["report"]
        except api_client.BackendUnavailable as exc:
            st.session_state["last_advisory_report"] = None
            st.session_state["last_advisory_error"] = str(exc)

    if st.session_state.get("last_advisory_report"):
        st.markdown(advisory_to_html(st.session_state["last_advisory_report"]), unsafe_allow_html=True)
    else:
        st.error(f"Could not generate the advisory report: "
                 f"{st.session_state.get('last_advisory_error', 'backend unreachable')}")

    st.markdown(f'<div style="font-size:0.8rem;color:{TEXT3};margin-top:0.8rem;">'
                f'Visit AI Advisory for feature-attribution charts, retriever comparison, '
                f'next-step guidance, and PDF export.</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns([1, 3])
    with col_a:
        if st.button("New Application", use_container_width=True):
            for k in ("last_submitted_id", "last_outcome", "last_prob", "last_risk_tier",
                      "last_advisory_report", "last_advisory_error"):
                st.session_state.pop(k, None)
            st.rerun()
    st.stop()

# form
with st.form("loan_form", clear_on_submit=False):

    # section 1: demographics
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

    # section 2: financial profile
    st.markdown(f'<div class="form-section">{_icon("banknotes",14,GOLD_LT)} Financial Profile</div>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    with f1:
        income = st.number_input("Annual Income (R)", min_value=5_000, max_value=10_000_000,
                                 value=180_000, step=5_000, format="%d",
                                 help="Total gross annual income in ZAR. The model was trained on "
                                      "real applications with incomes mostly below ~R2.4 million.")
    with f2:
        emp_exp = st.number_input("Employment Experience (years)", min_value=0, max_value=50,
                                  value=5, step=1)
    with f3:
        home_ownership = st.selectbox("Home Ownership",
            ["RENT", "MORTGAGE", "OWN", "OTHER"],
            format_func=lambda x: {"RENT":"Renting","MORTGAGE":"Mortgage","OWN":"Own","OTHER":"Other"}[x])

    # section 3: loan details
    st.markdown(f'<div class="form-section">{_icon("building-library",14,GOLD_LT)} Loan Details</div>', unsafe_allow_html=True)
    l1, l2, l3 = st.columns(3)
    with l1:
        loan_amnt = st.number_input("Loan Amount (R)", min_value=500, max_value=300_000,
                                    value=25_000, step=1_000, format="%d",
                                    help="Capped at R300,000 (~3x the largest loan amount in the "
                                         "training data, ~R104,000) to keep predictions within a "
                                         "range the model can reliably reason about.")
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

# process submission
if submitted and loan_pct > 1.0:
    st.error(
        f"Loan amount is {loan_pct*100:.0f}% of annual income, which exceeds the maximum "
        f"supported loan-to-income ratio of 100%. The classifier was not trained on ratios "
        f"this extreme (training max was 66%) and its predictions there are unreliable. "
        f"Reduce the loan amount or increase income before resubmitting."
    )
elif submitted:
    defaults_bool = prev_defaults == "Yes"

    payload = {
        "person_age":                     int(age),
        "person_gender":                  gender,
        "person_education":               education,
        "person_income":                  float(income),
        "person_emp_exp":                 int(emp_exp),
        "person_home_ownership":          home_ownership,
        "loan_amnt":                      float(loan_amnt),
        "loan_intent":                    loan_intent,
        "loan_grade":                     loan_grade,
        "loan_int_rate":                  float(loan_int_rate),
        "loan_percent_income":            loan_pct,
        "cb_person_cred_hist_length":     float(cred_hist),
        "credit_score":                   int(credit_score),
        "previous_loan_defaults_on_file": defaults_bool,
    }

    try:
        with st.spinner("Running credit assessment…"):
            result = api_client.submit_application(payload)
        st.session_state['last_submitted_id'] = result['display_code']
        st.session_state['last_outcome']      = result['outcome']
        st.session_state['last_prob']         = result['probability']
        st.session_state['last_risk_tier']    = result['risk_tier']
        st.session_state['selected_customer'] = result['display_code']
        st.rerun()
    except api_client.BackendUnavailable as exc:
        st.error(f"Could not reach the scoring backend: {exc}")

# sidebar tip
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
<div style="background:rgba(140,94,98,0.07);border:1px solid rgba(140,94,98,0.18);
border-radius:10px;padding:0.9rem;margin-top:0.8rem;">
  <div style="font-size:0.75rem;color:{DANGER_LT};font-weight:600;margin-bottom:0.4rem;
    display:flex;align-items:center;gap:0.4rem;">
    {_icon('exclamation-triangle',13,DANGER_LT)} Model Range Note</div>
  <div style="font-size:0.73rem;color:{TEXT2};line-height:1.6;">
    The classifier was trained on real applications with loan amounts up to
    ~R104,000 and a loan-to-income ratio never exceeding 0.66. Loan amount is
    capped at R300,000 and the loan-to-income ratio at 100% of annual income
    so submissions stay within a range the model can reliably reason about.
  </div>
</div>
""", unsafe_allow_html=True)
