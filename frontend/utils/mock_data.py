"""
utils/mock_data.py
Generates realistic mock data matching the LAPAS SQLAlchemy schema.
180 applicants with full feature sets, predictions, SHAP values and RAG explanations.
"""

import numpy as np
import pandas as pd
import uuid
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

N = 180

# enum value pools
GENDERS         = ['male', 'female', 'other']
EDUCATIONS      = ['High School', 'Bachelor', 'Diploma', 'Associate', 'Master', 'Doctor']
HOME_OWN        = ['MORTGAGE', 'OWN', 'RENT', 'OTHER']
INTENTS         = ['DEBTCONSOLIDATION', 'EDUCATION', 'HOME_IMPROVEMENT',
                   'MEDICAL', 'PERSONAL', 'VENTURE']
GRADES          = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

INTENT_LABELS   = {
    'DEBTCONSOLIDATION': 'Debt Consolidation',
    'EDUCATION':         'Education',
    'HOME_IMPROVEMENT':  'Home Improvement',
    'MEDICAL':           'Medical',
    'PERSONAL':          'Personal',
    'VENTURE':           'Venture',
}

_INTENT_ICON_NAMES = {
    'DEBTCONSOLIDATION': 'credit-card',
    'EDUCATION':         'academic-cap',
    'HOME_IMPROVEMENT':  'home',
    'MEDICAL':           'heart',
    'PERSONAL':          'user',
    'VENTURE':           'rocket-launch',
}

RISK_COLORS     = {'Low': '#5C8C6A', 'Medium': '#C4A87A', 'High': '#8C5E62'}
GRADE_ORDER     = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}

# core generator
def _build_raw() -> pd.DataFrame:
    ages      = np.random.normal(38, 11, N).clip(21, 70).round(1)
    genders   = np.random.choice(GENDERS, N, p=[0.47, 0.50, 0.03])
    edus      = np.random.choice(EDUCATIONS, N, p=[0.18, 0.38, 0.12, 0.10, 0.17, 0.05])
    incomes   = np.random.lognormal(12.21, 0.55, N).clip(50_000, 900_000).round(2)
    emp_exp   = np.random.randint(0, 26, N)
    home_own  = np.random.choice(HOME_OWN, N, p=[0.42, 0.18, 0.36, 0.04])
    loan_amt  = np.random.lognormal(10.08, 0.75, N).clip(1_500, 105_000).round(2)
    intents   = np.random.choice(INTENTS, N, p=[0.20, 0.18, 0.15, 0.12, 0.25, 0.10])
    grades    = np.random.choice(GRADES,  N, p=[0.18, 0.28, 0.25, 0.16, 0.08, 0.04, 0.01])
    int_rates = np.random.normal(12.5, 4.5, N).clip(4.5, 28).round(2)
    scores    = np.random.normal(640, 95, N).clip(300, 850).astype(int)
    cred_hist = np.random.normal(5.5, 3.5, N).clip(0.5, 30).round(1)
    defaults  = np.random.binomial(1, 0.22, N).astype(bool)
    loan_pct  = (loan_amt / incomes).clip(0.01, 0.95).round(4)

    base = datetime(2025, 1, 1)
    dates = [base + timedelta(days=random.randint(0, 530)) for _ in range(N)]

    return pd.DataFrame({
        'id':                           [str(uuid.uuid4())[:8].upper() for _ in range(N)],
        'person_age':                   ages,
        'person_gender':                genders,
        'person_education':             edus,
        'person_income':                incomes,
        'person_emp_exp':               emp_exp,
        'person_home_ownership':        home_own,
        'loan_amnt':                    loan_amt,
        'loan_intent':                  intents,
        'loan_grade':                   grades,
        'loan_int_rate':                int_rates,
        'loan_percent_income':          loan_pct,
        'cb_person_cred_hist_length':   cred_hist,
        'credit_score':                 scores,
        'previous_loan_defaults_on_file': defaults,
        'created_at':                   dates,
    })


def _add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['debt_to_income_ratio']       = (df['loan_amnt'] / df['person_income']).round(6)
    df['loan_to_income_ratio']       = (df['loan_amnt'] / df['person_income']).round(6)
    df['credit_hist_to_age_ratio']   = (df['cb_person_cred_hist_length'] / df['person_age']).round(6)
    df['monthly_income']             = (df['person_income'] / 12).round(2)
    df['monthly_loan_burden']        = (df['loan_amnt'] * df['loan_int_rate'] / 100 / 12).round(2)
    df['affordability_ratio']        = (df['monthly_loan_burden'] / df['monthly_income']).round(6)
    df['emp_to_age_ratio']           = (df['person_emp_exp'] / df['person_age'].clip(1)).round(6)
    df['young_inexperienced']        = (df['person_age'] < 25) & (df['person_emp_exp'] == 0)
    df['high_loan_burden_flag']      = df['loan_percent_income'] > 0.30
    df['thin_credit_file']           = df['cb_person_cred_hist_length'] < 2

    def score_tier(s):
        if s < 580: return 'Poor'
        if s < 670: return 'Fair'
        if s < 740: return 'Good'
        if s < 800: return 'Very Good'
        return 'Exceptional'

    df['credit_score_tier'] = df['credit_score'].apply(score_tier)

    def income_bucket(i):
        if i < 140_807:  return 'low'
        if i < 200_000:  return 'mid_low'
        if i < 285_733:  return 'medium'
        return 'high'

    df['income_bucket'] = df['person_income'].apply(income_bucket)

    grade_enc = {'A': 6, 'B': 5, 'C': 4, 'D': 3, 'E': 2, 'F': 1, 'G': 0}
    df['grade_encoded']          = df['loan_grade'].map(grade_enc)
    df['employment_stability']   = np.where(df['person_emp_exp'] >= 3, 'stable', 'unstable')

    own_enc = {'OTHER': 0, 'RENT': 1, 'MORTGAGE': 2, 'OWN': 3}
    df['homeownership_score']    = df['person_home_ownership'].map(own_enc)

    df['composite_risk_score'] = (
        (1 - (df['credit_score'] - 300) / 550) * 0.30 +
        df['loan_percent_income'].clip(0, 1) * 0.25 +
        df['previous_loan_defaults_on_file'].astype(float) * 0.25 +
        df['debt_to_income_ratio'].clip(0, 2) / 2 * 0.20
    ).clip(0, 1).round(6)

    return df


def _add_outcome(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    approval_score = (
        (df['credit_score'] - 300) / 550 * 0.38 +
        (1 - df['loan_percent_income'].clip(0, 1)) * 0.22 +
        (~df['previous_loan_defaults_on_file']).astype(float) * 0.22 +
        (df['grade_encoded'] / 6) * 0.10 +
        (df['person_emp_exp'].clip(0, 20) / 20) * 0.08
    ) + np.random.normal(0, 0.06, len(df))

    df['predicted_outcome']    = np.where(approval_score > 0.52, 'approved', 'rejected')
    df['approval_probability'] = approval_score.clip(0.02, 0.98).round(4)

    def risk_tier(prob):
        if prob >= 0.70: return 'Low'
        if prob >= 0.45: return 'Medium'
        return 'High'

    df['risk_tier'] = df['approval_probability'].apply(risk_tier)
    return df


def _add_shap(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rows = []
    for _, r in df.iterrows():
        s = {
            'credit_score':                     round((r['credit_score'] - 580) / 300 * 0.32, 4),
            'loan_percent_income':              round(-(r['loan_percent_income'] - 0.20) * 0.75, 4),
            'previous_loan_defaults_on_file':   round(-0.18 if r['previous_loan_defaults_on_file'] else 0.09, 4),
            'person_income':                    round((np.log(max(r['person_income'], 1)) - np.log(200_000)) / 10 * 0.18, 4),
            'loan_int_rate':                    round(-(r['loan_int_rate'] - 10) / 18 * 0.14, 4),
            'person_emp_exp':                   round((r['person_emp_exp'] - 5) / 25 * 0.10, 4),
            'debt_to_income_ratio':             round(-(r['debt_to_income_ratio'] - 0.30) * 0.20, 4),
            'cb_person_cred_hist_length':       round((r['cb_person_cred_hist_length'] - 4) / 25 * 0.08, 4),
            'loan_amnt':                        round(-(r['loan_amnt'] - 24_000) / 134_000 * 0.09, 4),
            'person_age':                       round((r['person_age'] - 35) / 30 * 0.05, 4),
        }
        # Sort by absolute value descending
        s_sorted = dict(sorted(s.items(), key=lambda x: abs(x[1]), reverse=True))
        rows.append(s_sorted)
    df['shap_values']       = rows
    df['top_shap_features'] = [list(s.keys())[:5] for s in rows]
    return df


def _add_explanations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def make_explanation(row):
        o = row['predicted_outcome']
        cs = row['credit_score']
        tier = row['credit_score_tier']
        inc = row['person_income']
        lpi = row['loan_percent_income']
        risk = row['risk_tier']
        prev = row['previous_loan_defaults_on_file']
        dti = row['debt_to_income_ratio']
        grade = row['loan_grade']
        emp = row['person_emp_exp']
        shap = row['shap_values']
        top_feat = sorted(shap.items(), key=lambda x: abs(x[1]), reverse=True)[:3]

        if o == 'approved':
            return f"""## Loan Application Assessment: APPROVED

**Risk Classification:** {risk} Risk | **Credit Score:** {cs} ({tier})

### Why This Application Was Approved

Your loan application has been assessed and approved based on a comprehensive analysis of your financial profile. The predictive model assigned an approval probability of **{row['approval_probability']*100:.1f}%**, placing this application in the **{risk.lower()} risk** category.

**Primary drivers of approval:**
- Your credit score of **{cs}** ({tier}) is the strongest positive signal, contributing significantly to the model's confidence.
- A loan-to-income ratio of **{lpi*100:.1f}%** is within acceptable bounds, indicating manageable repayment capacity.
- An annual income of **R{inc:,.0f}** provides a solid financial foundation for the requested loan amount.
{"- A clean repayment history with no previous defaults further strengthens the application." if not prev else ""}

**Top SHAP Feature Attributions:**
{chr(10).join(f"- {f.replace('_',' ').title()}: {'↑' if v > 0 else '↓'} {abs(v):.4f}" for f, v in top_feat)}

### Regulatory Grounding
Under FATF KYC Section 4.2, the customer due diligence assessment has been completed satisfactorily. In accordance with Basel III Article 147, the debt-to-income ratio of {dti:.2f} falls within the standard exposure threshold.

### Recommended Next Steps for Applicant
1. Maintain or improve your credit score by keeping credit utilisation below 30%.
2. Continue building your employment record — stability strengthens future applications.
3. Avoid taking on additional high-interest debt before loan drawdown.
"""
        else:
            return f"""## Loan Application Assessment: REJECTED

**Risk Classification:** {risk} Risk | **Credit Score:** {cs} ({tier})

### Why This Application Was Not Approved

The application was assessed and rejected based on a combination of risk factors identified by the predictive model. The approval probability was **{row['approval_probability']*100:.1f}%**, which falls below the minimum threshold for loan grade **{grade}**.

**Primary drivers of rejection:**
{f"- A credit score of **{cs}** ({tier}) indicates elevated credit risk relative to the requested loan amount." if cs < 650 else ""}
{f"- The loan-to-income ratio of **{lpi*100:.1f}%** suggests the monthly repayment burden may exceed sustainable limits." if lpi > 0.25 else ""}
{f"- A previous loan default on record is a significant risk flag that materially reduces approval probability." if prev else ""}
{f"- The debt-to-income ratio of **{dti:.2f}** indicates existing financial obligations that reduce residual repayment capacity." if dti > 0.35 else ""}

**Top SHAP Feature Attributions:**
{chr(10).join(f"- {f.replace('_',' ').title()}: {'↑' if v > 0 else '↓'} {abs(v):.4f}" for f, v in top_feat)}

### Regulatory Grounding
Under Basel III Article 147, the current risk profile exceeds the acceptable exposure threshold. FATF KYC Section 5.1 requires additional due diligence for applications in the high-risk tier.

### Best Next Steps to Improve Eligibility
1. **Improve credit score** — Target a score above 670 (currently {cs}). Pay down existing balances and ensure on-time payments for 6-12 months.
2. **Reduce loan amount** — Consider applying for R{max(1000, row["loan_amnt"]*0.65):,.0f} (35% less) to bring the loan-to-income ratio below 20%.
3. **Increase income documentation** — Additional income streams or a co-applicant can strengthen the income profile.
4. **Address defaults** — If a previous default exists, obtain a letter of explanation and evidence of resolved obligation.
5. **Re-apply in 6-12 months** — Allow time for credit profile improvements to register with credit bureaus.
"""

    df['llm_response'] = df.apply(make_explanation, axis=1)
    return df


# public api
_CACHE: pd.DataFrame | None = None


def get_data() -> pd.DataFrame:
    """Return the full mock dataset (cached after first call)."""
    global _CACHE
    if _CACHE is None:
        df = _build_raw()
        df = _add_engineered(df)
        df = _add_outcome(df)
        df = _add_shap(df)
        df = _add_explanations(df)
        _CACHE = df
    return _CACHE.copy()


def get_applicant(app_id: str) -> pd.Series | None:
    df = get_data()
    match = df[df['id'] == app_id]
    return match.iloc[0] if not match.empty else None


def intent_label(intent: str) -> str:
    return INTENT_LABELS.get(intent, intent)


def intent_icon(intent: str, size: int = 14) -> str:
    try:
        from styles.icons import icon as _svg
        return _svg(_INTENT_ICON_NAMES.get(intent, 'chart-bar'), size=size)
    except Exception:
        return '&#9632;'


def risk_color(tier: str) -> str:
    return RISK_COLORS.get(tier, '#A09890')
