import pandas as pd
import numpy as np
from helpers.file_loader import load_pandas_data
from database.schemas import IncomeBucketEnum,CreditScoreTierEnum
from config.settings import RAW_DATA_DIR,PROCESSED_DATA_DIR

CSV_DATA=RAW_DATA_DIR /'loan_data.csv'

def feature_engineering(df:pd.DataFrame)->pd.DataFrame:
    df=df.copy()

    # --- Derived Financial Ratios ---
    df['Debt_to_Income_Ratio']=df['loan_amnt'] / (df['person_income'] + 1)
    df['loan_to_income_ratio'] = df['loan_amnt'] / (df['person_income'] + 1)
    df['credit_history_to_age_ratio'] = df['cb_person_cred_hist_length'] / (df['person_age'] + 1)

    # --- Risk Flags ---

    df['is_high_risk']=(
        (df['loan_int_rate']>df['loan_int_rate'].median()) |
        (df['person_income'] < df['person_income'].median()) |
        (df['previous_loan_defaults_on_file'] == 'Yes')
    ).astype(int)

    # --- Income Buckets ---
    df['income_bucket']=pd.qcut(df['person_income'],q=len(IncomeBucketEnum),labels=[str(member.value) for member in IncomeBucketEnum])

    # --- Employment Stability ---
    df['employment_stability'] = np.where(df['person_emp_exp'] > 5, 'stable', 'unstable')
    # Credit score tier (interpretable for RAG explanations too)

    df['credit_score_tier']=pd.cut(df['credit_score'],bins=[0,580,670,740,800,850],labels=[str(member.value) for member in CreditScoreTierEnum])

    # Interaction: low credit score AND high interest rate = compounded risk
    df['credit_risk_interaction']=(
        (df['credit_score'] < 580).astype(int) * (df['loan_int_rate'] > df['loan_int_rate'].median()).astype(int)
    )

    # Monthly loan burden 
    df['monthly_loan_burden'] = df['loan_amnt'] / 36  # simplified

    # Affordability ratio: monthly burden vs monthly income
    df['monthly_income'] = df['person_income'] / 12
    df['affordability_ratio'] = df['monthly_loan_burden'] / (df['monthly_income'] + 1)

    # Flag: borrowing more than 30% of annual income (stress threshold)
    df['high_loan_burden_flag'] = (df['loan_percent_income'] > 0.30).astype(int)

    # Young borrower with no employment history — elevated risk profile
    df['young_inexperienced'] = (
    (df['person_age'] < 25) & (df['person_emp_exp'] == 0)
    ).astype(int)

    # Employment experience relative to age (how much of their life working)
    df['emp_to_age_ratio'] = df['person_emp_exp'] / (df['person_age'] + 1)

    # Loan amount relative to age (older borrowers often borrow more responsibly)
    df['loan_per_age'] = df['loan_amnt'] / (df['person_age'] + 1)


    # Short credit history flag (thin file = higher uncertainty)
    df['thin_credit_file'] = (df['cb_person_cred_hist_length'] < 2).astype(int)

    # Score-to-history ratio: high score in short time = positive signal
    df['score_per_history_year'] = df['credit_score'] / (df['cb_person_cred_hist_length'] + 1)

    df['composite_risk_score'] = (
    (1 - df['credit_score'] / 850) * 0.35 +          # credit quality
    df['loan_percent_income'] * 0.25 +                 # debt burden
    (1 / (df['person_emp_exp'] + 1)) * 0.20 +         # employment instability
    df['thin_credit_file'] * 0.10 +                    # thin file penalty
    (df['previous_loan_defaults_on_file'] == 'Yes').astype(int) * 0.10)

    ownership_score = {'OWN': 3, 'MORTGAGE': 2, 'RENT': 1, 'OTHER': 0}
    df['homeownership_score'] = df['person_home_ownership'].map(ownership_score)

    # Ownership × income: stable home + high income = lower risk
    df['stability_income_interaction'] = df['homeownership_score'] * np.log1p(df['person_income'])

    # Map intent to a risk tier (domain-informed)
    intent_risk_map = {
    'EDUCATION': 0,
    'HOME_IMPROVEMENT': 1,
    'VENTURE': 2,
    'PERSONAL': 3,
    'MEDICAL': 4,
    'DEBTCONSOLIDATION': 5}
    df['intent_risk_score'] = df['loan_intent'].map(intent_risk_map)
    """

    # --- Encode categorical variables ---
    categorical_cols = [
        'person_home_ownership',
        'loan_intent',
        'loan_grade',
        'previous_loan_defaults_on_file',
        'income_bucket',
        'employment_stability',
        'person_education'
    ]

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    """

    return df


if __name__=="__main__":
    df=load_pandas_data(CSV_DATA)
    print(len(df))
    processed_df=feature_engineering(df)
    processed_df.to_csv(PROCESSED_DATA_DIR/'processed.csv')

