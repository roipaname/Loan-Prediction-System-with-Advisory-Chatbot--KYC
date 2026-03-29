import pandas as pd
import numpy as np
from helpers.file_loader import load_pandas_data
from database.schemas import IncomeBucketEnum


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
    df['income_bucket']=pd.qcut(df['persone_income'],q=len(IncomeBucketEnum),labels=[str(member.value) for member in IncomeBucketEnum])

    # --- Employment Stability ---
    df['employment_stability'] = np.where(df['person_emp_exp'] > 5, 'stable', 'unstable')
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




