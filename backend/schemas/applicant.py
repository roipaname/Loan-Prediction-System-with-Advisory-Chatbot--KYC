from typing import Optional

from pydantic import BaseModel, Field

# caps keep inputs within the training data's range (loan_amnt maxed out
# around R104k, ratio never exceeded 0.66) — far beyond that the model just
# extrapolates and gives confidently wrong predictions. ratio cap is enforced
# in the /applicants router since it needs both fields together.
MAX_LOAN_AMNT = 300_000
MAX_PERSON_INCOME = 10_000_000
MAX_LOAN_TO_INCOME_RATIO = 1.0


class ApplicationCreate(BaseModel):
    person_age: int
    person_gender: str
    person_education: str
    person_income: float = Field(..., gt=0, le=MAX_PERSON_INCOME)
    person_emp_exp: int
    person_home_ownership: str
    loan_amnt: float = Field(..., gt=0, le=MAX_LOAN_AMNT)
    loan_intent: str
    loan_grade: str
    loan_int_rate: float
    cb_person_cred_hist_length: float
    credit_score: int
    previous_loan_defaults_on_file: bool
    loan_percent_income: Optional[float] = None


class ApplicationResult(BaseModel):
    display_code: str
    outcome: str
    probability: float
    risk_tier: str
