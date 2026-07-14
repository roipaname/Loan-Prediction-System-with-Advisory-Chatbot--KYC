from typing import Optional

from pydantic import BaseModel


class ApplicationCreate(BaseModel):
    person_age: int
    person_gender: str
    person_education: str
    person_income: float
    person_emp_exp: int
    person_home_ownership: str
    loan_amnt: float
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
