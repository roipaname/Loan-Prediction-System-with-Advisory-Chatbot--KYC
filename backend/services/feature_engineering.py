"""
backend/services/feature_engineering.py
========================================
Single-row equivalent of database/feature_eng.py's per-column steps, used to
score a brand-new application submitted through the frontend. Reuses the same
constants (INTENT_RISK_MAP, HOMEOWNERSHIP_SCORE_MAP, CREDIT_SCORE_TIERS, ...)
and the same _score_to_tier() helper as the batch pipeline so a walk-in
applicant is engineered identically to a bulk-loaded one, except for the two
dataset-relative signals (credit_risk_interaction, is_high_risk) which use
the precomputed ReferenceStats instead of a live median/quantile.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from database.feature_eng import (
    CREDIT_SCORE_TIERS,
    HOMEOWNERSHIP_SCORE_MAP,
    INCOME_LOW_MAX,
    INCOME_LOW_MED_MAX,
    INCOME_MED_MAX,
    INTENT_RISK_MAP,
    PIPELINE_VERSION,
    RISK_WEIGHTS,
    _score_to_tier,
)
from backend.services.reference_stats import ReferenceStats, get_reference_stats


def _income_bucket(income: float) -> str:
    if income <= INCOME_LOW_MAX:
        return "low"
    if income <= INCOME_LOW_MED_MAX:
        return "mid_low"
    if income <= INCOME_MED_MAX:
        return "medium"
    return "high"


def engineer_row(raw: Dict[str, Any], ref_stats: ReferenceStats | None = None) -> Dict[str, Any]:
    """
    Compute every EngineeredFeatures column for one raw applicant dict.

    `raw` must contain: person_age, person_income, person_emp_exp,
    person_home_ownership, loan_amnt, loan_intent, loan_int_rate,
    loan_percent_income, cb_person_cred_hist_length, credit_score,
    previous_loan_defaults_on_file.
    """
    ref_stats = ref_stats or get_reference_stats()

    age            = float(raw["person_age"])
    income         = float(raw["person_income"])
    emp_exp        = int(raw["person_emp_exp"])
    home_ownership = str(raw["person_home_ownership"])
    loan_amnt      = float(raw["loan_amnt"])
    loan_intent    = str(raw["loan_intent"])
    loan_int_rate  = float(raw["loan_int_rate"])
    loan_pct       = float(raw["loan_percent_income"])
    cred_hist      = float(raw["cb_person_cred_hist_length"])
    credit_score   = int(raw["credit_score"])
    prev_default   = bool(raw["previous_loan_defaults_on_file"])

    monthly_income      = income / 12.0
    monthly_loan_burden = loan_amnt * (1 + loan_int_rate / 100) / 12.0
    affordability_ratio = max(0.0, 1.0 - (monthly_loan_burden / monthly_income)) if monthly_income > 0 else 0.0
    credit_history_to_age_ratio = (cred_hist / age) if age > 0 else 0.0
    emp_to_age_ratio    = (emp_exp / age) if age > 0 else 0.0
    loan_per_age        = (loan_amnt / age) if age > 0 else 0.0
    young_inexperienced = age < 25 and emp_exp == 0

    credit_score_tier      = _score_to_tier(credit_score)
    thin_credit_file       = cred_hist < 2
    score_per_history_year = (credit_score / cred_hist) if cred_hist > 0 else 0.0
    credit_risk_interaction = (
        loan_int_rate > ref_stats.median_loan_int_rate
        and credit_score < ref_stats.median_credit_score
    )

    income_bucket         = _income_bucket(income)
    high_loan_burden_flag = loan_pct > 0.30

    employment_stability = "stable" if emp_exp >= 2 else "unstable"

    homeownership_score = HOMEOWNERSHIP_SCORE_MAP.get(home_ownership, 0)
    stability_income_interaction = homeownership_score * np.log1p(income)

    intent_risk_score = INTENT_RISK_MAP.get(loan_intent.upper(), 2)

    signals = {
        "debt_to_income_ratio":    loan_pct > 0.40,
        "loan_to_income_ratio":    loan_pct > 0.40,
        "thin_credit_file":        thin_credit_file,
        "credit_risk_interaction": credit_risk_interaction,
        "high_loan_burden_flag":   high_loan_burden_flag,
        "is_default_on_file":      prev_default,
        "young_inexperienced":     young_inexperienced,
    }
    composite_risk_score = round(
        sum(weight for key, weight in RISK_WEIGHTS.items() if signals.get(key)), 6
    )
    is_high_risk = composite_risk_score >= ref_stats.high_risk_threshold

    return {
        "debt_to_income_ratio":         loan_pct,
        "loan_to_income_ratio":         loan_pct,
        "credit_history_to_age_ratio":  round(credit_history_to_age_ratio, 6),
        "affordability_ratio":          round(affordability_ratio, 6),
        "monthly_loan_burden":          round(monthly_loan_burden, 2),
        "monthly_income":               round(monthly_income, 2),
        "emp_to_age_ratio":             round(emp_to_age_ratio, 6),
        "loan_per_age":                 round(loan_per_age, 4),
        "young_inexperienced":          young_inexperienced,
        "credit_score_tier":            credit_score_tier,
        "thin_credit_file":             thin_credit_file,
        "score_per_history_year":       round(score_per_history_year, 4),
        "credit_risk_interaction":      credit_risk_interaction,
        "income_bucket":                income_bucket,
        "high_loan_burden_flag":        high_loan_burden_flag,
        "employment_stability":         employment_stability,
        "is_high_risk":                 is_high_risk,
        "composite_risk_score":         composite_risk_score,
        "homeownership_score":          homeownership_score,
        "stability_income_interaction": round(float(stability_income_interaction), 6),
        "intent_risk_score":            intent_risk_score,
        "pipeline_version":             PIPELINE_VERSION,
    }
