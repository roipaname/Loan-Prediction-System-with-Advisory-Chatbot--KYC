"""
Scores a brand-new loan application end to end: feature engineering, DB
insert, model inference, LIME attribution, prediction insert.
"""
from __future__ import annotations

from typing import Any, Dict

from database import operations as ops
from database.schemas import (
    CreditScoreTierEnum,
    EmploymentStabilityEnum,
    GenderEnum,
    HomeOwnerShipEnum,
    IncomeBucketEnum,
    LoanGradeEnum,
    LoanIntentEnum,
    ModelAlgorithmEnum,
    PersonEducationEnum,
    PredictionOutcomeEnum,
)
from backend.deps import get_classifier, get_context_builder
from backend.services.feature_engineering import engineer_row
from backend.services.lime_service import explain_row
from backend.services.reference_stats import get_reference_stats
from src.ai_advisor.loan_context_builder import _prepare_single_row


def _applicant_orm_kwargs(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "person_age":                     raw["person_age"],
        "person_gender":                  GenderEnum(raw["person_gender"]),
        "person_education":               PersonEducationEnum(raw["person_education"]),
        "person_income":                  raw["person_income"],
        "person_emp_exp":                 raw["person_emp_exp"],
        "person_home_ownership":          HomeOwnerShipEnum(raw["person_home_ownership"]),
        "loan_amnt":                      raw["loan_amnt"],
        "loan_intent":                    LoanIntentEnum(raw["loan_intent"]),
        "loan_grade":                     LoanGradeEnum(raw["loan_grade"]),
        "loan_int_rate":                  raw["loan_int_rate"],
        "loan_percent_income":            raw["loan_percent_income"],
        "cb_person_cred_hist_length":     raw["cb_person_cred_hist_length"],
        "credit_score":                   raw["credit_score"],
        "previous_loan_defaults_on_file": raw["previous_loan_defaults_on_file"],
        "source_split":                   "walk_in",
    }


def _features_orm_kwargs(engineered: Dict[str, Any]) -> Dict[str, Any]:
    kw = dict(engineered)
    kw["credit_score_tier"]     = CreditScoreTierEnum(kw["credit_score_tier"])
    kw["income_bucket"]         = IncomeBucketEnum(kw["income_bucket"])
    kw["employment_stability"]  = EmploymentStabilityEnum(kw["employment_stability"])
    return kw


def score_new_application(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    raw: dict of the Customer Application form fields (see
    backend/schemas/applicant.py::ApplicationCreate for the exact shape).

    Returns {display_code, outcome, probability, risk_tier}.
    """
    raw = dict(raw)
    raw.setdefault(
        "loan_percent_income",
        round(raw["loan_amnt"] / max(raw["person_income"], 1), 4),
    )

    engineered = engineer_row(raw, get_reference_stats())

    applicant = ops.create_applicant(_applicant_orm_kwargs(raw))
    ops.create_features(applicant.id, _features_orm_kwargs(engineered))

    merged_row = {**raw, **engineered}
    ctx = get_context_builder().build(feature_row=merged_row)

    clf = get_classifier()
    x = _prepare_single_row(merged_row, clf.feature_names_)
    attribution = explain_row(clf, x)

    risk_tier = ctx["prediction"]["risk_tier"].replace(" Risk", "")

    model = ops.get_champion_model()
    if model is None:
        model = ops.create_model({
            "algorithm":   ModelAlgorithmEnum(ctx["model_algorithm"]),
            "is_champion": True,
        })

    ops.create_prediction({
        "applicant_id":         applicant.id,
        "model_id":             model.id,
        "predicted_outcome":    PredictionOutcomeEnum(ctx["prediction"]["outcome"]),
        "approval_probability": ctx["prediction"]["probability"],
        "risk_tier":            risk_tier,
        "shap_values":          attribution,
        "top_shap_features":    list(attribution.keys())[:10],
    })

    return {
        "display_code": applicant.display_code,
        "outcome":       ctx["prediction"]["outcome"],
        "probability":   ctx["prediction"]["probability"],
        "risk_tier":     risk_tier,
    }
