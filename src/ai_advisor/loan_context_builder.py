"""
Builds the structured context dict advisor.py needs for a loan applicant:
given a DB applicant UUID or a pre-computed feature row, reconstructs the
training-time feature vector, runs it through the best trained classifier,
and returns {applicant, engineered, prediction, feature_importance,
query_text, model_algorithm}.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

import numpy as np
import pandas as pd
from loguru import logger as log

from config.settings import BEST_MODEL_PATH, RANDOM_STATE
from src.classifier.classifier import LoanClassifier

# feature schema — mirrors scripts/train_model.py exactly

_TARGET_COL = "loan_status"
_DROP_COLS   = frozenset([_TARGET_COL, "pipeline_version"])

_CATEGORICAL_COLS: List[str] = [
    "person_gender",
    "person_education",
    "person_home_ownership",
    "loan_intent",
    "credit_score_tier",
    "income_bucket",
    "employment_stability",
]

_BOOL_COLS: List[str] = [
    "previous_loan_defaults_on_file",
    "young_inexperienced",
    "thin_credit_file",
    "credit_risk_interaction",
    "high_loan_burden_flag",
    "is_high_risk",
]

_HIGH_RISK_THRESHOLD   = 0.40
_MEDIUM_RISK_THRESHOLD = 0.70


def _probability_to_tier(prob: float) -> str:
    """Convert P(approved) to a human-readable risk tier label."""
    if prob >= _MEDIUM_RISK_THRESHOLD:
        return "Low Risk"
    if prob >= _HIGH_RISK_THRESHOLD:
        return "Medium Risk"
    return "High Risk"


def _prepare_single_row(
    row: Dict[str, Any],
    feature_names: List[str],
) -> np.ndarray:
    """
    Transform a raw feature dict into the model-compatible float64 vector,
    mirroring prepare_features() in scripts/train_model.py. One-hot encoding
    uses drop_first=False here (a single row would otherwise lose all dummies
    for its category) and reindexes to the training feature_names_, so the
    dropped reference category ends up correctly represented as all-zeros.
    """
    df = pd.DataFrame([row])

    df = df.drop(columns=[c for c in _DROP_COLS if c in df.columns], errors="ignore")

    for col in _BOOL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(int)

    cats_present = [c for c in _CATEGORICAL_COLS if c in df.columns]
    if cats_present:
        df = pd.get_dummies(df, columns=cats_present, drop_first=False)

    df = df.apply(pd.to_numeric, errors="coerce")

    df = df.reindex(columns=feature_names, fill_value=0)

    return df.values.astype(np.float64)


_FEATURE_DISPLAY: Dict[str, str] = {
    "credit_score":                  "Credit Score",
    "loan_amnt":                     "Loan Amount",
    "person_income":                 "Annual Income",
    "loan_int_rate":                 "Interest Rate",
    "loan_percent_income":           "Loan as % of Income",
    "cb_person_cred_hist_length":    "Credit History Length",
    "person_emp_exp":                "Employment Experience",
    "person_age":                    "Applicant Age",
    "debt_to_income_ratio":          "Debt-to-Income Ratio",
    "loan_to_income_ratio":          "Loan-to-Income Ratio",
    "affordability_ratio":           "Affordability Ratio",
    "monthly_loan_burden":           "Monthly Loan Burden",
    "monthly_income":                "Monthly Income",
    "composite_risk_score":          "Composite Risk Score",
    "emp_to_age_ratio":              "Employment-to-Age Ratio",
    "credit_history_to_age_ratio":   "Credit History-to-Age Ratio",
    "score_per_history_year":        "Score Per History Year",
    "loan_per_age":                  "Loan Per Year of Age",
    "homeownership_score":           "Home Ownership Score",
    "stability_income_interaction":  "Stability x Income",
    "intent_risk_score":             "Loan Intent Risk Score",
    "is_high_risk":                  "Overall High-Risk Flag",
    "high_loan_burden_flag":         "High Loan Burden Flag",
    "thin_credit_file":              "Thin Credit File Flag",
    "young_inexperienced":           "Young and Inexperienced Flag",
    "credit_risk_interaction":       "Credit Risk Interaction Flag",
    "previous_loan_defaults_on_file": "Previous Loan Default on Record",
}


def _readable_name(feature: str) -> str:
    """Map a model feature name (post-OHE) to a human-readable label."""
    if feature in _FEATURE_DISPLAY:
        return _FEATURE_DISPLAY[feature]
    # OHE dummy, e.g. "person_gender_male" -> "Gender: Male"
    for col in _CATEGORICAL_COLS:
        prefix = f"{col}_"
        if feature.startswith(prefix):
            category_val = feature[len(prefix):].replace("_", " ").title()
            col_display  = col.replace("_", " ").title()
            return f"{col_display}: {category_val}"
    return feature.replace("_", " ").title()


def _enum_val(obj: Any) -> Any:
    """Extract the .value string from a SQLAlchemy enum member, or return as-is."""
    return obj.value if hasattr(obj, "value") else obj


def _build_applicant_section(app: Any) -> Dict[str, Any]:
    """Extract human-readable fields from a LoanApplicant ORM object."""
    return {
        "age":                           float(app.person_age or 0),
        "gender":                        _enum_val(app.person_gender),
        "education":                     _enum_val(app.person_education),
        "annual_income":                 float(app.person_income or 0),
        "employment_experience_years":   int(app.person_emp_exp or 0),
        "home_ownership":                _enum_val(app.person_home_ownership),
        "loan_amount":                   float(app.loan_amnt or 0),
        "loan_intent":                   _enum_val(app.loan_intent),
        "loan_grade":                    _enum_val(app.loan_grade),
        "interest_rate_pct":             float(app.loan_int_rate or 0),
        "loan_percent_of_income":        float(app.loan_percent_income or 0),
        "credit_history_years":          float(app.cb_person_cred_hist_length or 0),
        "credit_score":                  int(app.credit_score or 0),
        "previous_default_on_record":    bool(app.previous_loan_defaults_on_file),
    }


def _build_engineered_section(feat: Any) -> Dict[str, Any]:
    """Extract human-readable fields from an EngineeredFeatures ORM object."""
    return {
        "debt_to_income_ratio":          float(feat.debt_to_income_ratio or 0),
        "loan_to_income_ratio":          float(feat.loan_to_income_ratio or 0),
        "credit_history_to_age_ratio":   float(feat.credit_history_to_age_ratio or 0),
        "affordability_ratio":           float(feat.affordability_ratio or 0),
        "monthly_loan_burden":           float(feat.monthly_loan_burden or 0),
        "monthly_income":                float(feat.monthly_income or 0),
        "emp_to_age_ratio":              float(feat.emp_to_age_ratio or 0),
        "loan_per_age":                  float(feat.loan_per_age or 0),
        "young_inexperienced":           bool(feat.young_inexperienced),
        "credit_score_tier":             _enum_val(feat.credit_score_tier),
        "thin_credit_file":              bool(feat.thin_credit_file),
        "score_per_history_year":        float(feat.score_per_history_year or 0),
        "credit_risk_interaction":       bool(feat.credit_risk_interaction),
        "income_bucket":                 _enum_val(feat.income_bucket),
        "high_loan_burden_flag":         bool(feat.high_loan_burden_flag),
        "employment_stability":          _enum_val(feat.employment_stability),
        "is_high_risk":                  bool(feat.is_high_risk),
        "composite_risk_score":          float(feat.composite_risk_score or 0),
        "homeownership_score":           int(feat.homeownership_score or 0),
        "stability_income_interaction":  float(feat.stability_income_interaction or 0),
        "intent_risk_score":             int(feat.intent_risk_score or 0),
    }


def _build_feature_row_from_db(app: Any, feat: Any) -> Dict[str, Any]:
    """Merge a LoanApplicant + EngineeredFeatures ORM row into a dict matching
    the loan_features.csv column schema."""
    return {
        # from LoanApplicant
        "person_age":                       float(app.person_age or 0),
        "person_gender":                    _enum_val(app.person_gender),
        "person_education":                 _enum_val(app.person_education),
        "person_income":                    float(app.person_income or 0),
        "person_emp_exp":                   int(app.person_emp_exp or 0),
        "person_home_ownership":            _enum_val(app.person_home_ownership),
        "loan_amnt":                        float(app.loan_amnt or 0),
        "loan_intent":                      _enum_val(app.loan_intent),
        "loan_int_rate":                    float(app.loan_int_rate or 0),
        "loan_percent_income":              float(app.loan_percent_income or 0),
        "cb_person_cred_hist_length":       float(app.cb_person_cred_hist_length or 0),
        "credit_score":                     int(app.credit_score or 0),
        "previous_loan_defaults_on_file":   bool(app.previous_loan_defaults_on_file),
        # from EngineeredFeatures
        "debt_to_income_ratio":             float(feat.debt_to_income_ratio or 0),
        "loan_to_income_ratio":             float(feat.loan_to_income_ratio or 0),
        "credit_history_to_age_ratio":      float(feat.credit_history_to_age_ratio or 0),
        "affordability_ratio":              float(feat.affordability_ratio or 0),
        "monthly_loan_burden":              float(feat.monthly_loan_burden or 0),
        "monthly_income":                   float(feat.monthly_income or 0),
        "emp_to_age_ratio":                 float(feat.emp_to_age_ratio or 0),
        "loan_per_age":                     float(feat.loan_per_age or 0),
        "young_inexperienced":              bool(feat.young_inexperienced),
        "credit_score_tier":                _enum_val(feat.credit_score_tier),
        "thin_credit_file":                 bool(feat.thin_credit_file),
        "score_per_history_year":           float(feat.score_per_history_year or 0),
        "credit_risk_interaction":          bool(feat.credit_risk_interaction),
        "income_bucket":                    _enum_val(feat.income_bucket),
        "high_loan_burden_flag":            bool(feat.high_loan_burden_flag),
        "employment_stability":             _enum_val(feat.employment_stability),
        "is_high_risk":                     bool(feat.is_high_risk),
        "composite_risk_score":             float(feat.composite_risk_score or 0),
        "homeownership_score":              int(feat.homeownership_score or 0),
        "stability_income_interaction":     float(feat.stability_income_interaction or 0),
        "intent_risk_score":                int(feat.intent_risk_score or 0),
    }


def _build_query_text(applicant: Dict, engineered: Dict, prediction: Dict) -> str:
    """Compose a natural-language query for vector store retrieval, surfacing
    the most decision-relevant features so retrieved docs match this case."""
    parts: List[str] = []
    outcome = prediction["outcome"]

    parts.append(f"loan {outcome}")

    credit_tier = str(engineered.get("credit_score_tier", "")).lower()
    if credit_tier:
        parts.append(f"{credit_tier} credit score")

    if engineered.get("is_high_risk"):
        parts.append("high risk applicant")

    if engineered.get("high_loan_burden_flag"):
        dti = engineered.get("loan_to_income_ratio", 0)
        parts.append(f"high loan burden loan-to-income ratio {dti:.2f}")

    if engineered.get("previous_default", False) or applicant.get("previous_default_on_record", False):
        parts.append("previous loan default on record")

    if engineered.get("thin_credit_file"):
        parts.append("thin credit file short credit history")

    if engineered.get("young_inexperienced"):
        parts.append("young inexperienced applicant no employment experience")

    stability = str(engineered.get("employment_stability", "")).lower()
    if stability == "unstable":
        parts.append("unstable employment")

    intent = str(applicant.get("loan_intent", "")).replace("_", " ").lower()
    if intent:
        parts.append(f"{intent} loan")

    if outcome == "rejected":
        parts.append("steps to improve loan eligibility reapplication strategy")

    return " ".join(parts)


class LoanContextBuilder:
    """Produces a structured context dict for a loan applicant. model_path is
    the trained model path without extension, defaults to BEST_MODEL_PATH."""

    def __init__(self, model_path: Optional[Union[str, Path]] = None) -> None:
        model_stem = Path(model_path) if model_path else BEST_MODEL_PATH.with_suffix("")
        if not model_stem.with_suffix(".joblib").exists():
            raise FileNotFoundError(
                f"Trained model not found at {model_stem}.joblib\n"
                "Run  uv run python -m scripts.train_model  to train and save the models first."
            )
        self.clf = LoanClassifier.load(model_stem)
        log.info(
            "LoanContextBuilder: loaded %s from %s",
            self.clf.algorithm, model_stem,
        )

    def build(
        self,
        applicant_id: Optional[Union[str, UUID]] = None,
        feature_row: Optional[Dict[str, Any]] = None,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Build the full context dict for an applicant. Supply exactly one
        of applicant_id or feature_row."""
        if applicant_id is not None:
            return self._build_from_db(applicant_id, threshold)
        if feature_row is not None:
            return self._build_from_row(feature_row, threshold)
        raise ValueError("Supply either applicant_id or feature_row.")

    def _build_from_db(
        self,
        applicant_id: Union[str, UUID],
        threshold: float,
    ) -> Dict[str, Any]:
        from database.operations import get_applicant, get_features

        if isinstance(applicant_id, str):
            from uuid import UUID as _UUID
            applicant_id = _UUID(applicant_id)

        app  = get_applicant(applicant_id)
        feat = get_features(applicant_id)

        if app is None:
            raise ValueError(f"No LoanApplicant found for id={applicant_id}")
        if feat is None:
            raise ValueError(
                f"No EngineeredFeatures found for applicant_id={applicant_id}.\n"
                "Run the feature engineering pipeline first."
            )

        applicant_section  = _build_applicant_section(app)
        engineered_section = _build_engineered_section(feat)
        raw_row            = _build_feature_row_from_db(app, feat)

        return self._finalise(raw_row, applicant_section, engineered_section, threshold, str(applicant_id))

    def _build_from_row(
        self,
        feature_row: Dict[str, Any],
        threshold: float,
    ) -> Dict[str, Any]:
        applicant_section = {
            "age":                           float(feature_row.get("person_age", 0)),
            "gender":                        feature_row.get("person_gender"),
            "education":                     feature_row.get("person_education"),
            "annual_income":                 float(feature_row.get("person_income", 0)),
            "employment_experience_years":   int(feature_row.get("person_emp_exp", 0)),
            "home_ownership":                feature_row.get("person_home_ownership"),
            "loan_amount":                   float(feature_row.get("loan_amnt", 0)),
            "loan_intent":                   feature_row.get("loan_intent"),
            "loan_grade":                    feature_row.get("loan_grade"),
            "interest_rate_pct":             float(feature_row.get("loan_int_rate", 0)),
            "loan_percent_of_income":        float(feature_row.get("loan_percent_income", 0)),
            "credit_history_years":          float(feature_row.get("cb_person_cred_hist_length", 0)),
            "credit_score":                  int(feature_row.get("credit_score", 0)),
            "previous_default_on_record":    bool(feature_row.get("previous_loan_defaults_on_file", False)),
        }
        engineered_section = {
            "debt_to_income_ratio":          float(feature_row.get("debt_to_income_ratio", 0)),
            "loan_to_income_ratio":          float(feature_row.get("loan_to_income_ratio", 0)),
            "credit_history_to_age_ratio":   float(feature_row.get("credit_history_to_age_ratio", 0)),
            "affordability_ratio":           float(feature_row.get("affordability_ratio", 0)),
            "monthly_loan_burden":           float(feature_row.get("monthly_loan_burden", 0)),
            "monthly_income":                float(feature_row.get("monthly_income", 0)),
            "emp_to_age_ratio":              float(feature_row.get("emp_to_age_ratio", 0)),
            "loan_per_age":                  float(feature_row.get("loan_per_age", 0)),
            "young_inexperienced":           bool(feature_row.get("young_inexperienced", False)),
            "credit_score_tier":             feature_row.get("credit_score_tier"),
            "thin_credit_file":              bool(feature_row.get("thin_credit_file", False)),
            "score_per_history_year":        float(feature_row.get("score_per_history_year", 0)),
            "credit_risk_interaction":       bool(feature_row.get("credit_risk_interaction", False)),
            "income_bucket":                 feature_row.get("income_bucket"),
            "high_loan_burden_flag":         bool(feature_row.get("high_loan_burden_flag", False)),
            "employment_stability":          feature_row.get("employment_stability"),
            "is_high_risk":                  bool(feature_row.get("is_high_risk", False)),
            "composite_risk_score":          float(feature_row.get("composite_risk_score", 0)),
            "homeownership_score":           int(feature_row.get("homeownership_score", 0)),
            "stability_income_interaction":  float(feature_row.get("stability_income_interaction", 0)),
            "intent_risk_score":             int(feature_row.get("intent_risk_score", 0)),
        }
        return self._finalise(feature_row, applicant_section, engineered_section, threshold, "WALK_IN")

    def _finalise(
        self,
        raw_row: Dict[str, Any],
        applicant: Dict[str, Any],
        engineered: Dict[str, Any],
        threshold: float,
        ref_id: str,
    ) -> Dict[str, Any]:
        feature_names = self.clf.feature_names_
        if feature_names is None:
            raise RuntimeError(
                "The loaded model has no feature_names_ attribute.\n"
                "Retrain with  uv run python -m scripts.train_model  "
                "to produce a model that stores feature names."
            )

        X = _prepare_single_row(raw_row, feature_names)
        proba = float(self.clf.predict_proba(X)[0, 1])
        outcome = "approved" if proba >= threshold else "rejected"
        risk_tier = _probability_to_tier(proba)

        prediction = {
            "outcome":     outcome,
            "probability": round(proba, 4),
            "confidence":  f"{proba * 100:.1f}%",
            "risk_tier":   risk_tier,
            "threshold":   threshold,
        }

        # not all models support this
        importance_list: List[Dict[str, Any]] = []
        try:
            fi_df = self.clf.get_feature_importance()
            if fi_df is not None:
                for _, row_fi in fi_df.head(15).iterrows():
                    importance_list.append({
                        "feature":       row_fi["feature"],
                        "importance":    round(float(row_fi["importance"]), 6),
                        "readable_name": _readable_name(row_fi["feature"]),
                    })
        except Exception as exc:
            log.warning("Feature importance unavailable for %s: %s", self.clf.algorithm, exc)

        query_text = _build_query_text(applicant, engineered, prediction)

        log.info(
            "Context built for applicant=%s: %s (P=%.4f, %s)",
            ref_id, outcome.upper(), proba, risk_tier,
        )

        return {
            "ref_id":              ref_id,
            "applicant":           applicant,
            "engineered":          engineered,
            "prediction":          prediction,
            "feature_importance":  importance_list,
            "query_text":          query_text,
            "model_algorithm":     self.clf.algorithm,
        }
