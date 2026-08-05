import uuid as _uuid
from typing import Optional, List, Dict, Any
from uuid import UUID

import pandas as pd

from database.connection import Connection
from database.schemas import (
    LoanApplicant,
    EngineeredFeatures,
    MLModel,
    ModelPrediction,
    RAGExplanation,
    PersonEducationEnum,
    HomeOwnerShipEnum,
    LoanIntentEnum,
    CreditScoreTierEnum,
    IncomeBucketEnum,
)

conn = Connection()


# generic helpers

def save(instance):
    with conn.get_db() as db:
        db.add(instance)
        db.flush()
        db.refresh(instance)
        db.expunge(instance)
        return instance


def get_by_id(model, obj_id: UUID):
    with conn.get_db() as db:
        obj = db.query(model).filter_by(id=obj_id).first()
        if obj:
            db.refresh(obj)
            db.expunge(obj)
        return obj


def list_all(model, limit: int = 100):
    with conn.get_db() as db:
        objs = db.query(model).limit(limit).all()
        for obj in objs:
            db.refresh(obj)
            db.expunge(obj)
        return objs


# applicant

def create_applicant(data: Dict[str, Any]) -> LoanApplicant:
    data = dict(data)
    data.setdefault("display_code", str(_uuid.uuid4())[:8].upper())
    applicant = LoanApplicant(**data)
    return save(applicant)


def get_applicant(applicant_id: UUID) -> Optional[LoanApplicant]:
    return get_by_id(LoanApplicant, applicant_id)


def get_applicant_by_code(display_code: str) -> Optional[LoanApplicant]:
    with conn.get_db() as db:
        obj = db.query(LoanApplicant).filter_by(display_code=display_code).first()
        if obj:
            db.refresh(obj)
            db.expunge(obj)
        return obj


# flat applicant + features + latest prediction join, for the frontend to
# filter/sort/paginate client-side

_APPLICANTS_JOIN_SQL = """
    SELECT
        a.display_code                       AS id,
        a.person_age, a.person_gender, a.person_education,
        a.person_income, a.person_emp_exp, a.person_home_ownership,
        a.loan_amnt, a.loan_intent, a.loan_grade, a.loan_int_rate,
        a.loan_percent_income, a.cb_person_cred_hist_length, a.credit_score,
        a.previous_loan_defaults_on_file, a.created_at,
        f.debt_to_income_ratio, f.affordability_ratio, f.composite_risk_score,
        f.credit_score_tier, f.thin_credit_file, f.income_bucket,
        p.predicted_outcome, p.approval_probability, p.risk_tier, p.shap_values,
        m.algorithm AS model_algorithm
    FROM loan_applicants a
    LEFT JOIN engineered_features f ON f.applicant_id = a.id
    LEFT JOIN model_predictions p   ON p.applicant_id = a.id
    LEFT JOIN ml_models m           ON m.id = p.model_id
    WHERE a.display_code IS NOT NULL
    ORDER BY a.created_at DESC
"""


# SQLAlchemy's Enum() stores the member NAME in postgres, not .value, and the
# ORM only translates that back on reads that go through it — raw SQL (below)
# gets the bare names, so remap them here.
_ENUM_NAME_TO_VALUE = {
    "person_education":        {m.name: m.value for m in PersonEducationEnum},
    "person_home_ownership":   {m.name: m.value for m in HomeOwnerShipEnum},
    "loan_intent":             {m.name: m.value for m in LoanIntentEnum},
    "credit_score_tier":       {m.name: m.value for m in CreditScoreTierEnum},
    "income_bucket":           {m.name: m.value for m in IncomeBucketEnum},
}


def get_applicants_flat(limit: int = 5000) -> pd.DataFrame:
    """Full applicant+features+prediction dataset as a flat DataFrame."""
    df = pd.read_sql(_APPLICANTS_JOIN_SQL, conn.engine)
    if limit:
        df = df.head(limit)
    for col in ("predicted_outcome", "risk_tier"):
        df[col] = df[col].astype(object)
    for col, mapping in _ENUM_NAME_TO_VALUE.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(df[col])
    return df


# engineered features

def create_features(applicant_id: UUID, data: Dict[str, Any]) -> EngineeredFeatures:
    features = EngineeredFeatures(applicant_id=applicant_id, **data)
    return save(features)


def get_features(applicant_id: UUID) -> Optional[EngineeredFeatures]:
    with conn.get_db() as db:
        obj = db.query(EngineeredFeatures).filter_by(applicant_id=applicant_id).first()
        if obj:
            db.refresh(obj)
            db.expunge(obj)
        return obj


# model

def create_model(data: Dict[str, Any]) -> MLModel:
    model = MLModel(**data)
    return save(model)


def get_model(model_id: UUID) -> Optional[MLModel]:
    return get_by_id(MLModel, model_id)


def get_champion_model() -> Optional[MLModel]:
    with conn.get_db() as db:
        obj = db.query(MLModel).filter_by(is_champion=True).first()
        if obj:
            db.refresh(obj)
            db.expunge(obj)
        return obj


# predictions

def create_prediction(data: Dict[str, Any]) -> ModelPrediction:
    prediction = ModelPrediction(**data)
    return save(prediction)


def get_prediction(prediction_id: UUID) -> Optional[ModelPrediction]:
    return get_by_id(ModelPrediction, prediction_id)


def get_applicant_predictions(applicant_id: UUID) -> List[ModelPrediction]:
    with conn.get_db() as db:
        objs = db.query(ModelPrediction).filter_by(applicant_id=applicant_id).all()
        for obj in objs:
            db.refresh(obj)
            db.expunge(obj)
        return objs


def get_latest_prediction(applicant_id: UUID) -> Optional[ModelPrediction]:
    with conn.get_db() as db:
        obj = (
            db.query(ModelPrediction)
            .filter_by(applicant_id=applicant_id)
            .order_by(ModelPrediction.predicted_at.desc())
            .first()
        )
        if obj:
            db.refresh(obj)
            db.expunge(obj)
        return obj


# RAG

def create_rag(data: Dict[str, Any]) -> RAGExplanation:
    rag = RAGExplanation(**data)
    return save(rag)


def upsert_rag(data: Dict[str, Any]) -> RAGExplanation:
    """
    RAGExplanation.prediction_id is unique (one-to-one with ModelPrediction),
    but a user can regenerate the advisory report (e.g. switch retriever) any
    number of times — update the existing row instead of inserting a duplicate.
    """
    with conn.get_db() as db:
        existing = db.query(RAGExplanation).filter_by(prediction_id=data["prediction_id"]).first()
        if existing:
            for key, value in data.items():
                setattr(existing, key, value)
            db.flush()
            db.refresh(existing)
            db.expunge(existing)
            return existing
        rag = RAGExplanation(**data)
        db.add(rag)
        db.flush()
        db.refresh(rag)
        db.expunge(rag)
        return rag


def get_rag(prediction_id: UUID) -> Optional[RAGExplanation]:
    with conn.get_db() as db:
        obj = db.query(RAGExplanation).filter_by(prediction_id=prediction_id).first()
        if obj:
            db.refresh(obj)
            db.expunge(obj)
        return obj


# seed / test block

from uuid import uuid4
from decimal import Decimal

from database.schemas import (
    GenderEnum,
    PersonEducationEnum,
    HomeOwnerShipEnum,
    LoanIntentEnum,
    LoanGradeEnum,
    CreditScoreTierEnum,
    EmploymentStabilityEnum,
    IncomeBucketEnum,
    ModelAlgorithmEnum,
    PredictionOutcomeEnum,
    RetrieverTypeEnum
)

if __name__ == "__main__":

    # 1. create applicant
    applicant_data = {
        "person_age": Decimal("28"),
        "person_gender": GenderEnum.male,
        "person_education": PersonEducationEnum.bachelor,
        "person_income": Decimal("45000"),
        "person_emp_exp": 5,
        "person_home_ownership": HomeOwnerShipEnum.rent,
        "loan_amnt": Decimal("12000"),
        "loan_intent": LoanIntentEnum.personal,
        "loan_grade": LoanGradeEnum.B,
        "loan_int_rate": Decimal("12.5"),
        "loan_percent_income": Decimal("0.26"),
        "cb_person_cred_hist_length": Decimal("6"),
        "credit_score": 690,
        "previous_loan_defaults_on_file": False,
        "loan_status": 1,
        "source_split": "test"
    }

    applicant = create_applicant(applicant_data)
    print("Applicant:", applicant)

    # 2. create features
    features_data = {
        "debt_to_income_ratio": Decimal("0.26"),
        "loan_to_income_ratio": Decimal("0.26"),
        "credit_history_to_age_ratio": Decimal("0.21"),
        "affordability_ratio": Decimal("0.74"),
        "monthly_loan_burden": Decimal("1000"),
        "monthly_income": Decimal("3750"),

        "emp_to_age_ratio": Decimal("0.18"),
        "loan_per_age": Decimal("428.57"),
        "young_inexperienced": False,

        "credit_score_tier": CreditScoreTierEnum.good,
        "thin_credit_file": False,
        "score_per_history_year": Decimal("115"),
        "credit_risk_interaction": False,

        "income_bucket": IncomeBucketEnum.MEDIUM,
        "high_loan_burden_flag": False,

        "employment_stability": EmploymentStabilityEnum.stable,

        "is_high_risk": False,
        "composite_risk_score": Decimal("0.32"),

        "homeownership_score": 1,
        "stability_income_interaction": Decimal("1.2"),

        "intent_risk_score": 2,
        "pipeline_version": "1.0.0"
    }

    features = create_features(applicant.id, features_data)
    print("Features:", features)

    # 3. create model
    model_data = {
        "algorithm": ModelAlgorithmEnum.logistic_regression,
        "is_from_scratch": True,
        "hyperparameters": {"lr": 0.01},

        "cv_accuracy": Decimal("0.82"),
        "cv_precision": Decimal("0.80"),
        "cv_recall": Decimal("0.78"),
        "cv_f1_weighted": Decimal("0.79"),
        "cv_auc_roc": Decimal("0.85"),

        "model_path": "/models/logreg.pkl",
        "is_champion": True
    }

    model = create_model(model_data)
    print("Model:", model)

    # 4. create prediction
    prediction_data = {
        "applicant_id": applicant.id,
        "model_id": model.id,
        "predicted_outcome": PredictionOutcomeEnum.approved,
        "approval_probability": Decimal("0.78"),
        "risk_tier": "Low",
        "shap_values": {"income": 0.2, "credit_score": 0.3},
        "top_shap_features": ["credit_score", "income"]
    }

    prediction = create_prediction(prediction_data)
    print("Prediction:", prediction)

    # 5. create rag explanation
    rag_data = {
        "prediction_id": prediction.id,
        "retriever_type": RetrieverTypeEnum.tfidf,
        "query_text": "Why was this loan approved?",
        "retrieval_k": 3,
        "retrieval_scores": {},

        "constructed_prompt": "Explain loan approval using policy.",
        "llm_model": "mistral",
        "llm_response": "The loan was approved due to stable income and good credit score.",
        "generation_latency_ms": 120
    }

    rag = create_rag(rag_data)
    print("RAG:", rag)

    # 6. fetch tests
    print("\n--- FETCH TESTS ---")

    print("Get Applicant:", get_applicant(applicant.id))
    print("Get Features:", get_features(applicant.id))
    print("Champion Model:", get_champion_model())
    print("Applicant Predictions:", get_applicant_predictions(applicant.id))
    print("Get RAG:", get_rag(prediction.id))