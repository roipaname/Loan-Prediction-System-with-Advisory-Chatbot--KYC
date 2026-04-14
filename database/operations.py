from typing import Optional, List, Dict, Any
from uuid import UUID

from database.connection import Connection
from database.schemas import (
    LoanApplicant,
    EngineeredFeatures,
    MLModel,
    ModelPrediction,
    RAGExplanation,
)

conn = Connection()


# ==============================================================================
# GENERIC HELPER
# ==============================================================================

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


# ==============================================================================
# APPLICANT
# ==============================================================================

def create_applicant(data: Dict[str, Any]) -> LoanApplicant:
    applicant = LoanApplicant(**data)
    return save(applicant)


def get_applicant(applicant_id: UUID) -> Optional[LoanApplicant]:
    return get_by_id(LoanApplicant, applicant_id)


# ==============================================================================
# ENGINEERED FEATURES
# ==============================================================================

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


# ==============================================================================
# MODEL
# ==============================================================================

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


# ==============================================================================
# PREDICTIONS
# ==============================================================================

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


# ==============================================================================
# RAG
# ==============================================================================

def create_rag(data: Dict[str, Any]) -> RAGExplanation:
    rag = RAGExplanation(**data)
    return save(rag)


def get_rag(prediction_id: UUID) -> Optional[RAGExplanation]:
    with conn.get_db() as db:
        obj = db.query(RAGExplanation).filter_by(prediction_id=prediction_id).first()
        if obj:
            db.refresh(obj)
            db.expunge(obj)
        return obj


# ==============================================================================
# SEED / TEST BLOCK
# ==============================================================================

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

    # =========================
    # 1. CREATE APPLICANT
    # =========================
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

    # =========================
    # 2. CREATE FEATURES
    # =========================
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

    # =========================
    # 3. CREATE MODEL
    # =========================
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

    # =========================
    # 4. CREATE PREDICTION
    # =========================
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

    # =========================
    # 5. CREATE RAG EXPLANATION
    # =========================
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

    # =========================
    # 6. FETCH TESTS
    # =========================
    print("\n--- FETCH TESTS ---")

    print("Get Applicant:", get_applicant(applicant.id))
    print("Get Features:", get_features(applicant.id))
    print("Champion Model:", get_champion_model())
    print("Applicant Predictions:", get_applicant_predictions(applicant.id))
    print("Get RAG:", get_rag(prediction.id))