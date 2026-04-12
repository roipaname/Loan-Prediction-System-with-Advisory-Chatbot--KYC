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
    """Generic save helper"""
    with conn.get_db() as db:
        db.add(instance)
        db.flush()
        return instance


def get_by_id(model, obj_id: UUID):
    with conn.get_db() as db:
        return db.query(model).filter_by(id=obj_id).first()


def list_all(model, limit: int = 100):
    with conn.get_db() as db:
        return db.query(model).limit(limit).all()


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
        return db.query(EngineeredFeatures).filter_by(applicant_id=applicant_id).first()


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
        return db.query(MLModel).filter_by(is_champion=True).first()


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
        return db.query(ModelPrediction).filter_by(applicant_id=applicant_id).all()


# ==============================================================================
# RAG
# ==============================================================================

def create_rag(data: Dict[str, Any]) -> RAGExplanation:
    rag = RAGExplanation(**data)
    return save(rag)


def get_rag(prediction_id: UUID) -> Optional[RAGExplanation]:
    with conn.get_db() as db:
        return db.query(RAGExplanation).filter_by(prediction_id=prediction_id).first()