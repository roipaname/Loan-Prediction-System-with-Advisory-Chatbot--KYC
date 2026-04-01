from sqlalchemy import (
    Column, Integer, String, Numeric, DateTime, ForeignKey,
    Boolean, UniqueConstraint, Enum, Text, Float, SmallInteger
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import mapped_column, Mapped, relationship
import enum
import uuid
from datetime import datetime
from typing import List, Optional


Base=declarative_base()


# ==============================================================================
# Database Enums
# ==============================================================================


class IncomeBucketEnum(enum.Enum):
    LOW = "low"
    LOW_MEDIUM = "mid_low"
    MEDIUM = "medium"
    HIGH = "high"
 
 
class GenderEnum(enum.Enum):
    male = "male"
    female = "female"
    other = "other"
 
 
class HomeOwnerShipEnum(enum.Enum):
    mortgage = "MORTGAGE"
    other = "OTHER"
    own = "OWN"
    rent = "RENT"
 
 
class PersonEducationEnum(enum.Enum):
    high_school = "High School"
    bachelor = "Bachelor"
    diploma = "Diploma"
    associate = "Associate"
    master = "Master"
    doctor = "Doctor"
 
 
class LoanIntentEnum(enum.Enum):
    debt_consolidation = "DEBTCONSOLIDATION"
    education = "EDUCATION"
    home_improvement = "HOME_IMPROVEMENT"
    medical = "MEDICAL"
    personal = "PERSONAL"
    venture = "VENTURE"
 
 
class LoanGradeEnum(enum.Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"
 
 
class CreditScoreTierEnum(enum.Enum):
    poor = "Poor"
    fair = "Fair"
    good = "Good"
    very_good = "Very Good"
    exceptional = "Exceptional"
 
 
class EmploymentStabilityEnum(enum.Enum):
    stable = "stable"
    unstable = "unstable"
 
 
class ModelAlgorithmEnum(enum.Enum):
    logistic_regression = "logistic_regression"
    decision_tree = "decision_tree"
    random_forest = "random_forest"
    gradient_boosting = "gradient_boosting"
    naive_bayes = "naive_bayes"
 
 
class PredictionOutcomeEnum(enum.Enum):
    approved = "approved"
    rejected = "rejected"
 
 
class RetrieverTypeEnum(enum.Enum):
    tfidf = "tfidf"
    vector = "vector"    


# ==============================================================================
# Core Tables
# ==============================================================================


class LoanApplicant(Base):
    """
    Raw applicant and loan application data.
    Mirrors the source CSV columns exactly — no derived values stored here.
    One row per loan application.
    """
    __tablename__ = "loan_applicants"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    # --- Demographic ---
    person_age = Column(Numeric(5, 1), nullable=False)
    person_gender = Column(Enum(GenderEnum), nullable=True)
    person_education = Column(Enum(PersonEducationEnum), nullable=True)

    # --- Financial ---
    person_income = Column(Numeric(12, 2), nullable=False)
    person_emp_exp = Column(Integer, nullable=False)             # years of employment experience
    person_home_ownership = Column(Enum(HomeOwnerShipEnum), nullable=False)

    # --- Loan Application ---
    loan_amnt = Column(Numeric(12, 2), nullable=False)
    loan_intent = Column(Enum(LoanIntentEnum), nullable=False)
    loan_grade = Column(Enum(LoanGradeEnum), nullable=True)
    loan_int_rate = Column(Numeric(5, 2), nullable=False)
    loan_percent_income = Column(Numeric(5, 4), nullable=False)  # pre-computed in source data

    # --- Credit Bureau ---
    cb_person_cred_hist_length = Column(Numeric(4, 1), nullable=False)
    credit_score = Column(Integer, nullable=False)
    previous_loan_defaults_on_file = Column(Boolean, nullable=False)  # Yes/No → True/False

    # --- Ground Truth Label (for training / evaluation) ---
    loan_status = Column(SmallInteger, nullable=True)            # 1 = approved, 0 = rejected

    # --- Metadata ---
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    source_split = Column(String(10), nullable=True)             # 'train', 'test', 'val'

    # --- Relationships ---
    engineered_features: Mapped[Optional["EngineeredFeatures"]] = relationship(
        "EngineeredFeatures", back_populates="applicant", uselist=False
    )
    predictions: Mapped[List["ModelPrediction"]] = relationship(
        "ModelPrediction", back_populates="applicant"
    )

    def __repr__(self):
        return f"<LoanApplicant id={self.id} income={self.person_income} loan={self.loan_amnt}>"


class EngineeredFeatures(Base):
    """
    All features derived by the feature_engineering pipeline.
    Stored separately from raw data to keep lineage clean and support re-computation.
    One-to-one with LoanApplicant.
    """
    __tablename__ = "engineered_features"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    applicant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("loan_applicants.id"), nullable=False, unique=True
    )

    # --- Financial Ratios ---
    debt_to_income_ratio = Column(Numeric(10, 6), nullable=True)
    loan_to_income_ratio = Column(Numeric(10, 6), nullable=True)   # same formula currently; kept for schema clarity
    credit_history_to_age_ratio = Column(Numeric(10, 6), nullable=True)
    affordability_ratio = Column(Numeric(10, 6), nullable=True)
    monthly_loan_burden = Column(Numeric(12, 2), nullable=True)
    monthly_income = Column(Numeric(12, 2), nullable=True)

    # --- Age & Experience ---
    emp_to_age_ratio = Column(Numeric(10, 6), nullable=True)
    loan_per_age = Column(Numeric(10, 4), nullable=True)
    young_inexperienced = Column(Boolean, nullable=True)           # age < 25 AND emp_exp == 0

    # --- Credit Quality ---
    credit_score_tier = Column(Enum(CreditScoreTierEnum), nullable=True)
    thin_credit_file = Column(Boolean, nullable=True)              # credit history < 2 years
    score_per_history_year = Column(Numeric(10, 4), nullable=True)
    credit_risk_interaction = Column(Boolean, nullable=True)       # low score AND high rate

    # --- Income & Loan Burden ---
    income_bucket = Column(Enum(IncomeBucketEnum), nullable=True)
    high_loan_burden_flag = Column(Boolean, nullable=True)         # loan_percent_income > 0.30

    # --- Employment ---
    employment_stability = Column(Enum(EmploymentStabilityEnum), nullable=True)

    # --- Risk Flags & Scores ---
    is_high_risk = Column(Boolean, nullable=True)
    composite_risk_score = Column(Numeric(8, 6), nullable=True)    # 0–1 weighted score

    # --- Homeownership ---
    homeownership_score = Column(SmallInteger, nullable=True)      # 0=OTHER, 1=RENT, 2=MORTGAGE, 3=OWN
    stability_income_interaction = Column(Numeric(12, 6), nullable=True)

    # --- Loan Intent ---
    intent_risk_score = Column(SmallInteger, nullable=True)        # 0 (lowest) – 5 (highest)

    # --- Metadata ---
    computed_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    pipeline_version = Column(String(20), nullable=True)           # e.g. "1.0.0" for reproducibility

    # --- Relationship ---
    applicant: Mapped["LoanApplicant"] = relationship(
        "LoanApplicant", back_populates="engineered_features"
    )

    def __repr__(self):
        return f"<EngineeredFeatures applicant={self.applicant_id} risk={self.composite_risk_score}>"


# ==============================================================================
# ML Pipeline Tables
# ==============================================================================

class MLModel(Base):
    """
    Registry of trained model artefacts.
    Tracks algorithm, hyperparameters, and evaluation metrics per training run.
    """
    __tablename__ = "ml_models"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    algorithm = Column(Enum(ModelAlgorithmEnum), nullable=False)
    is_from_scratch = Column(Boolean, default=False, nullable=False)  # True = custom impl
    hyperparameters = Column(JSONB, nullable=True)                    # grid-search best params

    # --- CV Evaluation Metrics ---
    cv_accuracy = Column(Numeric(6, 4), nullable=True)
    cv_precision = Column(Numeric(6, 4), nullable=True)
    cv_recall = Column(Numeric(6, 4), nullable=True)
    cv_f1_weighted = Column(Numeric(6, 4), nullable=True)
    cv_auc_roc = Column(Numeric(6, 4), nullable=True)

    # --- Fairness Metrics (Experiment 1) ---
    equalized_odds = Column(JSONB, nullable=True)     # {subgroup: value} per demographic
    demographic_parity = Column(JSONB, nullable=True)

    # --- Artefact ---
    model_path = Column(String(500), nullable=True)   # path to serialised .pkl / .joblib
    trained_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    is_champion = Column(Boolean, default=False)      # True = selected model for Pipeline 2

    predictions: Mapped[List["ModelPrediction"]] = relationship(
        "ModelPrediction", back_populates="model"
    )

    def __repr__(self):
        return f"<MLModel {self.algorithm} f1={self.cv_f1_weighted} champion={self.is_champion}>"


class ModelPrediction(Base):
    """
    Per-applicant prediction output from a trained MLModel.
    Stores probability, outcome, and SHAP feature attributions.
    """
    __tablename__ = "model_predictions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    applicant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("loan_applicants.id"), nullable=False
    )
    model_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("ml_models.id"), nullable=False
    )

    predicted_outcome = Column(Enum(PredictionOutcomeEnum), nullable=False)
    approval_probability = Column(Numeric(6, 4), nullable=False)   # P(approved)
    risk_tier = Column(String(20), nullable=True)                   # e.g. "High", "Medium", "Low"

    # SHAP: {feature_name: shap_value} for top-k drivers
    shap_values = Column(JSONB, nullable=True)
    top_shap_features = Column(ARRAY(String), nullable=True)        # ordered top-k feature names

    predicted_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # --- Relationships ---
    applicant: Mapped["LoanApplicant"] = relationship(
        "LoanApplicant", back_populates="predictions"
    )
    model: Mapped["MLModel"] = relationship(
        "MLModel", back_populates="predictions"
    )
    rag_explanation: Mapped[Optional["RAGExplanation"]] = relationship(
        "RAGExplanation", back_populates="prediction", uselist=False
    )

    __table_args__ = (
        UniqueConstraint("applicant_id", "model_id", name="uq_applicant_model"),
    )

    def __repr__(self):
        return f"<ModelPrediction applicant={self.applicant_id} outcome={self.predicted_outcome}>"


# ==============================================================================
# RAG Pipeline Tables
# ==============================================================================

class RetrievalDocument(Base):
    """
    Regulatory corpus documents used by the TF-IDF retriever (and vector baseline).
    Each row is a chunk of a source document.
    """
    __tablename__ = "retrieval_documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    source_name = Column(String(200), nullable=False)     # e.g. "FATF KYC Guidelines 2023"
    source_type = Column(String(100), nullable=True)      # e.g. "regulatory", "central_bank"
    chunk_index = Column(Integer, nullable=False)         # position within parent document
    chunk_text = Column(Text, nullable=False)
    token_count = Column(Integer, nullable=True)

    # Pre-computed TF-IDF vector stored as sparse representation
    tfidf_vector = Column(JSONB, nullable=True)           # {term: weight} sparse dict

    # For Experiment 2 vector-search baseline
    embedding_vector = Column(ARRAY(Float), nullable=True)  # all-MiniLM-L6-v2 embedding

    added_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    rag_explanations: Mapped[List["RAGExplanationChunk"]] = relationship(
        "RAGExplanationChunk", back_populates="document"
    )

    def __repr__(self):
        return f"<RetrievalDocument source={self.source_name} chunk={self.chunk_index}>"


class RAGExplanation(Base):
    """
    Full RAG pipeline output for a given prediction.
    Stores the constructed prompt, retrieved chunks, and final LLM response.
    One-to-one with ModelPrediction (champion model only).
    """
    __tablename__ = "rag_explanations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    prediction_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("model_predictions.id"), nullable=False, unique=True
    )

    # --- Retrieval Stage ---
    retriever_type = Column(Enum(RetrieverTypeEnum), nullable=False, default=RetrieverTypeEnum.tfidf)
    query_text = Column(Text, nullable=True)              # applicant query / trigger text
    retrieval_k = Column(SmallInteger, nullable=True)     # top-k chunks retrieved
    retrieval_scores = Column(JSONB, nullable=True)       # {chunk_id: score} for audit

    # --- Prompt Construction Stage ---
    constructed_prompt = Column(Text, nullable=True)      # full prompt sent to LLM

    # --- LLM Generation Stage ---
    llm_model = Column(String(100), nullable=True)        # e.g. "mistralai/Mistral-7B-Instruct-v0.2"
    llm_response = Column(Text, nullable=False)
    generation_latency_ms = Column(Integer, nullable=True)

    generated_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # --- Relationships ---
    prediction: Mapped["ModelPrediction"] = relationship(
        "ModelPrediction", back_populates="rag_explanation"
    )
    retrieved_chunks: Mapped[List["RAGExplanationChunk"]] = relationship(
        "RAGExplanationChunk", back_populates="explanation"
    )
    evaluator_assessments: Mapped[List["EvaluatorAssessment"]] = relationship(
        "EvaluatorAssessment", back_populates="explanation"
    )

    def __repr__(self):
        return f"<RAGExplanation prediction={self.prediction_id} retriever={self.retriever_type}>"


class RAGExplanationChunk(Base):
    """
    Junction table: which document chunks were retrieved for a given RAGExplanation.
    Enables per-chunk traceability and Precision@k / MRR computation.
    """
    __tablename__ = "rag_explanation_chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    explanation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("rag_explanations.id"), nullable=False
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("retrieval_documents.id"), nullable=False
    )
    rank = Column(SmallInteger, nullable=False)           # 1 = top retrieved chunk
    retrieval_score = Column(Numeric(8, 6), nullable=True)

    explanation: Mapped["RAGExplanation"] = relationship(
        "RAGExplanation", back_populates="retrieved_chunks"
    )
    document: Mapped["RetrievalDocument"] = relationship(
        "RetrievalDocument", back_populates="rag_explanations"
    )


# ==============================================================================
# Experiment 3: Human Evaluation Table
# ==============================================================================

class EvaluatorAssessment(Base):
    """
    Structured rubric scores from the panel of three evaluators (Experiment 3).
    Each evaluator scores each RAGExplanation on four dimensions.
    Cohen's Kappa is computed across rows grouped by explanation_id.
    """
    __tablename__ = "evaluator_assessments"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    explanation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("rag_explanations.id"), nullable=False
    )
    evaluator_id = Column(String(50), nullable=False)     # e.g. "evaluator_1"
    is_rag_system = Column(Boolean, nullable=False)       # True = RAG output; False = SHAP-only baseline

    # --- Rubric Scores (1–5 Likert scale) ---
    policy_traceability = Column(SmallInteger, nullable=False)   # cites relevant regulation
    factual_accuracy = Column(SmallInteger, nullable=False)      # no hallucinated claims
    completeness = Column(SmallInteger, nullable=False)          # covers key decision drivers
    actionability = Column(SmallInteger, nullable=False)         # applicant can act on it

    notes = Column(Text, nullable=True)
    assessed_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint("explanation_id", "evaluator_id", name="uq_explanation_evaluator"),
    )

    explanation: Mapped["RAGExplanation"] = relationship(
        "RAGExplanation", back_populates="evaluator_assessments"
    )

    def __repr__(self):
        return (
            f"<EvaluatorAssessment evaluator={self.evaluator_id} "
            f"traceability={self.policy_traceability} accuracy={self.factual_accuracy}>"
        )