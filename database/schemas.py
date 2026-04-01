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


 