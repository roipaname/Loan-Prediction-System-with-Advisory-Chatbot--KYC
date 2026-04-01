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
      
 