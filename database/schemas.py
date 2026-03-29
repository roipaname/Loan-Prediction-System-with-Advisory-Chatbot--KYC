from sqlalchemy import Column,Integer,String,Numeric,DateTime,ForeignKey,Boolean,UniqueConstraint,Enum,Text
from sqlalchemy.ext.declarative import declarative_base

from sqlalchemy.dialects.postgresql import UUID,JSONB,ARRAY
from sqlalchemy.orm import mapped_column,Mapped,relationship
import enum
import uuid
from datetime import datetime

from loguru import logger
from typing import List,Dict

Base=declarative_base()


# ==============================================================================
# Database Enums
# ==============================================================================


class IncomeBucketEnum(enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    LOW_MEDIUM = "mid_low"
    HIGH = "high"

class GenderEnum(enum.Enum):
    male="male"
    female="female"
    other="other"
class HomeOwnerShipEnum(enum.Enum):
    mortgage="mortgage"
    other="other"
    own="own"
    rent="rent"
class PersonEducationEnum(enum.Enum):
    high_school="high_school"
    bachelor="bachelor"
    diploma="diploma"
    associate="associate"
    master="master"
    doctor="doctor"
class LoanIntentEnum(enum.Enum):
    debt_consolidation="debt consolidation"
    education="education"
    home_improvement="home improvement"
    medical="medical"
    personal="personal"
    venture="venture"
    other="otheer"
