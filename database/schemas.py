from sqlalchemy import Column,Integer,String,Numeric,DateTime,ForeignKey,Boolean,UniqueConstraint,Enum,Text
from sqlalchemy.ext.declarative import declarative_base

from sqlalchemy.dialects.postgresql import UUID,JSONB,ARRAY
from sqlalchemy.orm import mapped_column,Mapped,relationship
import enum
import uuid
from datetime import datetime

from loguru import logger
from typing import List,Dict