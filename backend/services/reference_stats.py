"""
backend/services/reference_stats.py
====================================
database/feature_eng.py computes two engineered columns from dataset-wide
statistics (credit_risk_interaction from medians, is_high_risk from a 75th
percentile). Those statistics can't be recomputed on a single new-application
row, so we compute them once from the processed training data and reuse them
for every single-row scoring request.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from config.settings import PROCESSED_DATA_DIR


@dataclass(frozen=True)
class ReferenceStats:
    median_loan_int_rate: float
    median_credit_score: float
    high_risk_threshold: float


@lru_cache(maxsize=1)
def get_reference_stats() -> ReferenceStats:
    df = pd.read_csv(
        PROCESSED_DATA_DIR / "loan_features.csv",
        usecols=["loan_int_rate", "credit_score", "composite_risk_score"],
    )
    return ReferenceStats(
        median_loan_int_rate=float(df["loan_int_rate"].median()),
        median_credit_score=float(df["credit_score"].median()),
        high_risk_threshold=float(df["composite_risk_score"].quantile(0.75)),
    )
