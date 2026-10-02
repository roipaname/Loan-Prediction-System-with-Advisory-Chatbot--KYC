"""
credit_risk_interaction and is_high_risk depend on dataset-wide medians/
quantiles that can't be recomputed on a single row, so we compute them once
here and reuse them for every scoring request.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from config.settings import PROCESSED_DATA_DIR


@dataclass(frozen=True)
class ReferenceStats:
    median_loan_int_rate: float
    median_credit_score: float
    high_risk_threshold: float


def processed_features_path() -> Path:
    # full dataset preferred; the DB-reconstructed sample (scripts/
    # reconstruct_processed_from_db.py) is a fallback when it's unavailable
    path = PROCESSED_DATA_DIR / "loan_features.csv"
    if not path.exists():
        path = PROCESSED_DATA_DIR / "loan_features_from_db.csv"
    return path


@lru_cache(maxsize=1)
def get_reference_stats() -> ReferenceStats:
    df = pd.read_csv(
        processed_features_path(),
        usecols=["loan_int_rate", "credit_score", "composite_risk_score"],
    )
    return ReferenceStats(
        median_loan_int_rate=float(df["loan_int_rate"].median()),
        median_credit_score=float(df["credit_score"].median()),
        high_risk_threshold=float(df["composite_risk_score"].quantile(0.75)),
    )
