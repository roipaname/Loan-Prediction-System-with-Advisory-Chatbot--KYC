"""
database/insert_processed.py
=============================
Reads data/processed/loan_features.csv and bulk-inserts every row into:
  - loan_applicants      (LoanApplicant)
  - engineered_features  (EngineeredFeatures)

Features
--------
- Chunk-based inserts (default 500 rows) to avoid memory/transaction bloat
- Idempotent: skips rows whose (person_income, loan_amnt, credit_score,
  loan_percent_income) fingerprint already exists in loan_applicants
- Full enum coercion matching database/schemas.py exactly
- Progress logging via loguru
- CLI flags: --input, --chunk-size, --source-split, --dry-run

Usage
-----
    uv run python -m database.insert_processed
    uv run python -m database.insert_processed --input data/processed/loan_features.csv
    uv run python -m database.insert_processed --dry-run
    uv run python -m database.insert_processed --chunk-size 1000 --source-split train
"""

from __future__ import annotations

import argparse
import math
from decimal import Decimal
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from database.connection import Connection
from database.schemas import (
    CreditScoreTierEnum,
    EmploymentStabilityEnum,
    GenderEnum,
    HomeOwnerShipEnum,
    IncomeBucketEnum,
    LoanGradeEnum,
    LoanIntentEnum,
    LoanApplicant,
    EngineeredFeatures,
    PersonEducationEnum,
)
from config.settings import PROCESSED_DATA_DIR
# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_INPUT      = PROCESSED_DATA_DIR/"loan_features.csv"
DEFAULT_CHUNK_SIZE = 500

# ---------------------------------------------------------------------------
# Enum coercion maps  (CSV value → SQLAlchemy enum member)
# ---------------------------------------------------------------------------

GENDER_MAP: dict[str, GenderEnum] = {
    "male":   GenderEnum.male,
    "female": GenderEnum.female,
    "other":  GenderEnum.other,
}

EDUCATION_MAP: dict[str, PersonEducationEnum] = {
    "High School": PersonEducationEnum.high_school,
    "Bachelor":    PersonEducationEnum.bachelor,
    "Diploma":     PersonEducationEnum.diploma,
    "Associate":   PersonEducationEnum.associate,
    "Master":      PersonEducationEnum.master,
    "Doctor":      PersonEducationEnum.doctor,
    "Doctorate":   PersonEducationEnum.doctor,   # tolerate raw CSV spelling
}

HOMEOWNERSHIP_MAP: dict[str, HomeOwnerShipEnum] = {
    "MORTGAGE": HomeOwnerShipEnum.mortgage,
    "OTHER":    HomeOwnerShipEnum.other,
    "OWN":      HomeOwnerShipEnum.own,
    "RENT":     HomeOwnerShipEnum.rent,
}

INTENT_MAP: dict[str, LoanIntentEnum] = {
    "DEBTCONSOLIDATION": LoanIntentEnum.debt_consolidation,
    "EDUCATION":         LoanIntentEnum.education,
    "HOME_IMPROVEMENT":  LoanIntentEnum.home_improvement,
    "HOMEIMPROVEMENT":   LoanIntentEnum.home_improvement,  # raw CSV spelling
    "MEDICAL":           LoanIntentEnum.medical,
    "PERSONAL":          LoanIntentEnum.personal,
    "VENTURE":           LoanIntentEnum.venture,
}

CREDIT_TIER_MAP: dict[str, CreditScoreTierEnum] = {
    "Poor":        CreditScoreTierEnum.poor,
    "Fair":        CreditScoreTierEnum.fair,
    "Good":        CreditScoreTierEnum.good,
    "Very Good":   CreditScoreTierEnum.very_good,
    "Exceptional": CreditScoreTierEnum.exceptional,
}

EMPLOYMENT_MAP: dict[str, EmploymentStabilityEnum] = {
    "stable":   EmploymentStabilityEnum.stable,
    "unstable": EmploymentStabilityEnum.unstable,
}

INCOME_BUCKET_MAP: dict[str, IncomeBucketEnum] = {
    "low":     IncomeBucketEnum.LOW,
    "mid_low": IncomeBucketEnum.LOW_MEDIUM,
    "medium":  IncomeBucketEnum.MEDIUM,
    "high":    IncomeBucketEnum.HIGH,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dec(val, precision: int = 6) -> Optional[Decimal]:
    """Convert a value to Decimal, returning None for NaN/None."""
    if val is None:
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return Decimal(str(round(f, precision)))
    except (TypeError, ValueError):
        return None


def _bool(val) -> Optional[bool]:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes")
    try:
        return bool(int(val))
    except (TypeError, ValueError):
        return None


def _int(val) -> Optional[int]:
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _coerce_applicant(row: pd.Series, source_split: str) -> dict:
    return {
        "person_age":                       _dec(row["person_age"], 1),
        "person_gender":                    GENDER_MAP.get(str(row["person_gender"]).strip().lower()),
        "person_education":                 EDUCATION_MAP.get(str(row["person_education"]).strip()),
        "person_income":                    _dec(row["person_income"], 2),
        "person_emp_exp":                   _int(row["person_emp_exp"]),
        "person_home_ownership":            HOMEOWNERSHIP_MAP.get(str(row["person_home_ownership"]).strip().upper()),
        "loan_amnt":                        _dec(row["loan_amnt"], 2),
        "loan_intent":                      INTENT_MAP.get(str(row["loan_intent"]).strip().upper()),
        "loan_grade":                       None,   # not present in CSV
        "loan_int_rate":                    _dec(row["loan_int_rate"], 2),
        "loan_percent_income":              _dec(row["loan_percent_income"], 4),
        "cb_person_cred_hist_length":       _dec(row["cb_person_cred_hist_length"], 1),
        "credit_score":                     _int(row["credit_score"]),
        "previous_loan_defaults_on_file":   _bool(row["previous_loan_defaults_on_file"]),
        "loan_status":                      _int(row["loan_status"]),
        "source_split":                     source_split,
    }


def _coerce_features(row: pd.Series, applicant_id) -> dict:
    return {
        "applicant_id":                 applicant_id,
        # financial ratios
        "debt_to_income_ratio":         _dec(row["debt_to_income_ratio"]),
        "loan_to_income_ratio":         _dec(row["loan_to_income_ratio"]),
        "credit_history_to_age_ratio":  _dec(row["credit_history_to_age_ratio"]),
        "affordability_ratio":          _dec(row["affordability_ratio"]),
        "monthly_loan_burden":          _dec(row["monthly_loan_burden"], 2),
        "monthly_income":               _dec(row["monthly_income"], 2),
        # age & employment
        "emp_to_age_ratio":             _dec(row["emp_to_age_ratio"]),
        "loan_per_age":                 _dec(row["loan_per_age"], 4),
        "young_inexperienced":          _bool(row["young_inexperienced"]),
        # credit quality
        "credit_score_tier":            CREDIT_TIER_MAP.get(str(row["credit_score_tier"]).strip()),
        "thin_credit_file":             _bool(row["thin_credit_file"]),
        "score_per_history_year":       _dec(row["score_per_history_year"], 4),
        "credit_risk_interaction":      _bool(row["credit_risk_interaction"]),
        # income & burden
        "income_bucket":                INCOME_BUCKET_MAP.get(str(row["income_bucket"]).strip().lower()),
        "high_loan_burden_flag":        _bool(row["high_loan_burden_flag"]),
        # employment
        "employment_stability":         EMPLOYMENT_MAP.get(str(row["employment_stability"]).strip().lower()),
        # risk
        "is_high_risk":                 _bool(row["is_high_risk"]),
        "composite_risk_score":         _dec(row["composite_risk_score"], 6),
        # homeownership
        "homeownership_score":          _int(row["homeownership_score"]),
        "stability_income_interaction": _dec(row["stability_income_interaction"], 6),
        # intent
        "intent_risk_score":            _int(row["intent_risk_score"]),
        # metadata
        "pipeline_version":             str(row.get("pipeline_version", "1.0.0")),
    }


# ---------------------------------------------------------------------------
# Duplicate fingerprint check
# ---------------------------------------------------------------------------

def _build_existing_fingerprints(db) -> set[tuple]:
    """
    Load a lightweight fingerprint of every existing applicant row so we
    can skip duplicates without re-inserting.  Keyed on four stable columns.
    """
    rows = db.query(
        LoanApplicant.person_income,
        LoanApplicant.loan_amnt,
        LoanApplicant.credit_score,
        LoanApplicant.loan_percent_income,
    ).all()
    return {
        (float(r.person_income), float(r.loan_amnt),
         int(r.credit_score),    float(r.loan_percent_income))
        for r in rows
    }


def _fingerprint(row: pd.Series) -> tuple:
    return (
        float(row["person_income"]),
        float(row["loan_amnt"]),
        int(row["credit_score"]),
        float(row["loan_percent_income"]),
    )


# ---------------------------------------------------------------------------
# Core insert logic
# ---------------------------------------------------------------------------

def insert_chunk(
    db,
    chunk: pd.DataFrame,
    existing: set[tuple],
    source_split: str,
    dry_run: bool,
) -> tuple[int, int]:
    """
    Insert one DataFrame chunk.  Returns (inserted, skipped) counts.
    """
    inserted = skipped = 0

    for _, row in chunk.iterrows():
        fp = _fingerprint(row)
        if fp in existing:
            skipped += 1
            continue

        applicant_data = _coerce_applicant(row, source_split)
        applicant      = LoanApplicant(**applicant_data)

        if not dry_run:
            db.add(applicant)
            db.flush()   # populate applicant.id before features insert

        features_data = _coerce_features(row, applicant.id if not dry_run else None)
        features      = EngineeredFeatures(**features_data)

        if not dry_run:
            db.add(features)

        existing.add(fp)   # prevent intra-chunk duplicates too
        inserted += 1

    return inserted, skipped


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_insert(
    input_path:   str  = DEFAULT_INPUT,
    chunk_size:   int  = DEFAULT_CHUNK_SIZE,
    source_split: str  = "processed",
    dry_run:      bool = False,
) -> None:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Processed CSV not found: {path}")

    logger.info("Reading {}", path)
    df = pd.read_csv(path)
    total_rows = len(df)
    logger.info("Loaded {:,} rows", total_rows)

    if dry_run:
        logger.warning("DRY RUN — no data will be written to the database")

    conn = Connection()
    total_inserted = total_skipped = 0
    n_chunks = math.ceil(total_rows / chunk_size)

    with conn.get_db() as db:
        logger.info("Building duplicate fingerprint index …")
        existing = _build_existing_fingerprints(db)
        logger.info("Found {:,} existing applicant rows in DB", len(existing))

        for i, start in enumerate(range(0, total_rows, chunk_size), 1):
            chunk = df.iloc[start : start + chunk_size]
            logger.info("Processing chunk {}/{} (rows {}–{}) …",
                        i, n_chunks, start + 1, min(start + chunk_size, total_rows))

            ins, skp = insert_chunk(db, chunk, existing, source_split, dry_run)
            total_inserted += ins
            total_skipped  += skp

            if not dry_run:
                db.flush()
            logger.info("  chunk {}: inserted={}, skipped={}", i, ins, skp)

    # final summary
    logger.success("=" * 55)
    logger.success("INSERT COMPLETE")
    logger.success("  Total rows   : {:>8,}", total_rows)
    logger.success("  Inserted     : {:>8,}", total_inserted)
    logger.success("  Skipped (dup): {:>8,}", total_skipped)
    logger.success("  Dry run      : {}", dry_run)
    logger.success("=" * 55)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Insert processed loan features CSV into PostgreSQL."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Path to processed CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Rows per transaction chunk (default: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--source-split",
        default="processed",
        help="Value for source_split column (default: processed)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and coerce all rows but do not write to DB",
    )
    args = parser.parse_args()

    run_insert(
        input_path=args.input,
        chunk_size=args.chunk_size,
        source_split=args.source_split,
        dry_run=args.dry_run,
    )