"""
feature_eng.py
==============
Feature engineering pipeline for the Loan Prediction System.

Reads  : data/raw/loan_data.csv  (or any path passed via CLI / import)
Writes : data/processed/loan_features.csv

All derived columns align 1-to-1 with the EngineeredFeatures ORM schema
(database/schemas.py).  Column names, enum values, and flag semantics
are reproduced exactly so downstream DB-insert code can map directly.

Usage
-----
    python feature_eng.py                                  # default paths
    python feature_eng.py --input  path/to/raw.csv
                          --output path/to/processed.csv
                          --version 1.0.0
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from config.settings import PROCESSED_DATA_DIR,RAW_DATA_DIR
# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — kept in one place so they're easy to tune
# ---------------------------------------------------------------------------

PIPELINE_VERSION = "1.0.0"

# Income bucket thresholds (approx. dataset quartiles)
INCOME_LOW_MAX        = 47_204
INCOME_LOW_MED_MAX    = 67_048
INCOME_MED_MAX        = 95_789

# Credit score tier bands (standard US definitions)
CREDIT_SCORE_TIERS = [
    (800, 850, "Exceptional"),
    (740, 799, "Very Good"),
    (670, 739, "Good"),
    (580, 669, "Fair"),
    (300, 579, "Poor"),
]

# Loan-intent risk scores (0 = lowest risk, 5 = highest)
INTENT_RISK_MAP = {
    "EDUCATION":         0,
    "HOMEIMPROVEMENT":   1,
    "HOME_IMPROVEMENT":  1,   # tolerate both spellings
    "PERSONAL":          2,
    "MEDICAL":           3,
    "DEBTCONSOLIDATION": 4,
    "VENTURE":           5,
}

# Homeownership stability scores
HOMEOWNERSHIP_SCORE_MAP = {
    "OTHER":    0,
    "RENT":     1,
    "MORTGAGE": 2,
    "OWN":      3,
}

# Education normalisation — CSV uses "Doctorate", schema uses "Doctor"
EDUCATION_NORM_MAP = {
    "Doctorate": "Doctor",
}

# Loan-intent normalisation — CSV uses "HOMEIMPROVEMENT", schema uses "HOME_IMPROVEMENT"
INTENT_NORM_MAP = {
    "HOMEIMPROVEMENT": "HOME_IMPROVEMENT",
}

# Composite risk score weights (must sum to 1.0)
RISK_WEIGHTS = {
    "debt_to_income_ratio":      0.20,
    "loan_to_income_ratio":      0.15,
    "thin_credit_file":          0.10,
    "credit_risk_interaction":   0.15,
    "high_loan_burden_flag":     0.10,
    "is_default_on_file":        0.20,
    "young_inexperienced":       0.10,
}


# ---------------------------------------------------------------------------
# Step 1 — Load & validate raw data
# ---------------------------------------------------------------------------

def load_raw(path: str | Path=RAW_DATA_DIR/'loan_data.csv') -> pd.DataFrame:
    """Load the raw CSV and do light validation."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    log.info("Loading raw data from %s", path)
    df = pd.read_csv(path)
    log.info("Loaded %d rows × %d columns", *df.shape)

    required = {
        "person_age", "person_gender", "person_education",
        "person_income", "person_emp_exp", "person_home_ownership",
        "loan_amnt", "loan_intent", "loan_int_rate",
        "loan_percent_income", "cb_person_cred_hist_length",
        "credit_score", "previous_loan_defaults_on_file", "loan_status",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


# ---------------------------------------------------------------------------
# Step 2 — Clean & normalise raw columns
# ---------------------------------------------------------------------------

def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise string enums so they match SQLAlchemy enum values exactly,
    and cast booleans.
    """
    log.info("Cleaning raw columns …")
    df = df.copy()

    # --- Boolean conversions ---
    df["previous_loan_defaults_on_file"] = (
        df["previous_loan_defaults_on_file"]
        .str.strip()
        .map({"Yes": True, "No": False})
        .astype(bool)
    )

    # --- String normalisations ---
    df["person_education"] = (
        df["person_education"].str.strip().replace(EDUCATION_NORM_MAP)
    )
    df["loan_intent"] = (
        df["loan_intent"].str.strip().replace(INTENT_NORM_MAP)
    )
    df["person_home_ownership"] = df["person_home_ownership"].str.strip().str.upper()
    df["person_gender"]         = df["person_gender"].str.strip().str.lower()

    # --- Numeric coercions ---
    df["person_age"]                  = pd.to_numeric(df["person_age"],                  errors="coerce")
    df["person_income"]               = pd.to_numeric(df["person_income"],               errors="coerce")
    df["loan_amnt"]                   = pd.to_numeric(df["loan_amnt"],                   errors="coerce")
    df["loan_int_rate"]               = pd.to_numeric(df["loan_int_rate"],               errors="coerce")
    df["loan_percent_income"]         = pd.to_numeric(df["loan_percent_income"],         errors="coerce")
    df["cb_person_cred_hist_length"]  = pd.to_numeric(df["cb_person_cred_hist_length"],  errors="coerce")
    df["credit_score"]                = pd.to_numeric(df["credit_score"],                errors="coerce")
    df["person_emp_exp"]              = pd.to_numeric(df["person_emp_exp"],              errors="coerce")

    # --- Cap implausible ages (max >100 are outliers) ---
    outlier_ages = df["person_age"] > 100
    if outlier_ages.sum():
        log.warning("Capping %d age outliers (>100) to 100", outlier_ages.sum())
        df.loc[outlier_ages, "person_age"] = 100.0

    return df


# ---------------------------------------------------------------------------
# Step 3 — Financial ratio features
# ---------------------------------------------------------------------------

def add_financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    debt_to_income_ratio, loan_to_income_ratio, credit_history_to_age_ratio,
    affordability_ratio, monthly_loan_burden, monthly_income
    """
    log.info("Computing financial ratio features …")
    df = df.copy()

    # Monthly income helper (used in multiple features)
    df["monthly_income"] = df["person_income"] / 12.0

    # DTI / LTI — loan_percent_income is pre-computed in source data
    # We store it under both names for schema completeness
    df["debt_to_income_ratio"] = df["loan_percent_income"]
    df["loan_to_income_ratio"] = df["loan_percent_income"]

    # Monthly loan burden (approximate: annualise with simple interest)
    df["monthly_loan_burden"] = (
        df["loan_amnt"] * (1 + df["loan_int_rate"] / 100) / 12.0
    )

    # Affordability ratio — fraction of monthly income NOT consumed by loan
    df["affordability_ratio"] = np.where(
        df["monthly_income"] > 0,
        1.0 - (df["monthly_loan_burden"] / df["monthly_income"]),
        np.nan,
    )
    df["affordability_ratio"] = df["affordability_ratio"].clip(lower=0.0)

    # Credit history relative to age
    df["credit_history_to_age_ratio"] = np.where(
        df["person_age"] > 0,
        df["cb_person_cred_hist_length"] / df["person_age"],
        np.nan,
    )

    return df


# ---------------------------------------------------------------------------
# Step 4 — Age & employment features
# ---------------------------------------------------------------------------

def add_age_employment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    emp_to_age_ratio, loan_per_age, young_inexperienced
    """
    log.info("Computing age & employment features …")
    df = df.copy()

    df["emp_to_age_ratio"] = np.where(
        df["person_age"] > 0,
        df["person_emp_exp"] / df["person_age"],
        np.nan,
    )

    df["loan_per_age"] = np.where(
        df["person_age"] > 0,
        df["loan_amnt"] / df["person_age"],
        np.nan,
    )

    # Risky demographic: very young with zero work history
    df["young_inexperienced"] = (
        (df["person_age"] < 25) & (df["person_emp_exp"] == 0)
    ).astype(bool)

    return df


# ---------------------------------------------------------------------------
# Step 5 — Credit quality features
# ---------------------------------------------------------------------------

def _score_to_tier(score: float) -> str:
    for lo, hi, label in CREDIT_SCORE_TIERS:
        if lo <= score <= hi:
            return label
    return "Poor"  # fallback for scores below 300


def add_credit_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    credit_score_tier, thin_credit_file, score_per_history_year,
    credit_risk_interaction
    """
    log.info("Computing credit quality features …")
    df = df.copy()

    df["credit_score_tier"] = df["credit_score"].apply(_score_to_tier)

    # Thin file: fewer than 2 years of credit history
    df["thin_credit_file"] = (df["cb_person_cred_hist_length"] < 2).astype(bool)

    # Score density over credit history
    df["score_per_history_year"] = np.where(
        df["cb_person_cred_hist_length"] > 0,
        df["credit_score"] / df["cb_person_cred_hist_length"],
        np.nan,
    )

    # High-rate loan with low credit score — dangerous combination
    median_rate  = df["loan_int_rate"].median()
    median_score = df["credit_score"].median()
    df["credit_risk_interaction"] = (
        (df["loan_int_rate"] > median_rate) & (df["credit_score"] < median_score)
    ).astype(bool)

    return df


# ---------------------------------------------------------------------------
# Step 6 — Income & loan burden features
# ---------------------------------------------------------------------------

def add_income_burden_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    income_bucket (LOW / LOW_MEDIUM / MEDIUM / HIGH),
    high_loan_burden_flag
    """
    log.info("Computing income & loan burden features …")
    df = df.copy()

    def _income_bucket(inc: float) -> str:
        if inc <= INCOME_LOW_MAX:
            return "low"
        elif inc <= INCOME_LOW_MED_MAX:
            return "mid_low"
        elif inc <= INCOME_MED_MAX:
            return "medium"
        return "high"

    df["income_bucket"] = df["person_income"].apply(_income_bucket)

    # Loan consumes more than 30 % of gross income
    df["high_loan_burden_flag"] = (df["loan_percent_income"] > 0.30).astype(bool)

    return df


# ---------------------------------------------------------------------------
# Step 7 — Employment stability
# ---------------------------------------------------------------------------

def add_employment_stability(df: pd.DataFrame) -> pd.DataFrame:
    """
    employment_stability: 'stable' if emp_exp >= 2, else 'unstable'
    """
    log.info("Computing employment stability …")
    df = df.copy()

    df["employment_stability"] = np.where(
        df["person_emp_exp"] >= 2, "stable", "unstable"
    )
    return df


# ---------------------------------------------------------------------------
# Step 8 — Risk flags & composite score
# ---------------------------------------------------------------------------

def add_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    is_high_risk, composite_risk_score
    """
    log.info("Computing risk flags & composite score …")
    df = df.copy()

    # Individual binary risk signals normalised to [0, 1]
    signals = pd.DataFrame(index=df.index)
    signals["debt_to_income_ratio"]    = (df["debt_to_income_ratio"] > 0.40).astype(float)
    signals["loan_to_income_ratio"]    = (df["loan_to_income_ratio"] > 0.40).astype(float)
    signals["thin_credit_file"]        = df["thin_credit_file"].astype(float)
    signals["credit_risk_interaction"] = df["credit_risk_interaction"].astype(float)
    signals["high_loan_burden_flag"]   = df["high_loan_burden_flag"].astype(float)
    signals["is_default_on_file"]      = df["previous_loan_defaults_on_file"].astype(float)
    signals["young_inexperienced"]     = df["young_inexperienced"].astype(float)

    df["composite_risk_score"] = sum(
        signals[col] * weight
        for col, weight in RISK_WEIGHTS.items()
    ).round(6)

    # Flag applicants above the 75th-percentile risk score
    threshold = df["composite_risk_score"].quantile(0.75)
    df["is_high_risk"] = (df["composite_risk_score"] >= threshold).astype(bool)

    return df


# ---------------------------------------------------------------------------
# Step 9 — Homeownership & stability–income interaction
# ---------------------------------------------------------------------------

def add_homeownership_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    homeownership_score (0–3),
    stability_income_interaction
    """
    log.info("Computing homeownership features …")
    df = df.copy()

    df["homeownership_score"] = (
        df["person_home_ownership"]
        .map(HOMEOWNERSHIP_SCORE_MAP)
        .fillna(0)
        .astype(int)
    )

    # Stability proxy × normalised income (log-scaled to reduce skew)
    df["stability_income_interaction"] = (
        df["homeownership_score"] * np.log1p(df["person_income"])
    ).round(6)

    return df


# ---------------------------------------------------------------------------
# Step 10 — Loan intent risk score
# ---------------------------------------------------------------------------

def add_intent_risk(df: pd.DataFrame) -> pd.DataFrame:
    """intent_risk_score (0 = safest, 5 = riskiest)"""
    log.info("Computing intent risk scores …")
    df = df.copy()

    df["intent_risk_score"] = (
        df["loan_intent"]
        .str.upper()
        .map(INTENT_RISK_MAP)
        .fillna(2)   # default to mid-range if unknown
        .astype(int)
    )
    return df


# ---------------------------------------------------------------------------
# Step 11 — Metadata
# ---------------------------------------------------------------------------

def add_metadata(df: pd.DataFrame, version: str) -> pd.DataFrame:
    df = df.copy()
    df["pipeline_version"] = version
    return df


# ---------------------------------------------------------------------------
# Master pipeline
# ---------------------------------------------------------------------------

ENGINEERED_FEATURE_COLS = [
    # passthrough raw cols (for join key and labels)
    "person_age", "person_gender", "person_education",
    "person_income", "person_emp_exp", "person_home_ownership",
    "loan_amnt", "loan_intent", "loan_int_rate",
    "loan_percent_income", "cb_person_cred_hist_length",
    "credit_score", "previous_loan_defaults_on_file",
    "loan_status",
    # financial ratios
    "debt_to_income_ratio", "loan_to_income_ratio",
    "credit_history_to_age_ratio", "affordability_ratio",
    "monthly_loan_burden", "monthly_income",
    # age & employment
    "emp_to_age_ratio", "loan_per_age", "young_inexperienced",
    # credit quality
    "credit_score_tier", "thin_credit_file",
    "score_per_history_year", "credit_risk_interaction",
    # income & burden
    "income_bucket", "high_loan_burden_flag",
    # employment
    "employment_stability",
    # risk
    "is_high_risk", "composite_risk_score",
    # homeownership
    "homeownership_score", "stability_income_interaction",
    # intent
    "intent_risk_score",
    # metadata
    "pipeline_version",
]


def run_pipeline(
    input_path:  str | Path = "data/raw/loan_data.csv",
    output_path: str | Path = "data/processed/loan_features.csv",
    version:     str        = PIPELINE_VERSION,
) -> pd.DataFrame:
    """
    Execute the full feature engineering pipeline.

    Parameters
    ----------
    input_path  : path to the raw CSV
    output_path : where to write the enriched CSV
    version     : pipeline version tag embedded in every row

    Returns
    -------
    pd.DataFrame with all engineered columns
    """
    # --- Load ---
    df = load_raw(input_path)

    # --- Transform ---
    df = clean_raw(df)
    df = add_financial_ratios(df)
    df = add_age_employment_features(df)
    df = add_credit_features(df)
    df = add_income_burden_features(df)
    df = add_employment_stability(df)
    df = add_risk_features(df)
    df = add_homeownership_features(df)
    df = add_intent_risk(df)
    df = add_metadata(df, version)

    # --- Select & order final columns ---
    final_cols = [c for c in ENGINEERED_FEATURE_COLS if c in df.columns]
    df_out = df[final_cols].copy()

    # --- Save ---
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    log.info(
        "Saved %d rows × %d columns → %s",
        len(df_out), len(df_out.columns), output_path,
    )

    # --- Summary ---
    _print_summary(df_out)

    return df_out


def _print_summary(df: pd.DataFrame) -> None:
    """Print a concise pipeline output summary to the log."""
    log.info("=" * 60)
    log.info("PIPELINE SUMMARY")
    log.info("Rows             : %d", len(df))
    log.info("Columns          : %d", len(df.columns))
    log.info("High-risk rows   : %d (%.1f%%)",
             df["is_high_risk"].sum(),
             100 * df["is_high_risk"].mean())
    log.info("Approved (label) : %d (%.1f%%)",
             df["loan_status"].sum(),
             100 * df["loan_status"].mean())
    log.info("Income buckets   :\n%s", df["income_bucket"].value_counts().to_string())
    log.info("Credit tiers     :\n%s", df["credit_score_tier"].value_counts().to_string())
    log.info("Employment       :\n%s", df["employment_stability"].value_counts().to_string())
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the loan feature engineering pipeline."
    )
    parser.add_argument(
        "--input",
        default="data/raw/loan_data.csv",
        help="Path to raw CSV (default: data/raw/loan_data.csv)",
    )
    parser.add_argument(
        "--output",
        default="data/processed/loan_features.csv",
        help="Path for processed output CSV (default: data/processed/loan_features.csv)",
    )
    parser.add_argument(
        "--version",
        default=PIPELINE_VERSION,
        help=f"Pipeline version tag (default: {PIPELINE_VERSION})",
    )
    args = parser.parse_args()

    run_pipeline(
        input_path=args.input,
        output_path=args.output,
        version=args.version,
    )