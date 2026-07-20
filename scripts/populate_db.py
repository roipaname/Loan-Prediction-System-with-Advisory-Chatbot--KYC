"""
scripts/populate_db.py
=======================
One-time population of Postgres with a representative sample of the processed
dataset, plus a real prediction (LIME attribution included) for every row.

The `loan_applicants` table is gaining a new `display_code` column, so this
resets the schema first — the only rows lost are the 8 dummy rows from
database/operations.py's __main__ seed block, not real data.

The source CSV has no `loan_grade` column (feature_eng.py doesn't produce
one), so a synthetic A-G grade is assigned from loan_int_rate quantiles
(A = lowest rate/lowest risk ... G = highest), computed over the full
45k-row dataset before sampling.

Usage:
    .venv/bin/python -m scripts.populate_db
    .venv/bin/python -m scripts.populate_db --sample-size 2000
"""
from __future__ import annotations

import argparse
import time
import uuid

import numpy as np
import pandas as pd
from loguru import logger as log

from config.settings import MODELS_DIR, PROCESSED_DATA_DIR, RANDOM_STATE
from database.connection import Connection
from database.schemas import (
    Base,
    EngineeredFeatures,
    LoanApplicant,
    LoanGradeEnum,
    ModelAlgorithmEnum,
    ModelPrediction,
    PredictionOutcomeEnum,
)
from scripts.insert_processed import _coerce_applicant, _coerce_features

GRADE_LABELS = ["A", "B", "C", "D", "E", "F", "G"]


def _assign_synthetic_grades(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["loan_grade"] = pd.qcut(df["loan_int_rate"], 7, labels=GRADE_LABELS)
    return df


def _stratified_sample(df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    frac = sample_size / len(df)
    sampled = (
        df.groupby("loan_status", group_keys=False)
        .apply(lambda g: g.sample(frac=frac, random_state=RANDOM_STATE))
    )
    return sampled.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)


def reset_schema() -> None:
    conn = Connection()
    log.warning("Resetting schema (dropping + recreating all LAPAS tables) …")
    Base.metadata.drop_all(bind=conn.engine)
    Base.metadata.create_all(bind=conn.engine)
    log.success("Schema reset complete.")


def register_models(conn: Connection) -> dict[str, "MLModel"]:
    """Insert one MLModel row per algorithm from models/model_comparison.csv."""
    from database.schemas import MLModel

    csv_path = MODELS_DIR / "model_comparison.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found — run `.venv/bin/python -m scripts.train_model` first."
        )
    comparison = pd.read_csv(csv_path)

    models_by_algo: dict[str, MLModel] = {}
    with conn.get_db() as db:
        for _, row in comparison.iterrows():
            model = MLModel(
                algorithm=ModelAlgorithmEnum(row["algorithm"]),
                is_from_scratch=row["algorithm"] == "logistic_regression",
                cv_accuracy=round(float(row["accuracy"]), 4),
                cv_precision=round(float(row["precision"]), 4),
                cv_recall=round(float(row["recall"]), 4),
                cv_f1_weighted=round(float(row["f1_weighted"]), 4),
                cv_auc_roc=round(float(row["roc_auc"]), 4),
                model_path=str(MODELS_DIR / f"{row['algorithm']}.joblib"),
                is_champion=bool(row["is_champion"]),
            )
            db.add(model)
            db.flush()
            db.refresh(model)
            db.expunge(model)
            models_by_algo[row["algorithm"]] = model
    log.success("Registered {} MLModel rows.", len(models_by_algo))
    return models_by_algo


def populate(sample_size: int, chunk_size: int = 200) -> None:
    from backend.deps import get_classifier, get_context_builder
    from backend.services.lime_service import explain_row
    from src.ai_advisor.loan_context_builder import _prepare_single_row

    reset_schema()
    conn = Connection()

    log.info("Loading processed dataset …")
    df = pd.read_csv(PROCESSED_DATA_DIR / "loan_features.csv")
    df = _assign_synthetic_grades(df)
    sample = _stratified_sample(df, sample_size)
    log.info("Sampled {:,} of {:,} rows (stratified on loan_status).", len(sample), len(df))

    models_by_algo = register_models(conn)
    champion = next(m for m in models_by_algo.values() if m.is_champion)
    log.info("Champion model: {}", champion.algorithm.value)

    context_builder = get_context_builder()
    clf = get_classifier()

    n_rows = len(sample)
    start_time = time.time()
    inserted = 0

    for chunk_start in range(0, n_rows, chunk_size):
        chunk = sample.iloc[chunk_start : chunk_start + chunk_size]

        with conn.get_db() as db:
            for _, row in chunk.iterrows():
                row_dict = row.to_dict()

                applicant_data = _coerce_applicant(row, source_split="sample")
                applicant_data["loan_grade"] = LoanGradeEnum(str(row["loan_grade"]))
                applicant_data["display_code"] = str(uuid.uuid4())[:8].upper()
                applicant = LoanApplicant(**applicant_data)
                db.add(applicant)
                db.flush()

                features_data = _coerce_features(row, applicant.id)
                db.add(EngineeredFeatures(**features_data))

                ctx = context_builder.build(feature_row=row_dict)
                x = _prepare_single_row(row_dict, clf.feature_names_)
                attribution = explain_row(clf, x, num_samples=300)
                risk_tier = ctx["prediction"]["risk_tier"].replace(" Risk", "")

                db.add(ModelPrediction(
                    applicant_id=applicant.id,
                    model_id=champion.id,
                    predicted_outcome=PredictionOutcomeEnum(ctx["prediction"]["outcome"]),
                    approval_probability=ctx["prediction"]["probability"],
                    risk_tier=risk_tier,
                    shap_values=attribution,
                    top_shap_features=list(attribution.keys())[:10],
                ))

                inserted += 1

        elapsed = time.time() - start_time
        rate = inserted / elapsed if elapsed > 0 else 0
        log.info(
            "Progress: {}/{} ({:.1f}/s, {:.0f}s elapsed)",
            inserted, n_rows, rate, elapsed,
        )

    log.success("Done. Inserted {} applicants with predictions in {:.0f}s.", inserted, time.time() - start_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--chunk-size", type=int, default=200)
    args = parser.parse_args()
    populate(sample_size=args.sample_size, chunk_size=args.chunk_size)
