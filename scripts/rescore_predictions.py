"""
Re-scores every applicant already in Postgres with the current model and
decision logic, updating each one's latest model_predictions row in place
(outcome, approval probability, risk tier, LIME attribution). Unlike
scripts/populate_db.py it doesn't reset the DB or need the processed CSV
beyond LIME's background sample.

Run after changing how model output maps to a decision (see
LoanContextBuilder._finalise) or after swapping models/best_model.joblib.

    .venv/bin/python -m scripts.rescore_predictions
    .venv/bin/python -m scripts.rescore_predictions --dry-run
"""
from __future__ import annotations

import argparse
import time
from collections import Counter

from loguru import logger as log

from backend.services.lime_service import explain_row
from database import operations as ops
from database.connection import Connection
from database.schemas import LoanApplicant, ModelPrediction, PredictionOutcomeEnum
from src.ai_advisor.loan_context_builder import (
    LoanContextBuilder,
    _build_feature_row_from_db,
    _prepare_single_row,
)


def rescore(dry_run: bool = False, lime_samples: int = 300) -> None:
    builder = LoanContextBuilder()
    clf = builder.clf
    champion = ops.get_champion_model()
    if champion is None:
        raise RuntimeError("No champion model in ml_models — run scripts.populate_db first.")

    conn = Connection()
    with conn.get_db() as db:
        applicant_ids = [row[0] for row in db.query(LoanApplicant.id).all()]
    log.info("Re-scoring {:,} applicants (dry_run={})", len(applicant_ids), dry_run)

    changed = Counter()
    start = time.time()
    for i, applicant_id in enumerate(applicant_ids, 1):
        ctx = builder.build(applicant_id=applicant_id)
        pred = ctx["prediction"]
        outcome = PredictionOutcomeEnum(pred["outcome"])
        risk_tier = pred["risk_tier"].replace(" Risk", "")

        with conn.get_db() as db:
            row = (
                db.query(ModelPrediction)
                .filter_by(applicant_id=applicant_id)
                .order_by(ModelPrediction.predicted_at.desc())
                .first()
            )
            changed["flipped" if row is None or row.predicted_outcome != outcome else "same"] += 1
            if dry_run:
                continue

            raw_row = _build_feature_row_from_db(
                ops.get_applicant(applicant_id), ops.get_features(applicant_id)
            )
            x = _prepare_single_row(raw_row, clf.feature_names_)
            attribution = explain_row(clf, x, num_samples=lime_samples)
            if row is None:
                row = ModelPrediction(applicant_id=applicant_id, model_id=champion.id)
                db.add(row)
            row.predicted_outcome    = outcome
            row.approval_probability = pred["probability"]
            row.risk_tier            = risk_tier
            row.shap_values          = attribution
            row.top_shap_features    = list(attribution.keys())[:10]

        if i % 200 == 0 or i == len(applicant_ids):
            log.info("Progress: {}/{} ({:.0f}s)", i, len(applicant_ids), time.time() - start)

    log.success(
        "Done. {} outcomes changed, {} unchanged{}.",
        changed["flipped"], changed["same"], " (dry run, nothing written)" if dry_run else "",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-score all stored applicants.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--lime-samples", type=int, default=300)
    args = parser.parse_args()
    rescore(dry_run=args.dry_run, lime_samples=args.lime_samples)
