"""
Rebuilds a processed-features CSV from Postgres when data/processed/loan_features.csv
is unavailable (e.g. evicted by iCloud "Optimize Mac Storage").

The DB holds the stratified sample written by scripts/populate_db.py
(source_split='sample') — already ZAR-rescaled and engineered by
database/feature_eng.py — so rows are exported as-is, not re-engineered.
Walk-in applications submitted through the app are excluded by default.

Output columns and value spellings match feature_eng.ENGINEERED_FEATURE_COLS.

    .venv/bin/python -m scripts.reconstruct_processed_from_db
    .venv/bin/python -m scripts.reconstruct_processed_from_db --include-walk-ins
"""
from __future__ import annotations

import argparse
import enum
from pathlib import Path

import pandas as pd
from loguru import logger as log

from config.settings import PROCESSED_DATA_DIR
from database.connection import Connection
from database.feature_eng import ENGINEERED_FEATURE_COLS
from database.schemas import EngineeredFeatures, LoanApplicant

DEFAULT_OUTPUT = PROCESSED_DATA_DIR / "loan_features_from_db.csv"


def _plain(value):
    return value.value if isinstance(value, enum.Enum) else value


def reconstruct(output_path: Path = DEFAULT_OUTPUT, include_walk_ins: bool = False) -> pd.DataFrame:
    conn = Connection()
    rows = []
    with conn.get_db() as s:
        q = s.query(LoanApplicant, EngineeredFeatures).join(
            EngineeredFeatures, EngineeredFeatures.applicant_id == LoanApplicant.id
        )
        if not include_walk_ins:
            q = q.filter(LoanApplicant.source_split == "sample")
        for applicant, feats in q.all():
            row = {}
            for col in ENGINEERED_FEATURE_COLS:
                src = applicant if hasattr(LoanApplicant, col) else feats
                row[col] = _plain(getattr(src, col))
            rows.append(row)

    df = pd.DataFrame(rows, columns=ENGINEERED_FEATURE_COLS)
    if df.empty:
        raise RuntimeError("No rows found in the database — nothing to reconstruct.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    log.info(
        "Reconstructed {:,} rows × {} columns → {} (approved: {:.1%})",
        len(df), len(df.columns), output_path, df["loan_status"].mean(),
    )
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--include-walk-ins", action="store_true")
    args = parser.parse_args()
    reconstruct(args.output, args.include_walk_ins)
