import json
from pathlib import Path
from typing import Tuple

import pandas as pd
from fastapi import APIRouter, HTTPException

from config.settings import BASE_DIR, MODELS_DIR

router = APIRouter(tags=["comparisons"])

_MODEL_COMPARISON_CSV  = MODELS_DIR / "model_comparison.csv"
_RETRIEVAL_METRICS_CSV = BASE_DIR / "reports" / "tfidf_chroma_metrics.csv"

# path -> (mtime, records) — re-read only when the underlying CSV changes,
# since scripts/train_model.py and scripts/tfidf_chroma.py rewrite these
# files at retrain/rebenchmark time, not on a fixed schedule.
_csv_cache: dict[Path, Tuple[float, list]] = {}


def _read_csv_cached(path: Path) -> list:
    mtime = path.stat().st_mtime
    cached = _csv_cache.get(path)
    if cached and cached[0] == mtime:
        return cached[1]
    records = json.loads(pd.read_csv(path).to_json(orient="records"))
    _csv_cache[path] = (mtime, records)
    return records


@router.get("/comparisons/models")
def model_comparison():
    if not _MODEL_COMPARISON_CSV.exists():
        raise HTTPException(
            status_code=404,
            detail="models/model_comparison.csv not found — run `.venv/bin/python -m scripts.train_model` first.",
        )
    return _read_csv_cached(_MODEL_COMPARISON_CSV)


@router.get("/comparisons/retrieval")
def retrieval_comparison():
    if not _RETRIEVAL_METRICS_CSV.exists():
        raise HTTPException(
            status_code=404,
            detail="reports/tfidf_chroma_metrics.csv not found — run `.venv/bin/python -m scripts.tfidf_chroma` first.",
        )
    return _read_csv_cached(_RETRIEVAL_METRICS_CSV)
