import json

import pandas as pd
from fastapi import APIRouter, HTTPException

from config.settings import BASE_DIR, MODELS_DIR

router = APIRouter(tags=["comparisons"])

_MODEL_COMPARISON_CSV  = MODELS_DIR / "model_comparison.csv"
_RETRIEVAL_METRICS_CSV = BASE_DIR / "reports" / "tfidf_chroma_metrics.csv"


@router.get("/comparisons/models")
def model_comparison():
    if not _MODEL_COMPARISON_CSV.exists():
        raise HTTPException(
            status_code=404,
            detail="models/model_comparison.csv not found — run `.venv/bin/python -m scripts.train_model` first.",
        )
    df = pd.read_csv(_MODEL_COMPARISON_CSV)
    return json.loads(df.to_json(orient="records"))


@router.get("/comparisons/retrieval")
def retrieval_comparison():
    if not _RETRIEVAL_METRICS_CSV.exists():
        raise HTTPException(
            status_code=404,
            detail="reports/tfidf_chroma_metrics.csv not found — run `.venv/bin/python -m scripts.tfidf_chroma` first.",
        )
    df = pd.read_csv(_RETRIEVAL_METRICS_CSV)
    return json.loads(df.to_json(orient="records"))
