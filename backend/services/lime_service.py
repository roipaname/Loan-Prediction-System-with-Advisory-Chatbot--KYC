"""
backend/services/lime_service.py
=================================
Per-applicant local feature attribution using LIME (lime_tabular), not SHAP.
requirements.txt already documents why: "LIME replaces SHAP — no numba/llvmlite"
on this macOS Intel machine. LoanClassifier.get_feature_importance() only
exposes *global* importances, so LIME is what actually answers "why did THIS
applicant get this outcome" — the same question the UI's ModelPrediction
.shap_values column was designed to answer.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict

import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

from config.settings import PROCESSED_DATA_DIR, RANDOM_STATE
from src.ai_advisor.loan_context_builder import _prepare_single_row
from src.classifier.classifier import LoanClassifier

_BACKGROUND_SAMPLE_SIZE = 300


def _build_background_matrix(clf: LoanClassifier) -> np.ndarray:
    df = pd.read_csv(PROCESSED_DATA_DIR / "loan_features.csv")
    sample = df.sample(
        n=min(_BACKGROUND_SAMPLE_SIZE, len(df)), random_state=RANDOM_STATE
    )
    rows = sample.to_dict("records")
    return np.vstack([_prepare_single_row(r, clf.feature_names_) for r in rows])


@lru_cache(maxsize=1)
def get_explainer(clf: LoanClassifier) -> LimeTabularExplainer:
    background = _build_background_matrix(clf)
    return LimeTabularExplainer(
        training_data=background,
        feature_names=clf.feature_names_,
        class_names=["rejected", "approved"],
        mode="classification",
        discretize_continuous=True,
    )


def explain_row(
    clf: LoanClassifier,
    x_row: np.ndarray,
    num_features: int = 15,
    num_samples: int = 5000,
) -> Dict[str, float]:
    """Return {feature_name: local_weight} for the 'approved' class, top-k by |weight|."""
    explainer = get_explainer(clf)
    exp = explainer.explain_instance(
        x_row.reshape(-1), clf.predict_proba,
        num_features=num_features, num_samples=num_samples, labels=(1,),
    )
    weights = dict(exp.as_map()[1])
    result = {clf.feature_names_[idx]: round(float(w), 6) for idx, w in weights.items()}
    return dict(sorted(result.items(), key=lambda kv: abs(kv[1]), reverse=True))
