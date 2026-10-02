"""
Per-applicant local feature attribution via LIME, not SHAP (numba/llvmlite
is unstable on this machine). get_feature_importance() on the classifier
only gives global importances, this answers "why did this one applicant
get this outcome".
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict

import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

from backend.services.reference_stats import processed_features_path
from config.settings import RANDOM_STATE
from src.ai_advisor.loan_context_builder import _prepare_single_row
from src.classifier.classifier import LoanClassifier

_BACKGROUND_SAMPLE_SIZE = 300


def _build_background_matrix(clf: LoanClassifier) -> np.ndarray:
    df = pd.read_csv(processed_features_path())
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
        class_names=["repaid", "default"],
        mode="classification",
        discretize_continuous=True,
    )


def explain_row(
    clf: LoanClassifier,
    x_row: np.ndarray,
    num_features: int = 15,
    num_samples: int = 5000,
) -> Dict[str, float]:
    """Return {feature_name: local_weight} toward approval, top-k by |weight|."""
    explainer = get_explainer(clf)
    exp = explainer.explain_instance(
        x_row.reshape(-1), clf.predict_proba,
        num_features=num_features, num_samples=num_samples, labels=(1,),
    )
    # class 1 is default (see LoanContextBuilder._finalise); P(approve) =
    # 1 - P(default), so LIME's local weights toward approval are the negation
    weights = dict(exp.as_map()[1])
    result = {clf.feature_names_[idx]: round(-float(w), 6) for idx, w in weights.items()}
    return dict(sorted(result.items(), key=lambda kv: abs(kv[1]), reverse=True))
