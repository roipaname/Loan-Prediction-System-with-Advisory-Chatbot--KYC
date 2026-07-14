"""
scripts/train_model.py
======================
Train all loan classifiers on the processed feature CSV, evaluate each on a
held-out test set, persist every model to models/<algorithm>.joblib, and
promote the best performer to models/best_model.joblib.

All four algorithms supported by LoanClassifier are trained in one run:
  random_forest | xgboost | naive_bayes | logistic_regression

Best-model selection
--------------------
Default ranking metric: roc_auc (configurable via --metric or the
MODEL_SELECTION_METRIC env var).  The composite rank is also computed
across roc_auc, f1_weighted, avg_precision, accuracy, mcc, and brier_score
to give a holistic view of each model.

Artefacts produced
------------------
  models/<algorithm>.joblib          — every trained model
  models/<algorithm>.json            — metadata sidecar (params, timestamp)
  models/best_model.joblib           — copy of the winning model
  models/best_model.json             — metadata + selection provenance

Usage
-----
    uv run python -m scripts.train_model
    uv run python -m scripts.train_model --metric f1_weighted
    uv run python -m scripts.train_model --tune --tune-n-iter 30
    uv run python -m scripts.train_model --data data/processed/loan_features.csv
    uv run python -m scripts.train_model --test-size 0.25 --threshold 0.45
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger as log
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Settings expose paths and training defaults so values stay in one place
from config.settings import (
    BEST_MODEL_PATH,
    CV_FOLDS,
    MODEL_SELECTION_METRIC,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.classifier.classifier import LoanClassifier
from src.classifier.evaluate import compare_classifiers, evaluate_classifier, print_report


# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------
# These mirror ENGINEERED_FEATURE_COLS in database/feature_eng.py.
# Keeping them explicit here makes the feature contract obvious and lets
# the training script run without importing the full feature-engineering module.

TARGET_COL = "loan_status"

# Dropped before building the feature matrix
_DROP_COLS: frozenset[str] = frozenset([TARGET_COL, "pipeline_version"])

# Categorical columns that will be one-hot encoded.
# drop_first=True removes one dummy per group to avoid perfect multicollinearity.
_CATEGORICAL_COLS: List[str] = [
    "person_gender",
    "person_education",
    "person_home_ownership",
    "loan_intent",
    "credit_score_tier",
    "income_bucket",
    "employment_stability",
]

# Boolean columns cast to int (0/1) so tree models treat them as ordinal
# and gradient-descent models handle them without special encoding.
_BOOL_COLS: List[str] = [
    "previous_loan_defaults_on_file",
    "young_inexperienced",
    "thin_credit_file",
    "credit_risk_interaction",
    "high_loan_burden_flag",
    "is_high_risk",
]

# Algorithm-level settings:
#   scale_features — prepend StandardScaler; needed by gradient-descent (LR)
#                    and recommended for GaussianNB (not strictly required)
#   class_weight   — GaussianNB has no such parameter so we pass None for it
_ALGORITHM_CONFIG: Dict[str, Dict[str, Any]] = {
    "random_forest":       {"scale_features": False, "class_weight": "balanced"},
    "xgboost":             {"scale_features": False, "class_weight": "balanced"},
    "naive_bayes":         {"scale_features": True,  "class_weight": None},
    "logistic_regression": {"scale_features": True,  "class_weight": "balanced"},
}


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

def prepare_features(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Transform the processed loan CSV into a numeric feature matrix.

    Steps
    -----
    1. Separate target (loan_status) from features.
    2. Drop metadata-only columns (pipeline_version).
    3. Cast boolean flags to int — keeps trees from treating True/False
       as string categories and ensures gradient-based models get 0/1 float.
    4. One-hot encode categorical columns (drop_first reduces collinearity).
    5. Coerce stray object columns to numeric and median-impute NaNs.
       NaNs arise from division-based engineered features (e.g. affordability_ratio
       when monthly_income == 0) on a small fraction of rows.

    Parameters
    ----------
    df : raw processed CSV as a DataFrame

    Returns
    -------
    X             : float64 ndarray  (n_samples, n_features)
    y             : int ndarray      (n_samples,)   0=rejected, 1=approved
    feature_names : list[str]        column names in feature-matrix order
    """
    y = df[TARGET_COL].astype(int).values

    X = df.drop(columns=[c for c in _DROP_COLS if c in df.columns])

    # Boolean → 0 / 1
    for col in _BOOL_COLS:
        if col in X.columns:
            X = X.copy()  # avoid SettingWithCopyWarning
            X[col] = X[col].astype(int)

    # One-hot encode categoricals that exist in this particular CSV
    cats_present = [c for c in _CATEGORICAL_COLS if c in X.columns]
    if cats_present:
        X = pd.get_dummies(X, columns=cats_present, drop_first=True)

    # Coerce any remaining object/bool dtype columns that slipped through
    X = X.apply(pd.to_numeric, errors="coerce")

    # Median imputation is robust to outliers and preserves the marginal
    # distribution better than mean imputation for skewed financial data
    imputer = SimpleImputer(strategy="median")
    X_arr = imputer.fit_transform(X).astype(np.float64)
    feature_names = list(X.columns)

    class_counts = np.bincount(y)
    log.info("Feature matrix  : %d samples × %d features", *X_arr.shape)
    log.info(
        "Class distribution — rejected=%d (%.1f%%)  approved=%d (%.1f%%)",
        class_counts[0], 100 * class_counts[0] / len(y),
        class_counts[1], 100 * class_counts[1] / len(y),
    )
    return X_arr, y, feature_names


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_all(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    *,
    tune: bool = False,
    tune_strategy: str = "random",
    tune_n_iter: int = 20,
    tune_cv: int = CV_FOLDS,
    tune_scoring: str = "roc_auc",
    threshold: float = 0.5,
) -> Tuple[Dict[str, LoanClassifier], List[Dict[str, Any]]]:
    """
    Train (and optionally tune) every algorithm defined in _ALGORITHM_CONFIG.

    When tune=True, RandomizedSearchCV (or GridSearchCV) is run over the
    built-in search spaces defined in classifier.py before the final fit.

    Parameters
    ----------
    X_train, y_train : training partition
    X_test,  y_test  : held-out evaluation partition
    feature_names    : column names for feature-importance reporting
    tune             : run hyper-parameter search before fitting
    tune_strategy    : 'random' (fast, default) or 'grid' (exhaustive)
    tune_n_iter      : candidates for random search
    tune_cv          : CV folds during tuning
    tune_scoring     : sklearn scoring string used to rank candidates
    threshold        : probability cut-off for positive class in evaluation

    Returns
    -------
    classifiers  : {algorithm_name: fitted LoanClassifier}
    eval_results : list of evaluate_classifier() dicts, one per algorithm
    """
    classifiers: Dict[str, LoanClassifier] = {}
    eval_results: List[Dict[str, Any]] = []

    for algo, cfg in _ALGORITHM_CONFIG.items():
        log.info("─" * 60)
        log.info(
            "Training  %-22s  scale=%s  class_weight=%s",
            algo.upper(), cfg["scale_features"], cfg["class_weight"],
        )

        clf = LoanClassifier(
            algorithm=algo,
            random_state=RANDOM_STATE,
            **cfg,
        )

        if tune:
            log.info(
                "Tuning %s — strategy=%s  n_iter=%d  cv=%d  scoring=%s",
                algo, tune_strategy, tune_n_iter, tune_cv, tune_scoring,
            )
            clf.tune(
                X_train, y_train,
                strategy=tune_strategy,
                n_iter=tune_n_iter,
                cv=tune_cv,
                scoring=tune_scoring,
                refit=True,   # refit on full training set with best params
                verbose=0,
            )
            # tune() does not call fit() so feature_names_ isn't set
            clf.feature_names_ = feature_names
        else:
            clf.fit(X_train, y_train, feature_names=feature_names)

        metrics = evaluate_classifier(clf, X_test, y_test, threshold=threshold)
        classifiers[algo] = clf
        eval_results.append(metrics)

    return classifiers, eval_results


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_all_models(
    classifiers: Dict[str, LoanClassifier],
    eval_results: List[Dict[str, Any]],
    models_dir: Path,
    selection_metric: str,
) -> LoanClassifier:
    """
    Persist every trained classifier and promote the winner to best_model.

    Individual models  → <models_dir>/<algorithm>.joblib  +  .json
    Best model         → BEST_MODEL_PATH (.joblib)         +  .json

    The best_model.json sidecar is enriched with selection provenance
    (which metric won, all model scores, timestamp) so the decision is
    fully auditable without re-running training.

    Parameters
    ----------
    classifiers      : {algorithm_name: fitted LoanClassifier}
    eval_results     : list of dicts from evaluate_classifier()
    models_dir       : directory to write all model files
    selection_metric : primary metric used to crown the best model

    Returns
    -------
    The best-performing LoanClassifier instance.
    """
    models_dir.mkdir(parents=True, exist_ok=True)

    # --- Persist every model individually ---
    for algo, clf in classifiers.items():
        save_path = clf.save(models_dir / algo)
        log.info("Saved  %-22s → %s", algo, save_path)

    # --- Rank and identify the winner ---
    ranked = compare_classifiers(eval_results)
    log.info(
        "\nRanking by composite score:\n%s",
        ranked[["model", selection_metric, "composite_rank"]].to_string(),
    )

    # Primary sort: the user-chosen selection metric (higher = better)
    best_row = ranked.sort_values(selection_metric, ascending=False).iloc[0]
    best_algo  = str(best_row["algorithm"])
    best_score = float(best_row[selection_metric])
    best_clf   = classifiers[best_algo]

    log.info(
        "Winner: %-22s  %s=%.4f",
        best_algo.upper(), selection_metric, best_score,
    )

    # --- Save the full ranked comparison table for the frontend's Findings
    # & Comparisons page (best_model.json only stores the single selection
    # metric's scores, not the full metric suite for all algorithms) ---
    comparison = ranked.copy()
    comparison["is_champion"] = comparison["algorithm"] == best_algo
    comparison.to_csv(models_dir / "model_comparison.csv", index=True, index_label="rank")
    log.info("Saved model comparison table → %s", models_dir / "model_comparison.csv")

    # --- Save to the canonical best-model path (overwrites previous run) ---
    # BEST_MODEL_PATH ends in .joblib; strip suffix so LoanClassifier.save()
    # can append both .joblib and .json itself
    best_stem = BEST_MODEL_PATH.with_suffix("")
    best_clf.save(best_stem)

    # --- Annotate the JSON sidecar with selection provenance ---
    # LoanClassifier.save() already wrote algorithm + params; we extend it
    meta_path = best_stem.with_suffix(".json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update({
        "best_algorithm":   best_algo,
        "selection_metric": selection_metric,
        "selection_score":  round(best_score, 6),
        "all_scores": {
            r["algorithm"]: round(r.get(selection_metric, float("nan")), 6)
            for r in eval_results
        },
        "promoted_at": datetime.utcnow().isoformat() + "Z",
    })
    meta_path.write_text(json.dumps(meta, indent=2, default=str))

    log.success("Best model  → %s", BEST_MODEL_PATH)
    log.success("Load via:      LoanClassifier.load('%s')", best_stem)

    return best_clf


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--data",
        default=str(PROCESSED_DATA_DIR / "loan_features.csv"),
        metavar="PATH",
        help="Processed feature CSV (default: data/processed/loan_features.csv).",
    )
    p.add_argument(
        "--test-size", type=float, default=TEST_SIZE, metavar="FRAC",
        help="Fraction of data held out for evaluation (default: %(default)s).",
    )
    p.add_argument(
        "--metric",
        default=MODEL_SELECTION_METRIC,
        choices=["roc_auc", "f1_weighted", "f1_macro", "accuracy", "avg_precision", "mcc"],
        help="Primary metric used to select the best model (default: %(default)s).",
    )
    p.add_argument(
        "--threshold", type=float, default=0.5, metavar="P",
        help="Probability threshold for positive-class decision (default: %(default)s).",
    )
    p.add_argument(
        "--tune", action="store_true",
        help="Run hyper-parameter search before fitting each algorithm.",
    )
    p.add_argument(
        "--tune-strategy", choices=["random", "grid"], default="random",
        help="Search strategy — 'random' is faster; 'grid' is exhaustive (default: random).",
    )
    p.add_argument(
        "--tune-n-iter", type=int, default=20, metavar="N",
        help="Number of candidates for random search (default: %(default)s).",
    )
    p.add_argument(
        "--tune-cv", type=int, default=CV_FOLDS, metavar="K",
        help="CV folds used during tuning (default: %(default)s).",
    )
    p.add_argument(
        "--models-dir", default=str(MODELS_DIR), metavar="DIR",
        help="Directory where all models are saved (default: %(default)s).",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    # ------------------------------------------------------------------
    # Guard: processed data must exist before training
    # ------------------------------------------------------------------
    data_path = Path(args.data)
    if not data_path.exists():
        log.error("Data file not found: %s", data_path)
        log.error("Run the feature engineering pipeline first:")
        log.error("  uv run python -m database.feature_eng")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    log.info("Loading  %s", data_path)
    df = pd.read_csv(data_path)
    log.info("Loaded   %d rows × %d columns", *df.shape)

    # ------------------------------------------------------------------
    # Feature preparation
    # ------------------------------------------------------------------
    X, y, feature_names = prepare_features(df)

    # ------------------------------------------------------------------
    # Stratified train / test split
    # Stratify preserves the approval-rate ratio in both partitions,
    # which is important given the class imbalance in loan data.
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    log.info(
        "Split    %d train / %d test  "
        "(train approval=%.1f%%  test approval=%.1f%%)",
        len(y_train), len(y_test),
        100 * y_train.mean(), 100 * y_test.mean(),
    )

    # ------------------------------------------------------------------
    # Train all models
    # ------------------------------------------------------------------
    classifiers, eval_results = train_all(
        X_train, y_train,
        X_test,  y_test,
        feature_names=feature_names,
        tune=args.tune,
        tune_strategy=args.tune_strategy,
        tune_n_iter=args.tune_n_iter,
        tune_cv=args.tune_cv,
        tune_scoring=args.metric,
        threshold=args.threshold,
    )

    # ------------------------------------------------------------------
    # Report — individual scorecards + comparison table
    # ------------------------------------------------------------------
    print_report(eval_results)

    # ------------------------------------------------------------------
    # Save all models and promote the best
    # ------------------------------------------------------------------
    best_clf = save_all_models(
        classifiers,
        eval_results,
        models_dir=Path(args.models_dir),
        selection_metric=args.metric,
    )

    log.success("=" * 60)
    log.success("Training complete.  Best model: %s", best_clf.algorithm.upper())
    log.success("=" * 60)


if __name__ == "__main__":
    main()
