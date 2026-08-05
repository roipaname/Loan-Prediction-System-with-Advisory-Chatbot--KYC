"""
Standalone evaluation utilities for LoanClassifier. Extends the built-in
.evaluate() with average precision, Brier score, and MCC, plus multi-model
comparison and console reporting.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from loguru import logger as log
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.classifier.classifier import LoanClassifier


# metrics in the composite ranking (all higher-is-better); brier_score is
# handled separately since lower is better there
_RANK_METRICS: List[str] = ["roc_auc", "f1_weighted", "avg_precision", "accuracy", "mcc"]


def evaluate_classifier(
    clf: LoanClassifier,
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    *,
    threshold: float = 0.5,
    label: Optional[str] = None,
) -> Dict[str, Any]:
    """Full metric suite for one fitted LoanClassifier: everything in
    LoanClassifier.evaluate() plus average precision, Brier score, and MCC —
    accuracy alone is misleading on imbalanced loan-approval data."""
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values

    proba = clf.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)

    # need at least one positive sample
    try:
        auc = float(roc_auc_score(y, proba))
    except ValueError:
        auc = float("nan")

    try:
        ap = float(average_precision_score(y, proba))
    except ValueError:
        ap = float("nan")

    results: Dict[str, Any] = {
        "label":     label or clf.algorithm,
        "algorithm": clf.algorithm,
        "threshold": threshold,
        "accuracy":    float(accuracy_score(y, preds)),
        "precision":   float(precision_score(y, preds, zero_division=0)),
        "recall":      float(recall_score(y, preds, zero_division=0)),
        "f1_macro":    float(f1_score(y, preds, average="macro",    zero_division=0)),
        "f1_weighted": float(f1_score(y, preds, average="weighted", zero_division=0)),
        "roc_auc":       auc,
        "avg_precision": ap,
        "brier_score":   float(brier_score_loss(y, proba)),   # calibration quality, lower = better
        "mcc":           float(matthews_corrcoef(y, preds)),  # robust to class imbalance, range -1 to +1
        "confusion_matrix":      confusion_matrix(y, preds).tolist(),
        "classification_report": classification_report(y, preds, zero_division=0),
    }

    log.info(
        "[%s] acc=%.4f  AUC=%.4f  F1(w)=%.4f  AP=%.4f  MCC=%.4f",
        results["label"],
        results["accuracy"],
        results["roc_auc"],
        results["f1_weighted"],
        results["avg_precision"],
        results["mcc"],
    )
    return results


def compare_classifiers(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Rank multiple evaluate_classifier() result dicts side-by-side. Each
    metric is ranked independently and composite_rank is the average of
    those per-metric ranks (ties broken by roc_auc). Returns a DataFrame
    1-indexed by rank position, sorted best to worst (composite_rank asc).
    """
    if not results:
        raise ValueError("results list is empty")

    rows = [
        {
            "model":         r["label"],
            "algorithm":     r["algorithm"],
            "accuracy":      r["accuracy"],
            "precision":     r["precision"],
            "recall":        r["recall"],
            "f1_macro":      r["f1_macro"],
            "f1_weighted":   r["f1_weighted"],
            "roc_auc":       r["roc_auc"],
            "avg_precision": r["avg_precision"],
            "brier_score":   r["brier_score"],
            "mcc":           r["mcc"],
        }
        for r in results
    ]
    df = pd.DataFrame(rows)

    # brier score is lower-is-better, ranked separately below
    for col in _RANK_METRICS:
        df[f"_r_{col}"] = df[col].rank(ascending=False, method="min")
    df["_r_brier"] = df["brier_score"].rank(ascending=True, method="min")

    rank_cols = [f"_r_{c}" for c in _RANK_METRICS] + ["_r_brier"]
    df["composite_rank"] = df[rank_cols].mean(axis=1)
    df.drop(columns=rank_cols, inplace=True)

    df.sort_values("composite_rank", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.index += 1  # 1-based so df.index == rank position

    return df


def print_report(
    results: Union[Dict[str, Any], List[Dict[str, Any]]],
    *,
    show_classification_report: bool = True,
    show_confusion_matrix: bool = True,
) -> None:
    """Pretty-print evaluation results to stdout. Accepts a single dict or a
    list; a list also gets a side-by-side comparison table appended."""
    if isinstance(results, dict):
        results = [results]

    for r in results:
        _print_single(r, show_classification_report, show_confusion_matrix)

    if len(results) > 1:
        _print_comparison(results)


def _print_single(
    r: Dict[str, Any],
    show_clf_report: bool,
    show_cm: bool,
) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  {r['label'].upper()}")
    print(sep)
    print(f"  Threshold       : {r['threshold']:.2f}")
    print(f"  Accuracy        : {r['accuracy']:.4f}")
    print(f"  Precision       : {r['precision']:.4f}")
    print(f"  Recall          : {r['recall']:.4f}")
    print(f"  F1 (macro)      : {r['f1_macro']:.4f}")
    print(f"  F1 (weighted)   : {r['f1_weighted']:.4f}")
    print(f"  ROC-AUC         : {r['roc_auc']:.4f}")
    print(f"  Avg Precision   : {r['avg_precision']:.4f}")
    print(f"  Brier Score     : {r['brier_score']:.4f}  (lower = better)")
    print(f"  MCC             : {r['mcc']:.4f}")

    if show_cm:
        cm = r["confusion_matrix"]
        print(f"\n  Confusion Matrix:")
        print(f"    TN={cm[0][0]:>7,}  FP={cm[0][1]:>7,}")
        print(f"    FN={cm[1][0]:>7,}  TP={cm[1][1]:>7,}")

    if show_clf_report:
        print(f"\n  Classification Report:")
        for line in r["classification_report"].splitlines():
            print(f"    {line}")


def _print_comparison(results: List[Dict[str, Any]]) -> None:
    df = compare_classifiers(results)
    display_cols = [
        "model", "accuracy", "roc_auc", "f1_weighted",
        "avg_precision", "brier_score", "mcc", "composite_rank",
    ]
    print("\n" + "=" * 70)
    print("  MODEL COMPARISON  (rank 1 = best overall)")
    print("=" * 70)
    print(df[display_cols].to_string(float_format=lambda x: f"{x:.4f}"))
    print()
