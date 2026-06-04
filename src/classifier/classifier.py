

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.classifier.logistic_regression import CustomLogisticRegression

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

AlgorithmName = Literal["random_forest", "xgboost", "naive_bayes", "logistic_regression"]
TuningStrategy = Literal["grid", "random"]


# ---------------------------------------------------------------------------
# Default hyper-parameter search spaces
# ---------------------------------------------------------------------------

_DEFAULT_PARAM_GRIDS: Dict[str, Dict[str, list]] = {
    "random_forest": {
        "classifier__n_estimators":      [100, 200, 300],
        "classifier__max_depth":         [None, 10, 20, 30],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf":  [1, 2, 4],
        "classifier__max_features":      ["sqrt", "log2"],
    },
    "xgboost": {
        "classifier__n_estimators":  [100, 200, 300],
        "classifier__max_depth":     [3, 5, 7, 9],
        "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "classifier__subsample":     [0.6, 0.8, 1.0],
        "classifier__colsample_bytree": [0.6, 0.8, 1.0],
        "classifier__reg_alpha":     [0, 0.1, 1.0],
        "classifier__reg_lambda":    [1.0, 2.0, 5.0],
    },
    "naive_bayes": {
        "classifier__var_smoothing": np.logspace(-12, -6, 7).tolist(),
    },
    "logistic_regression": {
        "classifier__lr":        [0.001, 0.01, 0.05, 0.1],
        "classifier__C":         [0.01, 0.1, 1.0, 10.0],
        "classifier__penalty":   ["l1", "l2", "elasticnet"],
        "classifier__l1_ratio":  [0.25, 0.5, 0.75],
        "classifier__max_iter":  [300, 500],
    },
}

# Smaller random-search distributions (subset of the grid)
_DEFAULT_PARAM_DISTRIBUTIONS: Dict[str, Dict[str, Any]] = {
    "random_forest": {
        "classifier__n_estimators":      [50, 100, 200, 400],
        "classifier__max_depth":         [None, 5, 10, 20, 30],
        "classifier__min_samples_split": [2, 5, 10, 20],
        "classifier__min_samples_leaf":  [1, 2, 4, 8],
        "classifier__max_features":      ["sqrt", "log2", 0.5],
    },
    "xgboost": {
        "classifier__n_estimators":      [50, 100, 200, 400],
        "classifier__max_depth":         [3, 4, 5, 6, 7, 8, 9],
        "classifier__learning_rate":     [0.005, 0.01, 0.05, 0.1, 0.2, 0.3],
        "classifier__subsample":         [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "classifier__colsample_bytree":  [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "classifier__reg_alpha":         [0, 0.01, 0.1, 1.0, 5.0],
        "classifier__reg_lambda":        [0.5, 1.0, 2.0, 5.0, 10.0],
        "classifier__min_child_weight":  [1, 3, 5, 7],
    },
    "naive_bayes": {
        "classifier__var_smoothing": np.logspace(-12, -4, 20).tolist(),
    },
    "logistic_regression": {
        "classifier__lr":       [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
        "classifier__C":        [0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 50.0],
        "classifier__penalty":  ["l1", "l2", "elasticnet"],
        "classifier__l1_ratio": [0.1, 0.25, 0.5, 0.75, 0.9],
        "classifier__max_iter": [200, 300, 500, 800],
    },
}


# ---------------------------------------------------------------------------
# LoanClassifier
# ---------------------------------------------------------------------------

class LoanClassifier:
    """
    Unified loan-prediction classifier.

    Parameters
    ----------
    algorithm : str
        One of 'random_forest', 'xgboost', 'naive_bayes', 'logistic_regression'.
    random_state : int
        Seed used for all stochastic components.
    class_weight : {'balanced', None} or dict
        Class weighting strategy (not supported by naive_bayes; ignored there).
    calibrate : bool
        Wrap the fitted estimator in CalibratedClassifierCV (sigmoid method)
        to improve probability estimates.  Adds a second CV pass.
    scale_features : bool
        Prepend a StandardScaler in the pipeline.  Recommended for
        logistic_regression and naive_bayes.
    **model_kwargs
        Extra keyword arguments forwarded to the underlying estimator constructor.
    """

    def __init__(
        self,
        algorithm: AlgorithmName = "random_forest",
        random_state: int = 42,
        class_weight: Optional[Union[str, dict]] = "balanced",
        calibrate: bool = False,
        scale_features: bool = False,
        **model_kwargs: Any,
    ) -> None:
        if algorithm not in _DEFAULT_PARAM_GRIDS:
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. "
                f"Choose from: {list(_DEFAULT_PARAM_GRIDS)}"
            )

        self.algorithm     = algorithm
        self.random_state  = random_state
        self.class_weight  = class_weight
        self.calibrate     = calibrate
        self.scale_features = scale_features
        self.model_kwargs  = model_kwargs

        self.pipeline_: Optional[Pipeline] = None
        self.best_params_: Optional[dict]  = None
        self.feature_names_: Optional[list] = None
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Estimator factory
    # ------------------------------------------------------------------

    def _build_estimator(self, **overrides: Any) -> Any:
        """Instantiate the raw estimator (no scaler, no calibration)."""
        kwargs = {**self.model_kwargs, **overrides}

        if self.algorithm == "random_forest":
            return RandomForestClassifier(
                random_state=self.random_state,
                class_weight=self.class_weight,
                n_jobs=-1,
                **kwargs,
            )

        elif self.algorithm == "xgboost":
            # XGBoost uses scale_pos_weight instead of class_weight
            scale_pos_weight = kwargs.pop("scale_pos_weight", 1)
            if self.class_weight == "balanced":
                # Will be computed properly at fit time if y is available
                pass
            return XGBClassifier(
                random_state=self.random_state,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                use_label_encoder=False,
                n_jobs=-1,
                verbosity=0,
                **kwargs,
            )

        elif self.algorithm == "naive_bayes":
            return GaussianNB(**kwargs)

        else:  # logistic_regression
            return CustomLogisticRegression(
                random_state=self.random_state,
                class_weight=self.class_weight,
                **kwargs,
            )

    def _build_pipeline(self, estimator: Any) -> Pipeline:
        """Wrap estimator in optional scaler and return a Pipeline."""
        steps: list = []
        if self.scale_features:
            steps.append(("scaler", StandardScaler()))
        steps.append(("classifier", estimator))
        return Pipeline(steps)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        feature_names: Optional[list] = None,
    ) -> "LoanClassifier":
        """
        Fit the classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        feature_names : list, optional
            Column names for feature-importance reporting.

        Returns
        -------
        self
        """
        if isinstance(X, pd.DataFrame):
            feature_names = feature_names or list(X.columns)
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        self.feature_names_ = feature_names

        # For XGBoost + balanced, compute scale_pos_weight from y
        kwargs: Dict[str, Any] = {}
        if self.algorithm == "xgboost" and self.class_weight == "balanced":
            neg = np.sum(y == 0)
            pos = np.sum(y == 1)
            kwargs["scale_pos_weight"] = neg / max(pos, 1)
            log.info("XGBoost scale_pos_weight = %.3f", kwargs["scale_pos_weight"])

        estimator = self._build_estimator(**kwargs)

        if self.calibrate:
            cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            estimator = CalibratedClassifierCV(estimator, method="sigmoid", cv=cv_inner)

        self.pipeline_ = self._build_pipeline(estimator)
        self.pipeline_.fit(X, y)
        self._is_fitted = True
        log.info("Fitted %s classifier.", self.algorithm)
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        self._check_fitted()
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.pipeline_.predict(X)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        self._check_fitted()
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.pipeline_.predict_proba(X)

    # ------------------------------------------------------------------
    # Hyper-parameter tuning
    # ------------------------------------------------------------------

    def tune(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        strategy: TuningStrategy = "random",
        param_grid: Optional[dict] = None,
        scoring: str = "roc_auc",
        cv: int = 5,
        n_iter: int = 30,
        n_jobs: int = -1,
        refit: bool = True,
        verbose: int = 1,
    ) -> "LoanClassifier":
        """
        Run hyper-parameter search.

        Parameters
        ----------
        X, y : training data
        strategy : 'grid' (exhaustive) or 'random' (sampled, faster)
        param_grid : custom search space; defaults to built-in grids
        scoring : sklearn scoring string
        cv : number of CV folds
        n_iter : number of random candidates (strategy='random' only)
        n_jobs : parallel jobs
        refit : if True, refit the best model on the full dataset
        verbose : verbosity level for the searcher

        Returns
        -------
        self  (self.best_params_ is populated)
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # Build a fresh pipeline to search over
        estimator = self._build_estimator()
        pipeline  = self._build_pipeline(estimator)

        cv_splitter = StratifiedKFold(
            n_splits=cv, shuffle=True, random_state=self.random_state
        )

        if strategy == "grid":
            space = param_grid or _DEFAULT_PARAM_GRIDS[self.algorithm]
            searcher = GridSearchCV(
                pipeline,
                param_grid=space,
                scoring=scoring,
                cv=cv_splitter,
                n_jobs=n_jobs,
                refit=refit,
                verbose=verbose,
            )
        else:
            space = param_grid or _DEFAULT_PARAM_DISTRIBUTIONS[self.algorithm]
            searcher = RandomizedSearchCV(
                pipeline,
                param_distributions=space,
                n_iter=n_iter,
                scoring=scoring,
                cv=cv_splitter,
                n_jobs=n_jobs,
                refit=refit,
                random_state=self.random_state,
                verbose=verbose,
            )

        log.info(
            "Starting %s search for %s (scoring=%s, cv=%d) …",
            strategy, self.algorithm, scoring, cv,
        )
        searcher.fit(X, y)

        self.best_params_ = searcher.best_params_
        log.info("Best CV %s = %.4f", scoring, searcher.best_score_)
        log.info("Best params: %s", self.best_params_)

        if refit:
            self.pipeline_  = searcher.best_estimator_
            self._is_fitted = True

        return self

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Compute a full evaluation suite.

        Returns
        -------
        dict with keys:
            accuracy, roc_auc, f1_macro, f1_weighted,
            confusion_matrix, classification_report, threshold
        """
        self._check_fitted()
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        proba = self.predict_proba(X)[:, 1]
        preds = (proba >= threshold).astype(int)

        try:
            auc = roc_auc_score(y, proba)
        except ValueError:
            auc = float("nan")

        results = {
            "algorithm":              self.algorithm,
            "threshold":              threshold,
            "accuracy":               float(accuracy_score(y, preds)),
            "roc_auc":                float(auc),
            "f1_macro":               float(f1_score(y, preds, average="macro",   zero_division=0)),
            "f1_weighted":            float(f1_score(y, preds, average="weighted", zero_division=0)),
            "confusion_matrix":       confusion_matrix(y, preds).tolist(),
            "classification_report":  classification_report(y, preds, zero_division=0),
        }

        log.info(
            "[%s] accuracy=%.4f  AUC=%.4f  F1(w)=%.4f",
            self.algorithm, results["accuracy"], results["roc_auc"], results["f1_weighted"],
        )
        return results

    def cross_validate(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        cv: int = 5,
        scoring: str = "roc_auc",
    ) -> Dict[str, float]:
        """Run stratified k-fold cross-validation and return mean ± std."""
        self._check_fitted()
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        cv_splitter = StratifiedKFold(
            n_splits=cv, shuffle=True, random_state=self.random_state
        )
        scores = cross_val_score(
            self.pipeline_, X, y, cv=cv_splitter, scoring=scoring, n_jobs=-1
        )
        result = {"mean": float(scores.mean()), "std": float(scores.std()), "scores": scores.tolist()}
        log.info("CV %s: %.4f ± %.4f", scoring, result["mean"], result["std"])
        return result

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Return a DataFrame of feature importances sorted descending.

        Supports RandomForest (feature_importances_), XGBoost (feature_importances_),
        and CustomLogisticRegression (absolute coefficients).
        Naive Bayes does not expose feature importances — returns None.
        """
        self._check_fitted()
        estimator = self._get_base_estimator()

        importances: Optional[np.ndarray] = None

        if hasattr(estimator, "feature_importances_"):
            importances = estimator.feature_importances_
        elif hasattr(estimator, "coef_"):
            importances = np.abs(estimator.coef_).flatten()
        else:
            log.warning("%s does not expose feature importances.", self.algorithm)
            return None

        names = self.feature_names_ or [f"f{i}" for i in range(len(importances))]
        df = (
            pd.DataFrame({"feature": names, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        return df

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> Path:
        """
        Persist the classifier to disk.

        Saves two files:
          <path>.joblib  — serialised pipeline
          <path>.json    — metadata (algorithm, params, timestamp …)

        Parameters
        ----------
        path : file path *without* extension

        Returns
        -------
        Path to the .joblib file
        """
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_path = path.with_suffix(".joblib")
        meta_path  = path.with_suffix(".json")

        joblib.dump(self, model_path)

        metadata = {
            "algorithm":    self.algorithm,
            "best_params":  self.best_params_,
            "feature_names": self.feature_names_,
            "calibrate":    self.calibrate,
            "scale_features": self.scale_features,
            "saved_at":     datetime.utcnow().isoformat(),
        }
        meta_path.write_text(json.dumps(metadata, indent=2, default=str))

        log.info("Model saved → %s", model_path)
        return model_path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "LoanClassifier":
        """
        Load a previously saved classifier.

        Parameters
        ----------
        path : file path *without* extension  (same as used in save())

        Returns
        -------
        LoanClassifier instance
        """
        path       = Path(path)
        model_path = path.with_suffix(".joblib")

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        instance = joblib.load(model_path)
        log.info("Model loaded ← %s", model_path)
        return instance

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._is_fitted or self.pipeline_ is None:
            raise RuntimeError(
                "Classifier is not fitted yet. Call .fit() or .tune(refit=True) first."
            )

    def _get_base_estimator(self) -> Any:
        """Unwrap Pipeline (and optional CalibratedClassifierCV) to the raw estimator."""
        est = self.pipeline_.named_steps["classifier"]
        if isinstance(est, CalibratedClassifierCV):
            # After fit, calibrated_classifiers_ holds the base models
            est = est.estimator
        return est

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            f"LoanClassifier(algorithm={self.algorithm})",
            f"  fitted       : {self._is_fitted}",
            f"  calibrated   : {self.calibrate}",
            f"  scale_feats  : {self.scale_features}",
            f"  best_params  : {self.best_params_}",
            f"  feature_names: {self.feature_names_}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"LoanClassifier(algorithm={self.algorithm!r}, "
            f"fitted={self._is_fitted}, "
            f"calibrate={self.calibrate})"
        )
