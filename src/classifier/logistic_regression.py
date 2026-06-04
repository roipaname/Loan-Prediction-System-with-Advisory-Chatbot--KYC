"""
logistic_regression.py
======================
Custom optimised Logistic Regression for the Loan Prediction System.

Implements mini-batch SGD with:
  - L1 / L2 / ElasticNet regularisation
  - Adam adaptive learning-rate (momentum + RMSProp)
  - Early stopping on a held-out validation split
  - Class-weight balancing (handles the typical loan approval imbalance)
  - Full sklearn-compatible API (fit / predict / predict_proba / score)

Usage
-----
    from src.classifier.logistic_regression import CustomLogisticRegression

    clf = CustomLogisticRegression(
        lr=0.01, max_iter=500, penalty="elasticnet", l1_ratio=0.5,
        early_stopping=True, class_weight="balanced"
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        z >= 0,
        1.0 / (1.0 + np.exp(-z)),
        np.exp(z) / (1.0 + np.exp(z)),
    )


# ---------------------------------------------------------------------------
# Custom Logistic Regression
# ---------------------------------------------------------------------------

class CustomLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Logistic Regression trained via mini-batch SGD with Adam updates.

    Parameters
    ----------
    lr : float
        Initial learning rate (default 0.01).
    max_iter : int
        Maximum number of full-data passes (epochs).
    batch_size : int
        Mini-batch size; -1 uses full-batch gradient descent.
    penalty : {'l1', 'l2', 'elasticnet', None}
        Regularisation type.
    C : float
        Inverse regularisation strength.  Smaller → stronger regularisation.
    l1_ratio : float
        ElasticNet mixing parameter (0 = pure L2, 1 = pure L1).
    fit_intercept : bool
        Whether to fit a bias term.
    tol : float
        Convergence tolerance on loss improvement.
    early_stopping : bool
        Hold out `validation_fraction` of training data; stop when val-loss
        does not improve for `n_iter_no_change` consecutive epochs.
    n_iter_no_change : int
        Patience for early stopping.
    validation_fraction : float
        Fraction of training data used as validation split.
    class_weight : {'balanced', None} or dict
        Re-weight samples to handle class imbalance.
    use_adam : bool
        Use Adam update rule (momentum + RMSProp).
    beta1 : float
        Adam first-moment decay (default 0.9).
    beta2 : float
        Adam second-moment decay (default 0.999).
    epsilon : float
        Adam numerical stability constant.
    random_state : int, optional
        Seed for reproducibility.
    verbose : bool
        Log training progress every 50 epochs.
    """

    def __init__(
        self,
        lr: float = 0.01,
        max_iter: int = 500,
        batch_size: int = 256,
        penalty: Optional[Literal["l1", "l2", "elasticnet"]] = "l2",
        C: float = 1.0,
        l1_ratio: float = 0.5,
        fit_intercept: bool = True,
        tol: float = 1e-4,
        early_stopping: bool = True,
        n_iter_no_change: int = 10,
        validation_fraction: float = 0.1,
        class_weight: Optional[str | dict] = "balanced",
        use_adam: bool = True,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        random_state: Optional[int] = 42,
        verbose: bool = False,
    ) -> None:
        self.lr = lr
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.penalty = penalty
        self.C = C
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.validation_fraction = validation_fraction
        self.class_weight = class_weight
        self.use_adam = use_adam
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_sample_weights(self, y: np.ndarray) -> np.ndarray:
        """Return per-sample weights according to class_weight setting."""
        n_samples = len(y)
        classes, counts = np.unique(y, return_counts=True)

        if self.class_weight is None:
            return np.ones(n_samples)

        if self.class_weight == "balanced":
            weight_map = {
                cls: n_samples / (len(classes) * cnt)
                for cls, cnt in zip(classes, counts)
            }
        elif isinstance(self.class_weight, dict):
            weight_map = self.class_weight
        else:
            raise ValueError(f"Unknown class_weight: {self.class_weight!r}")

        return np.array([weight_map[yi] for yi in y])

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def _loss_and_grad(
        self,
        X_b: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
        sw: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Binary cross-entropy loss + gradient for a batch."""
        p = np.clip(_sigmoid(X_b @ w), 1e-15, 1 - 1e-15)
        bce = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        loss = np.dot(sw, bce) / sw.sum()
        grad = X_b.T @ ((p - y) * sw) / sw.sum()

        # Regularisation (skip bias at index 0)
        offset = 1 if self.fit_intercept else 0
        if self.penalty and self.C > 0:
            reg_w = w[offset:]
            lam = 1.0 / self.C
            if self.penalty == "l2":
                loss += 0.5 * lam * np.dot(reg_w, reg_w)
                grad[offset:] += lam * reg_w
            elif self.penalty == "l1":
                loss += lam * np.abs(reg_w).sum()
                grad[offset:] += lam * np.sign(reg_w)
            else:  # elasticnet
                r = self.l1_ratio
                loss += lam * (r * np.abs(reg_w).sum() + 0.5 * (1 - r) * np.dot(reg_w, reg_w))
                grad[offset:] += lam * (r * np.sign(reg_w) + (1 - r) * reg_w)

        return float(loss), grad

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CustomLogisticRegression":
        """Fit the model to training data."""
        X, y = check_X_y(X, y)
        rng = np.random.default_rng(self.random_state)

        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_
        y_enc = self.le_.transform(y).astype(float)

        # Optional validation split
        if self.early_stopping:
            n_val = max(1, int(len(y_enc) * self.validation_fraction))
            idx = rng.permutation(len(y_enc))
            val_idx, train_idx = idx[:n_val], idx[n_val:]
            X_val, y_val = X[val_idx], y_enc[val_idx]
            X_tr,  y_tr  = X[train_idx], y_enc[train_idx]
            sw_val = self._compute_sample_weights(y_val)
        else:
            X_tr, y_tr = X, y_enc
            X_val = y_val = sw_val = None

        sw_tr = self._compute_sample_weights(y_tr)
        X_b   = self._add_intercept(X_tr) if self.fit_intercept else X_tr
        w     = np.zeros(X_b.shape[1])

        # Adam moments
        m = np.zeros_like(w)
        v = np.zeros_like(w)
        t = 0

        bs = self.batch_size if self.batch_size > 0 else len(y_tr)
        best_val_loss = np.inf
        best_w        = w.copy()
        no_improve    = 0
        prev_loss     = np.inf

        self.loss_curve_:     list[float] = []
        self.val_loss_curve_: list[float] = []

        for epoch in range(self.max_iter):
            perm = rng.permutation(len(y_tr))
            X_b_s, y_s, sw_s = X_b[perm], y_tr[perm], sw_tr[perm]

            epoch_losses = []
            for start in range(0, len(y_tr), bs):
                sl = slice(start, start + bs)
                loss, grad = self._loss_and_grad(X_b_s[sl], y_s[sl], w, sw_s[sl])
                epoch_losses.append(loss)

                if self.use_adam:
                    t += 1
                    m = self.beta1 * m + (1 - self.beta1) * grad
                    v = self.beta2 * v + (1 - self.beta2) * grad ** 2
                    m_hat = m / (1 - self.beta1 ** t)
                    v_hat = v / (1 - self.beta2 ** t)
                    w -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
                else:
                    w -= self.lr * grad

            train_loss = float(np.mean(epoch_losses))
            self.loss_curve_.append(train_loss)

            if self.early_stopping and X_val is not None:
                X_val_b = self._add_intercept(X_val) if self.fit_intercept else X_val
                val_loss, _ = self._loss_and_grad(X_val_b, y_val, w, sw_val)
                self.val_loss_curve_.append(val_loss)

                if best_val_loss - val_loss > self.tol:
                    best_val_loss = val_loss
                    best_w = w.copy()
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= self.n_iter_no_change:
                        log.info("Early stopping at epoch %d (val_loss=%.6f)", epoch, val_loss)
                        w = best_w
                        break
            else:
                if abs(prev_loss - train_loss) < self.tol:
                    log.info("Convergence at epoch %d (loss=%.6f)", epoch, train_loss)
                    break
                prev_loss = train_loss

            if self.verbose and epoch % 50 == 0:
                log.info("Epoch %4d | train_loss=%.6f", epoch, train_loss)

        self.n_iter_    = epoch + 1
        self.coef_      = w[1:] if self.fit_intercept else w
        self.intercept_ = w[0]  if self.fit_intercept else 0.0
        self._w         = w
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates, shape (n_samples, 2)."""
        check_is_fitted(self)
        X = check_array(X)
        X_b = self._add_intercept(X) if self.fit_intercept else X
        p1 = _sigmoid(X_b @ self._w)
        return np.column_stack([1 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class labels."""
        proba = self.predict_proba(X)
        return self.le_.inverse_transform(np.argmax(proba, axis=1))

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return raw log-odds scores."""
        check_is_fitted(self)
        X = check_array(X)
        X_b = self._add_intercept(X) if self.fit_intercept else X
        return X_b @ self._w

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))

    def get_feature_importance(self, feature_names: Optional[list] = None) -> dict:
        """Return absolute coefficient magnitudes keyed by feature name / index."""
        check_is_fitted(self)
        importances = np.abs(self.coef_)
        if feature_names is not None:
            return dict(zip(feature_names, importances))
        return {i: float(v) for i, v in enumerate(importances)}
if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
    )

    print("=" * 70)
    print("CUSTOM LOGISTIC REGRESSION TEST")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load Dataset
    # ------------------------------------------------------------------
    data = load_breast_cancer()

    X = data.data
    y = data.target
    feature_names = data.feature_names

    print(f"Dataset: Breast Cancer")
    print(f"Samples : {X.shape[0]}")
    print(f"Features: {X.shape[1]}")

    # ------------------------------------------------------------------
    # Split Data
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ------------------------------------------------------------------
    # Scale Features
    # ------------------------------------------------------------------
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ------------------------------------------------------------------
    # Train Model
    # ------------------------------------------------------------------
    model = CustomLogisticRegression(
        lr=0.001,
        max_iter=1000,
        batch_size=64,
        penalty="elasticnet",
        l1_ratio=0.5,
        class_weight="balanced",
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
        verbose=False
    )

    print("\nTraining...")
    model.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")
    print(f"ROC AUC  : {roc_auc_score(y_test, y_prob):.4f}")

    print("\nClassification Report")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix")
    print(confusion_matrix(y_test, y_pred))

    print("\nTraining Information")
    print(f"Epochs Run: {model.n_iter_}")

    if len(model.loss_curve_) > 0:
        print(f"Final Training Loss: {model.loss_curve_[-1]:.6f}")

    if hasattr(model, "val_loss_curve_") and model.val_loss_curve_:
        print(f"Final Validation Loss: {model.val_loss_curve_[-1]:.6f}")

    print("\nTop 10 Features")

    importance = model.get_feature_importance(
        feature_names=feature_names.tolist()
    )

    top_features = sorted(
        importance.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]

    for i, (name, score) in enumerate(top_features, start=1):
        print(f"{i:2d}. {name:<35} {score:.6f}")