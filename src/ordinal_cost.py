from __future__ import annotations
from typing import List, Optional, Tuple, Any
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from src.data_utils import ORDERED_CLASSES


def create_ordinal_cost_matrix(
    base_cost_scale: float = 1.0,
    high_risk_underestimate_penalty: float = 2.0,
    classes: Optional[List[str]] = None,
) -> np.ndarray:
    if classes is None:
        classes = ORDERED_CLASSES
    n_classes = len(classes)
    loss_mat = np.zeros((n_classes, n_classes), dtype=np.float64)

    for i in range(n_classes):
        for j in range(n_classes):
            if i != j:
                # ordinal distance penalty
                dist = abs(i - j)
                cost = dist * base_cost_scale

                # underestimating high risk (i=2) to lower classes (j<2) incurs an extra penalty
                if i == 2 and j < 2:
                    cost += high_risk_underestimate_penalty * (2 - j)

                loss_mat[i, j] = cost

    return loss_mat


def align_probabilities(
    probs: np.ndarray,
    estimator_classes: np.ndarray | List[str],
    target_classes: Optional[List[str]] = None,
) -> np.ndarray:
    if target_classes is None:
        target_classes = ORDERED_CLASSES

    estimator_classes_list = list(estimator_classes)
    n_samples = probs.shape[0]
    n_target_classes = len(target_classes)

    aligned = np.zeros((n_samples, n_target_classes), dtype=np.float64)

    for target_idx, target_cls in enumerate(target_classes):
        if target_cls in estimator_classes_list:
            source_idx = estimator_classes_list.index(target_cls)
            aligned[:, target_idx] = probs[:, source_idx]
        else:
            aligned[:, target_idx] = 0.0

    if not np.all(np.isfinite(aligned)):
        raise ValueError("Probabilities contain NaN or infinite values.")
    aligned = np.clip(aligned, 0.0, 1.0)

    row_sums = aligned.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return aligned / row_sums


def predict_cost_sensitive(
    probs: np.ndarray,
    estimator_classes: np.ndarray | List[str],
    cost_matrix: np.ndarray,
    target_classes: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if target_classes is None:
        target_classes = ORDERED_CLASSES

    p = align_probabilities(probs, estimator_classes, target_classes)
    if cost_matrix.shape != (len(target_classes), len(target_classes)):
        raise ValueError(f"Cost matrix shape {cost_matrix.shape} does not match class count.")

    # expected loss for each candidate action: P @ CostMatrix
    expected_loss = np.matmul(p, cost_matrix)

    # picks the action that minimizes expected clinical loss
    best_action_indices = np.argmin(expected_loss, axis=1)
    predicted_labels = np.array([target_classes[idx] for idx in best_action_indices])

    return predicted_labels, expected_loss


class CostSensitiveClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        base_estimator: Any,
        base_cost_scale: float = 1.0,
        high_risk_underestimate_penalty: float = 2.0,
    ):
        self.base_estimator = base_estimator
        self.base_cost_scale = base_cost_scale
        self.high_risk_underestimate_penalty = high_risk_underestimate_penalty
        self.classes_ = np.array(ORDERED_CLASSES)
        self.cost_matrix_: Optional[np.ndarray] = None

    def fit(self, X, y):
        self.base_estimator.fit(X, y)
        self.classes_ = np.array(ORDERED_CLASSES)
        self.cost_matrix_ = create_ordinal_cost_matrix(
            base_cost_scale=self.base_cost_scale,
            high_risk_underestimate_penalty=self.high_risk_underestimate_penalty,
            classes=ORDERED_CLASSES,
        )
        return self

    def predict_proba(self, X) -> np.ndarray:
        raw_probs = self.base_estimator.predict_proba(X)
        return align_probabilities(raw_probs, self.base_estimator.classes_, ORDERED_CLASSES)

    def predict(self, X) -> np.ndarray:
        raw_probs = self.base_estimator.predict_proba(X)
        cost_mat = create_ordinal_cost_matrix(
            base_cost_scale=self.base_cost_scale,
            high_risk_underestimate_penalty=self.high_risk_underestimate_penalty,
            classes=ORDERED_CLASSES,
        )
        preds, _ = predict_cost_sensitive(
            raw_probs,
            self.base_estimator.classes_,
            cost_mat,
            ORDERED_CLASSES,
        )
        return preds
