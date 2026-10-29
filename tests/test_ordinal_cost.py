import numpy as np
from sklearn.tree import DecisionTreeClassifier
from src.ordinal_cost import (
    create_ordinal_cost_matrix,
    align_probabilities,
    predict_cost_sensitive,
    CostSensitiveClassifier,
)
from src.data_utils import ORDERED_CLASSES


def test_ordinal_cost_matrix_structure():
    cost_mat = create_ordinal_cost_matrix(base_cost_scale=1.0, high_risk_underestimate_penalty=3.0)
    assert cost_mat.shape == (3, 3)
    assert np.all(np.diag(cost_mat) == 0.0)

    # true high risk (index 2) predicting low (index 0): 2*1.0 + 3.0*(2-0) = 8.0
    assert cost_mat[2, 0] == 8.0
    # true high risk (index 2) predicting mid (index 1): 1*1.0 + 3.0*(2-1) = 4.0
    assert cost_mat[2, 1] == 4.0
    # true low risk (index 0) predicting high (index 2): 2*1.0 = 2.0
    assert cost_mat[0, 2] == 2.0

    # underestimating high risk must cost strictly more than overestimating low risk
    assert cost_mat[2, 0] > cost_mat[0, 2]


def test_align_probabilities():
    raw_classes = ["mid risk", "high risk", "low risk"]
    raw_probs = np.array([
        [0.2, 0.7, 0.1],
        [0.8, 0.1, 0.1],
    ])
    aligned = align_probabilities(raw_probs, raw_classes, ORDERED_CLASSES)

    # ordered classes are [0: low risk, 1: mid risk, 2: high risk]
    assert np.isclose(aligned[0, 0], 0.1)
    assert np.isclose(aligned[0, 1], 0.2)
    assert np.isclose(aligned[0, 2], 0.7)

    assert np.isclose(aligned[1, 0], 0.1)
    assert np.isclose(aligned[1, 1], 0.8)
    assert np.isclose(aligned[1, 2], 0.1)


def test_predict_cost_sensitive_penalty_effect():
    # under heavy penalty, ambiguous cases with 40% high risk must bias to high risk
    probs = np.array([[0.15, 0.45, 0.40]])
    cost_mat = create_ordinal_cost_matrix(base_cost_scale=1.0, high_risk_underestimate_penalty=5.0)
    preds, _ = predict_cost_sensitive(probs, ORDERED_CLASSES, cost_mat, ORDERED_CLASSES)

    assert preds[0] == "high risk"


def test_cost_sensitive_classifier_fit_predict():
    X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0], [6.0, 7.0]])
    y = np.array(["low risk", "low risk", "mid risk", "mid risk", "high risk", "high risk"])
    base = DecisionTreeClassifier(random_state=42)
    clf = CostSensitiveClassifier(base, high_risk_underestimate_penalty=4.0)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert len(preds) == len(y)
    assert set(preds).issubset(set(ORDERED_CLASSES))
