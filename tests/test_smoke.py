import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from src.data_utils import CANONICAL_FEATURES, ORDERED_CLASSES
from src.features import engineer_features
from src.ordinal_cost import CostSensitiveClassifier
from src.evaluation import evaluate_predictions


def test_end_to_end_smoke_synthetic():
    np.random.seed(42)
    n_samples = 30
    synth_df = pd.DataFrame({
        "Age": np.random.randint(18, 55, size=n_samples),
        "SystolicBP": np.random.randint(90, 160, size=n_samples),
        "DiastolicBP": np.random.randint(60, 100, size=n_samples),
        "BS": np.random.uniform(5.5, 15.0, size=n_samples),
        "BodyTemp": np.random.uniform(98.0, 102.0, size=n_samples),
        "HeartRate": np.random.randint(60, 95, size=n_samples),
    })
    y_synth = pd.Series(np.random.choice(ORDERED_CLASSES, size=n_samples))

    feat_df = engineer_features(synth_df)
    assert feat_df.shape[1] == len(CANONICAL_FEATURES) + 6

    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(feat_df, y_synth)
    preds = clf.predict(feat_df)
    probs = clf.predict_proba(feat_df)

    metrics = evaluate_predictions(
        y_true=y_synth,
        y_pred=preds,
        y_proba=probs,
        estimator_classes=clf.classes_,
        model_name="SmokeModel",
    )
    assert "accuracy" in metrics
    assert "high_risk_recall" in metrics
    assert "macro_f1" in metrics
    assert "confusion_matrix" in metrics

    cost_clf = CostSensitiveClassifier(clf, high_risk_underestimate_penalty=2.0)
    cost_clf.fit(feat_df, y_synth)
    cost_preds = cost_clf.predict(feat_df)
    assert len(cost_preds) == n_samples
