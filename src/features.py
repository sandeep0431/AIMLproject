from __future__ import annotations
from typing import List
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from src.data_utils import CANONICAL_FEATURES

ENGINEERED_FEATURE_NAMES: List[str] = [
    "PulsePressure",
    "MeanArterialPressure",
    "BP_Product",
    "Age_BS_Interaction",
    "High_BP_Flag",
    "High_BS_Flag",
]

ALL_FEATURE_NAMES: List[str] = CANONICAL_FEATURES + ENGINEERED_FEATURE_NAMES


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # copy defensively so input dataframe is unchanged
    res = df.copy()

    res["PulsePressure"] = res["SystolicBP"] - res["DiastolicBP"]
    res["MeanArterialPressure"] = (res["SystolicBP"] + 2.0 * res["DiastolicBP"]) / 3.0
    res["BP_Product"] = res["SystolicBP"] * res["DiastolicBP"]
    res["Age_BS_Interaction"] = res["Age"] * res["BS"]
    
    # systolic >= 140 or diastolic >= 90 mmhg clinical hypertension threshold
    res["High_BP_Flag"] = (
        (res["SystolicBP"] >= 140) | (res["DiastolicBP"] >= 90)
    ).astype(int)
    
    # bs >= 7.0 mmol/l threshold for high blood sugar
    res["High_BS_Flag"] = (res["BS"] >= 7.0).astype(int)

    return res


class MaternalFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, include_engineered: bool = True):
        self.include_engineered = include_engineered
        self.feature_names_in_: List[str] = []

    def fit(self, X: pd.DataFrame | np.ndarray, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = CANONICAL_FEATURES[: X.shape[1]]
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X_df = pd.DataFrame(X, columns=CANONICAL_FEATURES[: X.shape[1]])
        else:
            X_df = X.copy()

        if self.include_engineered:
            return engineer_features(X_df)
        return X_df

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if self.include_engineered:
            return np.array(ALL_FEATURE_NAMES)
        return np.array(CANONICAL_FEATURES)
