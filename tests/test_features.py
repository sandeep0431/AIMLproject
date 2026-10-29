import pandas as pd
from src.features import engineer_features, MaternalFeatureEngineer, ENGINEERED_FEATURE_NAMES


def test_engineer_features_formulas():
    df = pd.DataFrame({
        "Age": [30, 40],
        "SystolicBP": [140, 110],
        "DiastolicBP": [90, 70],
        "BS": [8.0, 6.0],
        "BodyTemp": [98.0, 100.0],
        "HeartRate": [70, 80],
    })
    feat_df = engineer_features(df)

    assert feat_df.loc[0, "PulsePressure"] == 50
    assert feat_df.loc[1, "PulsePressure"] == 40
    assert round(feat_df.loc[0, "MeanArterialPressure"], 2) == round((140 + 180) / 3.0, 2)
    assert feat_df.loc[0, "BP_Product"] == 140 * 90
    assert feat_df.loc[0, "Age_BS_Interaction"] == 30 * 8.0
    assert feat_df.loc[0, "High_BP_Flag"] == 1
    assert feat_df.loc[1, "High_BP_Flag"] == 0
    assert feat_df.loc[0, "High_BS_Flag"] == 1
    assert feat_df.loc[1, "High_BS_Flag"] == 0


def test_engineer_features_no_mutation():
    df = pd.DataFrame({
        "Age": [25],
        "SystolicBP": [120],
        "DiastolicBP": [80],
        "BS": [6.5],
        "BodyTemp": [98.6],
        "HeartRate": [75],
    })
    original_cols = list(df.columns)
    _ = engineer_features(df)
    assert list(df.columns) == original_cols
    for col in ENGINEERED_FEATURE_NAMES:
        assert col not in df.columns


def test_maternal_feature_engineer_transformer():
    df = pd.DataFrame({
        "Age": [25, 30],
        "SystolicBP": [120, 130],
        "DiastolicBP": [80, 85],
        "BS": [6.5, 7.2],
        "BodyTemp": [98.6, 99.0],
        "HeartRate": [70, 75],
    })
    transformer = MaternalFeatureEngineer(include_engineered=True)
    out = transformer.fit_transform(df)
    assert isinstance(out, pd.DataFrame)
    assert out.shape[1] == 12
    assert "PulsePressure" in out.columns
