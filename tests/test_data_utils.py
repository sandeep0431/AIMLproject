import pandas as pd
from src.data_utils import (
    normalize_columns,
    normalize_target_series,
    validate_schema,
)


def test_normalize_columns():
    raw_cols = ["age", "Systolic_BP", "DIASTOLIC", "BS", "body_temp", "Heart_Rate", "risk_level"]
    mapping = normalize_columns(raw_cols)
    assert mapping["age"] == "Age"
    assert mapping["Systolic_BP"] == "SystolicBP"
    assert mapping["DIASTOLIC"] == "DiastolicBP"
    assert mapping["BS"] == "BS"
    assert mapping["body_temp"] == "BodyTemp"
    assert mapping["Heart_Rate"] == "HeartRate"
    assert mapping["risk_level"] == "RiskLevel"


def test_normalize_target_series():
    raw_labels = pd.Series(["low risk", "Low", "mid risk", "medium", "high risk", "High", "invalid"])
    cleaned = normalize_target_series(raw_labels)
    assert cleaned.iloc[0] == "low risk"
    assert cleaned.iloc[1] == "low risk"
    assert cleaned.iloc[2] == "mid risk"
    assert cleaned.iloc[3] == "mid risk"
    assert cleaned.iloc[4] == "high risk"
    assert cleaned.iloc[5] == "high risk"
    assert cleaned.iloc[6] == "invalid"


def test_validate_schema_valid():
    df = pd.DataFrame({
        "Age": [25, 30],
        "SystolicBP": [120, 130],
        "DiastolicBP": [80, 85],
        "BS": [6.5, 7.2],
        "BodyTemp": [98.6, 99.0],
        "HeartRate": [70, 75],
        "RiskLevel": ["low risk", "high risk"],
    })
    is_valid, errors = validate_schema(df, require_target=True)
    assert is_valid is True
    assert len(errors) == 0


def test_validate_schema_missing_column():
    df = pd.DataFrame({
        "Age": [25],
        "SystolicBP": [120],
        "RiskLevel": ["low risk"],
    })
    is_valid, errors = validate_schema(df, require_target=True)
    assert is_valid is False
    assert any("DiastolicBP" in err for err in errors)


def test_validate_schema_invalid_class():
    df = pd.DataFrame({
        "Age": [25],
        "SystolicBP": [120],
        "DiastolicBP": [80],
        "BS": [6.5],
        "BodyTemp": [98.6],
        "HeartRate": [70],
        "RiskLevel": ["extreme critical danger"],
    })
    is_valid, errors = validate_schema(df, require_target=True)
    assert is_valid is False
    assert any("invalid class labels" in err for err in errors)
