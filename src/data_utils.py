from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd

CANONICAL_FEATURES: List[str] = [
    "Age",
    "SystolicBP",
    "DiastolicBP",
    "BS",
    "BodyTemp",
    "HeartRate",
]

CANONICAL_TARGET: str = "RiskLevel"

ORDERED_CLASSES: List[str] = [
    "low risk",
    "mid risk",
    "high risk",
]

COLUMN_NAME_MAP: Dict[str, str] = {
    "age": "Age",
    "systolicbp": "SystolicBP",
    "systolic_bp": "SystolicBP",
    "systolic": "SystolicBP",
    "diastolicbp": "DiastolicBP",
    "diastolic_bp": "DiastolicBP",
    "diastolic": "DiastolicBP",
    "bs": "BS",
    "bloodsugar": "BS",
    "blood_sugar": "BS",
    "bodytemp": "BodyTemp",
    "body_temp": "BodyTemp",
    "temp": "BodyTemp",
    "heartrate": "HeartRate",
    "heart_rate": "HeartRate",
    "hr": "HeartRate",
    "risklevel": "RiskLevel",
    "risk_level": "RiskLevel",
    "risk": "RiskLevel",
}

LABEL_MAP: Dict[str, str] = {
    "low risk": "low risk",
    "low": "low risk",
    "lowrisk": "low risk",
    "mid risk": "mid risk",
    "medium risk": "mid risk",
    "mid": "mid risk",
    "medium": "mid risk",
    "midrisk": "mid risk",
    "high risk": "high risk",
    "high": "high risk",
    "highrisk": "high risk",
}


def _clean_str(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).strip().lower())


def normalize_columns(columns: List[str]) -> Dict[str, str]:
    mapping = {}
    for col in columns:
        cleaned = _clean_str(col)
        mapping[col] = COLUMN_NAME_MAP.get(cleaned, col)
    return mapping


def normalize_target_series(series: pd.Series) -> pd.Series:
    def _map_val(val: Any) -> str:
        if pd.isna(val):
            return "unknown"
        cleaned = _clean_str(val)
        if cleaned in LABEL_MAP:
            return LABEL_MAP[cleaned]
        cleaned_with_space = " ".join(str(val).strip().lower().split())
        return LABEL_MAP.get(cleaned_with_space, str(val).strip().lower())

    return series.apply(_map_val)


def validate_schema(df: pd.DataFrame, require_target: bool = True) -> Tuple[bool, List[str]]:
    errors = []
    for feat in CANONICAL_FEATURES:
        if feat not in df.columns:
            errors.append(f"Missing required feature column: '{feat}'")
        elif not pd.api.types.is_numeric_dtype(df[feat]):
            try:
                pd.to_numeric(df[feat])
            except Exception as e:
                errors.append(f"Feature column '{feat}' must be numeric: {e}")

    if require_target:
        if CANONICAL_TARGET not in df.columns:
            errors.append(f"Missing required target column: '{CANONICAL_TARGET}'")
        else:
            invalid = set(df[CANONICAL_TARGET].unique()) - set(ORDERED_CLASSES)
            if invalid:
                errors.append(f"Target column contains invalid class labels: {invalid}")

    return len(errors) == 0, errors


def load_dataset(
    path: Optional[str | Path] = None,
    allow_download: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    metadata: Dict[str, Any] = {
        "source": None,
        "raw_rows": 0,
        "raw_cols": 0,
        "retrieved_from_uci": False,
        "errors": [],
    }

    df: Optional[pd.DataFrame] = None
    candidate_paths = []
    if path is not None:
        candidate_paths.append(Path(path))
    candidate_paths.extend([
        Path("dataset/Maternal Health Risk Data Set.csv"),
        Path("data/maternal_health_risk_clean.csv"),
        Path("dataset/maternal_health_risk.csv"),
    ])

    for p in candidate_paths:
        if p.exists() and p.is_file():
            try:
                df = pd.read_csv(p)
                metadata["source"] = str(p)
                break
            except Exception as e:
                metadata["errors"].append(f"Failed reading {p}: {e}")

    # uci dataset id 863 fallback if local file is missing
    if df is None and allow_download:
        try:
            from ucimlrepo import fetch_ucirepo
            maternal_health = fetch_ucirepo(id=863)
            df = pd.concat([maternal_health.data.features, maternal_health.data.targets], axis=1)
            metadata["source"] = "UCI ML Repository (ID 863 via ucimlrepo)"
            metadata["retrieved_from_uci"] = True
        except Exception as e:
            metadata["errors"].append(f"UCI download failed: {e}")

    if df is None:
        raise FileNotFoundError(
            "Could not load Maternal Health Risk dataset from candidate local paths "
            f"or UCI repository. Checked paths: {[str(p) for p in candidate_paths]}."
        )

    metadata["raw_rows"] = len(df)
    metadata["raw_cols"] = len(df.columns)

    df = df.rename(columns=normalize_columns(list(df.columns)))

    if CANONICAL_TARGET in df.columns:
        df[CANONICAL_TARGET] = normalize_target_series(df[CANONICAL_TARGET])

    for col in CANONICAL_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    is_valid, validation_errors = validate_schema(df, require_target=(CANONICAL_TARGET in df.columns))
    if not is_valid:
        raise ValueError(f"Dataset validation failed: {validation_errors}")

    metadata["canonical_columns"] = list(df.columns)
    return df, metadata
