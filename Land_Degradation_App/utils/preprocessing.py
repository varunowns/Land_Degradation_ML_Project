"""
Preprocessing and feature transformation for live inference.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from utils.config import (
    ID_COLUMNS,
    LEAKAGE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    TARGET_COLUMN,
)


REQUIRED_INPUT_COLUMNS = NUMERIC_FEATURE_COLUMNS + ["District"]
OPTIONAL_METADATA_COLUMNS = ID_COLUMNS + ["Year"]


def get_feature_schema() -> dict[str, str]:
    """Return the expected input schema for prediction requests."""
    schema = {col: "float" for col in NUMERIC_FEATURE_COLUMNS}
    schema["District"] = "string (one of 20 UP districts)"
    return schema


def get_district_list() -> list[str]:
    """Return sorted district names from the training dataset."""
    from utils.data_loader import load_ldi_dataset

    try:
        df = load_ldi_dataset()
        return sorted(df["District"].unique().tolist())
    except FileNotFoundError:
        return []


def validate_input_columns(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Validate that a DataFrame contains required columns and no forbidden ones."""
    errors: list[str] = []
    missing = [col for col in REQUIRED_INPUT_COLUMNS if col not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {', '.join(missing)}")

    forbidden = [col for col in LEAKAGE_COLUMNS + [TARGET_COLUMN] if col in df.columns]
    if forbidden:
        errors.append(
            f"Columns must not be supplied as inputs: {', '.join(forbidden)}"
        )

    districts = get_district_list()
    if districts:
        unknown = sorted(set(df["District"].dropna()) - set(districts))
        if unknown:
            errors.append(f"Unknown districts: {', '.join(unknown[:5])}{'…' if len(unknown) > 5 else ''}")

    return len(errors) == 0, errors


def build_model_input(raw_input: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a raw input DataFrame for the saved ``preprocessor_lr`` transformer.

    Expects numeric feature columns plus ``District``.
    """
    frame = raw_input[REQUIRED_INPUT_COLUMNS].copy()
    for col in NUMERIC_FEATURE_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if frame[NUMERIC_FEATURE_COLUMNS].isnull().any().any():
        raise ValueError("Non-numeric values detected in feature columns.")
    return frame


def transform_features(raw_input: pd.DataFrame, preprocessor: Any) -> pd.DataFrame:
    """Transform raw user input into model-ready feature matrix."""
    model_input = build_model_input(raw_input)
    transformed = preprocessor.transform(model_input)
    feature_names = _get_feature_names(preprocessor)
    return pd.DataFrame(transformed, columns=feature_names)


def _get_feature_names(preprocessor: Any) -> list[str]:
    """Extract feature names from a fitted ColumnTransformer."""
    names: list[str] = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder":
            continue
        if name == "num":
            names.extend(list(columns))
        elif name == "district" and hasattr(transformer, "get_feature_names_out"):
            names.extend(transformer.get_feature_names_out(["District"]).tolist())
    return names


def describe_preprocessing_pipeline() -> dict[str, Any]:
    """Return a human-readable summary of the training-time preprocessing steps."""
    return {
        "numeric_scaling": "StandardScaler (Logistic Regression path)",
        "district_encoding": "OneHotEncoder (drop_first=True, 19 dummies)",
        "excluded_columns": ID_COLUMNS + LEAKAGE_COLUMNS + [TARGET_COLUMN],
        "train_test_split": "Stratified 80/20 (random_state=42)",
    }


def get_sample_template_row() -> pd.DataFrame:
    """Return a one-row sample for CSV template download."""
    from utils.data_loader import get_district_medians

    medians = get_district_medians()
    row = medians.iloc[0].to_dict()
    return pd.DataFrame([row])
