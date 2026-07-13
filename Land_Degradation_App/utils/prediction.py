"""
Prediction interface — single and batch inference using saved artefacts.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd

from utils.config import CLASS_ORDER
from utils.model_loader import load_model_artifacts
from utils.preprocessing import build_model_input, validate_input_columns


def predict_single(features: dict[str, Any]) -> dict[str, Any]:
    """
    Predict degradation class for a single grid observation.

    Parameters
    ----------
    features:
        Dictionary of feature name → value pairs (must include ``District``).

    Returns
    -------
    dict
        Keys: ``degradation_class``, ``probabilities``, ``confidence``.
    """
    df = pd.DataFrame([features])
    result_df = predict_batch(df)
    row = result_df.iloc[0]
    probabilities = {cls: float(row[f"prob_{cls}"]) for cls in CLASS_ORDER}
    return format_prediction_result(str(row["predicted_class"]), probabilities)


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict degradation classes for a batch of observations.

    Returns the original DataFrame augmented with prediction columns.
    """
    is_valid, errors = validate_input_columns(df)
    if not is_valid:
        raise ValueError("; ".join(errors))

    artifacts = load_model_artifacts()
    model_input = build_model_input(df)
    X = artifacts.preprocessor.transform(model_input)

    pred_encoded = artifacts.model.predict(X)
    pred_proba = artifacts.model.predict_proba(X)
    pred_labels = artifacts.label_encoder.inverse_transform(pred_encoded)

    output = df.copy().reset_index(drop=True)
    output["predicted_class"] = pred_labels
    for idx, cls in enumerate(artifacts.label_encoder.classes_):
        output[f"prob_{cls}"] = pred_proba[:, idx]
    output["confidence"] = pred_proba.max(axis=1)
    return output


def format_prediction_result(
    label: str,
    probabilities: dict[str, float],
) -> dict[str, Any]:
    """Format a raw model output into a display-friendly dictionary."""
    return {
        "degradation_class": label,
        "probabilities": probabilities,
        "confidence": max(probabilities.values()) if probabilities else 0.0,
    }


def predictions_to_csv(df: pd.DataFrame) -> bytes:
    """Serialise prediction results to CSV bytes for download."""
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue()


def get_class_color(label: str) -> str:
    """Return hex colour for a degradation class badge."""
    from utils.config import COLORS

    return COLORS.get(label.lower(), COLORS["text_muted"])
