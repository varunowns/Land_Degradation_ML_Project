"""
End-to-end data refresh and inference pipeline for the Streamlit app.

This module fetches new Earth Engine data, rebuilds the derived datasets, and
runs inference with saved model artefacts only. It never retrains models.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib
import pandas as pd

from auto_pipeline import run_data_pipeline
from config_manager import mark_update_requested, set_update_pipeline_status, update_last_run
from gee_fetcher import fetch_all_data
from gee_auth_manager import check_gee_status

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

LDI_DATASET_PATH = DATA_DIR / "ldi_dataset.csv"
LATEST_PREDICTIONS_PATH = RESULTS_DIR / "latest_predictions.csv"

BEST_MODEL_PATH = MODELS_DIR / "tuned_logistic_regression.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor_lr.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"

NUMERIC_FEATURE_COLUMNS = [
    "Area_km2",
    "BareLand",
    "Builtup",
    "Cropland",
    "Grassland",
    "TreeCover",
    "Water",
    "Wetland",
    "Shrubland",
    "NDVI_mean",
    "Rainfall_mean",
    "Temperature_mean",
    "SoilMoisture_mean",
]
REQUIRED_INPUT_COLUMNS = NUMERIC_FEATURE_COLUMNS + ["District"]

ProgressCallback = Callable[[str], None]


def _notify(callback: ProgressCallback | None, message: str) -> None:
    if callback:
        callback(message)


def _load_model_artifacts() -> tuple[Any, Any, Any]:
    missing = [
        path
        for path in (BEST_MODEL_PATH, PREPROCESSOR_PATH, LABEL_ENCODER_PATH)
        if not path.exists()
    ]
    if missing:
        names = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Required saved model artefacts are missing: {names}")

    model = joblib.load(BEST_MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    return model, preprocessor, label_encoder


def _validate_prediction_frame(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_INPUT_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Updated LDI dataset is missing required columns: {', '.join(missing)}")

    numeric_frame = df[NUMERIC_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if numeric_frame.isnull().any().any():
        bad_columns = numeric_frame.columns[numeric_frame.isnull().any()].tolist()
        raise ValueError(f"Updated LDI dataset contains non-numeric values in: {', '.join(bad_columns)}")


def _run_saved_model_predictions() -> dict[str, Any]:
    if not LDI_DATASET_PATH.exists():
        raise FileNotFoundError(f"Updated LDI dataset not found: {LDI_DATASET_PATH}")

    df = pd.read_csv(LDI_DATASET_PATH)
    if df.empty:
        raise ValueError(f"Updated LDI dataset is empty: {LDI_DATASET_PATH}")
    if "Year" not in df.columns:
        raise ValueError("Updated LDI dataset is missing required metadata column: Year")

    latest_year = int(pd.to_numeric(df["Year"], errors="coerce").max())
    latest_df = df[df["Year"] == latest_year].copy().reset_index(drop=True)
    _validate_prediction_frame(latest_df)

    model, preprocessor, label_encoder = _load_model_artifacts()
    model_input = latest_df[REQUIRED_INPUT_COLUMNS].copy()
    for column in NUMERIC_FEATURE_COLUMNS:
        model_input[column] = pd.to_numeric(model_input[column], errors="coerce")

    transformed = preprocessor.transform(model_input)
    pred_encoded = model.predict(transformed)
    pred_proba = model.predict_proba(transformed)
    pred_labels = label_encoder.inverse_transform(pred_encoded)

    output = latest_df.copy()
    output["predicted_class"] = pred_labels
    output["y_pred"] = pred_labels
    if "Degradation_Class" in output.columns:
        output["y_true"] = output["Degradation_Class"]
        output["correct"] = output["predicted_class"] == output["Degradation_Class"]

    for index, class_name in enumerate(label_encoder.classes_):
        output[f"prob_{class_name}"] = pred_proba[:, index]
    output["confidence"] = pred_proba.max(axis=1)
    output["model"] = BEST_MODEL_PATH.stem

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(LATEST_PREDICTIONS_PATH, index=False)

    return {
        "latest_year": latest_year,
        "prediction_rows": len(output),
        "latest_predictions": str(LATEST_PREDICTIONS_PATH),
    }


def run_full_update(
    progress_callback: ProgressCallback | None = None,
    *,
    year: int | None = None,
) -> dict[str, Any]:
    """
    Fetch latest GEE data, rebuild datasets, and run saved-model inference.

    Returns a dictionary whose ``status`` is ``SUCCESS`` or ``FAILED``. The
    pipeline marks the persisted config status as running/completed/failed.
    """
    requested = mark_update_requested()
    requested_at = requested.get("update_requested_at")
    fetch_year = year or datetime.now(timezone.utc).year

    try:
        set_update_pipeline_status("RUNNING")

        # Check GEE status before fetching data
        if check_gee_status() != "CONNECTED":
            set_update_pipeline_status("FAILED")
            return {
                "status": "FAILED",
                "requested_at": requested_at,
                "fetch_year": fetch_year,
                "message": "Google Earth Engine authentication required",
            }

        _notify(progress_callback, "Fetching satellite data...")
        fetched = fetch_all_data(fetch_year)

        _notify(progress_callback, "Processing features...")
        datasets = run_data_pipeline()

        _notify(progress_callback, "Generating predictions...")
        predictions = _run_saved_model_predictions()

        # Save prediction run to history
        try:
            from history_manager import save_prediction_history
            latest_df = pd.read_csv(LATEST_PREDICTIONS_PATH)
            save_prediction_history(latest_df)
        except Exception as hist_exc:
            import logging
            logging.getLogger("update_pipeline").error(f"Failed to record prediction history: {hist_exc}")

        # Run AI model health monitor check
        try:
            from model_monitor import check_model_health
            check_model_health()
        except Exception as health_exc:
            import logging
            logging.getLogger("update_pipeline").error(f"Failed to run model health monitor: {health_exc}")

        updated = update_last_run()
        _notify(progress_callback, "Dashboard updated.")

        return {
            "status": "SUCCESS",
            "requested_at": requested_at,
            "last_update": updated.get("last_update"),
            "fetch_year": fetch_year,
            "fetched": {name: str(path) for name, path in fetched.items()},
            "datasets": datasets,
            **predictions,
            "message": "Dataset update pipeline finished successfully.",
        }
    except Exception as exc:
        set_update_pipeline_status("FAILED")
        return {
            "status": "FAILED",
            "requested_at": requested_at,
            "fetch_year": fetch_year,
            "message": f"Dataset update pipeline failed: {exc}",
            "error": str(exc),
        }


def trigger_data_update_pipeline(progress_callback: ProgressCallback | None = None) -> dict[str, Any]:
    """Backward-compatible wrapper used by older Settings page code."""
    return run_full_update(progress_callback=progress_callback)
