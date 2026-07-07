"""Run ML predictions on the latest GEE-exported dataset."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import CONFIG, EXPORT_COLUMNS, METADATA_PATH
from .env import PROJECT_ROOT

logger = logging.getLogger(__name__)

LATEST_DATASET_PATH = CONFIG.export_directory / CONFIG.dataset_filename
PREDICTIONS_PATH = CONFIG.export_directory / "predictions.csv"
PREDICTION_HISTORY_PATH = CONFIG.export_directory / "prediction_history.json"
MODEL_FILENAME = "tuned_logistic_regression.pkl"


def _import_prediction_utils():
    """Import app prediction utilities without modifying the Streamlit UI."""
    import sys

    app_root = PROJECT_ROOT / "Land_Degradation_App"
    if str(app_root) not in sys.path:
        sys.path.insert(0, str(app_root))
    from utils.prediction import predict_batch, predictions_to_csv  # type: ignore

    return predict_batch, predictions_to_csv


def validate_dataset_file(path: Path) -> pd.DataFrame:
    """Load and validate the latest dataset CSV."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Run update_dataset() first."
        )

    df = pd.read_csv(path)
    missing = [col for col in EXPORT_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"latest_dataset.csv missing columns: {missing}")

    ordered = df[EXPORT_COLUMNS].copy()
    if ordered.isnull().any().any():
        null_cols = ordered.columns[ordered.isnull().any()].tolist()
        raise ValueError(f"Null values in latest dataset columns: {null_cols}")

    logger.info("Validated latest dataset: %d rows, %d columns", len(ordered), len(ordered.columns))
    return ordered


def run_predictions(dataset_path: Path | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Generate predictions and history metadata for the latest dataset.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        Prediction dataframe and history metadata dictionary.
    """
    predict_batch, predictions_to_csv = _import_prediction_utils()
    source_path = dataset_path or LATEST_DATASET_PATH
    input_df = validate_dataset_file(source_path)

    logger.info("Running model inference on %d rows…", len(input_df))
    results = predict_batch(input_df)

    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(PREDICTIONS_PATH, index=False)
    logger.info("Saved predictions to %s", PREDICTIONS_PATH)

    metadata: dict[str, Any] = {}
    if METADATA_PATH.exists():
        with METADATA_PATH.open("r", encoding="utf-8") as fh:
            metadata = json.load(fh)

    class_counts = results["predicted_class"].value_counts().to_dict()
    history_entry = {
        "prediction_timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_timestamp": metadata.get("last_update"),
        "dataset_acquisition_date": metadata.get("acquisition_date"),
        "model_filename": MODEL_FILENAME,
        "total_predictions": int(len(results)),
        "class_distribution": class_counts,
        "average_confidence": float(results["confidence"].mean()),
        "source_dataset": str(source_path),
        "predictions_file": str(PREDICTIONS_PATH),
    }

    history: list[dict[str, Any]] = []
    if PREDICTION_HISTORY_PATH.exists():
        with PREDICTION_HISTORY_PATH.open("r", encoding="utf-8") as fh:
            loaded = json.load(fh)
            history = loaded if isinstance(loaded, list) else loaded.get("history", [])

    history.append(history_entry)
    with PREDICTION_HISTORY_PATH.open("w", encoding="utf-8") as fh:
        json.dump({"history": history, "latest": history_entry}, fh, indent=2)

    logger.info("Saved prediction history to %s", PREDICTION_HISTORY_PATH)
    return results, history_entry
