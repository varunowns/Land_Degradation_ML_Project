"""
Prediction history manager to track long-term model predictions over multiple pipeline runs.
"""

from __future__ import annotations

import logging
import secrets
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
HISTORY_PATH = RESULTS_DIR / "prediction_history.csv"

logger = logging.getLogger("history_manager")


def generate_run_id() -> str:
    """Generate a unique run ID using the current time and a random hex suffix."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rand_suffix = secrets.token_hex(3)
    return f"run_{timestamp}_{rand_suffix}"


def save_prediction_history(predictions_df: pd.DataFrame) -> str:
    """
    Append predictions to the prediction_history.csv file.
    Only successful runs should call this.
    """
    if predictions_df.empty:
        logger.warning("Empty predictions dataframe provided. History not saved.")
        return ""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = generate_run_id()
    now_timestamp = datetime.now(timezone.utc).isoformat()

    # Normalize columns
    df_copy = predictions_df.copy()
    df_copy["Run_ID"] = run_id
    df_copy["Timestamp"] = now_timestamp

    # Resolve predicted class column
    pred_col = next((col for col in ["predicted_class", "y_pred", "Predicted_Class"] if col in df_copy.columns), None)
    if pred_col:
        df_copy["Predicted_Class"] = df_copy[pred_col]
    else:
        df_copy["Predicted_Class"] = "Unknown"

    # Resolve confidence column
    conf_col = next((col for col in ["confidence", "Confidence"] if col in df_copy.columns), None)
    if conf_col:
        df_copy["Confidence"] = df_copy[conf_col]
    else:
        df_copy["Confidence"] = 1.0

    # Ensure required columns are present
    required_cols = [
        "Run_ID",
        "Timestamp",
        "District",
        "Grid_ID",
        "Year",
        "Predicted_Class",
        "Confidence"
    ]
    for col in required_cols:
        if col not in df_copy.columns:
            df_copy[col] = None

    history_df = df_copy[required_cols]

    # Append to file
    header = not HISTORY_PATH.exists()
    try:
        history_df.to_csv(HISTORY_PATH, mode="a", index=False, header=header)
        logger.info(f"Successfully appended {len(history_df)} predictions to prediction history (Run_ID: {run_id}).")
    except Exception as exc:
        logger.error(f"Failed to save prediction history: {exc}")
        raise

    return run_id


def load_prediction_history() -> pd.DataFrame:
    """Load the full prediction history as a pandas DataFrame."""
    if not HISTORY_PATH.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(HISTORY_PATH)
    except Exception as exc:
        logger.error(f"Failed to load prediction history: {exc}")
        return pd.DataFrame()
