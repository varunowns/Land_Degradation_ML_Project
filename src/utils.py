"""
Shared utilities for the Land Degradation ML pipeline.

Provides path configuration, logging setup, and common evaluation helpers.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
PLOTS_DIR = PROJECT_ROOT / "plots"
REPORTS_DIR = PROJECT_ROOT / "reports"
RESULTS_DIR = PROJECT_ROOT / "results"

LDI_DATASET_PATH = DATA_DIR / "ldi_dataset.csv"

# ---------------------------------------------------------------------------
# Reproducibility and modelling configuration
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
CLASS_ORDER = ["Low", "Moderate", "High"]

# Columns never used as model inputs
ID_COLUMNS = ["Grid_ID"]
TARGET_COLUMN = "Degradation_Class"
LEAKAGE_COLUMNS = ["LDI"]  # Used to derive the target; must not be a feature
METADATA_COLUMNS = ["District", "Year"]  # Kept for error analysis; Year excluded from X

EXCLUDE_FROM_FEATURES = ID_COLUMNS + [TARGET_COLUMN] + LEAKAGE_COLUMNS

# Primary metric for model ranking (macro average handles multiclass balance)
PRIMARY_METRIC = "f1_macro"

# Plot defaults
PLOT_DPI = 300
PLOT_SAVE_KW = {"dpi": PLOT_DPI, "bbox_inches": "tight"}


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """Configure and return a module-level logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


def ensure_directories() -> None:
    """Create output directories if they do not exist."""
    for directory in (MODELS_DIR, PLOTS_DIR, REPORTS_DIR, RESULTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def save_json(data: dict[str, Any], path: Path) -> None:
    """Persist a dictionary as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, default=str)


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Compute standard multiclass classification metrics.

    Uses macro averaging for precision, recall, and F1 unless noted.
    ROC-AUC uses one-vs-rest (OvR) strategy when probabilities are supplied.
    """
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    if y_prob is not None:
        metrics["roc_auc"] = float(
            roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        )

    metrics["classification_report"] = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_ORDER,
        output_dict=True,
        zero_division=0,
    )
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()

    return metrics


def time_model_fit(model: Any, X_train: np.ndarray, y_train: np.ndarray) -> tuple[Any, float]:
    """Fit a model and return it with training time in seconds."""
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start
    return model, train_time


def time_model_predict(
    model: Any, X: np.ndarray
) -> tuple[np.ndarray, np.ndarray | None, float]:
    """Predict with a fitted model and return labels, probabilities, and elapsed time."""
    start = time.perf_counter()
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X) if hasattr(model, "predict_proba") else None
    predict_time = time.perf_counter() - start
    return y_pred, y_prob, predict_time


def save_artifact(obj: Any, filename: str) -> Path:
    """Save a Python object with joblib under models/."""
    ensure_directories()
    path = MODELS_DIR / filename
    joblib.dump(obj, path)
    return path
