"""
AI Model Health and Data Drift Detection monitor.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

REFERENCE_DATA_PATH = DATA_DIR / "ldi_dataset.csv"
LATEST_DATA_PATH = RESULTS_DIR / "latest_predictions.csv"
HEALTH_REPORT_PATH = RESULTS_DIR / "model_health.json"

FEATURES_TO_MONITOR = [
    "NDVI_mean",
    "Rainfall_mean",
    "Temperature_mean",
    "SoilMoisture_mean",
]

logger = logging.getLogger("model_monitor")


def check_model_health() -> dict[str, Any]:
    """
    Compare latest prediction features with baseline to detect data drift.
    Exports the report to results/model_health.json.
    """
    logger.info("Starting AI Model Health check...")
    report: dict[str, Any] = {
        "overall_status": "UNKNOWN",
        "last_checked": datetime.now(timezone.utc).isoformat(),
        "highest_drift_feature": "None",
        "highest_drift_percent": 0.0,
        "features": {},
    }

    # Handle missing files
    if not REFERENCE_DATA_PATH.exists():
        msg = f"Reference baseline file missing: {REFERENCE_DATA_PATH}"
        logger.error(msg)
        report["overall_status"] = "DRIFT_DETECTED"
        report["message"] = msg
        _save_health_report(report)
        return report

    if not LATEST_DATA_PATH.exists():
        msg = f"Latest predictions file missing: {LATEST_DATA_PATH}"
        logger.error(msg)
        report["overall_status"] = "WARNING"
        report["message"] = msg
        _save_health_report(report)
        return report

    try:
        ref_df = pd.read_csv(REFERENCE_DATA_PATH)
        latest_df = pd.read_csv(LATEST_DATA_PATH)

        if ref_df.empty or latest_df.empty:
            msg = "One or both data sources are empty."
            logger.error(msg)
            report["overall_status"] = "WARNING"
            report["message"] = msg
            _save_health_report(report)
            return report

        max_drift = 0.0
        max_drift_feature = "None"
        features_report = {}

        for feature in FEATURES_TO_MONITOR:
            # Handle missing columns
            if feature not in ref_df.columns or feature not in latest_df.columns:
                logger.warning(f"Feature '{feature}' missing from reference or latest dataset. Skipping.")
                continue

            ref_vals = pd.to_numeric(ref_df[feature], errors="coerce").dropna()
            latest_vals = pd.to_numeric(latest_df[feature], errors="coerce").dropna()

            if ref_vals.empty or latest_vals.empty:
                logger.warning(f"Feature '{feature}' contains no valid numeric data. Skipping.")
                continue

            ref_mean = float(ref_vals.mean())
            latest_mean = float(latest_vals.mean())

            # Calculate drift with zero check
            if abs(ref_mean) < 1e-9:
                drift_percent = 0.0 if abs(latest_mean) < 1e-9 else 100.0
            else:
                drift_percent = (abs(latest_mean - ref_mean) / abs(ref_mean)) * 100.0

            drift_percent = round(drift_percent, 2)
            features_report[feature] = {
                "training_mean": round(ref_mean, 4),
                "latest_mean": round(latest_mean, 4),
                "drift_percent": drift_percent,
            }

            if drift_percent > max_drift:
                max_drift = drift_percent
                max_drift_feature = feature

        report["features"] = features_report
        report["highest_drift_feature"] = max_drift_feature
        report["highest_drift_percent"] = max_drift

        # Classification
        if max_drift <= 10.0:
            report["overall_status"] = "HEALTHY"
        elif max_drift <= 25.0:
            report["overall_status"] = "WARNING"
        else:
            report["overall_status"] = "DRIFT_DETECTED"

        logger.info(f"Model health status: {report['overall_status']} (Max drift: {max_drift}% on {max_drift_feature})")

    except Exception as exc:
        msg = f"Exception occurred during health monitoring check: {exc}"
        logger.exception(msg)
        report["overall_status"] = "DRIFT_DETECTED"
        report["message"] = msg

    _save_health_report(report)
    return report


def _save_health_report(report: dict[str, Any]) -> None:
    """Save the health report to JSON."""
    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with HEALTH_REPORT_PATH.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)
    except Exception as exc:
        logger.error(f"Failed to write model health report JSON: {exc}")
