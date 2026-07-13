"""
Satellite dataset health monitoring and update scheduling metrics.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from config_manager import load_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LDI_DATASET_PATH = DATA_DIR / "ldi_dataset.csv"

logger = logging.getLogger("data_health")


def calculate_next_update(last_update: str | None, frequency: str) -> str | None:
    """Calculate the ISO timestamp of the next expected update."""
    if not last_update:
        return None

    try:
        dt = datetime.fromisoformat(last_update)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        if frequency == "daily":
            delta = timedelta(days=1)
        elif frequency == "8_days":
            delta = timedelta(days=8)
        elif frequency == "monthly":
            delta = timedelta(days=30)
        else:
            delta = timedelta(days=8)

        next_dt = dt + delta
        return next_dt.isoformat()
    except Exception as exc:
        logger.error(f"Error calculating next update time: {exc}")
        return None


def check_dataset_health() -> dict[str, Any]:
    """
    Check dataset integrity and timeliness, returning a health report.
    """
    config = load_config()
    freq = config.get("data_update_frequency", "8_days")
    last_update = config.get("last_update")

    report = {
        "dataset_status": "FRESH",
        "total_records": 0,
        "start_year": "",
        "end_year": "",
        "last_update": last_update or "",
        "next_update": "",
        "message": "Dataset is fresh and healthy.",
    }

    if not LDI_DATASET_PATH.exists():
        report["dataset_status"] = "CRITICAL"
        report["message"] = "Dataset file ldi_dataset.csv is missing."
        return report

    try:
        df = pd.read_csv(LDI_DATASET_PATH)
        if df.empty:
            report["dataset_status"] = "CRITICAL"
            report["message"] = "Dataset is empty."
            return report

        report["total_records"] = int(len(df))
        if "Year" in df.columns:
            report["start_year"] = int(df["Year"].min())
            report["end_year"] = int(df["Year"].max())

        # Status Logic
        if not last_update:
            report["dataset_status"] = "OUTDATED"
            report["message"] = "No update timestamp available. Update is required."
        else:
            last_run = datetime.fromisoformat(last_update)
            if last_run.tzinfo is None:
                last_run = last_run.replace(tzinfo=timezone.utc)

            now = datetime.now(timezone.utc)

            if freq == "daily":
                delta = timedelta(days=1)
            elif freq == "8_days":
                delta = timedelta(days=8)
            elif freq == "monthly":
                delta = timedelta(days=30)
            else:
                delta = timedelta(days=8)

            if now - last_run >= delta:
                report["dataset_status"] = "OUTDATED"
                report["message"] = f"Dataset is outdated based on the frequency '{freq}'."
            else:
                report["dataset_status"] = "FRESH"
                report["message"] = "Dataset is fresh."

        # Next update time
        next_up = calculate_next_update(last_update, freq)
        report["next_update"] = next_up or ""

    except Exception as exc:
        report["dataset_status"] = "CRITICAL"
        report["message"] = f"Error performing dataset health check: {exc}"

    return report
