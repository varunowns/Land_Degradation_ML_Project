"""Dataset validation utilities for live GEE exports."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .config import EXPORT_COLUMNS, METADATA_PATH, MODEL_FEATURE_COLUMNS

logger = logging.getLogger(__name__)


def validate_exported_dataset(path: Path, metadata_path: Path | None = None) -> dict[str, Any]:
    """
    Validate exported dataset schema, types, missing values, and ranges.

    Returns a structured report dictionary.
    """
    report: dict[str, Any] = {
        "dataset_exists": path.exists(),
        "metadata_exists": metadata_path.exists() if metadata_path else False,
        "errors": [],
        "warnings": [],
    }

    if not path.exists():
        report["errors"].append(f"Dataset file not found: {path}")
        return report

    df = pd.read_csv(path)
    report["shape"] = list(df.shape)
    report["columns"] = df.columns.tolist()
    report["column_order_matches"] = df.columns.tolist() == EXPORT_COLUMNS

    missing_cols = [c for c in EXPORT_COLUMNS if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in EXPORT_COLUMNS]
    if missing_cols:
        report["errors"].append(f"Missing columns: {missing_cols}")
    if extra_cols:
        report["warnings"].append(f"Extra columns: {extra_cols}")

    null_counts = df.isnull().sum()
    report["null_counts"] = null_counts.to_dict()
    if null_counts.any():
        report["errors"].append(f"Null values present: {null_counts[null_counts > 0].to_dict()}")

    dtype_map = {col: str(df[col].dtype) for col in EXPORT_COLUMNS if col in df.columns}
    report["dtypes"] = dtype_map

    ranges: dict[str, dict[str, float]] = {}
    for col in MODEL_FEATURE_COLUMNS:
        if col in df.columns:
            ranges[col] = {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean()),
            }
    report["feature_ranges"] = ranges

    # Sanity checks
    if "NDVI_mean" in df.columns:
        if df["NDVI_mean"].min() < -1.1 or df["NDVI_mean"].max() > 1.1:
            report["warnings"].append("NDVI_mean outside expected [-1, 1] range.")
    if "Rainfall_mean" in df.columns and (df["Rainfall_mean"] < 0).any():
        report["errors"].append("Negative rainfall values detected.")
    if "Year" in df.columns:
        report["years"] = sorted(df["Year"].unique().tolist())

    report["district_count"] = int(df["District"].nunique()) if "District" in df.columns else None
    report["grid_count"] = int(df["Grid_ID"].nunique()) if "Grid_ID" in df.columns else None

    if metadata_path and metadata_path.exists():
        import json

        with metadata_path.open("r", encoding="utf-8") as fh:
            report["metadata"] = json.load(fh)

    report["passed"] = len(report["errors"]) == 0
    logger.info("Dataset validation %s", "passed" if report["passed"] else "failed")
    return report
