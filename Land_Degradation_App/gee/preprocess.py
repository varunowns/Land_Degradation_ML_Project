"""Preprocessing utilities for Earth Engine-derived land degradation features."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from .config import EXPORT_COLUMNS, FULL_LULC_COLUMNS, MODEL_FEATURE_COLUMNS

logger = logging.getLogger(__name__)


def apply_legacy_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the legacy post-export preprocessing used for training data."""
    expected_columns = [
        "Grid_ID",
        "District",
        "Year",
        "Area_km2",
        *FULL_LULC_COLUMNS,
        "NDVI_mean",
        "Rainfall_mean",
        "Temperature_mean",
        "SoilMoisture_mean",
    ]
    missing = [column for column in expected_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required legacy columns: {missing}")

    processed = df[expected_columns].copy()

    duplicates = int(processed.duplicated().sum())
    if duplicates > 0:
        logger.warning("Dropping %d duplicate rows to match the legacy workflow.", duplicates)
        processed = processed.drop_duplicates()

    processed["NDVI_mean"] = processed.groupby(["Year", "District"])["NDVI_mean"].transform(
        lambda series: series.fillna(series.median())
    )
    if processed["NDVI_mean"].isnull().any():
        processed["NDVI_mean"] = processed["NDVI_mean"].fillna(processed["NDVI_mean"].median())

    return processed


def validate_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and reorder the export frame to the model-compatible schema."""
    missing = [column for column in EXPORT_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in generated dataset: {missing}")

    ordered = df.loc[:, EXPORT_COLUMNS].copy()
    ordered["Grid_ID"] = ordered["Grid_ID"].astype(str)
    ordered["District"] = ordered["District"].astype(str)
    ordered["Year"] = pd.to_numeric(ordered["Year"], errors="raise").astype(int)

    for column in MODEL_FEATURE_COLUMNS:
        ordered[column] = pd.to_numeric(ordered[column], errors="coerce")

    if ordered[EXPORT_COLUMNS].isnull().any().any():
        null_columns = ordered.columns[ordered.isnull().any()].tolist()
        raise ValueError(f"Null values detected in export columns: {null_columns}")

    logger.info("Validated generated dataset schema with %d samples.", len(ordered))
    return ordered


def build_metadata(
    df: pd.DataFrame,
    acquisition_date: str,
    cloud_cover: float,
    satellite: str,
    last_update: str,
) -> dict[str, Any]:
    """Create metadata for the latest exported dataset."""
    return {
        "last_update": last_update,
        "acquisition_date": acquisition_date,
        "satellite": satellite,
        "cloud_cover": float(cloud_cover),
        "total_samples": int(len(df)),
    }


def dataframe_from_feature_collection(feature_collection: Any) -> pd.DataFrame:
    """Convert an Earth Engine feature collection to a pandas DataFrame."""
    try:
        records = feature_collection.getInfo().get("features", [])
    except Exception as exc:
        logger.exception("Failed to materialize Earth Engine feature collection.")
        raise RuntimeError("Unable to read Earth Engine feature collection output.") from exc

    rows = [feature.get("properties", {}) for feature in records]
    if not rows:
        raise ValueError("Earth Engine returned no samples for the requested AOI and date range.")

    return pd.DataFrame(rows)
