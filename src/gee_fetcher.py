"""
Google Earth Engine data fetcher for the land degradation project.

This module exports annual grid-level CSV files for:
- NDVI
- Rainfall
- Temperature
- Soil Moisture
- LULC

Primary outputs are written to ``Land_Degradation/Raw/`` using lowercase names
such as ``ndvi_2025.csv``. For backward compatibility with
``phase1_data_prep.py``, matching legacy copies are also written to
``Land_Degradation/`` with the historic naming scheme, e.g. ``NDVI_2025.csv``.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from utils import setup_logging

try:
    import ee
except ImportError:  # pragma: no cover - depends on optional runtime dependency
    ee = None

logger = setup_logging(__name__, level=logging.INFO)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LAND_DEGRADATION_DIR = PROJECT_ROOT / "Land_Degradation"
RAW_OUTPUT_DIR = LAND_DEGRADATION_DIR / "Raw"

SCALAR_OUTPUT_COLUMNS = ["system:index", "Area_km2", "District", "Grid_ID", "mean", ".geo"]
LULC_OUTPUT_COLUMNS = [
    "system:index",
    "Area_km2",
    "BareLand",
    "Builtup",
    "Cropland",
    "District",
    "Grassland",
    "Grid_ID",
    "Mangroves",
    "MossLichen",
    "Shrubland",
    "SnowIce",
    "TreeCover",
    "Water",
    "Wetland",
    ".geo",
]

LEGACY_FILENAMES = {
    "ndvi": "NDVI",
    "rainfall": "Rainfall",
    "temperature": "Temperature",
    "soil_moisture": "SoilMoisture",
    "lulc": "LULC",
}

LULC_CLASS_MAP = {
    "BareLand": 7,
    "Builtup": 6,
    "Cropland": 4,
    "Grassland": 2,
    "Shrubland": 5,
    "SnowIce": 8,
    "TreeCover": 1,
    "Water": 0,
    "Wetland": 3,
}


def authenticate_gee() -> None:
    """Authenticate and initialize the Earth Engine client."""
    if ee is None:
        raise ImportError(
            "earthengine-api is not installed. Install it with `pip install earthengine-api`."
        )

    project = os.getenv("EARTHENGINE_PROJECT")
    try:
        ee.Initialize(project=project) if project else ee.Initialize()
        logger.info("Google Earth Engine initialized successfully.")
        return
    except Exception as exc:
        logger.warning("Initial Earth Engine initialization failed: %s", exc)

    try:
        ee.Authenticate()
        ee.Initialize(project=project) if project else ee.Initialize()
        logger.info("Google Earth Engine authenticated and initialized.")
    except Exception as exc:
        logger.exception("Unable to authenticate with Google Earth Engine.")
        raise RuntimeError("Google Earth Engine authentication failed.") from exc


def fetch_ndvi(year: int) -> Path:
    """Fetch annual mean NDVI and write CSV outputs."""
    logger.info("Fetching NDVI for %s", year)
    start, end = _year_date_range(year)
    collection = ee.ImageCollection("MODIS/061/MOD13Q1").filterDate(start, end).select("NDVI")
    _ensure_non_empty_collection(collection, "NDVI", year)
    image = collection.mean().multiply(0.0001).rename("mean")
    df = _reduce_scalar_image(image, year, scale=250)
    return _write_feature_output(df, "ndvi", year, SCALAR_OUTPUT_COLUMNS)


def fetch_rainfall(year: int) -> Path:
    """Fetch annual rainfall totals and write CSV outputs."""
    logger.info("Fetching rainfall for %s", year)
    start, end = _year_date_range(year)
    collection = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterDate(start, end).select(
        "precipitation"
    )
    _ensure_non_empty_collection(collection, "Rainfall", year)
    image = collection.sum().rename("mean")
    df = _reduce_scalar_image(image, year, scale=5566)
    return _write_feature_output(df, "rainfall", year, SCALAR_OUTPUT_COLUMNS)


def fetch_temperature(year: int) -> Path:
    """Fetch annual mean 2m air temperature in Celsius and write CSV outputs."""
    logger.info("Fetching temperature for %s", year)
    start, end = _year_date_range(year)
    collection = (
        ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
        .filterDate(start, end)
        .select("temperature_2m")
    )
    _ensure_non_empty_collection(collection, "Temperature", year)
    image = collection.mean().subtract(273.15).rename("mean")
    df = _reduce_scalar_image(image, year, scale=11132)
    return _write_feature_output(df, "temperature", year, SCALAR_OUTPUT_COLUMNS)


def fetch_soil_moisture(year: int) -> Path:
    """Fetch annual mean surface soil moisture and write CSV outputs."""
    logger.info("Fetching soil moisture for %s", year)
    start, end = _year_date_range(year)
    collection = (
        ee.ImageCollection("NASA_USDA/HSL/SMAP10KM_soil_moisture")
        .filterDate(start, end)
        .select("ssm")
    )
    _ensure_non_empty_collection(collection, "Soil Moisture", year)
    image = collection.mean().rename("mean")
    df = _reduce_scalar_image(image, year, scale=10000)
    return _write_feature_output(df, "soil_moisture", year, SCALAR_OUTPUT_COLUMNS)


def fetch_lulc(year: int) -> Path:
    """Fetch annual LULC percentages and write CSV outputs."""
    logger.info("Fetching LULC for %s", year)
    start, end = _year_date_range(year)
    collection = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterDate(start, end).select(
        "label"
    )
    _ensure_non_empty_collection(collection, "LULC", year)
    label_mode = collection.mode().rename("label")

    class_bands = []
    for column, class_id in LULC_CLASS_MAP.items():
        band = label_mode.eq(class_id).multiply(100).rename(column)
        class_bands.append(band)

    class_bands.append(ee.Image.constant(0).rename("Mangroves"))
    class_bands.append(ee.Image.constant(0).rename("MossLichen"))
    lulc_image = ee.Image.cat(class_bands)

    df = _reduce_multiband_image(lulc_image, year, scale=10)
    return _write_feature_output(df, "lulc", year, LULC_OUTPUT_COLUMNS)


def fetch_all_data(year: int) -> dict[str, Path]:
    """Fetch all annual GEE layers for the study grid."""
    logger.info("Starting full GEE fetch for %s", year)
    authenticate_gee()
    RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, Path] = {}
    failures: dict[str, str] = {}

    jobs = {
        "ndvi": fetch_ndvi,
        "rainfall": fetch_rainfall,
        "temperature": fetch_temperature,
        "soil_moisture": fetch_soil_moisture,
        "lulc": fetch_lulc,
    }
    for name, func in jobs.items():
        try:
            results[name] = func(year)
        except Exception as exc:
            failures[name] = str(exc)
            logger.exception("Failed to fetch %s for %s", name, year)

    if failures:
        details = "; ".join(f"{name}: {message}" for name, message in failures.items())
        raise RuntimeError(f"One or more GEE fetch jobs failed for {year}: {details}")

    logger.info("Completed full GEE fetch for %s", year)
    return results


def _year_date_range(year: int) -> tuple[str, str]:
    """Return an inclusive-exclusive annual date window."""
    current_utc = datetime.now(timezone.utc)
    if year > current_utc.year:
        raise ValueError(f"Cannot fetch data for future year {year}.")

    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    if year == current_utc.year:
        end = current_utc + timedelta(days=1)
    else:
        end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _ensure_non_empty_collection(collection: Any, label: str, year: int) -> None:
    """Raise a clear error when a dataset has no images for the requested year."""
    count = int(collection.size().getInfo())
    if count == 0:
        raise ValueError(f"No {label} imagery available in Earth Engine for {year}.")


def _load_template_grid() -> pd.DataFrame:
    """
    Load a legacy LULC CSV to reuse the study grid and metadata.

    This keeps output compatible with the existing project schema without
    rebuilding the 5 km grid from scratch.
    """
    candidates = sorted(LAND_DEGRADATION_DIR.glob("LULC_*.csv"), reverse=True)
    if not candidates:
        raise FileNotFoundError(
            f"No legacy LULC template found in {LAND_DEGRADATION_DIR}. "
            "At least one existing LULC CSV is required to define the study grid."
        )

    template_path = candidates[0]
    logger.info("Using template grid from %s", template_path)
    df = pd.read_csv(template_path, usecols=["system:index", "Area_km2", "District", "Grid_ID", ".geo"])
    district_count = df["District"].nunique()
    if district_count != 20:
        logger.warning("Expected 20 study districts, found %d in template grid.", district_count)
    return df


def _build_template_feature_collection(template_df: pd.DataFrame) -> Any:
    """Build an Earth Engine FeatureCollection from the existing grid polygons."""
    features = []
    for system_index, area_km2, district, grid_id, geojson_text in template_df.itertuples(
        index=False, name=None
    ):
        geometry = ee.Geometry(json.loads(geojson_text))
        feature = ee.Feature(
            geometry,
            {
                "system:index": system_index,
                "Area_km2": float(area_km2),
                "District": district,
                "Grid_ID": grid_id,
                ".geo": geojson_text,
            },
        )
        features.append(feature)
    logger.info("Prepared %d grid features for Earth Engine sampling.", len(features))
    return ee.FeatureCollection(features)


def _reduce_scalar_image(image: Any, year: int, scale: int) -> pd.DataFrame:
    """Reduce a single-band image over the study grid and return a dataframe."""
    template_df = _load_template_grid()
    feature_collection = _build_template_feature_collection(template_df)
    reduced = image.reduceRegions(
        collection=feature_collection,
        reducer=ee.Reducer.mean(),
        scale=scale,
        tileScale=4,
    )
    df = _feature_collection_to_dataframe(reduced)
    return _coerce_output_columns(df, SCALAR_OUTPUT_COLUMNS)


def _reduce_multiband_image(image: Any, year: int, scale: int) -> pd.DataFrame:
    """Reduce a multi-band image over the study grid and return a dataframe."""
    template_df = _load_template_grid()
    feature_collection = _build_template_feature_collection(template_df)
    reduced = image.reduceRegions(
        collection=feature_collection,
        reducer=ee.Reducer.mean(),
        scale=scale,
        tileScale=4,
    )
    df = _feature_collection_to_dataframe(reduced)
    return _coerce_output_columns(df, LULC_OUTPUT_COLUMNS)


def _feature_collection_to_dataframe(feature_collection: Any) -> pd.DataFrame:
    """Download a FeatureCollection as a pandas DataFrame with a safe fallback."""
    try:
        computed = ee.data.computeFeatures(
            {"expression": feature_collection, "fileFormat": "PANDAS_DATAFRAME"}
        )
        if isinstance(computed, pd.DataFrame):
            return computed
    except Exception as exc:
        logger.warning("computeFeatures dataframe download failed, falling back to paging: %s", exc)

    size = int(feature_collection.size().getInfo())
    rows: list[dict[str, Any]] = []
    page_size = 500
    for start in range(0, size, page_size):
        batch = feature_collection.toList(page_size, start).getInfo()
        for feature in batch:
            rows.append(feature.get("properties", {}))
    return pd.DataFrame(rows)


def _coerce_output_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Align Earth Engine output to the project's expected column order."""
    working = df.copy()
    for column in columns:
        if column not in working.columns:
            working[column] = 0.0 if column not in {"system:index", "District", "Grid_ID", ".geo"} else None
    working = working[columns]
    return working


def _write_feature_output(df: pd.DataFrame, feature_name: str, year: int, columns: list[str]) -> Path:
    """Write raw and legacy-compatible CSV outputs."""
    RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_OUTPUT_DIR / f"{feature_name}_{year}.csv"
    legacy_stem = LEGACY_FILENAMES[feature_name]
    legacy_path = LAND_DEGRADATION_DIR / f"{legacy_stem}_{year}.csv"

    export_df = _coerce_output_columns(df, columns)
    export_df.to_csv(raw_path, index=False)
    export_df.to_csv(legacy_path, index=False)

    logger.info("Saved %s output to %s and legacy copy to %s", feature_name, raw_path, legacy_path)
    return raw_path
