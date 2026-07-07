"""Automated Earth Engine data acquisition and export workflow."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .authenticate import authenticate_gee
from .config import (
    AOI_FALLBACK_CSV_PATH,
    CONFIG,
    EXPORT_COLUMNS,
    FULL_LULC_COLUMNS,
    NATIVE_SCALES,
    RAINFALL_COLLECTION_ID,
    S2_COLLECTION_ID,
    SATELLITE_NAME,
    SOIL_MOISTURE_COLLECTION_ID,
    TEMPERATURE_COLLECTION_ID,
    WORLDCOVER_2020_ID,
    WORLDCOVER_2021_ID,
    WORLDCOVER_CLASS_MAP,
)
from .preprocess import (
    apply_legacy_preprocessing,
    build_metadata,
    dataframe_from_feature_collection,
    validate_feature_columns,
)

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    """Configure default logging once for the pipeline."""
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )


def _load_feature_collection_from_asset(ee: object, asset_id: str) -> Any:
    """Load a feature collection asset."""
    return ee.FeatureCollection(asset_id)


def _load_feature_collection_from_csv(ee: object, csv_path: Path) -> Any:
    """Build a feature collection from the legacy exported grid CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Legacy grid CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, usecols=["Grid_ID", "District", "Area_km2", ".geo"])
    features = []
    for _, row in df.drop_duplicates(subset=["Grid_ID"]).iterrows():
        geometry_json = json.loads(row[".geo"])
        geometry = ee.Geometry(geometry_json)
        feature = ee.Feature(
            geometry,
            {
                CONFIG.grid_id_property: row["Grid_ID"],
                CONFIG.district_property: row["District"],
                CONFIG.area_property: float(row["Area_km2"]),
            },
        )
        features.append(feature)
    logger.info("Loaded %d grid features from %s", len(features), csv_path)
    return ee.FeatureCollection(features)


def _load_grid(ee: object) -> Any:
    """Load the project grid feature collection."""
    if CONFIG.grid_asset_id:
        logger.info("Loading grid asset: %s", CONFIG.grid_asset_id)
        return _load_feature_collection_from_asset(ee, CONFIG.grid_asset_id)

    logger.warning(
        "Grid asset not configured. Falling back to legacy CSV geometry from %s",
        AOI_FALLBACK_CSV_PATH,
    )
    return _load_feature_collection_from_csv(ee, AOI_FALLBACK_CSV_PATH)


def _load_aoi(ee: object, grid_fc: Any) -> Any:
    """Load the AOI boundary or derive it from the grid asset."""
    if CONFIG.aoi_asset_id:
        logger.info("Loading AOI asset: %s", CONFIG.aoi_asset_id)
        return _load_feature_collection_from_asset(ee, CONFIG.aoi_asset_id)
    return ee.FeatureCollection(grid_fc).geometry()


def _mask_s2_clouds(image: Any) -> Any:
    """Apply Sentinel-2 QA60 cloud masking and scale reflectance."""
    qa60 = image.select("QA60")
    cloud_bit = 1 << 10
    cirrus_bit = 1 << 11
    mask = qa60.bitwiseAnd(cloud_bit).eq(0).And(qa60.bitwiseAnd(cirrus_bit).eq(0))
    return image.updateMask(mask).divide(10000).copyProperties(image, image.propertyNames())


def _date_window_for_year(year: int) -> tuple[str, str]:
    """Return the full calendar-year date window."""
    return f"{year}-01-01", f"{year + 1}-01-01"


def _get_worldcover_image(ee: object, year: int) -> Any:
    """Return the WorldCover image used in the legacy workflow."""
    worldcover_id = WORLDCOVER_2020_ID if year == 2020 else WORLDCOVER_2021_ID
    return ee.Image(worldcover_id).select("Map")


def _reduce_image_over_grid(
    ee: object,
    image: Any,
    grid_fc: Any,
    scale: int,
    extra_properties: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Reduce a single image over the project grid and return a DataFrame."""

    def set_properties(feature: Any) -> Any:
        feature = feature.set(
            {
                "Grid_ID": feature.get(CONFIG.grid_id_property),
                "District": feature.get(CONFIG.district_property),
                "Area_km2": feature.get(CONFIG.area_property),
            }
        )
        if extra_properties:
            feature = feature.set(extra_properties)
        return feature

    reduced_fc = image.reduceRegions(
        collection=grid_fc,
        reducer=ee.Reducer.mean(),
        scale=scale,
        crs=CONFIG.export_crs,
        tileScale=4,
    ).map(set_properties)
    return dataframe_from_feature_collection(reduced_fc)


def _fetch_ndvi_table(ee: object, grid_fc: Any, aoi: Any, year: int) -> tuple[pd.DataFrame, float, str]:
    """Fetch annual NDVI means over the project grid."""
    start_date, end_date = _date_window_for_year(year)
    collection = (
        ee.ImageCollection(S2_COLLECTION_ID)
        .filterDate(start_date, end_date)
        .filterBounds(aoi)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", CONFIG.cloud_threshold))
        .map(_mask_s2_clouds)
    )

    image_count = int(collection.size().getInfo())
    if image_count == 0:
        raise ValueError(f"No Sentinel-2 imagery matched the configured filters for {year}.")

    ndvi_image = (
        collection
        .map(lambda image: image.normalizedDifference(["B8", "B4"]).rename("mean"))
        .median()
    )
    df = _reduce_image_over_grid(
        ee,
        ndvi_image,
        grid_fc,
        NATIVE_SCALES["ndvi"],
        {"Year": year},
    )
    cloud_cover = float(
        ee.Number(collection.aggregate_mean("CLOUDY_PIXEL_PERCENTAGE")).getInfo()
    )
    acquisition_date = str(
        ee.Date(collection.sort("system:time_start", False).first().get("system:time_start"))
        .format("YYYY-MM-dd")
        .getInfo()
    )
    logger.info("Fetched NDVI table for %d using %d Sentinel-2 scenes.", year, image_count)
    return df, cloud_cover, acquisition_date


def _fetch_rainfall_table(ee: object, grid_fc: Any, aoi: Any, year: int) -> pd.DataFrame:
    """Fetch annual rainfall totals over the project grid."""
    start_date, end_date = _date_window_for_year(year)
    image = (
        ee.ImageCollection(RAINFALL_COLLECTION_ID)
        .filterDate(start_date, end_date)
        .filterBounds(aoi)
        .select("precipitation")
        .sum()
        .rename("mean")
    )
    logger.info("Fetched rainfall table for %d.", year)
    return _reduce_image_over_grid(
        ee,
        image,
        grid_fc,
        NATIVE_SCALES["rainfall"],
        {"Year": year},
    )


def _fetch_temperature_table(ee: object, grid_fc: Any, aoi: Any, year: int) -> pd.DataFrame:
    """Fetch annual mean 2m air temperature in Celsius over the project grid."""
    start_date, end_date = _date_window_for_year(year)
    image = (
        ee.ImageCollection(TEMPERATURE_COLLECTION_ID)
        .filterDate(start_date, end_date)
        .filterBounds(aoi)
        .select("temperature_2m")
        .mean()
        .subtract(273.15)
        .rename("mean")
    )
    logger.info("Fetched temperature table for %d.", year)
    return _reduce_image_over_grid(
        ee,
        image,
        grid_fc,
        NATIVE_SCALES["temperature"],
        {"Year": year},
    )


def _fetch_soil_moisture_table(ee: object, grid_fc: Any, aoi: Any, year: int) -> pd.DataFrame:
    """Fetch annual mean soil moisture over the project grid."""
    start_date, end_date = _date_window_for_year(year)
    image = (
        ee.ImageCollection(SOIL_MOISTURE_COLLECTION_ID)
        .filterDate(start_date, end_date)
        .filterBounds(aoi)
        .select("ssm")
        .mean()
        .rename("mean")
    )
    logger.info("Fetched soil moisture table for %d.", year)
    return _reduce_image_over_grid(
        ee,
        image,
        grid_fc,
        NATIVE_SCALES["soil_moisture"],
        {"Year": year},
    )


def _fetch_lulc_table(ee: object, grid_fc: Any, year: int) -> pd.DataFrame:
    """Fetch annual WorldCover land-cover percentages over the project grid."""
    worldcover_image = _get_worldcover_image(ee, year)
    base_df: pd.DataFrame | None = None

    for class_code, column_name in WORLDCOVER_CLASS_MAP.items():
        image = worldcover_image.eq(class_code).multiply(100).rename("mean")
        class_df = _reduce_image_over_grid(
            ee,
            image,
            grid_fc,
            NATIVE_SCALES["lulc"],
            {"Year": year},
        )
        class_df = class_df.rename(columns={"mean": column_name})
        keep = ["Grid_ID", column_name]
        if base_df is None:
            keep = ["Grid_ID", "District", "Area_km2", "Year", column_name]
        class_df = class_df[keep]
        base_df = class_df if base_df is None else pd.merge(base_df, class_df, on="Grid_ID", how="inner")

    if base_df is None:
        raise ValueError(f"Failed to build LULC table for {year}.")

    ordered = ["Grid_ID", "District", "Year", "Area_km2", *FULL_LULC_COLUMNS]
    logger.info("Fetched LULC table for %d.", year)
    return base_df[ordered]


def _merge_yearly_tables(
    year: int,
    lulc_df: pd.DataFrame,
    ndvi_df: pd.DataFrame,
    rainfall_df: pd.DataFrame,
    temperature_df: pd.DataFrame,
    soil_moisture_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge yearly tables exactly like the legacy manual workflow."""
    merged = lulc_df.copy()

    variable_frames = [
        ("NDVI_mean", ndvi_df),
        ("Rainfall_mean", rainfall_df),
        ("Temperature_mean", temperature_df),
        ("SoilMoisture_mean", soil_moisture_df),
    ]

    for column_name, frame in variable_frames:
        reduced = frame[["Grid_ID", "mean"]].rename(columns={"mean": column_name})
        merged = pd.merge(merged, reduced, on="Grid_ID", how="inner")

    merged["Year"] = year
    return merged


def _build_dataset(ee: object, grid_fc: Any, aoi: Any) -> tuple[pd.DataFrame, float, str]:
    """Build the multi-year dataset matching the legacy GEE workflow."""
    yearly_frames: list[pd.DataFrame] = []
    cloud_cover_by_year: list[float] = []
    acquisition_dates: list[str] = []

    for year in CONFIG.study_years:
        lulc_df = _fetch_lulc_table(ee, grid_fc, year)
        ndvi_df, cloud_cover, acquisition_date = _fetch_ndvi_table(ee, grid_fc, aoi, year)
        rainfall_df = _fetch_rainfall_table(ee, grid_fc, aoi, year)
        temperature_df = _fetch_temperature_table(ee, grid_fc, aoi, year)
        soil_moisture_df = _fetch_soil_moisture_table(ee, grid_fc, aoi, year)

        yearly_frames.append(
            _merge_yearly_tables(
                year,
                lulc_df,
                ndvi_df,
                rainfall_df,
                temperature_df,
                soil_moisture_df,
            )
        )
        cloud_cover_by_year.append(cloud_cover)
        acquisition_dates.append(acquisition_date)

    master_df = pd.concat(yearly_frames, ignore_index=True)
    master_df = apply_legacy_preprocessing(master_df)
    export_df = validate_feature_columns(master_df)
    export_df = export_df[EXPORT_COLUMNS].copy()
    return export_df, sum(cloud_cover_by_year) / len(cloud_cover_by_year), max(acquisition_dates)


def _export_dataset(df: pd.DataFrame) -> Path:
    """Save the latest dataset CSV to disk."""
    CONFIG.export_directory.mkdir(parents=True, exist_ok=True)
    output_path = CONFIG.export_directory / CONFIG.dataset_filename
    df.to_csv(output_path, index=False)
    logger.info("Exported processed dataset to %s", output_path)
    return output_path


def _export_metadata(metadata: dict[str, Any]) -> Path:
    """Save dataset metadata to disk."""
    CONFIG.export_directory.mkdir(parents=True, exist_ok=True)
    output_path = CONFIG.export_directory / CONFIG.metadata_filename
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    logger.info("Saved metadata to %s", output_path)
    return output_path


def update_dataset() -> pd.DataFrame:
    """Run the complete Earth Engine dataset refresh workflow."""
    _setup_logging()
    logger.info("Starting automated Earth Engine dataset update.")

    try:
        ee = authenticate_gee()
        grid_fc = _load_grid(ee)
        aoi = _load_aoi(ee, grid_fc)
        dataset_df, cloud_cover, acquisition_date = _build_dataset(ee, grid_fc, aoi)
        _export_dataset(dataset_df)

        metadata = build_metadata(
            df=dataset_df,
            acquisition_date=acquisition_date,
            cloud_cover=cloud_cover,
            satellite=SATELLITE_NAME,
            last_update=datetime.now(timezone.utc).isoformat(),
        )
        _export_metadata(metadata)
        logger.info("Automated Earth Engine dataset update completed successfully.")
        return dataset_df
    except Exception:
        logger.exception("Automated Earth Engine dataset update failed.")
        raise
