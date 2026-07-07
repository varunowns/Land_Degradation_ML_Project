"""Configuration for the automated Google Earth Engine data pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Load .env before reading environment variables
try:
    from dotenv import load_dotenv

    _ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
    if _ENV_PATH.exists():
        load_dotenv(_ENV_PATH, override=False)
except ImportError:
    pass


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
EXPORT_CSV_PATH = DATA_DIR / "latest_dataset.csv"
METADATA_PATH = DATA_DIR / "metadata.json"
AOI_FALLBACK_CSV_PATH = PROJECT_ROOT / "Land_Degradation" / "LULC_2020.csv"


@dataclass(frozen=True)
class GEEConfig:
    """Runtime configuration for the Earth Engine pipeline."""

    gee_project: str | None
    grid_asset_id: str | None
    aoi_asset_id: str | None
    study_years: tuple[int, ...]
    cloud_threshold: float
    export_directory: Path
    dataset_filename: str
    metadata_filename: str
    export_crs: str
    grid_id_property: str
    district_property: str
    area_property: str


CONFIG = GEEConfig(
    gee_project=os.getenv("GEE_PROJECT_ID"),
    grid_asset_id=os.getenv("GEE_GRID_ASSET_ID"),
    aoi_asset_id=os.getenv("GEE_AOI_ASSET_ID"),
    study_years=(2020, 2021, 2022, 2023, 2024),
    cloud_threshold=20.0,
    export_directory=DATA_DIR,
    dataset_filename=EXPORT_CSV_PATH.name,
    metadata_filename=METADATA_PATH.name,
    export_crs="EPSG:3857",
    grid_id_property="Grid_ID",
    district_property="District",
    area_property="Area_km2",
)


FULL_LULC_COLUMNS: list[str] = [
    "BareLand",
    "Builtup",
    "Cropland",
    "Grassland",
    "TreeCover",
    "Water",
    "Wetland",
    "Shrubland",
    "Mangroves",
    "MossLichen",
    "SnowIce",
]

MODEL_FEATURE_COLUMNS: list[str] = [
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

LULC_OUTPUT_COLUMNS: list[str] = [
    "Grid_ID",
    "District",
    "Year",
    "Area_km2",
    *FULL_LULC_COLUMNS,
]

EXPORT_COLUMNS: list[str] = [
    "Grid_ID",
    "District",
    "Year",
    *MODEL_FEATURE_COLUMNS,
]

SATELLITE_NAME = "Sentinel-2 Surface Reflectance"
S2_COLLECTION_ID = "COPERNICUS/S2_SR_HARMONIZED"
RAINFALL_COLLECTION_ID = "UCSB-CHG/CHIRPS/DAILY"
TEMPERATURE_COLLECTION_ID = "ECMWF/ERA5_LAND/MONTHLY_AGGR"
SOIL_MOISTURE_COLLECTION_ID = "NASA_USDA/HSL/SMAP10KM_soil_moisture"
WORLDCOVER_2020_ID = "ESA/WorldCover/v100/2020"
WORLDCOVER_2021_ID = "ESA/WorldCover/v200/2021"

WORLDCOVER_CLASS_MAP: dict[int, str] = {
    60: "BareLand",
    50: "Builtup",
    40: "Cropland",
    30: "Grassland",
    10: "TreeCover",
    80: "Water",
    90: "Wetland",
    20: "Shrubland",
    95: "Mangroves",
    100: "MossLichen",
    70: "SnowIce",
}

NATIVE_SCALES: dict[str, int] = {
    "lulc": 10,
    "ndvi": 10,
    "rainfall": 5566,
    "temperature": 11132,
    "soil_moisture": 10000,
}
