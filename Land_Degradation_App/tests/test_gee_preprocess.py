from pathlib import Path

import pandas as pd

from gee.config import EXPORT_COLUMNS, FULL_LULC_COLUMNS
from gee.preprocess import apply_legacy_preprocessing, build_metadata, validate_feature_columns


def test_validate_feature_columns_preserves_order() -> None:
    df = pd.DataFrame(
        [
            {
                "Grid_ID": "GRID_1",
                "District": "Agra",
                "Year": 2024,
                "Area_km2": 25.0,
                "BareLand": 1.0,
                "Builtup": 2.0,
                "Cropland": 70.0,
                "Grassland": 3.0,
                "TreeCover": 10.0,
                "Water": 5.0,
                "Wetland": 0.5,
                "Shrubland": 0.2,
                "NDVI_mean": 0.45,
                "Rainfall_mean": 1000.0,
                "Temperature_mean": 25.0,
                "SoilMoisture_mean": 0.22,
            }
        ]
    )

    result = validate_feature_columns(df)
    assert result.columns.tolist() == EXPORT_COLUMNS


def test_build_metadata_shape() -> None:
    df = pd.DataFrame({"Grid_ID": ["GRID_1", "GRID_2"]})
    metadata = build_metadata(
        df=df,
        acquisition_date="2024-12-31",
        cloud_cover=12.5,
        satellite="Sentinel-2 Surface Reflectance",
        last_update="2026-07-05T00:00:00+00:00",
    )

    assert metadata == {
        "last_update": "2026-07-05T00:00:00+00:00",
        "acquisition_date": "2024-12-31",
        "satellite": "Sentinel-2 Surface Reflectance",
        "cloud_cover": 12.5,
        "total_samples": 2,
    }


def test_legacy_merge_matches_training_dataset_shape() -> None:
    project_root = Path(__file__).resolve().parents[2]
    yearly_frames: list[pd.DataFrame] = []

    for year in (2020, 2021, 2022, 2023, 2024):
        base = pd.read_csv(project_root / "Land_Degradation" / f"LULC_{year}.csv")
        base["Year"] = year
        base = base.rename(
            columns={
                "BareLand": "BareLand",
                "Builtup": "Builtup",
                "Cropland": "Cropland",
                "Grassland": "Grassland",
                "TreeCover": "TreeCover",
                "Water": "Water",
                "Wetland": "Wetland",
                "Shrubland": "Shrubland",
                "Mangroves": "Mangroves",
                "MossLichen": "MossLichen",
                "SnowIce": "SnowIce",
            }
        )
        base = base[
            ["Grid_ID", "District", "Year", "Area_km2", *FULL_LULC_COLUMNS]
        ].copy()

        for variable in ("NDVI", "Rainfall", "Temperature", "SoilMoisture"):
            variable_df = pd.read_csv(project_root / "Land_Degradation" / f"{variable}_{year}.csv")
            variable_df = variable_df[["Grid_ID", "mean"]].rename(
                columns={"mean": f"{variable}_mean"}
            )
            base = pd.merge(base, variable_df, on="Grid_ID", how="inner")

        yearly_frames.append(base)

    merged = pd.concat(yearly_frames, ignore_index=True)
    processed = apply_legacy_preprocessing(merged)
    export_df = validate_feature_columns(processed)
    reference_df = pd.read_csv(project_root / "data" / "working_dataset.csv")

    assert export_df.shape == reference_df.shape
    assert export_df.columns.tolist() == reference_df.columns.tolist()
