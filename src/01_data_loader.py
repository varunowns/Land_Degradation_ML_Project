"""
Data loading for the Land Degradation supervised classification pipeline.

Loads the pre-built ``ldi_dataset.csv`` and validates schema expectations.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from utils import (
    CLASS_ORDER,
    EXCLUDE_FROM_FEATURES,
    LDI_DATASET_PATH,
    METADATA_COLUMNS,
    TARGET_COLUMN,
    setup_logging,
)

logger = setup_logging(__name__)


@dataclass
class LoadedDataset:
    """Container for the raw LDI dataset and derived metadata."""

    dataframe: pd.DataFrame
    feature_columns: list[str]
    target_column: str
    metadata_columns: list[str]


def load_ldi_dataset(path: str | None = None) -> pd.DataFrame:
    """
    Load ``ldi_dataset.csv`` without modification.

    Parameters
    ----------
    path:
        Optional override for the dataset location. Defaults to ``data/ldi_dataset.csv``.

    Returns
    -------
    pd.DataFrame
        Full dataset including identifiers, features, LDI, and target.
    """
    dataset_path = LDI_DATASET_PATH if path is None else path
    logger.info("Loading dataset from %s", dataset_path)
    df = pd.read_csv(dataset_path)
    _validate_dataset(df)
    logger.info("Loaded %d rows and %d columns", df.shape[0], df.shape[1])
    return df


def _validate_dataset(df: pd.DataFrame) -> None:
    """Raise informative errors if required columns or class labels are missing."""
    required = set(EXCLUDE_FROM_FEATURES + METADATA_COLUMNS + ["Year", "Area_km2"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    observed_classes = set(df[TARGET_COLUMN].unique())
    expected_classes = set(CLASS_ORDER)
    if observed_classes != expected_classes:
        raise ValueError(
            f"Unexpected degradation classes: {observed_classes}. Expected {expected_classes}."
        )


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return numeric/environmental feature columns for modelling.

    Excludes identifiers, the LDI leakage column, and the string target.
    District is handled separately during preprocessing (one-hot encoding).
    """
    exclude = set(EXCLUDE_FROM_FEATURES + METADATA_COLUMNS)
    features = [col for col in df.columns if col not in exclude]
    logger.info("Identified %d base feature columns: %s", len(features), features)
    return features


def prepare_loaded_dataset(path: str | None = None) -> LoadedDataset:
    """Load and annotate the dataset for downstream preprocessing."""
    df = load_ldi_dataset(path)
    return LoadedDataset(
        dataframe=df,
        feature_columns=get_feature_columns(df),
        target_column=TARGET_COLUMN,
        metadata_columns=METADATA_COLUMNS.copy(),
    )


if __name__ == "__main__":
    loaded = prepare_loaded_dataset()
    print(loaded.dataframe.head())
    print(f"Features: {loaded.feature_columns}")
