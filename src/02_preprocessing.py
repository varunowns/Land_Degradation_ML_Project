"""
Feature preprocessing for degradation-class classification.

Design note — District encoding
--------------------------------
District is **included** via **one-hot encoding** (``drop_first=True``).

Rationale:
- Uttar Pradesh contains 20 districts with distinct agro-climatic zones.
  Excluding District would discard useful spatial context captured at the
  administrative scale of this study.
- **Label encoding is rejected** because it imposes a false ordinal relationship
  on nominal district IDs, which distorts Logistic Regression coefficients.
- **One-hot encoding** treats each district as an independent spatial indicator,
  which is statistically appropriate for linear and tree-based models alike.
  With 19 dummy variables (20 districts, one reference category), dimensionality
  remains manageable relative to ~22 645 observations.

Grid_ID is excluded (identifier). LDI is excluded (target leakage).
"""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from utils import (
    CLASS_ORDER,
    METADATA_COLUMNS,
    MODELS_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
    ensure_directories,
    save_artifact,
    setup_logging,
)

logger = setup_logging(__name__)


@dataclass
class PreparedData:
    """Train/test splits with aligned metadata for error analysis."""

    X_train_lr: pd.DataFrame
    X_test_lr: pd.DataFrame
    X_train_tree: pd.DataFrame
    X_test_tree: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names_lr: list[str]
    feature_names_tree: list[str]
    metadata_train: pd.DataFrame
    metadata_test: pd.DataFrame
    label_encoder: LabelEncoder
    preprocessor_lr: ColumnTransformer
    preprocessor_tree: ColumnTransformer
    raw_train: pd.DataFrame
    raw_test: pd.DataFrame


def _build_lr_preprocessor(numeric_features: list[str]) -> ColumnTransformer:
    """Scale numeric features and one-hot encode District for linear models."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "district",
                OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                ["District"],
            ),
        ],
        remainder="drop",
    )


def _build_tree_preprocessor(numeric_features: list[str]) -> ColumnTransformer:
    """One-hot encode District; leave numeric features unscaled for tree models."""
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            (
                "district",
                OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                ["District"],
            ),
        ],
        remainder="drop",
    )


def _get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Extract human-readable feature names after transformation."""
    names: list[str] = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder":
            continue
        if name == "num":
            names.extend(list(columns))
        elif name == "district" and hasattr(transformer, "get_feature_names_out"):
            names.extend(transformer.get_feature_names_out(["District"]).tolist())
    return names


def prepare_train_test_split(
    df: pd.DataFrame,
    numeric_features: list[str],
) -> PreparedData:
    """
    Build stratified train/test splits and fitted preprocessors.

    Logistic Regression receives StandardScaler-transformed numerics.
    Tree-based models receive unscaled numerics with the same District encoding.
    """
    ensure_directories()

    label_encoder = LabelEncoder()
    label_encoder.fit(CLASS_ORDER)
    y = label_encoder.transform(df[TARGET_COLUMN].values)

    feature_frame = df[numeric_features + ["District"]].copy()
    metadata = df[METADATA_COLUMNS + ["Grid_ID"]].copy()

    (
        raw_train,
        raw_test,
        y_train,
        y_test,
        meta_train,
        meta_test,
    ) = train_test_split(
        feature_frame,
        y,
        metadata,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor_lr = _build_lr_preprocessor(numeric_features)
    preprocessor_tree = _build_tree_preprocessor(numeric_features)

    X_train_lr = preprocessor_lr.fit_transform(raw_train)
    X_test_lr = preprocessor_lr.transform(raw_test)
    X_train_tree = preprocessor_tree.fit_transform(raw_train)
    X_test_tree = preprocessor_tree.transform(raw_test)

    feature_names_lr = _get_feature_names(preprocessor_lr)
    feature_names_tree = _get_feature_names(preprocessor_tree)

    logger.info(
        "Train/test split: %d / %d | LR features: %d | Tree features: %d",
        len(y_train),
        len(y_test),
        len(feature_names_lr),
        len(feature_names_tree),
    )

    save_artifact(preprocessor_lr, "preprocessor_lr.pkl")
    save_artifact(preprocessor_tree, "preprocessor_tree.pkl")
    save_artifact(label_encoder, "label_encoder.pkl")

    prepared = PreparedData(
        X_train_lr=pd.DataFrame(X_train_lr, columns=feature_names_lr),
        X_test_lr=pd.DataFrame(X_test_lr, columns=feature_names_lr),
        X_train_tree=pd.DataFrame(X_train_tree, columns=feature_names_tree),
        X_test_tree=pd.DataFrame(X_test_tree, columns=feature_names_tree),
        y_train=y_train,
        y_test=y_test,
        feature_names_lr=feature_names_lr,
        feature_names_tree=feature_names_tree,
        metadata_train=meta_train.reset_index(drop=True),
        metadata_test=meta_test.reset_index(drop=True),
        label_encoder=label_encoder,
        preprocessor_lr=preprocessor_lr,
        preprocessor_tree=preprocessor_tree,
        raw_train=raw_train.reset_index(drop=True),
        raw_test=raw_test.reset_index(drop=True),
    )
    save_artifact(prepared, "train_test_splits.pkl")
    return prepared


def build_lr_pipeline(numeric_features: list[str]) -> Pipeline:
    """Return a sklearn Pipeline with scaling + Logistic Regression."""
    return Pipeline(
        steps=[
            ("preprocess", _build_lr_preprocessor(numeric_features)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                ),
            ),
        ]
    )


if __name__ == "__main__":
    import importlib

    loader = importlib.import_module("01_data_loader")
    loaded = loader.prepare_loaded_dataset()
    prepared = prepare_train_test_split(loaded.dataframe, loaded.feature_columns)
    print("LR:", prepared.X_train_lr.shape, "Tree:", prepared.X_train_tree.shape)
