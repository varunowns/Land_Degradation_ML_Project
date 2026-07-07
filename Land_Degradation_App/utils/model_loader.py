"""
Model artefact loading utilities.

Provides cached, read-only access to pre-trained models and encoders.
Prediction logic is intentionally **not** implemented here — this module
only handles discovery and loading of saved joblib files.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import streamlit as st

from utils.config import (
    APP_ROOT,
    BEST_MODEL_FILENAME,
    LABEL_ENCODER_FILENAME,
    ML_PROJECT_ROOT,
    MODELS_DIR,
    PREPROCESSOR_LR_FILENAME,
)


def _model_path(filename: str) -> Path:
    """Resolve a model file from app or ML project directory."""
    for base in (APP_ROOT / "models", ML_PROJECT_ROOT / "models"):
        candidate = base / filename
        if candidate.exists():
            return candidate
    return MODELS_DIR / filename


class ModelArtifacts:
    """Container for loaded model artefacts."""

    def __init__(
        self,
        model: Any,
        preprocessor: Any,
        label_encoder: Any,
        model_path: Path,
    ) -> None:
        self.model = model
        self.preprocessor = preprocessor
        self.label_encoder = label_encoder
        self.model_path = model_path


def get_models_directory() -> Path:
    """Return the path to the models directory."""
    return MODELS_DIR


def list_available_models() -> list[str]:
    """List all ``.pkl`` model files available in the models directory."""
    if not MODELS_DIR.exists():
        return []
    return sorted(p.name for p in MODELS_DIR.glob("*.pkl"))


def verify_artifacts_exist() -> dict[str, bool]:
    """
    Check whether required artefact files are present on disk.

    Returns
    -------
    dict[str, bool]
        Mapping of artefact name to existence flag.
    """
    artefacts = {
        "best_model": _model_path(BEST_MODEL_FILENAME),
        "preprocessor_lr": _model_path(PREPROCESSOR_LR_FILENAME),
        "label_encoder": _model_path(LABEL_ENCODER_FILENAME),
    }
    return {name: path.exists() for name, path in artefacts.items()}


@st.cache_resource(show_spinner="Loading model artefacts…")
def load_model_artifacts() -> ModelArtifacts:
    """
    Load the best tuned model and its companion preprocessors.

    Raises
    ------
    FileNotFoundError
        If any required artefact is missing from ``models/``.

    Notes
    -----
    Cached with ``@st.cache_resource`` so artefacts are loaded once per session.
    Models are **never** retrained or modified by this function.
    """
    model_path = _model_path(BEST_MODEL_FILENAME)
    preprocessor_path = _model_path(PREPROCESSOR_LR_FILENAME)
    encoder_path = _model_path(LABEL_ENCODER_FILENAME)

    missing = [
        str(p)
        for p in (model_path, preprocessor_path, encoder_path)
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Required model artefacts not found: {', '.join(missing)}"
        )

    return ModelArtifacts(
        model=joblib.load(model_path),
        preprocessor=joblib.load(preprocessor_path),
        label_encoder=joblib.load(encoder_path),
        model_path=model_path,
    )


def get_model_metadata() -> dict[str, str]:
    """Return static metadata about the deployed model (no inference)."""
    return {
        "model_name": "Tuned Logistic Regression",
        "model_file": BEST_MODEL_FILENAME,
        "task": "Multiclass classification",
        "classes": "Low, Moderate, High",
        "features": "13 environmental + 19 district (one-hot)",
    }
