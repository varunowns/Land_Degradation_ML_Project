"""
Cached dataset loading for the Streamlit application.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

from utils.config import LDI_DATASET_PATH, WORKING_DATASET_PATH


@st.cache_data(show_spinner="Loading LDI dataset…")
def load_ldi_dataset() -> pd.DataFrame:
    """Load the full LDI dataset (read-only)."""
    if not LDI_DATASET_PATH.exists():
        raise FileNotFoundError(f"LDI dataset not found: {LDI_DATASET_PATH}")
    return pd.read_csv(LDI_DATASET_PATH)


@st.cache_data(show_spinner="Loading working dataset…")
def load_working_dataset() -> pd.DataFrame:
    """Load the working (cleaned) dataset."""
    if not WORKING_DATASET_PATH.exists():
        raise FileNotFoundError(f"Working dataset not found: {WORKING_DATASET_PATH}")
    return pd.read_csv(WORKING_DATASET_PATH)


@st.cache_data
def get_district_medians() -> pd.DataFrame:
    """Return per-district median values for form defaults (model inputs only)."""
    from utils.preprocessing import REQUIRED_INPUT_COLUMNS

    df = load_ldi_dataset()
    numeric = [c for c in REQUIRED_INPUT_COLUMNS if c != "District" and c in df.columns]
    grouped = df.groupby("District")[numeric].median().reset_index()
    return grouped
