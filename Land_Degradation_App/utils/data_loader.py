"""
Cached dataset loading for the Streamlit application.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

from utils.config import LDI_DATASET_PATH, WORKING_DATASET_PATH
from config_manager import get_cache_ttl_seconds, update_last_run

CACHE_TTL_SECONDS = get_cache_ttl_seconds()


@st.cache_data(show_spinner="Loading LDI dataset…", ttl=CACHE_TTL_SECONDS)
def load_ldi_dataset() -> pd.DataFrame:
    """Load the full LDI dataset (read-only)."""
    if not LDI_DATASET_PATH.exists():
        raise FileNotFoundError(f"LDI dataset not found: {LDI_DATASET_PATH}")
    df = pd.read_csv(LDI_DATASET_PATH)
    update_last_run()
    return df


@st.cache_data(show_spinner="Loading working dataset…", ttl=CACHE_TTL_SECONDS)
def load_working_dataset() -> pd.DataFrame:
    """Load the working (cleaned) dataset."""
    if not WORKING_DATASET_PATH.exists():
        raise FileNotFoundError(f"Working dataset not found: {WORKING_DATASET_PATH}")
    df = pd.read_csv(WORKING_DATASET_PATH)
    update_last_run()
    return df


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def get_district_medians() -> pd.DataFrame:
    """Return per-district median values for form defaults (model inputs only)."""
    from utils.preprocessing import REQUIRED_INPUT_COLUMNS

    df = load_ldi_dataset()
    numeric = [c for c in REQUIRED_INPUT_COLUMNS if c != "District" and c in df.columns]
    grouped = df.groupby("District")[numeric].median().reset_index()
    return grouped
