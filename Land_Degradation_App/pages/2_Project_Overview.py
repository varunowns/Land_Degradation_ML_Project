"""Project Overview — methodology and dataset description."""

from __future__ import annotations

import streamlit as st

from utils.config import (
    GRID_RESOLUTION,
    LEAKAGE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    STUDY_AREA,
    STUDY_YEARS,
    TARGET_COLUMN,
)
from utils.preprocessing import describe_preprocessing_pipeline, get_district_list
from utils.ui import render_placeholder, setup_page

setup_page(
    "Project Overview",
    subtitle="Methodology, study design, and data sources",
)

tab_method, tab_data, tab_features = st.tabs(
    ["Methodology", "Dataset", "Feature Engineering"]
)

with tab_method:
    st.markdown(
        """
        ### Objective
        Predict **land degradation severity** (Low / Moderate / High) across Uttar Pradesh
        using remote-sensing features extracted from **Google Earth Engine** and a
        supervised machine-learning classifier.

        ### Workflow
        1. **GEE Extraction** — LULC, NDVI, rainfall, temperature, soil moisture (2020–2024)
        2. **LDI Construction** — Composite Land Degradation Index with quantile classification
        3. **ML Pipeline** — Stratified train/test split, 5 baseline models, top-2 tuning
        4. **Deployment** — This Streamlit application (inference & visualisation)

        ### Study Design
        """
    )
    c1, c2 = st.columns(2)
    c1.info(f"**Area:** {STUDY_AREA}")
    c2.info(f"**Resolution:** {GRID_RESOLUTION}")
    c1.info(f"**Period:** {STUDY_YEARS}")
    c2.info(f"**Target:** `{TARGET_COLUMN}`")

with tab_data:
    st.markdown("### Dataset Summary")
    districts = get_district_list()
    if districts:
        st.success(f"**{len(districts)} districts** loaded from training data.")
        st.dataframe(
            {"District": districts},
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("Dataset not found in `data/ldi_dataset.csv`.")
    render_placeholder(
        "Interactive dataset explorer with filtering and summary statistics "
        "will be added in the EDA Dashboard phase.",
    )

with tab_features:
    st.markdown("### Input Features")
    st.markdown(f"**{len(NUMERIC_FEATURE_COLUMNS)} numeric environmental features:**")
    st.code(", ".join(NUMERIC_FEATURE_COLUMNS))

    pipeline = describe_preprocessing_pipeline()
    st.markdown("### Preprocessing Pipeline")
    for key, value in pipeline.items():
        st.markdown(f"- **{key.replace('_', ' ').title()}:** {value}")

    st.markdown("### Excluded Columns")
    st.markdown(
        f"- **Target leakage:** `{', '.join(LEAKAGE_COLUMNS)}`\n"
        f"- **Target label:** `{TARGET_COLUMN}`\n"
        "- **Identifier:** `Grid_ID`"
    )
