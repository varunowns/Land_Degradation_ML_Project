"""Home — landing page with project summary and quick navigation."""

from __future__ import annotations

import streamlit as st

from utils.config import CLASS_ORDER, GRID_RESOLUTION, N_DISTRICTS, PAGES, STUDY_AREA, STUDY_YEARS
from utils.model_loader import get_model_metadata, verify_artifacts_exist
from utils.ui import setup_page

setup_page(
    "Land Degradation Prediction System",
    subtitle=f"Machine-learning powered monitoring for {STUDY_AREA} ({STUDY_YEARS})",
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Study Area", STUDY_AREA)
col2.metric("Districts", N_DISTRICTS)
col3.metric("Grid Resolution", GRID_RESOLUTION)
col4.metric("Classes", " / ".join(CLASS_ORDER))

st.markdown("---")
st.markdown("### Deployed Model")
metadata = get_model_metadata()
artifact_status = verify_artifacts_exist()

info_col, status_col = st.columns([2, 1])
with info_col:
    st.markdown(
        f"""
        | Property | Value |
        |----------|-------|
        | **Model** | {metadata['model_name']} |
        | **Task** | {metadata['task']} |
        | **Classes** | {metadata['classes']} |
        | **Features** | {metadata['features']} |
        """
    )
with status_col:
    st.markdown("**Artefact Status**")
    for name, exists in artifact_status.items():
        st.markdown(f"{'✅' if exists else '❌'} `{name}`")

st.markdown("### Explore the Application")
nav_cols = st.columns(2)
for idx, page in enumerate(PAGES[1:], start=0):
    col = nav_cols[idx % 2]
    with col:
        with st.container(border=True):
            st.markdown(f"#### {page['icon']} {page['label']}")
            st.page_link(page["file"], label=f"Open {page['label']} →")

st.success(
    "All application features are live: Single & batch prediction, EDA, "
    "model performance, SHAP explainability, and the Geographic Dashboard."
)
