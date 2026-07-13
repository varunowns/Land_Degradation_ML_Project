"""EDA Dashboard — interactive exploratory data analysis."""

from __future__ import annotations

import streamlit as st

from utils.config import CLASS_ORDER, NUMERIC_FEATURE_COLUMNS, STUDY_YEARS_LIST
from utils.data_loader import load_ldi_dataset
from utils.plotting import (
    create_boxplot_by_class,
    create_correlation_heatmap,
    create_degradation_percentage_pie,
    create_histogram,
    create_scatter,
)
from utils.ui import render_section_divider, setup_page

setup_page(
    "EDA Dashboard",
    subtitle="Exploratory analysis of land degradation indicators across Uttar Pradesh",
)

try:
    df = load_ldi_dataset()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
render_section_divider("Environmental Filters", "🌱")
with st.container(border=True):
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        year_filter = st.multiselect("Year", STUDY_YEARS_LIST, default=STUDY_YEARS_LIST)
    with filter_col2:
        district_filter = st.multiselect(
            "District",
            sorted(df["District"].unique()),
            default=sorted(df["District"].unique()),
        )
    with filter_col3:
        class_filter = st.multiselect("Degradation Class", CLASS_ORDER, default=CLASS_ORDER)

filtered = df[
    df["Year"].isin(year_filter)
    & df["District"].isin(district_filter)
    & df["Degradation_Class"].isin(class_filter)
]

# ---------------------------------------------------------------------------
# Overview KPIs
# ---------------------------------------------------------------------------
render_section_divider("Monitoring Metrics", "🌿")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Observations", f"{len(filtered):,}")
k2.metric("Mean LDI", f"{filtered['LDI'].mean():.4f}")
k3.metric("Mean NDVI", f"{filtered['NDVI_mean'].mean():.4f}")
k4.metric("Mean Rainfall", f"{filtered['Rainfall_mean'].mean():.1f} mm")
k5.metric("Mean Temperature", f"{filtered['Temperature_mean'].mean():.2f} °C")

tab_dist, tab_class, tab_corr, tab_scatter = st.tabs(
    ["Distributions", "By Class", "Correlations", "Scatter Analysis"]
)

env_features = [
    "LDI", "NDVI_mean", "Rainfall_mean", "Temperature_mean",
    "SoilMoisture_mean", "BareLand", "TreeCover", "Cropland",
]

with tab_dist:
    with st.container(border=True):
        feature = st.selectbox("Feature", env_features, key="eda_hist")
        st.plotly_chart(create_histogram(filtered, feature), use_container_width=True)

        year_trend = filtered.groupby("Year")["LDI"].mean().reset_index()
        st.plotly_chart(
            __import__("utils.plotting", fromlist=["create_line_trend"]).create_line_trend(
                year_trend, "Year", "LDI", "Mean LDI by Year"
            ),
            use_container_width=True,
        )

with tab_class:
    with st.container(border=True):
        box_feature = st.selectbox("Feature", env_features, key="eda_box")
        st.plotly_chart(
            create_boxplot_by_class(filtered, box_feature),
            use_container_width=True,
        )
        st.plotly_chart(
            create_degradation_percentage_pie(filtered, "Degradation_Class"),
            use_container_width=True,
        )

with tab_corr:
    with st.container(border=True):
        corr_cols = st.multiselect(
            "Features for correlation",
            env_features,
            default=env_features[:6],
            key="eda_corr",
        )
        if len(corr_cols) >= 2:
            st.plotly_chart(
                create_correlation_heatmap(filtered, corr_cols),
                use_container_width=True,
            )

with tab_scatter:
    with st.container(border=True):
        sc1, sc2 = st.columns(2)
        with sc1:
            x_var = st.selectbox("X axis", env_features, index=env_features.index("Temperature_mean"))
            y_var = st.selectbox("Y axis", env_features, index=0)
        st.plotly_chart(
            create_scatter(
                filtered.sample(min(3000, len(filtered)), random_state=42),
                x_var,
                y_var,
                color="Degradation_Class",
                title=f"{y_var} vs {x_var}",
            ),
            use_container_width=True,
        )

        district_ldi = filtered.groupby("District")["LDI"].mean().sort_values(ascending=False)
        st.markdown("#### 🌱 Mean LDI by District")
        st.bar_chart(district_ldi)
