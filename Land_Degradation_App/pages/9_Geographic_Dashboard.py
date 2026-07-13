"""Geographic Dashboard — spatial exploration of land degradation across UP."""

from __future__ import annotations

import streamlit as st

from utils.config import COLORS, STUDY_YEARS_LIST
from utils.data_loader import load_ldi_dataset
from utils.geography import (
    aggregate_district_stats,
    build_animated_choropleth,
    build_choropleth,
    get_district_detail,
    get_district_yearly_trend,
    get_geojson_source_info,
    get_kpi_summary,
    get_metric_options,
    load_up_geojson,
)
from utils.plotting import create_bar_ranking, create_line_trend, create_scatter, export_figure_png
from utils.prediction import predict_batch
from utils.ui import render_section_divider, setup_page

setup_page(
    "Geographic Dashboard",
    subtitle="Explore spatial patterns of land degradation across Uttar Pradesh",
)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.markdown("### 🌱 Map Controls")
year_options = ["All Years"] + STUDY_YEARS_LIST
selected_year = st.sidebar.selectbox("Year", year_options, index=0)

metric_options = get_metric_options()
metric_labels = [m[0] for m in metric_options]
metric_cols = [m[1] for m in metric_options]
selected_metric_label = st.sidebar.selectbox("Map Metric", metric_labels, index=0)
selected_metric_col = metric_cols[metric_labels.index(selected_metric_label)]

animated_mode = st.sidebar.toggle("Animated Playback (2020→2024)", value=False)

with st.sidebar.expander("GeoJSON Source"):
    st.caption(get_geojson_source_info())

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
try:
    geojson = load_up_geojson()
    stats = aggregate_district_stats(selected_year)
except Exception as exc:
    st.error(f"Failed to load geographic data: {exc}")
    st.stop()

# ---------------------------------------------------------------------------
# KPI cards
# ---------------------------------------------------------------------------
kpis = get_kpi_summary(stats)
render_section_divider("State Monitoring Overview", "🌱")
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Most Degraded", kpis.get("most_degraded", "—"))
k2.metric("Its Avg LDI", f"{kpis.get('most_degraded_ldi', 0):.4f}")
k3.metric("Least Degraded", kpis.get("least_degraded", "—"))
k4.metric("Avg LDI", f"{kpis.get('avg_ldi', 0):.4f}")
k5.metric("Avg NDVI", f"{kpis.get('avg_ndvi', 0):.4f}")
k6.metric("Avg Rainfall", f"{kpis.get('avg_rainfall', 0):.0f} mm")

# ---------------------------------------------------------------------------
# Choropleth map
# ---------------------------------------------------------------------------
render_section_divider("District Choropleth Map", "🌿")

if animated_mode:
    fig = build_animated_choropleth(geojson, selected_metric_col, selected_metric_label)
else:
    fig = build_choropleth(stats, geojson, selected_metric_col, selected_metric_label)

with st.container(border=True):
    st.plotly_chart(fig, use_container_width=True)

export_col1, export_col2 = st.columns(2)
png_bytes = export_figure_png(fig)
with export_col1:
    if png_bytes:
        st.download_button(
            "Export Map as PNG",
            data=png_bytes,
            file_name=f"up_degradation_map_{selected_year}.png",
            mime="image/png",
        )
    else:
        st.caption("Install `kaleido` for PNG export: `pip install kaleido`")

with export_col2:
    st.download_button(
        "Download District Summary (CSV)",
        data=stats.to_csv(index=False).encode("utf-8"),
        file_name=f"district_summary_{selected_year}.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------------------------
# Rankings
# ---------------------------------------------------------------------------
render_section_divider("District Risk Rankings", "🌾")
rank_col1, rank_col2 = st.columns(2)
with rank_col1:
    with st.container(border=True):
        st.plotly_chart(
            create_bar_ranking(
                stats, "District", "LDI",
                f"Top 10 Most Degraded Districts ({selected_year})",
                ascending=False,
            ),
            use_container_width=True,
        )
with rank_col2:
    with st.container(border=True):
        st.plotly_chart(
            create_bar_ranking(
                stats, "District", "LDI",
                f"Top 10 Least Degraded Districts ({selected_year})",
                ascending=True,
            ),
            use_container_width=True,
        )

# ---------------------------------------------------------------------------
# Trends & scatter
# ---------------------------------------------------------------------------
render_section_divider("Temporal & Environmental Relationships", "🌱")
yearly = load_ldi_dataset().groupby("Year").agg(LDI=("LDI", "mean")).reset_index()
trend_col, scatter_col = st.columns(2)
with trend_col:
    with st.container(border=True):
        st.plotly_chart(
            create_line_trend(yearly, "Year", "LDI", "State-wide Mean LDI by Year", COLORS["accent"]),
            use_container_width=True,
        )

sample_df = load_ldi_dataset()
if selected_year != "All Years":
    sample_df = sample_df[sample_df["Year"] == int(selected_year)]
sample_df = sample_df.sample(min(2000, len(sample_df)), random_state=42)

with scatter_col:
    with st.container(border=True):
        scatter_choice = st.selectbox(
            "Scatter X-axis",
            ["Temperature_mean", "NDVI_mean", "Rainfall_mean"],
            key="geo_scatter_x",
        )
        st.plotly_chart(
            create_scatter(
                sample_df, scatter_choice, "LDI",
                color="Degradation_Class",
                title=f"LDI vs {scatter_choice}",
            ),
            use_container_width=True,
        )

sc_row1, sc_row2 = st.columns(2)
with sc_row1:
    with st.container(border=True):
        st.plotly_chart(
            create_scatter(
                sample_df, "NDVI_mean", "LDI",
                color="Degradation_Class", title="LDI vs NDVI",
            ),
            use_container_width=True,
        )
with sc_row2:
    with st.container(border=True):
        st.plotly_chart(
            create_scatter(
                sample_df, "Rainfall_mean", "LDI",
                color="Degradation_Class", title="LDI vs Rainfall",
            ),
            use_container_width=True,
        )

# ---------------------------------------------------------------------------
# District drill-down
# ---------------------------------------------------------------------------
render_section_divider("District Drill-Down", "🌳")
selected_district = st.selectbox(
    "Select District",
    sorted(stats["District"].tolist()),
    key="geo_district_select",
)

detail = get_district_detail(selected_district, selected_year)
trend = get_district_yearly_trend(selected_district)

d1, d2, d3 = st.columns(3)
d1.metric("Grids", len(detail))
d2.metric("Mean LDI", f"{detail['LDI'].mean():.4f}")
d3.metric("Dominant Class", detail["Degradation_Class"].mode().iloc[0])

prof_col, pred_col = st.columns(2)
with prof_col:
    st.markdown("#### Environmental Profile")
    profile = detail[
        ["NDVI_mean", "Rainfall_mean", "Temperature_mean",
         "SoilMoisture_mean", "BareLand", "TreeCover"]
    ].mean()
    st.dataframe(profile.to_frame("Mean Value").T, use_container_width=True)

    st.markdown("#### Historical LDI Trend")
    st.plotly_chart(
        create_line_trend(trend, "Year", "LDI", f"LDI Trend — {selected_district}"),
        use_container_width=True,
    )

with pred_col:
    st.markdown("#### Prediction Summary")
    try:
        from utils.preprocessing import REQUIRED_INPUT_COLUMNS

        pred_input = detail[REQUIRED_INPUT_COLUMNS].head(50).copy()
        preds = predict_batch(pred_input)
        summary = preds["predicted_class"].value_counts()
        st.bar_chart(summary)
        st.caption(f"Model predictions for up to 50 grids in {selected_district}.")
        agree = (preds["predicted_class"] == detail.head(50)["Degradation_Class"].values).mean()
        st.metric("Agreement with LDI Class", f"{agree:.1%}")
    except Exception as exc:
        st.warning(f"Prediction summary unavailable: {exc}")

st.markdown("#### District Grid Sample")
st.dataframe(detail.head(20), use_container_width=True, hide_index=True)
