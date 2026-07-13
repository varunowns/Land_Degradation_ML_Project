"""Home — landing page with project summary and quick navigation."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from utils.config import (
    CLASS_ORDER,
    GRID_RESOLUTION,
    LATEST_PREDICTIONS_PATH,
    N_DISTRICTS,
    PAGES,
    PREDICTIONS_PATH,
    STUDY_AREA,
    STUDY_YEARS,
)
from config_manager import load_config
from data_health import check_dataset_health
from history_manager import load_prediction_history
from utils.model_loader import get_model_metadata, verify_artifacts_exist
from utils.plotting import (
    create_degradation_count_by_district,
    create_degradation_percentage_pie,
    create_latest_ndvi_distribution,
)
from utils.ui import render_section_divider, setup_page

FREQUENCY_LABELS = {
    "daily": "Daily",
    "8_days": "Every 8 Days",
    "monthly": "Monthly",
}


def _format_timestamp(value: str | None) -> str:
    if not value:
        return "Not available"
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return value
    return parsed.strftime("%d %b %Y, %I:%M %p")


def _load_latest_predictions() -> tuple[pd.DataFrame, str]:
    if LATEST_PREDICTIONS_PATH.exists():
        return pd.read_csv(LATEST_PREDICTIONS_PATH), str(LATEST_PREDICTIONS_PATH)
    if PREDICTIONS_PATH.exists():
        return pd.read_csv(PREDICTIONS_PATH), str(PREDICTIONS_PATH)
    return pd.DataFrame(), str(LATEST_PREDICTIONS_PATH)


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((column for column in candidates if column in df.columns), None)

setup_page(
    "Land Degradation Prediction System",
    subtitle=f"Machine-learning powered monitoring for {STUDY_AREA} ({STUDY_YEARS})",
)

render_section_divider("Monitoring Footprint", "🌱")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Study Area", STUDY_AREA)
col2.metric("District Network", N_DISTRICTS)
col3.metric("Grid Resolution", GRID_RESOLUTION)
col4.metric("Risk Classes", " / ".join(CLASS_ORDER))

render_section_divider("System Status", "🌿")
settings = load_config()
predictions_df, predictions_source = _load_latest_predictions()
class_col = _first_existing_column(predictions_df, ["predicted_class", "Degradation_Class", "y_pred"])
district_col = _first_existing_column(predictions_df, ["District", "district"])
year_col = _first_existing_column(predictions_df, ["Year", "year"])
ndvi_col = _first_existing_column(predictions_df, ["NDVI_mean", "NDVI", "ndvi"])

latest_year = "Not available"
if year_col and not predictions_df.empty:
    latest_year = str(int(predictions_df[year_col].max()))

s1, s2, s3, s4 = st.columns(4)
s1.metric("Last Data Update", _format_timestamp(settings.get("last_update")))
s2.metric(
    "Update Frequency",
    FREQUENCY_LABELS.get(settings.get("data_update_frequency"), "Every 8 Days"),
)
s3.metric("Total Predictions", f"{len(predictions_df):,}")
s4.metric("Latest Prediction Year", latest_year)
st.caption(f"Prediction source: `{predictions_source}`")

render_section_divider("Environmental Data Health", "🌱")
try:
    health = check_dataset_health()
    status_label = health["dataset_status"]
    status_color_map = {
        "FRESH": "Fresh 🟢",
        "OUTDATED": "Outdated 🟠",
        "CRITICAL": "Critical 🔴",
    }
    status_display = status_color_map.get(status_label, "Unknown ❓")

    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Dataset Status", status_display)
    h2.metric("Satellite Observations", f"{health['total_records']:,}")

    start_y = health["start_year"]
    end_y = health["end_year"]
    coverage_display = f"{start_y}–{end_y}" if start_y and end_y else "Not available"
    h3.metric("Coverage", coverage_display)

    next_up_ts = health["next_update"]
    next_up_display = _format_timestamp(next_up_ts) if next_up_ts else "Not scheduled"
    h4.metric("Next Update", next_up_display)

    if health["message"]:
        st.caption(f"Health Message: {health['message']}")
except Exception as e:
    st.error(f"Error checking environmental data health: {e}")


if predictions_df.empty:
    st.warning("No latest prediction file found yet. Expected `results/latest_predictions.csv`.")
elif not class_col:
    st.warning("Prediction class column not found. Expected `predicted_class`, `Degradation_Class`, or `y_pred`.")
else:
    render_section_divider("Latest AI Monitoring Signals", "🌾")
    with st.container(border=True):
        chart_col1, chart_col2 = st.columns([2, 1])
        with chart_col1:
            if district_col:
                st.plotly_chart(
                    create_degradation_count_by_district(predictions_df, district_col, class_col),
                    use_container_width=True,
                )
            else:
                st.info("District-wise chart unavailable because no District column was found.")
        with chart_col2:
            st.plotly_chart(
                create_degradation_percentage_pie(predictions_df, class_col),
                use_container_width=True,
            )

    if ndvi_col:
        with st.container(border=True):
            st.plotly_chart(
                create_latest_ndvi_distribution(predictions_df, ndvi_col),
                use_container_width=True,
            )
    else:
        st.info("Latest NDVI distribution unavailable because no NDVI column was found.")

render_section_divider("AI Model Health Monitor", "🤖")
try:
    from model_monitor import check_model_health, HEALTH_REPORT_PATH
    import json
    
    # Generate the report if it does not exist
    if not HEALTH_REPORT_PATH.exists():
        check_model_health()
        
    if HEALTH_REPORT_PATH.exists():
        with HEALTH_REPORT_PATH.open("r", encoding="utf-8") as f:
            health_data = json.load(f)
            
        status = health_data.get("overall_status", "UNKNOWN")
        status_map = {
            "HEALTHY": "Healthy 🟢",
            "WARNING": "Warning 🟠",
            "DRIFT_DETECTED": "Retraining Recommended 🔴"
        }
        status_display = status_map.get(status, "Unknown ❓")
        
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Model Status", status_display)
        mc2.metric("Highest Drift Feature", health_data.get("highest_drift_feature", "None"))
        mc3.metric("Maximum Drift Percentage", f"{health_data.get('highest_drift_percent', 0.0):.2f}%")
        
        last_checked = health_data.get("last_checked")
        last_checked_display = _format_timestamp(last_checked) if last_checked else "Not available"
        mc4.metric("Last Health Check", last_checked_display)
        
        features_dict = health_data.get("features", {})
        if features_dict:
            drift_df = pd.DataFrame([
                {"Feature": feat, "Drift %": info["drift_percent"]}
                for feat, info in features_dict.items()
            ])
            
            fig_drift = px.bar(
                drift_df,
                x="Feature",
                y="Drift %",
                title="Feature Drift Percentage relative to Baseline",
                text="Drift %"
            )
            fig_drift.update_traces(
                marker_color="#8B5E3C", 
                textposition="outside",
                texttemplate='%{text:.2f}%'
            )
            fig_drift.update_layout(
                template="plotly_white",
                yaxis=dict(ticksuffix="%")
            )
            
            with st.container(border=True):
                st.plotly_chart(fig_drift, use_container_width=True)
        else:
            st.info("No feature drift data available.")
    else:
        st.warning("Model health report could not be loaded or generated.")
except Exception as e:
    st.error(f"Error rendering AI Model Health Monitor: {e}")

render_section_divider("Deployed Model", "🌳")
metadata = get_model_metadata()
artifact_status = verify_artifacts_exist()

info_col, status_col = st.columns([2, 1])
with info_col:
    with st.container(border=True):
        st.markdown(
            f"""
            <div class="ld-card-title">🌱 Model Intelligence Layer</div>

            | Property | Value |
            |----------|-------|
            | **Model** | {metadata['model_name']} |
            | **Task** | {metadata['task']} |
            | **Classes** | {metadata['classes']} |
            | **Features** | {metadata['features']} |
            """,
            unsafe_allow_html=True,
        )
with status_col:
    st.markdown("**Artefact Status**")
    for name, exists in artifact_status.items():
        st.markdown(f"{'✅' if exists else '❌'} `{name}`")

render_section_divider("Long-Term Degradation Monitoring", "📈")
try:
    history_df = load_prediction_history()
    if history_df.empty:
        st.info("Monitoring history will appear after future updates.")
    else:
        # Format Timestamp for readable dates on chart axis
        history_df['Date'] = pd.to_datetime(history_df['Timestamp']).dt.strftime('%d %b %H:%M')
        
        # Color mapping aligning with brown earth theme
        color_map = {
            "Low": "#7CB342",
            "Moderate": "#F59E0B",
            "High": "#D32F2F"
        }
        
        # Chart 1: Degradation class trend over update dates
        class_trend = history_df.groupby(['Date', 'Predicted_Class']).size().reset_index(name='Count')
        fig1 = px.line(
            class_trend,
            x='Date',
            y='Count',
            color='Predicted_Class',
            color_discrete_map=color_map,
            title="Degradation Class Distribution Trend over Updates",
            markers=True
        )
        fig1.update_layout(template='plotly_white', xaxis_title="Update Date", yaxis_title="Grid Cells Count")
        
        # Chart 2: District risk evolution
        high_mod_df = history_df[history_df['Predicted_Class'].isin(['High', 'Moderate'])]
        if not high_mod_df.empty:
            district_evo = high_mod_df.groupby(['Date', 'District']).size().reset_index(name='Risk_Grid_Count')
            fig2 = px.bar(
                district_evo,
                x='District',
                y='Risk_Grid_Count',
                color='Date',
                barmode='group',
                title="District-wise Risk Grid Count (Moderate/High) Evolution",
                color_discrete_sequence=["#8B5E3C", "#A67C52", "#C49A6C", "#D7CCC8", "#6F4E37"]
            )
            fig2.update_layout(template='plotly_white', xaxis_title="District", yaxis_title="Risk Grid Cells Count")
        else:
            fig2 = None

        # Chart 3: Number of high degradation areas over time
        high_df = history_df[history_df['Predicted_Class'] == 'High']
        if not high_df.empty:
            high_trend = high_df.groupby('Date').size().reset_index(name='High_Risk_Count')
            fig3 = px.line(
                high_trend,
                x='Date',
                y='High_Risk_Count',
                title="Total High Degradation Grid Cells Over Time",
                markers=True
            )
            fig3.update_traces(line_color="#E53935", marker=dict(size=8, color="#E53935"))
            fig3.update_layout(template='plotly_white', xaxis_title="Update Date", yaxis_title="High Risk Grid Cells")
        else:
            fig3 = None
            
        with st.container(border=True):
            st.plotly_chart(fig1, use_container_width=True)
            
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            with st.container(border=True):
                if fig2:
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("No High/Moderate risk predictions recorded yet to display evolution.")
        with col_c2:
            with st.container(border=True):
                if fig3:
                    st.plotly_chart(fig3, use_container_width=True)
                else:
                    st.info("No High risk predictions recorded yet to display trend.")
except Exception as e:
    st.error(f"Error rendering long-term monitoring visualizations: {e}")

render_section_divider("Explore the Monitoring Console", "🌱")
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
