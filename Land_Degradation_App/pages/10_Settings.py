"""Settings page for dataset refresh controls."""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from utils.ui import setup_page
from config_manager import load_config, update_frequency
from update_pipeline import run_full_update
from gee_auth_manager import check_gee_status, authenticate_gee

FREQUENCY_OPTIONS = {
    "Daily": "daily",
    "Every 8 Days": "8_days",
    "Monthly": "monthly",
}


def _format_timestamp(value: str | None) -> str:
    """Format stored ISO timestamps for display."""
    if not value:
        return "Not available"

    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return value

    tz_label = parsed.tzname() or "UTC"
    return parsed.strftime("%d %b %Y, %I:%M %p") + f" {tz_label}"


setup_page(
    "Settings",
    subtitle="Manage dataset refresh cadence and trigger manual update requests",
)

config = load_config()
current_frequency = config["data_update_frequency"]
current_label = next(
    (label for label, value in FREQUENCY_OPTIONS.items() if value == current_frequency),
    "Every 8 Days",
)

st.markdown("### Data Update Frequency")
selected_label = st.radio(
    "Choose how often the dataset should refresh",
    options=list(FREQUENCY_OPTIONS.keys()),
    index=list(FREQUENCY_OPTIONS.keys()).index(current_label),
)

selected_value = FREQUENCY_OPTIONS[selected_label]
if selected_value != current_frequency:
    config = update_frequency(selected_value)
    current_frequency = config["data_update_frequency"]
    st.success(f"Data update frequency saved as {selected_label}.")

status_col, update_col = st.columns(2)
status_col.metric("Current Frequency", selected_label if selected_value == current_frequency else current_label)
update_col.metric("Last Dataset Update", _format_timestamp(config.get("last_update")))

st.caption(f"Pipeline status: {config.get('update_pipeline_status', 'idle').title()}")

st.markdown("### Earth Observation Infrastructure")
with st.container(border=True):
    # Ensure config has latest values
    config = load_config()
    current_project_id = config.get("gee_project_id", "")
    
    st.markdown("**Google Cloud Project ID**")
    project_id_input = st.text_input(
        "Enter your Google Cloud Project ID for Earth Engine", 
        value=current_project_id,
        placeholder="land-degradation-123456",
        label_visibility="collapsed"
    )
    
    if st.button("Save Project ID", type="secondary"):
        from config_manager import save_config
        config["gee_project_id"] = project_id_input
        save_config(config)
        st.success("Project ID saved successfully.")
        st.rerun()

    st.divider()

    gee_status = check_gee_status()
    # Reload config to get the latest last_gee_check timestamp updated by check_gee_status
    config = load_config()
    
    col_status, col_action = st.columns([3, 2])
    with col_status:
        st.markdown("**Google Earth Engine (GEE)**")
        if gee_status == "CONNECTED":
            st.markdown("#### Connected 🟢")
        else:
            st.markdown("#### Not Connected 🔴")
        st.caption(f"Last connection check: {_format_timestamp(config.get('last_gee_check'))}")
        
    with col_action:
        st.write("")  # padding
        st.write("")  # padding
        if st.button("Authenticate / Reconnect GEE", type="secondary", use_container_width=True):
            with st.spinner("Starting GEE authentication flow..."):
                status, message = authenticate_gee()
            if status:
                st.success("Google Earth Engine connected successfully")
                check_gee_status()
                st.rerun()
            else:
                st.error(message)

with st.container(border=True):
    st.markdown("### Manual Update")
    st.write("Trigger a dataset refresh request immediately.")

    if st.button("Run Update Now", type="primary", use_container_width=True):
        progress_placeholder = st.empty()
        log_lines = []

        def show_progress(msg: str):
            log_lines.append(msg)
            progress_placeholder.markdown("\n".join(f"✓ {line}" if idx < len(log_lines) - 1 or "updated" in line.lower() else f"🔄 **{line}**" for idx, line in enumerate(log_lines)))

        try:
            result = run_full_update(progress_callback=show_progress)
            status = result.get("status")
            if status == "SUCCESS":
                st.success(result.get("message", "Dataset update pipeline finished successfully."))
                st.cache_data.clear()
            else:
                st.error(result.get("message", "Dataset update pipeline failed."))
        except Exception as exc:
            st.error(f"Dataset update pipeline failed with exception: {exc}")
            result = {"status": "FAILED", "requested_at": datetime.now().isoformat()}

        st.caption(f"Last request: {_format_timestamp(result.get('requested_at'))}")
