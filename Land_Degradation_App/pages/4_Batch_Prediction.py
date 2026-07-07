"""Batch Prediction — CSV upload, bulk inference, and download."""

from __future__ import annotations

import streamlit as st

from utils.model_loader import verify_artifacts_exist
from utils.prediction import predict_batch, predictions_to_csv
from utils.preprocessing import (
    REQUIRED_INPUT_COLUMNS,
    get_feature_schema,
    get_sample_template_row,
    validate_input_columns,
)
from utils.ui import setup_page

setup_page(
    "Batch Prediction",
    subtitle="Upload a CSV file to predict degradation classes in bulk",
)

if not all(verify_artifacts_exist().values()):
    st.error("Model artefacts missing. Ensure `models/` contains the trained joblib files.")
    st.stop()

st.markdown("### Expected CSV Format")
schema = get_feature_schema()
st.dataframe(
    {"Column": list(schema.keys()), "Type / Description": list(schema.values())},
    use_container_width=True,
    hide_index=True,
)

template_df = get_sample_template_row()
st.download_button(
    label="Download Sample Template (CSV)",
    data=template_df.to_csv(index=False).encode("utf-8"),
    file_name="prediction_template.csv",
    mime="text/csv",
)

uploaded = st.file_uploader(
    "Upload CSV for batch prediction",
    type=["csv"],
    help="Required columns: " + ", ".join(REQUIRED_INPUT_COLUMNS),
)

if uploaded is not None:
    try:
        input_df = __import__("pandas").read_csv(uploaded)
        st.markdown(f"**{len(input_df)}** rows loaded.")

        is_valid, errors = validate_input_columns(input_df)
        if not is_valid:
            for err in errors:
                st.error(err)
        else:
            if st.button("Run Batch Prediction", type="primary"):
                with st.spinner("Running inference…"):
                    results_df = predict_batch(input_df)
                st.session_state["batch_results"] = results_df
                st.success(f"Predicted {len(results_df)} rows successfully.")

        if "batch_results" in st.session_state:
            results_df = st.session_state["batch_results"]
            st.markdown("### Prediction Results")
            st.dataframe(results_df.head(100), use_container_width=True)
            if len(results_df) > 100:
                st.caption(f"Showing first 100 of {len(results_df)} rows.")

            class_counts = results_df["predicted_class"].value_counts()
            st.bar_chart(class_counts)

            st.download_button(
                label="Download Predictions (CSV)",
                data=predictions_to_csv(results_df),
                file_name="batch_predictions.csv",
                mime="text/csv",
            )
    except Exception as exc:
        st.error(f"Error processing file: {exc}")
