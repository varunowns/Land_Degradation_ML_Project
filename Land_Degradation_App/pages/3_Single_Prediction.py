"""Single Prediction — interactive form with live inference."""

from __future__ import annotations

import streamlit as st

from utils.config import CLASS_ORDER, COLORS, NUMERIC_FEATURE_COLUMNS
from utils.data_loader import get_district_medians
from utils.model_loader import verify_artifacts_exist
from utils.prediction import predict_single
from utils.preprocessing import get_district_list, get_feature_schema
from utils.ui import setup_page

setup_page(
    "Single Prediction",
    subtitle="Predict degradation class for an individual grid observation",
)

if not all(verify_artifacts_exist().values()):
    st.error("Model artefacts missing. Ensure `models/` contains the trained joblib files.")
    st.stop()

districts = get_district_list()
medians = get_district_medians()
defaults = medians.set_index("District") if not medians.empty else None

with st.form("single_prediction_form"):
    st.markdown("### Environmental Inputs")
    selected_district = st.selectbox("District", options=districts, index=0)

    base = defaults.loc[selected_district] if defaults is not None else {}
    form_col1, form_col2 = st.columns(2)
    feature_values: dict[str, float] = {"District": selected_district}

    half = len(NUMERIC_FEATURE_COLUMNS) // 2 + len(NUMERIC_FEATURE_COLUMNS) % 2
    schema = get_feature_schema()

    with form_col1:
        for feature in NUMERIC_FEATURE_COLUMNS[:half]:
            default_val = float(base.get(feature, 0.0)) if len(base) else 0.0
            feature_values[feature] = st.number_input(
                feature,
                value=default_val,
                format="%.4f",
                help=schema.get(feature, ""),
            )

    with form_col2:
        for feature in NUMERIC_FEATURE_COLUMNS[half:]:
            default_val = float(base.get(feature, 0.0)) if len(base) else 0.0
            feature_values[feature] = st.number_input(
                feature,
                value=default_val,
                format="%.4f",
                help=schema.get(feature, ""),
            )

    submitted = st.form_submit_button("Predict Degradation Class", type="primary")

if submitted:
    try:
        result = predict_single(feature_values)
        label = result["degradation_class"]
        color = COLORS.get(label.lower(), COLORS["primary"])

        st.markdown("---")
        st.markdown("### Prediction Result")
        r1, r2, r3 = st.columns(3)
        r1.metric("Predicted Class", label)
        r2.metric("Confidence", f"{result['confidence']:.1%}")
        r3.metric("Model", "Tuned Logistic Regression")

        st.markdown(
            f'<div style="background:{color}22;border-left:4px solid {color};'
            f'padding:1rem;border-radius:8px;margin:1rem 0;">'
            f'<strong>Degradation Class: {label}</strong></div>',
            unsafe_allow_html=True,
        )

        st.markdown("#### Class Probabilities")
        prob_cols = st.columns(len(CLASS_ORDER))
        for col, cls in zip(prob_cols, CLASS_ORDER):
            prob = result["probabilities"].get(cls, 0.0)
            col.progress(prob, text=f"{cls}: {prob:.1%}")

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

st.caption(f"Output classes: **{' · '.join(CLASS_ORDER)}** · District defaults use training-set medians.")
