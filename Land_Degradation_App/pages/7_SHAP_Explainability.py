"""SHAP Explainability — feature importance and SHAP visualisations."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from utils.config import FEATURE_IMPORTANCE_PATH, PLOTS_DIR, SHAP_IMPORTANCE_PATH
from utils.plotting import create_shap_bar, load_static_plot
from utils.ui import setup_page

setup_page(
    "SHAP Explainability",
    subtitle="Understand which environmental drivers influence degradation predictions",
)

tab_shap, tab_perm, tab_static = st.tabs(
    ["SHAP Importance", "Permutation Importance", "Static Plots"]
)

with tab_shap:
    if SHAP_IMPORTANCE_PATH.exists():
        shap_df = pd.read_csv(SHAP_IMPORTANCE_PATH)
        shap_df = shap_df.loc[:, ~shap_df.columns.duplicated()]
        
        if "mean_abs_shap" not in shap_df.columns and "importance_mean" in shap_df.columns:
            shap_df = shap_df.rename(columns={"importance_mean": "mean_abs_shap"})
            
        if "feature" not in shap_df.columns or "mean_abs_shap" not in shap_df.columns:
            st.warning("SHAP feature importance data unavailable.")
        else:
            st.plotly_chart(
                create_shap_bar(shap_df, "Mean |SHAP| — Feature Impact"),
                use_container_width=True,
            )
            st.dataframe(shap_df, use_container_width=True, hide_index=True)
    else:
        st.warning("`shap_importance.csv` not found.")

    st.info(
        "**Interpretation:** SHAP values quantify each feature's contribution to "
        "predictions. Larger mean |SHAP| indicates stronger influence. "
        "Temperature, rainfall, and soil moisture typically dominate."
    )

with tab_perm:
    if FEATURE_IMPORTANCE_PATH.exists():
        fi_df = pd.read_csv(FEATURE_IMPORTANCE_PATH)
        fi_df = fi_df.loc[:, ~fi_df.columns.duplicated()]
        
        if "mean_abs_shap" not in fi_df.columns and "importance_mean" in fi_df.columns:
            fi_df = fi_df.rename(columns={"importance_mean": "mean_abs_shap"})
            
        if "feature" not in fi_df.columns or "mean_abs_shap" not in fi_df.columns:
            st.warning("SHAP feature importance data unavailable.")
        else:
            env_only = fi_df[~fi_df["feature"].str.startswith("District_")].head(15)
            st.plotly_chart(
                create_shap_bar(
                    env_only,
                    "Permutation Importance (Environmental Features)",
                ),
                use_container_width=True,
            )
            st.dataframe(fi_df.head(20), use_container_width=True, hide_index=True)
    else:
        st.warning("`feature_importance.csv` not found.")

with tab_static:
    interp_dir = PLOTS_DIR / "interpretability"
    if not interp_dir.exists():
        interp_dir = Path(__file__).resolve().parent.parent.parent / "plots" / "interpretability"

    if interp_dir.exists():
        for plot_path in sorted(interp_dir.glob("*.png")):
            img = load_static_plot(plot_path)
            if img:
                st.image(img, caption=plot_path.stem.replace("_", " ").title(), use_container_width=True)
    else:
        st.warning("Pre-generated interpretability plots not found.")

st.markdown("---")
st.markdown(
    """
    ### Figure Guide
    - **SHAP Summary** — Red = high feature value pushing toward higher degradation class
    - **Permutation Importance** — Drop in F1 when feature is shuffled
    - **Partial Dependence** — Marginal effect of each variable on predicted class
    """
)
