"""Model Performance — baseline comparison and evaluation metrics (shell only)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from utils.config import BASELINE_COMPARISON_PATH, CV_COMPARISON_PATH, PLOTS_DIR
from utils.plotting import apply_plotly_theme, create_metric_bar_chart, load_static_plot
from utils.ui import render_placeholder, setup_page

setup_page(
    "Model Performance",
    subtitle="Baseline comparison, cross-validation scores, and evaluation plots",
)

# ---------------------------------------------------------------------------
# Baseline comparison table (read-only from saved results)
# ---------------------------------------------------------------------------
st.markdown("### Baseline Model Comparison")

if BASELINE_COMPARISON_PATH.exists():
    comparison_df = pd.read_csv(BASELINE_COMPARISON_PATH)
    st.dataframe(
        comparison_df.style.format(
            {
                "accuracy": "{:.4f}",
                "precision": "{:.4f}",
                "recall": "{:.4f}",
                "f1_score": "{:.4f}",
                "roc_auc": "{:.4f}",
                "train_time_sec": "{:.2f}",
                "predict_time_sec": "{:.4f}",
            },
            subset=[
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "roc_auc",
                "train_time_sec",
                "predict_time_sec",
            ],
        ),
        use_container_width=True,
        hide_index=True,
    )

    best = comparison_df.iloc[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Best Model", str(best["model"]).replace("_", " ").title())
    m2.metric("F1-Score", f"{best['f1_score']:.4f}")
    m3.metric("Accuracy", f"{best['accuracy']:.4f}")
    m4.metric("ROC-AUC", f"{best['roc_auc']:.4f}")
else:
    st.warning(f"Comparison table not found at `{BASELINE_COMPARISON_PATH}`.")

# ---------------------------------------------------------------------------
# CV scores
# ---------------------------------------------------------------------------
st.markdown("### Cross-Validation Scores")
if CV_COMPARISON_PATH.exists():
    cv_df = pd.read_csv(CV_COMPARISON_PATH)
    st.dataframe(cv_df, use_container_width=True, hide_index=True)
else:
    st.info("CV comparison file not yet copied to `data/cv_comparison.csv`.")

# ---------------------------------------------------------------------------
# Static evaluation plots
# ---------------------------------------------------------------------------
st.markdown("### Evaluation Plots")
eval_dir = PLOTS_DIR / "evaluation"
if eval_dir.exists():
    plot_files = sorted(eval_dir.glob("*.png"))
    if plot_files:
        cols = st.columns(2)
        for idx, plot_path in enumerate(plot_files[:4]):
            img_bytes = load_static_plot(plot_path)
            if img_bytes:
                cols[idx % 2].image(img_bytes, caption=plot_path.stem.replace("_", " ").title())
    else:
        render_placeholder("Evaluation plots will appear here once copied to plots/evaluation/.")
else:
    render_placeholder("Evaluation plot directory not found.")

apply_plotly_theme()
if BASELINE_COMPARISON_PATH.exists():
    best_row = comparison_df.iloc[0]
    st.plotly_chart(
        create_metric_bar_chart(
            labels=["Accuracy", "F1-Score", "ROC-AUC"],
            values=[
                float(best_row["accuracy"]),
                float(best_row["f1_score"]),
                float(best_row["roc_auc"]),
            ],
            title="Best Model — Key Metrics (Preview Chart)",
        ),
        use_container_width=True,
    )
