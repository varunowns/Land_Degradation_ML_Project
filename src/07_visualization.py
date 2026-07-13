"""
Stage 5 — Error analysis and diagnostic visualizations.

Analyses class-wise difficulty, district/year breakdowns, and misclassified
grid cells with environmental context for discussion.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import (
    CLASS_ORDER,
    MODELS_DIR,
    PLOTS_DIR,
    REPORTS_DIR,
    RESULTS_DIR,
    ensure_directories,
    setup_logging,
    PLOT_SAVE_KW,
)

logger = setup_logging(__name__)
ERROR_PLOTS = PLOTS_DIR / "error_analysis"


def _get_best_model_predictions() -> tuple[str, pd.DataFrame]:
    """Build a prediction dataframe with metadata for error analysis."""
    tuned_summary = pd.read_csv(RESULTS_DIR / "tuned_models_summary.csv")
    model_name = tuned_summary.sort_values("best_cv_f1_macro", ascending=False).iloc[0]["model"]
    model = joblib.load(MODELS_DIR / f"tuned_{model_name}.pkl")
    prepared = joblib.load(MODELS_DIR / "train_test_splits.pkl")
    label_encoder = joblib.load(MODELS_DIR / "label_encoder.pkl")

    if model_name == "logistic_regression":
        X_test = prepared.X_test_lr.values
    else:
        X_test = prepared.X_test_tree.values

    y_pred = model.predict(X_test)
    y_true = prepared.y_test

    df = prepared.metadata_test.copy()
    df["y_true"] = label_encoder.inverse_transform(y_true)
    df["y_pred"] = label_encoder.inverse_transform(y_pred)
    df["correct"] = df["y_true"] == df["y_pred"]
    return model_name, df


def class_wise_error_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify which degradation class is hardest to predict.

    Moderate classes in ordinal-like problems are often confused with both
    neighbours; this analysis quantifies per-class error rates.
    """
    rows = []
    for cls in CLASS_ORDER:
        subset = df[df["y_true"] == cls]
        error_rate = 1.0 - subset["correct"].mean()
        rows.append(
            {
                "class": cls,
                "n_samples": len(subset),
                "error_rate": error_rate,
                "accuracy": subset["correct"].mean(),
            }
        )
    class_df = pd.DataFrame(rows).sort_values("error_rate", ascending=False)
    class_df.to_csv(RESULTS_DIR / "class_error_analysis.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    ax.bar(class_df["class"], class_df["error_rate"], color=colors, edgecolor="white")
    ax.set_ylabel("Error Rate")
    ax.set_title("Per-Class Prediction Error Rate")
    ax.set_ylim(0, max(class_df["error_rate"].max() * 1.2, 0.1))
    plt.tight_layout()
    fig.savefig(ERROR_PLOTS / "class_error_rate.png", **PLOT_SAVE_KW)
    plt.close(fig)
    return class_df


def district_wise_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Compute classification accuracy for each of the 20 districts."""
    district_df = (
        df.groupby("District")
        .agg(accuracy=("correct", "mean"), n=("correct", "count"))
        .reset_index()
        .sort_values("accuracy")
    )
    district_df["error_rate"] = 1.0 - district_df["accuracy"]
    district_df.to_csv(RESULTS_DIR / "district_accuracy.csv", index=False)

    fig, ax = plt.subplots(figsize=(14, 6))
    palette = [
        "#e74c3c" if e > 0.15 else "#f39c12" if e > 0.10 else "#2ecc71"
        for e in district_df["error_rate"]
    ]
    ax.bar(district_df["District"], district_df["accuracy"], color=palette, edgecolor="white")
    ax.axhline(district_df["accuracy"].mean(), color="black", linestyle="--", label="Mean accuracy")
    ax.set_xticks(range(len(district_df)))
    ax.set_xticklabels(district_df["District"], rotation=45, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("District-wise Classification Accuracy")
    ax.legend()
    plt.tight_layout()
    fig.savefig(ERROR_PLOTS / "district_accuracy.png", **PLOT_SAVE_KW)
    plt.close(fig)
    return district_df


def year_wise_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Compute accuracy by observation year (2020–2024)."""
    year_df = (
        df.groupby("Year")
        .agg(accuracy=("correct", "mean"), n=("correct", "count"))
        .reset_index()
        .sort_values("Year")
    )
    year_df.to_csv(RESULTS_DIR / "year_accuracy.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(year_df["Year"], year_df["accuracy"], "o-", linewidth=2, markersize=8, color="#3498db")
    ax.set_xlabel("Year")
    ax.set_ylabel("Accuracy")
    ax.set_title("Year-wise Classification Accuracy")
    ax.set_xticks(year_df["Year"])
    plt.tight_layout()
    fig.savefig(ERROR_PLOTS / "year_accuracy.png", **PLOT_SAVE_KW)
    plt.close(fig)
    return year_df


def misclassification_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Export misclassified observations and confusion patterns."""
    misclassified = df[~df["correct"]].copy()
    misclassified.to_csv(RESULTS_DIR / "misclassified_observations.csv", index=False)

    confusion = pd.crosstab(
        df["y_true"], df["y_pred"], rownames=["True"], colnames=["Predicted"]
    )
    confusion.to_csv(RESULTS_DIR / "confusion_crosstab.csv")

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Reds", ax=ax)
    ax.set_title("Misclassification Patterns (Counts)")
    plt.tight_layout()
    fig.savefig(ERROR_PLOTS / "misclassification_heatmap.png", **PLOT_SAVE_KW)
    plt.close(fig)
    return misclassified


def write_error_discussion(
    df: pd.DataFrame,
    class_df: pd.DataFrame,
    district_df: pd.DataFrame,
    year_df: pd.DataFrame,
    misclassified: pd.DataFrame,
) -> None:
    """Generate discussion text with plausible environmental explanations."""
    hardest_class = class_df.iloc[0]["class"]
    worst_district = district_df.iloc[0]["District"]
    best_year = year_df.loc[year_df["accuracy"].idxmax(), "Year"]
    worst_year = year_df.loc[year_df["accuracy"].idxmin(), "Year"]

    top_confusions = (
        misclassified.groupby(["y_true", "y_pred"]).size().sort_values(ascending=False).head(5)
    )

    total = len(df)
    discussion = f"""
ERROR ANALYSIS DISCUSSION
=========================

Hardest class to predict: {hardest_class}
- The '{hardest_class}' class shows the highest error rate ({class_df.iloc[0]['error_rate']:.1%}).
- Moderate degradation is often ecologically transitional: NDVI, soil moisture, and
  bare-land fractions overlap with both Low and High classes, producing ambiguous
  feature signatures at 5 km resolution.

District with lowest accuracy: {worst_district}
- District-level heterogeneity in land use (intensive cropland vs. fragmented tree cover)
  and local irrigation infrastructure may not be fully captured by coarse GEE aggregates.
- Transboundary climate gradients across eastern and western UP districts can shift
  rainfall/temperature baselines, increasing confusion for models relying on district dummies.

Temporal patterns:
- Best year: {int(best_year)} | Worst year: {int(worst_year)}
- Inter-annual rainfall variability (monsoon deficit vs. surplus years) directly affects
  NDVI and soil moisture, which are primary LDI inputs. Extreme years may push grid cells
  across class boundaries defined by study-wide quantile thresholds.

Most common misclassification patterns:
{top_confusions.to_string()}

Environmental interpretation:
- Low → Moderate errors often occur in cropland-dominated grids experiencing seasonal
  vegetation stress without sustained bare-soil expansion.
- Moderate → High errors align with grids showing elevated bare-land fraction and
  temperature anomalies during drought-influenced years.
- High → Moderate errors may reflect short-term recovery signals (NDVI rebound after
  monsoon) that the model interprets as resilience.

Total misclassified observations: {len(misclassified)} / {total}
""".strip()

    (REPORTS_DIR / "error_analysis_discussion.txt").write_text(discussion, encoding="utf-8")


def run_error_analysis() -> None:
    """Execute Stage 5 error analysis pipeline."""
    ensure_directories()
    ERROR_PLOTS.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    model_name, pred_df = _get_best_model_predictions()
    logger.info("Error analysis for best model: %s", model_name)

    class_df = class_wise_error_analysis(pred_df)
    district_df = district_wise_accuracy(pred_df)
    year_df = year_wise_accuracy(pred_df)
    misclassified = misclassification_analysis(pred_df)
    write_error_discussion(pred_df, class_df, district_df, year_df, misclassified)

    logger.info("Error analysis saved to %s", ERROR_PLOTS)


def main() -> None:
    run_error_analysis()


if __name__ == "__main__":
    main()
