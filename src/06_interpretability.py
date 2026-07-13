"""
Stage 4 — Model interpretability.

Generates SHAP values, native/permutation feature importance,
and partial dependence plots for the best tuned model.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import PartialDependenceDisplay, permutation_importance

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import (
    MODELS_DIR,
    PLOTS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    RESULTS_DIR,
    ensure_directories,
    setup_logging,
    PLOT_SAVE_KW,
)

logger = setup_logging(__name__)
INTERP_PLOTS = PLOTS_DIR / "interpretability"

# Core environmental drivers for PDP (excluding one-hot district dummies)
PDP_FEATURES = [
    "NDVI_mean",
    "Rainfall_mean",
    "Temperature_mean",
    "SoilMoisture_mean",
    "BareLand",
    "TreeCover",
    "Cropland",
]


def _select_best_model() -> tuple[str, Any, Any]:
    """Pick the highest-ranked tuned model from Stage 2."""
    tuned_summary = pd.read_csv(RESULTS_DIR / "tuned_models_summary.csv")
    model_name = tuned_summary.sort_values("best_cv_f1_macro", ascending=False).iloc[0]["model"]
    model = joblib.load(MODELS_DIR / f"tuned_{model_name}.pkl")
    prepared = joblib.load(MODELS_DIR / "train_test_splits.pkl")
    return model_name, model, prepared


def compute_permutation_importance(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    model_name: str,
) -> pd.DataFrame:
    """
    Permutation importance on the test set.

    Measures how much F1-macro drops when each feature is randomly shuffled,
    providing a model-agnostic ranking of predictive drivers.
    """
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=15,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scoring="f1_macro",
    )
    fi_df = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
    )
    fi_df.to_csv(RESULTS_DIR / "permutation_importance.csv", index=False)

    top = fi_df.head(20)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["feature"][::-1], top["importance_mean"][::-1], color="#2ecc71")
    ax.set_xlabel("Decrease in F1-macro when permuted")
    ax.set_title("Permutation Importance (Best Tuned Model)")
    plt.tight_layout()
    fig.savefig(INTERP_PLOTS / "permutation_importance.png", **PLOT_SAVE_KW)
    plt.close(fig)
    return fi_df


def compute_shap_values(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
    model_name: str,
) -> pd.DataFrame:
    """
    Compute SHAP values for the best model.

    Uses TreeExplainer for tree models and KernelExplainer (background sample)
    for linear models. SHAP quantifies each feature's marginal contribution to
    individual predictions relative to the model's baseline expectation.
    """
    sample_size = min(500, len(X_test))
    rng = np.random.default_rng(RANDOM_STATE)
    test_idx = rng.choice(len(X_test), size=sample_size, replace=False)
    X_sample = X_test[test_idx]

    if model_name == "logistic_regression":
        explainer = shap.LinearExplainer(model, X_train, feature_names=feature_names)
        shap_values = explainer.shap_values(X_sample)
    elif model_name in {"random_forest", "gradient_boosting", "decision_tree", "xgboost"}:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    else:
        background = shap.sample(X_train, min(100, len(X_train)), random_state=RANDOM_STATE)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_sample, nsamples=50)

    # Summary beeswarm plot (use class 2 = High degradation as reference view)
    plt.figure(figsize=(10, 7))
    if isinstance(shap_values, list):
        plot_values = shap_values[2] if len(shap_values) > 2 else shap_values[0]
        shap.summary_plot(plot_values, X_sample, feature_names=feature_names, show=False)
    else:
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.title("SHAP Summary — Feature Impact on Predictions (High class)")
    plt.tight_layout()
    plt.savefig(INTERP_PLOTS / "shap_summary.png", **PLOT_SAVE_KW)
    plt.close()

    # Aggregate mean |SHAP| across classes and samples for export
    if isinstance(shap_values, list):
        stacked = np.stack([np.abs(sv) for sv in shap_values], axis=0)
        mean_abs = stacked.mean(axis=(0, 1))
    else:
        mean_abs = np.abs(shap_values).mean(axis=0)

    if mean_abs.ndim > 1:
        mean_abs = mean_abs.mean(axis=tuple(range(1, mean_abs.ndim)))

    shap_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    shap_df = shap_df.sort_values("mean_abs_shap", ascending=False)
    shap_df.to_csv(RESULTS_DIR / "shap_importance.csv", index=False)
    return shap_df


def plot_partial_dependence(
    model: Any,
    X_train: pd.DataFrame,
    feature_names: list[str],
) -> None:
    """
    Partial Dependence Plots for key environmental variables.

    PDPs show the marginal effect of a feature on predicted class probability
    after averaging out all other features — useful for understanding
    directionality (e.g., declining NDVI → higher degradation probability).
    """
    available = [f for f in PDP_FEATURES if f in feature_names]
    if not available:
        logger.warning("No PDP features found in feature matrix.")
        return

    indices = [feature_names.index(f) for f in available]
    n_plot = min(6, len(indices))
    fig, ax = plt.subplots(figsize=(14, 10))
    PartialDependenceDisplay.from_estimator(
        model,
        X_train.values,
        features=indices[:n_plot],
        feature_names=feature_names,
        target=2,  # High degradation class
        ax=ax,
        n_cols=3,
    )
    fig.suptitle(
        "Partial Dependence Plots — Drivers of High Degradation Class",
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(INTERP_PLOTS / "partial_dependence.png", **PLOT_SAVE_KW)
    plt.close(fig)


def write_interpretability_notes() -> None:
    """Generate plain-language explanations for each interpretability figure."""
    notes = """
INTERPRETABILITY FIGURE GUIDE
=============================

1. permutation_importance.png
   Shows how much the model's F1-macro score drops when each feature is
   randomly permuted. Larger drops indicate stronger predictive influence.
   This method is model-agnostic and accounts for feature interactions.

2. shap_summary.png
   SHAP (SHapley Additive exPlanations) values decompose each prediction into
   per-feature contributions. Red points indicate high feature values pushing
   predictions toward a class; blue points indicate low values. Features are
   ranked by mean absolute SHAP impact across the sample.

3. partial_dependence.png
   Displays the marginal relationship between selected environmental variables
   (NDVI, rainfall, temperature, soil moisture, LULC fractions) and the
   model's predicted response. Upward trends for BareLand/Temperature suggest
   increased degradation likelihood; downward trends for NDVI/TreeCover
   indicate vegetation resilience suppressing degradation classification.

4. feature_importance_*.png (from Stage 3)
   Native Gini-based importance for tree models; reflects how often and how
   effectively each feature splits nodes during ensemble construction.
""".strip()

    path = REPORTS_DIR / "interpretability_notes.txt"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(notes, encoding="utf-8")


def run_interpretability() -> None:
    """Execute Stage 4 interpretability pipeline."""
    ensure_directories()
    INTERP_PLOTS.mkdir(parents=True, exist_ok=True)

    model_name, model, prepared = _select_best_model()
    logger.info("Interpretability analysis for: %s", model_name)

    if model_name == "logistic_regression":
        X_train = prepared.X_train_lr.values
        X_test = prepared.X_test_lr.values
        feature_names = prepared.feature_names_lr
        X_train_df = prepared.X_train_lr
    else:
        X_train = prepared.X_train_tree.values
        X_test = prepared.X_test_tree.values
        feature_names = prepared.feature_names_tree
        X_train_df = prepared.X_train_tree

    perm_df = compute_permutation_importance(
        model, X_test, prepared.y_test, feature_names, model_name
    )
    shap_df = compute_shap_values(model, X_train, X_test, feature_names, model_name)
    plot_partial_dependence(model, X_train_df, feature_names)

    combined = perm_df.merge(shap_df, on="feature", how="outer")
    combined.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)

    write_interpretability_notes()
    logger.info("Interpretability outputs saved to %s and %s", INTERP_PLOTS, RESULTS_DIR)


def main() -> None:
    run_interpretability()


if __name__ == "__main__":
    main()
