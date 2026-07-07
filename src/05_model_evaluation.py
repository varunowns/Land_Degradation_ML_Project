"""
Stage 3 — Model evaluation: baseline vs tuned comparison.

Generates cross-validation scores, confusion matrices, ROC curves,
precision-recall curves, learning curves, and feature-importance plots.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve
from sklearn.preprocessing import label_binarize

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import (
    CLASS_ORDER,
    CV_FOLDS,
    MODELS_DIR,
    PLOTS_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    compute_classification_metrics,
    ensure_directories,
    setup_logging,
    time_model_predict,
    PLOT_SAVE_KW,
)

logger = setup_logging(__name__)
EVAL_PLOTS = PLOTS_DIR / "evaluation"


def _load_model_pair(model_name: str) -> tuple[Any, Any]:
    """Load baseline and tuned versions of a model."""
    baseline = joblib.load(MODELS_DIR / f"baseline_{model_name}.pkl")
    tuned_path = MODELS_DIR / f"tuned_{model_name}.pkl"
    tuned = joblib.load(tuned_path) if tuned_path.exists() else None
    return baseline, tuned


def _get_xy(prepared: Any, model_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return train/test feature matrices for the model family."""
    if model_name == "logistic_regression":
        return prepared.X_train_lr.values, prepared.X_test_lr.values
    return prepared.X_train_tree.values, prepared.X_test_tree.values


def cross_validate_models(prepared: Any, model_names: list[str]) -> pd.DataFrame:
    """Compare 5-fold CV F1-macro for baseline and tuned variants."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows: list[dict[str, Any]] = []

    for model_name in model_names:
        baseline, tuned = _load_model_pair(model_name)
        X_train, _ = _get_xy(prepared, model_name)

        base_scores = cross_val_score(
            clone(baseline), X_train, prepared.y_train, cv=cv, scoring="f1_macro", n_jobs=-1
        )
        rows.append(
            {
                "model": model_name,
                "variant": "baseline",
                "cv_f1_mean": base_scores.mean(),
                "cv_f1_std": base_scores.std(),
            }
        )

        if tuned is not None:
            tuned_scores = cross_val_score(
                clone(tuned), X_train, prepared.y_train, cv=cv, scoring="f1_macro", n_jobs=-1
            )
            rows.append(
                {
                    "model": model_name,
                    "variant": "tuned",
                    "cv_f1_mean": tuned_scores.mean(),
                    "cv_f1_std": tuned_scores.std(),
                }
            )

    cv_df = pd.DataFrame(rows)
    cv_df.to_csv(RESULTS_DIR / "cv_comparison.csv", index=False)
    return cv_df


def evaluate_on_test(
    prepared: Any, model_name: str, model: Any
) -> dict[str, Any]:
    """Evaluate a single model on the held-out test set."""
    _, X_test = _get_xy(prepared, model_name)
    y_pred, y_prob, predict_time = time_model_predict(model, X_test)
    metrics = compute_classification_metrics(prepared.y_test, y_pred, y_prob)
    metrics["predict_time_sec"] = predict_time
    return metrics


def plot_confusion_matrices(
    prepared: Any, model_name: str, baseline: Any, tuned: Any | None
) -> None:
    """Side-by-side confusion matrices for baseline and tuned models."""
    _, X_test = _get_xy(prepared, model_name)
    n_panels = 2 if tuned is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    for ax, model, title in zip(
        axes,
        [baseline, tuned] if tuned is not None else [baseline],
        ["Baseline", "Tuned"] if tuned is not None else ["Baseline"],
    ):
        y_pred = model.predict(X_test)
        ConfusionMatrixDisplay.from_predictions(
            prepared.y_test,
            y_pred,
            display_labels=CLASS_ORDER,
            cmap="Blues",
            ax=ax,
            colorbar=False,
        )
        ax.set_title(f"{model_name} — {title}")

    plt.suptitle(f"Confusion Matrices: {model_name}", fontweight="bold")
    plt.tight_layout()
    fig.savefig(EVAL_PLOTS / f"confusion_{model_name}.png", **PLOT_SAVE_KW)
    plt.close(fig)


def plot_roc_curves(prepared: Any, model_name: str, models: dict[str, Any]) -> None:
    """Multiclass one-vs-rest ROC curves."""
    _, X_test = _get_xy(prepared, model_name)
    y_bin = label_binarize(prepared.y_test, classes=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(8, 6))

    for label, model in models.items():
        y_prob = model.predict_proba(X_test)
        for class_idx, class_name in enumerate(CLASS_ORDER):
            fpr, tpr, _ = roc_curve(y_bin[:, class_idx], y_prob[:, class_idx])
            roc_auc = auc(fpr, tpr)
            ax.plot(
                fpr,
                tpr,
                label=f"{label} — {class_name} (AUC={roc_auc:.3f})",
                alpha=0.8,
            )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves (OvR): {model_name}")
    ax.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    fig.savefig(EVAL_PLOTS / f"roc_{model_name}.png", **PLOT_SAVE_KW)
    plt.close(fig)


def plot_precision_recall(prepared: Any, model_name: str, models: dict[str, Any]) -> None:
    """Multiclass precision-recall curves."""
    _, X_test = _get_xy(prepared, model_name)
    y_bin = label_binarize(prepared.y_test, classes=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(8, 6))

    for label, model in models.items():
        y_prob = model.predict_proba(X_test)
        for class_idx, class_name in enumerate(CLASS_ORDER):
            precision, recall, _ = precision_recall_curve(
                y_bin[:, class_idx], y_prob[:, class_idx]
            )
            ap = average_precision_score(y_bin[:, class_idx], y_prob[:, class_idx])
            ax.plot(
                recall,
                precision,
                label=f"{label} — {class_name} (AP={ap:.3f})",
                alpha=0.8,
            )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curves: {model_name}")
    ax.legend(fontsize=7, loc="lower left")
    plt.tight_layout()
    fig.savefig(EVAL_PLOTS / f"pr_{model_name}.png", **PLOT_SAVE_KW)
    plt.close(fig)


def plot_learning_curve(prepared: Any, model_name: str, model: Any) -> None:
    """Learning curve for the tuned (or baseline) model."""
    X_train, _ = _get_xy(prepared, model_name)
    train_sizes, train_scores, val_scores = learning_curve(
        clone(model),
        X_train,
        prepared.y_train,
        cv=CV_FOLDS,
        scoring="f1_macro",
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 8),
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, train_scores.mean(axis=1), "o-", label="Training F1-macro")
    ax.fill_between(
        train_sizes,
        train_scores.mean(axis=1) - train_scores.std(axis=1),
        train_scores.mean(axis=1) + train_scores.std(axis=1),
        alpha=0.15,
    )
    ax.plot(train_sizes, val_scores.mean(axis=1), "o-", label="Validation F1-macro")
    ax.fill_between(
        train_sizes,
        val_scores.mean(axis=1) - val_scores.std(axis=1),
        val_scores.mean(axis=1) + val_scores.std(axis=1),
        alpha=0.15,
    )
    ax.set_xlabel("Training samples")
    ax.set_ylabel("F1-macro")
    ax.set_title(f"Learning Curve: {model_name}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(EVAL_PLOTS / f"learning_curve_{model_name}.png", **PLOT_SAVE_KW)
    plt.close(fig)


def plot_feature_importance(
    prepared: Any, model_name: str, model: Any, feature_names: list[str]
) -> pd.DataFrame:
    """Tree-based feature importance or permutation importance for linear models."""
    _, X_test = _get_xy(prepared, model_name)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        method = "native"
    else:
        result = permutation_importance(
            model,
            X_test,
            prepared.y_test,
            n_repeats=10,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            scoring="f1_macro",
        )
        importances = result.importances_mean
        method = "permutation"

    fi_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(20)
    )
    fi_df.to_csv(RESULTS_DIR / f"feature_importance_{model_name}.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=fi_df, y="feature", x="importance", ax=ax, palette="viridis")
    ax.set_title(f"Top Feature Importance ({method}): {model_name}")
    plt.tight_layout()
    fig.savefig(EVAL_PLOTS / f"feature_importance_{model_name}.png", **PLOT_SAVE_KW)
    plt.close(fig)
    return fi_df


def run_evaluation(model_names: list[str] | None = None) -> pd.DataFrame:
    """Execute full Stage 3 evaluation pipeline."""
    ensure_directories()
    EVAL_PLOTS.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    prepared = joblib.load(MODELS_DIR / "train_test_splits.pkl")
    if model_names is None:
        tuned_summary = pd.read_csv(RESULTS_DIR / "tuned_models_summary.csv")
        model_names = tuned_summary["model"].tolist()

    cv_df = cross_validate_models(prepared, model_names)
    test_rows: list[dict[str, Any]] = []

    for model_name in model_names:
        baseline, tuned = _load_model_pair(model_name)
        best_model = tuned if tuned is not None else baseline
        feature_names = (
            prepared.feature_names_lr
            if model_name == "logistic_regression"
            else prepared.feature_names_tree
        )

        for variant, model in [("baseline", baseline), ("tuned", tuned)]:
            if model is None:
                continue
            metrics = evaluate_on_test(prepared, model_name, model)
            test_rows.append(
                {
                    "model": model_name,
                    "variant": variant,
                    "accuracy": metrics["accuracy"],
                    "f1_score": metrics["f1_score"],
                    "roc_auc": metrics.get("roc_auc"),
                }
            )

        plot_confusion_matrices(prepared, model_name, baseline, tuned)
        model_dict = {"baseline": baseline}
        if tuned is not None:
            model_dict["tuned"] = tuned
        plot_roc_curves(prepared, model_name, model_dict)
        plot_precision_recall(prepared, model_name, model_dict)
        plot_learning_curve(prepared, model_name, best_model)
        plot_feature_importance(prepared, model_name, best_model, feature_names)

    test_df = pd.DataFrame(test_rows)
    test_df.to_csv(RESULTS_DIR / "baseline_vs_tuned_test.csv", index=False)
    logger.info("Evaluation complete. Plots saved to %s", EVAL_PLOTS)
    return cv_df


def main() -> None:
    run_evaluation()


if __name__ == "__main__":
    main()
