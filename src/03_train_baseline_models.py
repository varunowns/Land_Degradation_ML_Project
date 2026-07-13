"""
Stage 1 — Baseline model training and comparison.

Trains five classifiers with default hyperparameters (no tuning), evaluates
each on the held-out test set, ranks models, and persists all artifacts.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Allow running as script from src/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import (
    MODELS_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    compute_classification_metrics,
    ensure_directories,
    save_artifact,
    save_json,
    setup_logging,
    time_model_fit,
    time_model_predict,
)

logger = setup_logging(__name__)


def _get_baseline_models() -> dict[str, tuple[Any, str]]:
    """
    Return baseline estimators and their data key ('lr' or 'tree').

    Default hyperparameters only — tuning is deferred to Stage 2.
    """
    models: dict[str, tuple[Any, str]] = {
        "logistic_regression": (
            LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_STATE,
                class_weight="balanced",
            ),
            "lr",
        ),
        "decision_tree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
            "tree",
        ),
        "random_forest": (
            RandomForestClassifier(
                n_estimators=200,
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1,
            ),
            "tree",
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            "tree",
        ),
    }

    try:
        from xgboost import XGBClassifier

        models["xgboost"] = (
            XGBClassifier(
                n_estimators=200,
                random_state=RANDOM_STATE,
                eval_metric="mlogloss",
                verbosity=0,
            ),
            "tree",
        )
    except ImportError:
        logger.warning("XGBoost not installed — skipping.")

    return models


def train_baselines(prepared: Any) -> pd.DataFrame:
    """Train all baseline models and return a ranked comparison table."""
    ensure_directories()
    models = _get_baseline_models()
    rows: list[dict[str, Any]] = []
    all_predictions: dict[str, dict[str, Any]] = {}

    for model_name, (estimator, data_key) in models.items():
        logger.info("Training baseline: %s", model_name)

        if data_key == "lr":
            X_train = prepared.X_train_lr.values
            X_test = prepared.X_test_lr.values
        else:
            X_train = prepared.X_train_tree.values
            X_test = prepared.X_test_tree.values

        fitted, train_time = time_model_fit(estimator, X_train, prepared.y_train)
        y_pred, y_prob, predict_time = time_model_predict(fitted, X_test)

        metrics = compute_classification_metrics(prepared.y_test, y_pred, y_prob)
        row = {
            "model": model_name,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "roc_auc": metrics.get("roc_auc"),
            "train_time_sec": train_time,
            "predict_time_sec": predict_time,
        }
        rows.append(row)

        save_artifact(fitted, f"baseline_{model_name}.pkl")
        save_json(
            {
                "classification_report": metrics["classification_report"],
                "confusion_matrix": metrics["confusion_matrix"],
                "train_time_sec": train_time,
                "predict_time_sec": predict_time,
            },
            RESULTS_DIR / f"baseline_{model_name}_report.json",
        )

        all_predictions[model_name] = {
            "y_pred": y_pred,
            "y_prob": y_prob,
            "data_key": data_key,
        }
        logger.info(
            "%s — F1: %.4f | Acc: %.4f | Train: %.2fs",
            model_name,
            row["f1_score"],
            row["accuracy"],
            train_time,
        )

    comparison = pd.DataFrame(rows).sort_values("f1_score", ascending=False)
    comparison.insert(0, "rank", range(1, len(comparison) + 1))
    comparison.to_csv(RESULTS_DIR / "baseline_comparison.csv", index=False)
    save_artifact(all_predictions, "baseline_predictions.pkl")
    save_artifact(comparison, "baseline_comparison.pkl")

    logger.info("Baseline comparison saved to %s", RESULTS_DIR / "baseline_comparison.csv")
    return comparison


def main() -> pd.DataFrame:
    """Entry point for Stage 1."""
    loader = importlib.import_module("01_data_loader")
    preprocessing = importlib.import_module("02_preprocessing")

    loaded = loader.prepare_loaded_dataset()
    prepared = preprocessing.prepare_train_test_split(
        loaded.dataframe, loaded.feature_columns
    )

    comparison = train_baselines(prepared)
    print("\n=== BASELINE MODEL COMPARISON ===")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return comparison


if __name__ == "__main__":
    main()
