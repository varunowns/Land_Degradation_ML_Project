"""
Stage 2 — Hyperparameter tuning for the two best baseline models.

Uses RandomizedSearchCV with 5-fold stratified cross-validation.
Only the top-two performers from Stage 1 are tuned to avoid wasted compute.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import (
    CV_FOLDS,
    MODELS_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    ensure_directories,
    load_json,
    save_artifact,
    save_json,
    setup_logging,
)

logger = setup_logging(__name__)

N_ITER_SEARCH = 20
SCORING = "f1_macro"
CV_JOBS = 1  # Avoid parallel fold exhaustion on memory-constrained systems


def _search_spaces() -> dict[str, dict[str, list[Any]]]:
    """Reasonable hyperparameter search spaces per model family."""
    return {
        "logistic_regression": {
            "C": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            "solver": ["lbfgs", "saga"],
            "max_iter": [1000, 2000, 3000],
        },
        "decision_tree": {
            "max_depth": [3, 5, 7, 10, 15, None],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 8],
            "criterion": ["gini", "entropy"],
        },
        "random_forest": {
            "n_estimators": [100, 200, 300, 400],
            "max_depth": [5, 10, 15, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", 0.5],
        },
        "gradient_boosting": {
            "n_estimators": [100, 200, 300],
            "max_depth": [2, 3, 4, 5],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "min_samples_leaf": [1, 2, 4],
        },
        "xgboost": {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 4, 5, 6],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "reg_alpha": [0.0, 0.1, 0.5],
            "reg_lambda": [1.0, 1.5, 2.0],
        },
    }


def _get_estimator(model_name: str) -> Any:
    """Instantiate a fresh estimator for the given model key."""
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    factories = {
        "logistic_regression": lambda: LogisticRegression(
            random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "decision_tree": lambda: DecisionTreeClassifier(
            random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "random_forest": lambda: RandomForestClassifier(
            random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1
        ),
        "gradient_boosting": lambda: GradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    if model_name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(random_state=RANDOM_STATE, eval_metric="mlogloss", verbosity=0)
    return factories[model_name]()


def tune_top_models(prepared: Any, top_n: int = 2) -> pd.DataFrame:
    """
    Tune the ``top_n`` best baseline models identified in Stage 1.

    Returns a DataFrame summarising tuned CV scores and best parameters.
    """
    ensure_directories()
    comparison_path = RESULTS_DIR / "baseline_comparison.csv"
    if not comparison_path.exists():
        raise FileNotFoundError("Run Stage 1 (03_train_baseline_models.py) first.")

    baseline_ranking = pd.read_csv(comparison_path)
    top_models = baseline_ranking.head(top_n)["model"].tolist()
    logger.info("Tuning top %d models: %s", top_n, top_models)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    spaces = _search_spaces()
    tuned_rows: list[dict[str, Any]] = []

    for model_name in top_models:
        tuned_path = MODELS_DIR / f"tuned_{model_name}.pkl"
        summary_path = RESULTS_DIR / f"tuned_{model_name}_summary.json"

        if tuned_path.exists() and summary_path.exists():
            logger.info("Skipping %s — tuned model already exists.", model_name)
            summary = load_json(summary_path)
            tuned_rows.append(
                {
                    "model": model_name,
                    "best_cv_f1_macro": float(summary["best_cv_score"]),
                    "best_params": str(summary["best_params"]),
                }
            )
            continue

        data_key = "lr" if model_name == "logistic_regression" else "tree"
        X_train = (
            prepared.X_train_lr.values.astype(np.float32)
            if data_key == "lr"
            else prepared.X_train_tree.values.astype(np.float32)
        )

        search = RandomizedSearchCV(
            estimator=_get_estimator(model_name),
            param_distributions=spaces[model_name],
            n_iter=N_ITER_SEARCH,
            scoring=SCORING,
            cv=cv,
            n_jobs=CV_JOBS,
            random_state=RANDOM_STATE,
            verbose=1,
        )
        logger.info("RandomizedSearchCV for %s ...", model_name)
        search.fit(X_train, prepared.y_train)

        tuned = search.best_estimator_
        save_artifact(tuned, f"tuned_{model_name}.pkl")
        save_artifact(search, f"tuned_{model_name}_search.pkl")
        save_json(
            {
                "best_params": search.best_params_,
                "best_cv_score": float(search.best_score_),
                "scoring": SCORING,
            },
            RESULTS_DIR / f"tuned_{model_name}_summary.json",
        )

        tuned_rows.append(
            {
                "model": model_name,
                "best_cv_f1_macro": float(search.best_score_),
                "best_params": str(search.best_params_),
            }
        )
        logger.info(
            "%s tuned — CV F1-macro: %.4f | params: %s",
            model_name,
            search.best_score_,
            search.best_params_,
        )

    tuned_df = pd.DataFrame(tuned_rows)
    tuned_df.to_csv(RESULTS_DIR / "tuned_models_summary.csv", index=False)
    save_artifact(tuned_df, "tuned_models_summary.pkl")
    return tuned_df


def main() -> pd.DataFrame:
    """Entry point for Stage 2."""
    prepared = joblib.load(MODELS_DIR / "train_test_splits.pkl")
    tuned = tune_top_models(prepared, top_n=2)
    print("\n=== TUNED MODEL SUMMARY ===")
    print(tuned.to_string(index=False))
    return tuned


if __name__ == "__main__":
    main()
