"""
Automatic ML retraining and prediction pipeline orchestration.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from utils import RESULTS_DIR, setup_logging

logger = setup_logging(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_PATH = RESULTS_DIR / "predictions.csv"


def run_prediction_pipeline() -> str:
    """
    Run the full Phase 4 ML pipeline and return the predictions CSV path.
    """
    logger.info("Started: ML prediction pipeline")
    try:
        subprocess.run(
            [sys.executable, "run_phase4.py", "--stage", "all"],
            cwd=PROJECT_ROOT,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        logger.exception("Failed: ML prediction pipeline")
        raise RuntimeError(
            f"ML prediction pipeline failed with exit code {exc.returncode}"
        ) from exc

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"Expected predictions output was not created: {PREDICTIONS_PATH}")

    logger.info("Completed: ML prediction pipeline")
    return str(PREDICTIONS_PATH)


if __name__ == "__main__":
    print(run_prediction_pipeline())
