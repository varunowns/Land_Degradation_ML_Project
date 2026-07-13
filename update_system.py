"""
End-to-end update orchestration for Streamlit settings actions.
"""

from __future__ import annotations

import ast
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from auto_pipeline import run_data_pipeline
from auto_predict import run_prediction_pipeline
from config_manager import (
    get_update_frequency,
    mark_update_requested,
    set_update_pipeline_status,
    update_last_run,
)
from gee_fetcher import fetch_all_data


def _load_supported_years() -> list[int]:
    """Read the hardcoded Phase 1 year list to stay compatible with legacy preprocessing."""
    phase1_path = PROJECT_ROOT / "phase1_data_prep.py"
    source = phase1_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(phase1_path))

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "years":
                    if isinstance(node.value, ast.List):
                        return [
                            int(elt.value)
                            for elt in node.value.elts
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, int)
                        ]
    raise ValueError(f"Could not determine supported years from {phase1_path}")


def _resolve_fetch_year() -> int:
    """Fetch the latest year supported by the existing preprocessing pipeline."""
    years = _load_supported_years()
    if not years:
        raise ValueError("No supported years were found in phase1_data_prep.py")
    return max(years)


def run_full_update() -> dict[str, Any]:
    """
    Run the full automated update flow for data refresh and predictions.
    """
    config = mark_update_requested()
    frequency = get_update_frequency()
    fetch_year = _resolve_fetch_year()

    print(f"Update frequency: {frequency}")
    print(f"Fetching latest Google Earth Engine data for {fetch_year}...")
    set_update_pipeline_status("running")

    try:
        gee_outputs = fetch_all_data(fetch_year)

        print("Processing dataset...")
        pipeline_outputs = run_data_pipeline()

        print("Running prediction...")
        predictions_path = run_prediction_pipeline()

        updated = update_last_run()
        print("Update completed.")
        return {
            "status": "completed",
            "update_frequency": frequency,
            "requested_at": config.get("update_requested_at"),
            "fetched_year": fetch_year,
            "gee_outputs": {name: str(path) for name, path in gee_outputs.items()},
            "pipeline_outputs": pipeline_outputs,
            "predictions_path": predictions_path,
            "last_update": updated.get("last_update"),
        }
    except Exception as exc:
        set_update_pipeline_status("failed")
        print(f"Update failed: {exc}")
        raise


if __name__ == "__main__":
    result = run_full_update()
    print(result)
