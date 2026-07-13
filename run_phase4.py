"""
Orchestrator for Phase 4 ML pipeline stages.

Usage:
    python run_phase4.py --stage 1
    python run_phase4.py --stage all
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))


def _load_module(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, SRC / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAGES = {
    "1": ("03_train_baseline_models.py", "stage1"),
    "2": ("04_hyperparameter_tuning.py", "stage2"),
    "3": ("05_model_evaluation.py", "stage3"),
    "4": ("06_interpretability.py", "stage4"),
    "5": ("07_visualization.py", "stage5"),
    "6": ("08_export_results.py", "stage6"),
}


def run_stage(stage: str) -> None:
    filename, mod_name = STAGES[stage]
    module = _load_module(filename, mod_name)
    module.main()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 4 ML pipeline stages")
    parser.add_argument(
        "--stage",
        choices=["1", "2", "3", "4", "5", "6", "all"],
        default="1",
        help="Pipeline stage to execute (default: 1)",
    )
    args = parser.parse_args()

    if args.stage == "all":
        for stage in STAGES:
            print(f"\n{'=' * 60}\nRunning Stage {stage}\n{'=' * 60}")
            run_stage(stage)
    else:
        run_stage(args.stage)


if __name__ == "__main__":
    main()
