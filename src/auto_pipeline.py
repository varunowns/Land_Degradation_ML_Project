"""
Automatic preprocessing pipeline for the legacy Phase 1-3 scripts.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
GEE_DIR = PROJECT_ROOT / "Land_Degradation"

ROOT_MASTER_DATASET = PROJECT_ROOT / "master_dataset.csv"
ROOT_WORKING_DATASET = PROJECT_ROOT / "working_dataset.csv"
ROOT_LDI_DATASET = PROJECT_ROOT / "ldi_dataset.csv"

DATA_MASTER_DATASET = DATA_DIR / "master_dataset.csv"
DATA_WORKING_DATASET = DATA_DIR / "working_dataset.csv"
DATA_LDI_DATASET = DATA_DIR / "ldi_dataset.csv"

EXPECTED_GEE_PREFIXES = ("LULC", "NDVI", "Rainfall", "SoilMoisture", "Temperature")


def _build_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = _build_logger()


def _check_latest_gee_csv_files() -> list[Path]:
    csv_files = sorted(GEE_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No GEE CSV files found in {GEE_DIR}")

    missing_prefixes = [
        prefix for prefix in EXPECTED_GEE_PREFIXES if not any(path.name.startswith(prefix) for path in csv_files)
    ]
    if missing_prefixes:
        missing = ", ".join(missing_prefixes)
        raise FileNotFoundError(f"Missing expected GEE CSV groups: {missing}")

    latest_file = max(csv_files, key=lambda path: path.stat().st_mtime)
    logger.info("Started: checking latest GEE CSV files")
    logger.info(
        "Completed: found %d GEE CSV files; latest is %s",
        len(csv_files),
        latest_file.name,
    )
    return csv_files


def _run_script(script_name: str) -> None:
    logger.info("Started: %s", script_name)
    try:
        subprocess.run(
            [sys.executable, script_name],
            cwd=PROJECT_ROOT,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        logger.exception("Failed: %s", script_name)
        raise RuntimeError(f"{script_name} failed with exit code {exc.returncode}") from exc
    logger.info("Completed: %s", script_name)


def _copy_dataset(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Expected dataset was not created: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _sync_phase_outputs(include_ldi: bool = False) -> None:
    logger.info("Started: syncing generated datasets")
    _copy_dataset(ROOT_MASTER_DATASET, ROOT_WORKING_DATASET)
    _copy_dataset(ROOT_MASTER_DATASET, DATA_MASTER_DATASET)
    _copy_dataset(ROOT_WORKING_DATASET, DATA_WORKING_DATASET)
    if include_ldi:
        _copy_dataset(ROOT_LDI_DATASET, DATA_LDI_DATASET)
    logger.info("Completed: syncing generated datasets")


def run_data_pipeline() -> dict[str, str]:
    """
    Run the legacy Phase 1-3 preprocessing flow and refresh data/ datasets.
    """
    logger.info("Started: data preprocessing pipeline")
    try:
        _check_latest_gee_csv_files()
        _run_script("phase1_data_prep.py")
        _sync_phase_outputs()
        _run_script("phase2_eda.py")
        _run_script("phase3_ldi.py")
        _sync_phase_outputs(include_ldi=True)
    except Exception:
        logger.exception("Failed: data preprocessing pipeline")
        raise

    logger.info("Completed: data preprocessing pipeline")
    return {
        "master_dataset": str(DATA_MASTER_DATASET),
        "working_dataset": str(DATA_WORKING_DATASET),
        "ldi_dataset": str(DATA_LDI_DATASET),
    }


if __name__ == "__main__":
    run_data_pipeline()
