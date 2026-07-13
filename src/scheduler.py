"""
Background scheduler to automatically check and run the dataset refresh/inference pipeline.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler

from config_manager import load_config
from update_pipeline import run_full_update

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOGS_DIR / "scheduler.log"

# Setup Logger
logger = logging.getLogger("scheduler")
logger.setLevel(logging.INFO)

if not logger.handlers:
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

_scheduler: BackgroundScheduler | None = None


def check_update_required() -> bool:
    """Calculate if a new update is required based on last_update and frequency."""
    config = load_config()
    last_update = config.get("last_update")
    if not last_update:
        logger.info("Scheduler: No last update timestamp found. Update is required.")
        return True

    try:
        last_run = datetime.fromisoformat(last_update)
    except ValueError:
        logger.warning(f"Scheduler: Failed to parse last_update '{last_update}'. Update is required.")
        return True

    if last_run.tzinfo is None:
        last_run = last_run.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
    freq = config.get("data_update_frequency", "8_days")

    if freq == "daily":
        delta = timedelta(days=1)
    elif freq == "8_days":
        delta = timedelta(days=8)
    elif freq == "monthly":
        delta = timedelta(days=30)
    else:
        delta = timedelta(days=8)

    time_since_update = now - last_run
    required = time_since_update >= delta
    logger.info(
        f"Scheduler check: Frequency is '{freq}'. Time since last update: {time_since_update}. "
        f"Update required: {required}."
    )
    return required


def _scheduled_job() -> None:
    """Check update conditions and trigger run_full_update() if required."""
    try:
        config = load_config()
        status = config.get("update_pipeline_status")

        if check_update_required():
            if status == "RUNNING":
                logger.warning("Scheduler check: Update is required, but pipeline status is already RUNNING. Skipping.")
                return

            logger.info("Scheduler check: Update is required. Running full update pipeline...")
            
            def log_progress(msg: str) -> None:
                logger.info(f"Pipeline Progress: {msg}")

            result = run_full_update(progress_callback=log_progress)
            logger.info(f"Scheduler check: Pipeline finished with status: {result.get('status')}")
        else:
            logger.info("Scheduler check: Update is not required yet.")
    except Exception as exc:
        logger.exception(f"Scheduler check: Error during background update check: {exc}")


def start_scheduler() -> None:
    """Start the background scheduler task."""
    global _scheduler
    if _scheduler is not None and _scheduler.running:
        logger.info("Scheduler: Already running.")
        return

    logger.info("Scheduler: Starting background scheduler...")
    _scheduler = BackgroundScheduler(timezone="UTC")
    
    # Run the check every 5 minutes, triggering the first check immediately
    _scheduler.add_job(
        _scheduled_job,
        "interval",
        minutes=5,
        id="degradation_update_job",
        next_run_time=datetime.now(timezone.utc)
    )
    _scheduler.start()
    logger.info("Scheduler: Background scheduler started successfully.")


def stop_scheduler() -> None:
    """Stop the background scheduler task."""
    global _scheduler
    if _scheduler is not None and _scheduler.running:
        logger.info("Scheduler: Stopping background scheduler...")
        _scheduler.shutdown()
        _scheduler = None
        logger.info("Scheduler: Background scheduler stopped.")
    else:
        logger.info("Scheduler: Already stopped or not initialized.")
