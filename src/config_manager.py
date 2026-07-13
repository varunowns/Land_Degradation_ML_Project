"""
Configuration manager for project-wide update frequency settings.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
CONFIG_PATH = CONFIG_DIR / "settings.json"

ALLOWED_FREQUENCIES = {"daily", "8_days", "monthly"}
ALLOWED_PIPELINE_STATUSES = {"idle", "queued", "running", "completed", "failed", "SUCCESS", "FAILED", "RUNNING"}
DEFAULT_CONFIG: dict[str, Any] = {
    "data_update_frequency": "8_days",
    "last_update": None,
    "auto_fetch": True,
    "update_requested_at": None,
    "update_pipeline_status": "idle",
    "gee_project_id": "",
    "gee_status": "NOT_CONNECTED",
    "last_gee_check": None,
}

_FREQUENCY_TO_DELTA = {
    "daily": timedelta(days=1),
    "8_days": timedelta(days=8),
    "monthly": timedelta(days=30),
}


def _normalise_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Merge user config with defaults and coerce invalid values."""
    merged = DEFAULT_CONFIG.copy()
    if isinstance(config, dict):
        merged.update(config)

    if merged["data_update_frequency"] not in ALLOWED_FREQUENCIES:
        merged["data_update_frequency"] = DEFAULT_CONFIG["data_update_frequency"]

    if merged.get("update_pipeline_status") not in ALLOWED_PIPELINE_STATUSES:
        merged["update_pipeline_status"] = DEFAULT_CONFIG["update_pipeline_status"]

    merged["auto_fetch"] = bool(merged.get("auto_fetch", DEFAULT_CONFIG["auto_fetch"]))
    merged["last_update"] = merged.get("last_update")
    merged["update_requested_at"] = merged.get("update_requested_at")
    merged["gee_project_id"] = merged.get("gee_project_id", "")
    merged["gee_status"] = merged.get("gee_status", "NOT_CONNECTED")
    merged["last_gee_check"] = merged.get("last_gee_check")
    return merged


def load_config() -> dict[str, Any]:
    """
    Load the JSON config, creating it with defaults if it does not exist.
    """
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as file:
            config = json.load(file)
    except FileNotFoundError:
        config = DEFAULT_CONFIG.copy()
        save_config(config)
        return config
    except (json.JSONDecodeError, OSError):
        config = DEFAULT_CONFIG.copy()
        save_config(config)
        return config

    normalised = _normalise_config(config)
    if normalised != config:
        save_config(normalised)
    return normalised


def save_config(config: dict[str, Any]) -> dict[str, Any]:
    """Persist config to disk after validating supported fields."""
    normalised = _normalise_config(config)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open("w", encoding="utf-8") as file:
        json.dump(normalised, file, indent=2)
    return normalised


def update_frequency(value: str) -> dict[str, Any]:
    """Update the configured data refresh frequency."""
    if value not in ALLOWED_FREQUENCIES:
        allowed = ", ".join(sorted(ALLOWED_FREQUENCIES))
        raise ValueError(f"Unsupported frequency '{value}'. Allowed values: {allowed}")

    config = load_config()
    config["data_update_frequency"] = value
    return save_config(config)


def get_update_frequency() -> str:
    """Return the currently configured update frequency."""
    return load_config()["data_update_frequency"]


def update_last_run() -> dict[str, Any]:
    """Stamp the config with the current UTC time."""
    config = load_config()
    config["last_update"] = datetime.now(timezone.utc).isoformat()
    config["update_pipeline_status"] = "SUCCESS"
    return save_config(config)


def mark_update_requested() -> dict[str, Any]:
    """Mark that a manual or scheduled dataset refresh has been requested."""
    config = load_config()
    config["update_requested_at"] = datetime.now(timezone.utc).isoformat()
    config["update_pipeline_status"] = "RUNNING"
    return save_config(config)


def set_update_pipeline_status(status: str) -> dict[str, Any]:
    """Persist the current data update pipeline status."""
    if status not in ALLOWED_PIPELINE_STATUSES:
        allowed = ", ".join(sorted(ALLOWED_PIPELINE_STATUSES))
        raise ValueError(f"Unsupported pipeline status '{status}'. Allowed values: {allowed}")

    config = load_config()
    config["update_pipeline_status"] = status
    return save_config(config)


def get_cache_ttl_seconds() -> int:
    """Map the configured frequency to a cache TTL in seconds."""
    frequency = get_update_frequency()
    return int(_FREQUENCY_TO_DELTA[frequency].total_seconds())


def is_auto_fetch_enabled() -> bool:
    """Return whether automatic refresh/fetch behavior is enabled."""
    return bool(load_config()["auto_fetch"])


def is_update_due(reference_time: datetime | None = None) -> bool:
    """Return True when data should be refreshed based on the configured interval."""
    config = load_config()
    last_update = config.get("last_update")
    if not last_update:
        return True

    try:
        last_run = datetime.fromisoformat(last_update)
    except ValueError:
        return True

    if last_run.tzinfo is None:
        last_run = last_run.replace(tzinfo=timezone.utc)

    now = reference_time or datetime.now(timezone.utc)
    return now - last_run >= _FREQUENCY_TO_DELTA[config["data_update_frequency"]]
