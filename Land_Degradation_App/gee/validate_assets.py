"""Validate Google Earth Engine project configuration and assets."""

from __future__ import annotations

import logging
from typing import Any

from .config import CONFIG
from .env import get_env_status, require_env_vars

logger = logging.getLogger(__name__)


def check_credentials_exist() -> bool:
    """Return True if Earth Engine credential file exists locally."""
    from pathlib import Path

    credential_paths = [
        Path.home() / ".config" / "earthengine" / "credentials",
        Path.home() / ".config" / "earthengine" / "credentials.json",
    ]
    return any(path.exists() for path in credential_paths)


def validate_authentication(ee: object) -> dict[str, Any]:
    """
    Verify that the Earth Engine session is authenticated and initialized.

    Returns a status dictionary; raises on hard failures.
    """
    status: dict[str, Any] = {
        "credentials_file_present": check_credentials_exist(),
        "initialized": False,
        "project_id": CONFIG.gee_project,
        "error": None,
    }
    try:
        # Lightweight call to confirm API access
        ee.data.getAssetRoots()
        status["initialized"] = True
        logger.info("Earth Engine authentication verified.")
    except Exception as exc:
        status["error"] = str(exc)
        logger.error("Earth Engine authentication check failed: %s", exc)
        raise RuntimeError(
            "Earth Engine authentication failed. Run:\n"
            "  python -m gee.authenticate\n"
            "or:\n"
            "  earthengine authenticate\n"
            "Then set GEE_PROJECT_ID and re-run."
        ) from exc
    return status


def validate_assets(ee: object) -> dict[str, Any]:
    """
    Validate project ID, grid asset, and optional AOI asset.

    Returns structured status for operational reporting.
    """
    env = require_env_vars(("GEE_PROJECT_ID", "GEE_GRID_ASSET_ID"))
    status: dict[str, Any] = {
        "project_id": env["GEE_PROJECT_ID"],
        "grid_asset_id": env["GEE_GRID_ASSET_ID"],
        "aoi_asset_id": CONFIG.aoi_asset_id,
        "grid_readable": False,
        "grid_feature_count": None,
        "grid_properties_ok": False,
        "aoi_readable": None,
        "aoi_feature_count": None,
        "errors": [],
    }

    try:
        grid_info = ee.data.getAsset(env["GEE_GRID_ASSET_ID"])
        status["grid_asset_type"] = grid_info.get("type")
        grid_fc = ee.FeatureCollection(env["GEE_GRID_ASSET_ID"])
        count = int(grid_fc.size().getInfo())
        status["grid_readable"] = True
        status["grid_feature_count"] = count

        sample = grid_fc.first().getInfo()
        props = sample.get("properties", {}) if sample else {}
        required_props = {CONFIG.grid_id_property, CONFIG.district_property, CONFIG.area_property}
        missing_props = required_props - set(props.keys())
        if missing_props:
            status["errors"].append(
                f"Grid asset missing required properties: {sorted(missing_props)}"
            )
        else:
            status["grid_properties_ok"] = True
        logger.info("Grid asset validated: %d features", count)
    except Exception as exc:
        status["errors"].append(f"Grid asset validation failed: {exc}")
        logger.error("Grid asset validation failed: %s", exc)

    if CONFIG.aoi_asset_id:
        try:
            aoi_info = ee.data.getAsset(CONFIG.aoi_asset_id)
            status["aoi_asset_type"] = aoi_info.get("type")
            aoi_fc = ee.FeatureCollection(CONFIG.aoi_asset_id)
            status["aoi_readable"] = True
            status["aoi_feature_count"] = int(aoi_fc.size().getInfo())
            logger.info("AOI asset validated: %d features", status["aoi_feature_count"])
        except Exception as exc:
            status["aoi_readable"] = False
            status["errors"].append(f"AOI asset validation failed: {exc}")
            logger.error("AOI asset validation failed: %s", exc)
    else:
        status["aoi_readable"] = "not_configured"
        status["note"] = "AOI not configured; grid geometry will be used as AOI."

    if status["errors"]:
        raise RuntimeError("\n".join(status["errors"]))

    return status


def collect_environment_status() -> dict[str, Any]:
    """Collect environment variable status without requiring values."""
    env = get_env_status()
    return {
        "GEE_PROJECT_ID": {"set": bool(env["GEE_PROJECT_ID"]), "value": env["GEE_PROJECT_ID"]},
        "GEE_GRID_ASSET_ID": {"set": bool(env["GEE_GRID_ASSET_ID"]), "value": env["GEE_GRID_ASSET_ID"]},
        "GEE_AOI_ASSET_ID": {"set": bool(env["GEE_AOI_ASSET_ID"]), "value": env["GEE_AOI_ASSET_ID"]},
        "credentials_file_present": check_credentials_exist(),
    }
