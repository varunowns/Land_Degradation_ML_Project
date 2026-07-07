"""Environment variable loading for the GEE pipeline."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"


def load_project_env() -> None:
    """
    Load environment variables from ``.env`` if present.

    OS-level environment variables take precedence over ``.env`` values.
    """
    if ENV_FILE.exists():
        load_dotenv(ENV_FILE, override=False)
    elif ENV_EXAMPLE.exists():
        # Do not auto-load example values; only document their presence.
        pass


def get_env_status() -> dict[str, str | None]:
    """Return current GEE-related environment variable values."""
    load_project_env()
    return {
        "GEE_PROJECT_ID": os.getenv("GEE_PROJECT_ID"),
        "GEE_GRID_ASSET_ID": os.getenv("GEE_GRID_ASSET_ID"),
        "GEE_AOI_ASSET_ID": os.getenv("GEE_AOI_ASSET_ID"),
    }


def require_env_vars(required: tuple[str, ...] = ("GEE_PROJECT_ID", "GEE_GRID_ASSET_ID")) -> dict[str, str]:
    """
    Validate required environment variables and return their values.

    Raises
    ------
    RuntimeError
        If any required variable is missing, with remediation guidance.
    """
    load_project_env()
    values: dict[str, str] = {}
    missing: list[str] = []

    guidance = {
        "GEE_PROJECT_ID": (
            "Your Google Cloud project ID registered with Earth Engine. "
            "Obtain it from https://console.cloud.google.com/ or the Earth Engine "
            "Code Editor (Assets tab → project name)."
        ),
        "GEE_GRID_ASSET_ID": (
            "Earth Engine asset path to your 5 km grid FeatureCollection, e.g. "
            "'projects/YOUR_PROJECT/assets/up_land_degradation_grid'. "
            "Upload or create this asset in the Earth Engine Code Editor / CLI. "
            "Each feature must include properties: Grid_ID, District, Area_km2."
        ),
        "GEE_AOI_ASSET_ID": (
            "Optional Earth Engine asset for the study-area boundary FeatureCollection. "
            "If omitted, the pipeline uses the grid geometry as the AOI."
        ),
    }

    for name in required:
        value = os.getenv(name)
        if not value or not value.strip():
            missing.append(name)
        else:
            values[name] = value.strip()

    if missing:
        lines = ["Missing required environment variables:"]
        for name in missing:
            lines.append(f"  - {name}: {guidance.get(name, 'Set this variable before running the pipeline.')}")
        lines.append("")
        lines.append("Configure via either:")
        lines.append(f"  1. Copy .env.example to {ENV_FILE.name} and fill in values")
        lines.append("  2. Set OS environment variables in PowerShell:")
        lines.append('       $env:GEE_PROJECT_ID = "your-project-id"')
        raise RuntimeError("\n".join(lines))

    return values
