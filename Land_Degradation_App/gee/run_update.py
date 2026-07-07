#!/usr/bin/env python
"""
Execute the complete live Google Earth Engine dataset refresh workflow.

Usage (from repository root)::

    python Land_Degradation_App/gee/run_update.py

Requires:
    - earthengine-api installed and authenticated
    - GEE_PROJECT_ID and GEE_GRID_ASSET_ID environment variables set
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from gee.env import load_project_env, require_env_vars  # noqa: E402
from gee.fetch_data import update_dataset  # noqa: E402
from gee.validate_assets import collect_environment_status, validate_assets  # noqa: E402
from gee.authenticate import authenticate_gee  # noqa: E402


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger("gee.run_update")

    try:
        load_project_env()
        env_status = collect_environment_status()
        logger.info("Environment status: %s", env_status)

        require_env_vars(("GEE_PROJECT_ID", "GEE_GRID_ASSET_ID"))
        ee = authenticate_gee()
        validate_assets(ee)

        logger.info("Starting live Earth Engine dataset update…")
        df = update_dataset()
        logger.info("Dataset update complete: %d rows exported.", len(df))
        return 0
    except Exception as exc:
        logger.error("Dataset update failed: %s", exc)
        logger.error(
            "Remediation:\n"
            "  1. pip install earthengine-api python-dotenv\n"
            "  2. earthengine authenticate\n"
            "  3. Set GEE_PROJECT_ID and GEE_GRID_ASSET_ID in .env\n"
            "  4. Ensure Earth Engine API is enabled for your GCP project"
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
