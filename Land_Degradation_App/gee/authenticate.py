"""Authentication helpers for Google Earth Engine."""

from __future__ import annotations

import logging

from .config import CONFIG

logger = logging.getLogger(__name__)


def authenticate_gee() -> object:
    """Authenticate and initialize a Google Earth Engine session."""
    try:
        import ee  # type: ignore
    except ImportError as exc:
        logger.exception("Earth Engine package is not installed.")
        raise RuntimeError(
            "Missing dependency: install the `earthengine-api` package to use the GEE pipeline."
        ) from exc

    try:
        if CONFIG.gee_project:
            ee.Initialize(project=CONFIG.gee_project)
        else:
            ee.Initialize()
        logger.info("Initialized Google Earth Engine session.")
        return ee
    except Exception:
        logger.warning("Initial Earth Engine initialization failed; attempting authentication.")
        try:
            ee.Authenticate()
            if CONFIG.gee_project:
                ee.Initialize(project=CONFIG.gee_project)
            else:
                ee.Initialize()
            logger.info("Authenticated and initialized Google Earth Engine session.")
            return ee
        except Exception as exc:
            logger.exception("Unable to authenticate with Google Earth Engine.")
            raise RuntimeError(
                "Google Earth Engine authentication failed. Ensure credentials are configured.\n"
                "Run: earthengine authenticate\n"
                "Or:  python Land_Degradation_App/gee/authenticate.py"
            ) from exc


def main() -> int:
    """CLI entry point for Earth Engine authentication."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    try:
        authenticate_gee()
        print("Earth Engine authentication successful.")
        return 0
    except Exception as exc:
        print(f"Authentication failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
