"""
Google Earth Engine authentication and connection status manager.
"""

from __future__ import annotations

import logging
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path

try:
    import ee
except ImportError:
    ee = None

from config_manager import load_config, save_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOGS_DIR / "gee.log"

# Setup Logger
logger = logging.getLogger("gee_auth")
logger.setLevel(logging.INFO)

if not logger.handlers:
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | Project: %(project_id)s | %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def _log_with_project(level: int, msg: str, project_id: str = "UNKNOWN"):
    """Helper to inject project_id into log records."""
    logger.log(level, msg, extra={"project_id": project_id})


def get_gee_project_id() -> str:
    config = load_config()
    project_id = config.get("gee_project_id", "")
    if project_id:
        project_id = project_id.strip()
    return project_id


def check_gee_status() -> str:
    """
    Attempt to initialize Earth Engine.
    Returns 'CONNECTED' if successful, otherwise 'NOT_CONNECTED'.
    Saves the status and timestamp to config/settings.json.
    """
    project_id = get_gee_project_id()
    _log_with_project(logging.INFO, "Checking Google Earth Engine connection status...", project_id)
    
    if ee is None:
        status = "NOT_CONNECTED"
        _log_with_project(logging.ERROR, "earthengine-api package is not installed.", project_id)
    else:
        try:
            if not project_id:
                raise ValueError("Google Cloud Project ID missing")
                
            ee.Initialize(project=project_id)
            status = "CONNECTED"
            _log_with_project(logging.INFO, "Google Earth Engine connection verified: CONNECTED.", project_id)
        except Exception as exc:
            status = "NOT_CONNECTED"
            _log_with_project(logging.ERROR, f"Google Earth Engine is NOT_CONNECTED. Details:\n{traceback.format_exc()}", project_id)

    # Update config
    try:
        config = load_config()
        config["gee_status"] = status
        config["last_gee_check"] = datetime.now(timezone.utc).isoformat()
        save_config(config)
    except Exception as exc:
        _log_with_project(logging.ERROR, f"Failed to update settings.json with GEE status: {exc}", project_id)

    return status


def initialize_gee() -> tuple[bool, str]:
    """
    Initialize Google Earth Engine client using configured project ID.
    """
    project_id = get_gee_project_id()
    _log_with_project(logging.INFO, "Initializing Google Earth Engine...", project_id)
    
    if ee is None:
        return False, "earthengine-api is not installed"
        
    if not project_id:
        return False, "Google Cloud Project ID missing"

    try:
        ee.Initialize(project=project_id)
        
        config = load_config()
        config["gee_status"] = "CONNECTED"
        save_config(config)
        
        _log_with_project(logging.INFO, "Google Earth Engine initialized successfully.", project_id)
        return True, "CONNECTED"
        
    except Exception as e:
        _log_with_project(logging.ERROR, f"Google Earth Engine initialization failed:\n{traceback.format_exc()}", project_id)
        
        config = load_config()
        config["gee_status"] = "NOT_CONNECTED"
        save_config(config)
        
        return False, str(e)


def authenticate_gee() -> tuple[bool, str]:
    """
    Authenticate and initialize Google Earth Engine.
    """
    project_id = get_gee_project_id()
    _log_with_project(logging.INFO, "Triggering Google Earth Engine authentication flow...", project_id)
    
    if ee is None:
        return False, "earthengine-api is not installed"

    try:
        # Run Authenticate flow with localhost mode to avoid stdin blocking in Streamlit
        # Do not store credentials
        ee.Authenticate(auth_mode='localhost')
        
        # Initialize immediately after authentication
        return initialize_gee()
        
    except Exception as e:
        _log_with_project(logging.ERROR, f"Google Earth Engine authentication flow failed:\n{traceback.format_exc()}", project_id)
        return False, str(e)
