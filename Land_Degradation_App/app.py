"""
Land Degradation Prediction System — Application Entry Point.

Run with::

    streamlit run app.py

This script bootstraps global configuration and redirects to the Home page.
All feature pages live under ``pages/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path to import from src/
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from utils.ui import configure_app

configure_app()

try:
    from src.scheduler import start_scheduler
    start_scheduler()
except ImportError:
    pass  # Scheduler optional for app startup

st.switch_page("pages/1_Home.py")
