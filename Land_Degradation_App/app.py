"""
Land Degradation Prediction System — Application Entry Point.

Run with::

    streamlit run app.py

This script bootstraps global configuration and redirects to the Home page.
All feature pages live under ``pages/``.
"""

from __future__ import annotations

import streamlit as st

from utils.ui import configure_app

configure_app()
st.switch_page("pages/1_Home.py")
