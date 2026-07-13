"""Utility package for the Land Degradation Streamlit application."""

from utils.config import APP_TITLE, CLASS_ORDER, COLORS, PAGES
import logging

def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """Configure and return a module-level logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger

__all__ = [
    "APP_TITLE",
    "CLASS_ORDER",
    "COLORS",
    "PAGES",
    "setup_logging",
]
