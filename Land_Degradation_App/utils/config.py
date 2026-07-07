"""
Application configuration: paths, constants, and design tokens.

Paths resolve to ``Land_Degradation_App/`` first, then fall back to the
parent ML project directory so artefacts are never duplicated or regenerated.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
APP_ROOT = Path(__file__).resolve().parent.parent
ML_PROJECT_ROOT = APP_ROOT.parent


def _resolve(*relative: str) -> Path:
    """Return the first existing path among app-local and ML-project locations."""
    for base in (APP_ROOT, ML_PROJECT_ROOT):
        candidate = base.joinpath(*relative)
        if candidate.exists():
            return candidate
    return APP_ROOT.joinpath(*relative)


MODELS_DIR = ML_PROJECT_ROOT / "models"
if (APP_ROOT / "models").is_dir() and any((APP_ROOT / "models").glob("*.pkl")):
    MODELS_DIR = APP_ROOT / "models"
DATA_DIR = _resolve("data") if _resolve("data").is_dir() else APP_ROOT / "data"
PLOTS_DIR = _resolve("plots") if _resolve("plots").is_dir() else APP_ROOT / "plots"
ASSETS_DIR = APP_ROOT / "assets"
GEO_DIR = ASSETS_DIR / "geo"

LDI_DATASET_PATH = _resolve("data", "ldi_dataset.csv")
if not LDI_DATASET_PATH.exists():
    LDI_DATASET_PATH = ML_PROJECT_ROOT / "ldi_dataset.csv"

WORKING_DATASET_PATH = _resolve("data", "working_dataset.csv")
if not WORKING_DATASET_PATH.exists():
    WORKING_DATASET_PATH = ML_PROJECT_ROOT / "working_dataset.csv"

BASELINE_COMPARISON_PATH = _resolve("data", "baseline_comparison.csv")
if not BASELINE_COMPARISON_PATH.exists():
    BASELINE_COMPARISON_PATH = ML_PROJECT_ROOT / "results" / "baseline_comparison.csv"

PREDICTIONS_PATH = _resolve("data", "predictions.csv")
if not PREDICTIONS_PATH.exists():
    PREDICTIONS_PATH = ML_PROJECT_ROOT / "results" / "predictions.csv"

FEATURE_IMPORTANCE_PATH = _resolve("data", "feature_importance.csv")
if not FEATURE_IMPORTANCE_PATH.exists():
    FEATURE_IMPORTANCE_PATH = ML_PROJECT_ROOT / "results" / "feature_importance.csv"

SHAP_IMPORTANCE_PATH = _resolve("data", "shap_importance.csv")
if not SHAP_IMPORTANCE_PATH.exists():
    SHAP_IMPORTANCE_PATH = ML_PROJECT_ROOT / "results" / "shap_importance.csv"

CV_COMPARISON_PATH = _resolve("data", "cv_comparison.csv")
if not CV_COMPARISON_PATH.exists():
    CV_COMPARISON_PATH = ML_PROJECT_ROOT / "results" / "cv_comparison.csv"

DISTRICT_ACCURACY_PATH = _resolve("data", "district_accuracy.csv")
if not DISTRICT_ACCURACY_PATH.exists():
    DISTRICT_ACCURACY_PATH = ML_PROJECT_ROOT / "results" / "district_accuracy.csv"

# GeoJSON cache (downloaded once — see utils/geography.py for source)
UP_GEOJSON_CACHE = GEO_DIR / "uttar_pradesh_districts.geojson"

# Trained artefact filenames (read-only — do not retrain)
BEST_MODEL_FILENAME = "tuned_logistic_regression.pkl"
PREPROCESSOR_LR_FILENAME = "preprocessor_lr.pkl"
LABEL_ENCODER_FILENAME = "label_encoder.pkl"

# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------
APP_TITLE = "Land Degradation Prediction System"
APP_ICON = "🌍"
CLASS_ORDER = ["Low", "Moderate", "High"]
STUDY_YEARS_LIST = [2020, 2021, 2022, 2023, 2024]
STUDY_AREA = "Uttar Pradesh, India"
GRID_RESOLUTION = "5 km × 5 km"
STUDY_YEARS = "2020–2024"
N_DISTRICTS = 20

TARGET_COLUMN = "Degradation_Class"
LEAKAGE_COLUMNS = ["LDI"]
ID_COLUMNS = ["Grid_ID"]

NUMERIC_FEATURE_COLUMNS = [
    "Area_km2",
    "BareLand",
    "Builtup",
    "Cropland",
    "Grassland",
    "TreeCover",
    "Water",
    "Wetland",
    "Shrubland",
    "NDVI_mean",
    "Rainfall_mean",
    "Temperature_mean",
    "SoilMoisture_mean",
]

# Choropleth metric options: (label, column_name)
MAP_METRICS = [
    ("Average LDI", "LDI"),
    ("Average NDVI", "NDVI_mean"),
    ("Average Rainfall", "Rainfall_mean"),
    ("Average Temperature", "Temperature_mean"),
    ("Average Soil Moisture", "SoilMoisture_mean"),
    ("Average Bare Land", "BareLand"),
    ("Average Tree Cover", "TreeCover"),
]

# Map dataset district names → GeoJSON district names (2011 Datameet boundaries)
DISTRICT_GEOJSON_ALIASES: dict[str, str] = {
    "Allahabad": "Allahabad",
    "Ambedkar Nagar": "Ambedkar Nagar",
    "Bulandshahr": "Bulandshahr",
    "Chitrakoot": "Chitrakoot",
    "Badaun": "Budaun",
}

GEOJSON_SOURCE_DOC = (
    "Datameet Community Maps — India District Boundaries (2011 Census shapefile). "
    "Downloaded from https://github.com/datameet/maps/tree/master/Districts/Census_2011, "
    "converted to GeoJSON with GeoPandas, filtered to 20 study districts. "
    "License: CC BY 4.0."
)

# Legacy URL kept for documentation only; actual download uses shapefile parts above.
GEOJSON_SOURCE_URL = (
    "https://github.com/datameet/maps/tree/master/Districts/Census_2011"
)

# ---------------------------------------------------------------------------
# Design system
# ---------------------------------------------------------------------------
COLORS = {
    "primary": "#2E7D32",
    "primary_dark": "#1B5E20",
    "secondary": "#1565C0",
    "accent": "#F57C00",
    "accent_light": "#FFB74D",
    "background": "#FAFAFA",
    "surface": "#FFFFFF",
    "text": "#263238",
    "text_muted": "#546E7A",
    "border": "#E0E0E0",
    "success": "#43A047",
    "warning": "#FB8C00",
    "danger": "#E53935",
    "low": "#43A047",
    "moderate": "#FB8C00",
    "high": "#E53935",
}

PLOTLY_TEMPLATE = "plotly_white"
PLOTLY_COLOR_SEQUENCE = [
    COLORS["primary"],
    COLORS["secondary"],
    COLORS["accent"],
    COLORS["primary_dark"],
    COLORS["accent_light"],
]

PLOTLY_SEQUENTIAL_LD = [
    [0.0, COLORS["low"]],
    [0.5, COLORS["moderate"]],
    [1.0, COLORS["high"]],
]

# ---------------------------------------------------------------------------
# Page registry
# ---------------------------------------------------------------------------
PAGES = [
    {"file": "pages/1_Home.py", "label": "Home", "icon": "🏠"},
    {"file": "pages/2_Project_Overview.py", "label": "Project Overview", "icon": "📋"},
    {"file": "pages/3_Single_Prediction.py", "label": "Single Prediction", "icon": "🎯"},
    {"file": "pages/4_Batch_Prediction.py", "label": "Batch Prediction", "icon": "📂"},
    {"file": "pages/5_Model_Performance.py", "label": "Model Performance", "icon": "📊"},
    {"file": "pages/6_EDA_Dashboard.py", "label": "EDA Dashboard", "icon": "🔍"},
    {"file": "pages/7_SHAP_Explainability.py", "label": "SHAP Explainability", "icon": "🧠"},
    {"file": "pages/9_Geographic_Dashboard.py", "label": "Geographic Dashboard", "icon": "🗺️"},
    {"file": "pages/8_About.py", "label": "About", "icon": "ℹ️"},
]
