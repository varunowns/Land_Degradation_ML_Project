"""About — credits, technology stack, and references."""

from __future__ import annotations

import streamlit as st

from utils.config import APP_TITLE, STUDY_AREA, STUDY_YEARS
from utils.geography import get_geojson_source_info
from utils.ui import setup_page

setup_page(
    "About",
    subtitle="Technology stack, acknowledgements, and project information",
    show_nav=False,
)

st.markdown(
    f"""
    ### {APP_TITLE}

    A **B.Tech Major Project** applying machine learning and Google Earth Engine
    remote-sensing data to classify land degradation severity across **{STUDY_AREA}**
    for the period **{STUDY_YEARS}**.

    ---

    ### Technology Stack

    | Layer | Tools |
    |-------|-------|
    | **Frontend** | Streamlit (multipage, wide layout) |
    | **Visualisation** | Plotly, Matplotlib (static exports) |
    | **ML / Data** | scikit-learn, XGBoost, pandas, joblib |
    | **Explainability** | SHAP |
    | **Geospatial** | Plotly Choropleth, Datameet GeoJSON |
    | **Data Source** | Google Earth Engine |

    ---

    ### Geographic Data Source

    {get_geojson_source_info()}

    ---

    ### Application Pages

    | Page | Capability |
    |------|------------|
    | Single Prediction | Live inference from environmental inputs |
    | Batch Prediction | CSV upload & download |
    | Model Performance | Baseline comparison & evaluation plots |
    | EDA Dashboard | Interactive distributions, correlations, scatter |
    | SHAP Explainability | Feature importance & static SHAP plots |
    | Geographic Dashboard | Choropleth maps, trends, district drill-down |

    ---

    ### Disclaimer

    Degradation classes were derived from a composite Land Degradation Index (LDI)
    built from the same remote-sensing variables used as model inputs. High model
    accuracy reflects this relationship and should be interpreted alongside
    independent validation where available.
    """
)

st.caption("© 2026 Land Degradation Prediction Project · Built with Streamlit")
