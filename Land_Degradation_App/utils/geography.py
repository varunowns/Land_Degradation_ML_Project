"""
Geographic utilities: GeoJSON loading, district aggregation, and choropleth maps.

GeoJSON source
--------------
India district boundaries (2011 Census) from the **Datameet Community Maps**
shapefile repository. Shapefile components are downloaded once, converted to
GeoJSON with GeoPandas, filtered to Uttar Pradesh study districts, and cached
locally at ``assets/geo/uttar_pradesh_districts.geojson``.

Source: https://github.com/datameet/maps/tree/master/Districts/Census_2011
License: CC BY 4.0
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config_manager import (
    get_cache_ttl_seconds,
    is_auto_fetch_enabled,
    is_update_due,
    update_last_run,
)
from utils.config import (
    COLORS,
    DISTRICT_GEOJSON_ALIASES,
    GEOJSON_SOURCE_DOC,
    MAP_METRICS,
    STUDY_YEARS_LIST,
    UP_GEOJSON_CACHE,
)
from utils.data_loader import load_ldi_dataset
from utils.plotting import apply_plotly_theme

DATAMEET_SHP_BASE = (
    "https://github.com/datameet/maps/raw/master/Districts/Census_2011/"
)
SHAPEFILE_PARTS = ("2011_Dist.shp", "2011_Dist.dbf", "2011_Dist.shx", "2011_Dist.prj")
CACHE_TTL_SECONDS = get_cache_ttl_seconds()


def _normalize_name(name: str) -> str:
    """Normalise district names for fuzzy matching."""
    return name.lower().replace("_", " ").replace("-", " ").strip()


def dataset_to_geojson_name(district: str) -> str:
    """Map dataset district label to GeoJSON district property."""
    return DISTRICT_GEOJSON_ALIASES.get(district, district)


def geojson_to_dataset_name(geo_name: str, dataset_districts: list[str]) -> str | None:
    """Match a GeoJSON district name back to a dataset district label."""
    norm_geo = _normalize_name(geo_name)
    for d in dataset_districts:
        if _normalize_name(d) == norm_geo:
            return d
        if _normalize_name(dataset_to_geojson_name(d)) == norm_geo:
            return d
    return None


def _download_shapefile_parts(dest_dir: Path) -> Path:
    """Download Datameet 2011 district shapefile components to ``dest_dir``."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for part in SHAPEFILE_PARTS:
        target = dest_dir / part
        if target.exists() and target.stat().st_size > 0:
            continue
        url = DATAMEET_SHP_BASE + part
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "LandDegradationApp/1.0 (academic project)"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            target.write_bytes(response.read())
    return dest_dir / SHAPEFILE_PARTS[0]


def _load_cached_geojson() -> dict[str, Any]:
    """Load the cached GeoJSON from disk."""
    with UP_GEOJSON_CACHE.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _resolve_shapefile_path(shp_dir: Path) -> Path | None:
    """Return a local shapefile path if all components are already present."""
    shp_path = shp_dir / SHAPEFILE_PARTS[0]
    if shp_path.exists() and all((shp_dir / part).exists() for part in SHAPEFILE_PARTS):
        return shp_path
    return None


def _build_study_district_geojson() -> dict[str, Any]:
    """
    Download, convert, and filter district boundaries for the 20 study districts.

    Cached in memory and on disk at ``UP_GEOJSON_CACHE``.
    """
    UP_GEOJSON_CACHE.parent.mkdir(parents=True, exist_ok=True)

    import geopandas as gpd

    shp_dir = UP_GEOJSON_CACHE.parent / "datameet_shp"
    cached_geojson_exists = UP_GEOJSON_CACHE.exists()
    auto_fetch_enabled = is_auto_fetch_enabled()
    refresh_due = is_update_due()

    if cached_geojson_exists and not (auto_fetch_enabled and refresh_due):
        return _load_cached_geojson()

    shp_path = _resolve_shapefile_path(shp_dir)
    if shp_path is None and auto_fetch_enabled:
        shp_path = _download_shapefile_parts(shp_dir)
    elif shp_path is None and cached_geojson_exists:
        return _load_cached_geojson()
    elif shp_path is None:
        raise FileNotFoundError(
            f"GeoJSON cache not found and auto-fetch is disabled: {UP_GEOJSON_CACHE}"
        )

    gdf = gpd.read_file(shp_path)

    dataset_districts = sorted(load_ldi_dataset()["District"].unique().tolist())
    geojson_names_needed = {dataset_to_geojson_name(d) for d in dataset_districts}

    up = gdf[gdf["ST_NM"].str.contains("Uttar", na=False)].copy()
    up = up[up["DISTRICT"].isin(geojson_names_needed)]

    up["dataset_district"] = up["DISTRICT"].apply(
        lambda g: geojson_to_dataset_name(g, dataset_districts) or g
    )

    geo = json.loads(up.to_json())
    with UP_GEOJSON_CACHE.open("w", encoding="utf-8") as fh:
        json.dump(geo, fh)
    update_last_run()

    return geo


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def load_up_geojson() -> dict[str, Any]:
    """Load cached UP study-district GeoJSON (Streamlit-cached)."""
    return _build_study_district_geojson()


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def aggregate_district_stats(year: str | int = "All Years") -> pd.DataFrame:
    """
    Aggregate grid-level data to district level.

    Parameters
    ----------
    year:
        Specific year int or ``"All Years"`` for temporal mean.
    """
    df = load_ldi_dataset()
    if year != "All Years":
        df = df[df["Year"] == int(year)]

    grouped = df.groupby("District").agg(
        LDI=("LDI", "mean"),
        NDVI_mean=("NDVI_mean", "mean"),
        Rainfall_mean=("Rainfall_mean", "mean"),
        Temperature_mean=("Temperature_mean", "mean"),
        SoilMoisture_mean=("SoilMoisture_mean", "mean"),
        BareLand=("BareLand", "mean"),
        TreeCover=("TreeCover", "mean"),
        dominant_class=("Degradation_Class", lambda s: s.mode().iloc[0]),
        grid_count=("Grid_ID", "count"),
    ).reset_index()

    grouped["Year"] = year if year != "All Years" else "All Years"
    grouped["geo_district"] = grouped["District"].map(dataset_to_geojson_name)
    return grouped


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def build_choropleth(
    stats: pd.DataFrame,
    geojson: dict[str, Any],
    metric_col: str,
    metric_label: str,
) -> go.Figure:
    """Create an interactive Plotly choropleth for UP study districts."""
    apply_plotly_theme()

    if stats.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data for selected filters", showarrow=False)
        return fig

    hover_data = {
        "District": True,
        "Year": True,
        "LDI": ":.4f",
        "dominant_class": True,
        "NDVI_mean": ":.4f",
        "Rainfall_mean": ":.1f",
        "Temperature_mean": ":.2f",
        "SoilMoisture_mean": ":.4f",
        "TreeCover": ":.2f",
        "BareLand": ":.2f",
        metric_col: ":.4f",
        "geo_district": False,
        "grid_count": True,
    }

    fig = px.choropleth(
        stats,
        geojson=geojson,
        locations="geo_district",
        featureidkey="properties.DISTRICT",
        color=metric_col,
        hover_name="District",
        hover_data=hover_data,
        color_continuous_scale=[
            [0.0, COLORS["low"]],
            [0.5, COLORS["moderate"]],
            [1.0, COLORS["high"]],
        ],
        title=f"{metric_label} by District — {stats['Year'].iloc[0]}",
        labels={metric_col: metric_label},
    )
    fig.update_geos(fitbounds="locations", visible=False, projection_type="mercator")
    fig.update_layout(
        height=560,
        margin=dict(l=0, r=0, t=60, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def build_animated_choropleth(
    geojson: dict[str, Any],
    metric_col: str = "LDI",
    metric_label: str = "Average LDI",
) -> go.Figure:
    """Build year-by-year animated choropleth (2020 → 2024)."""
    apply_plotly_theme()
    frames_data = [aggregate_district_stats(year) for year in STUDY_YEARS_LIST]
    all_stats = pd.concat(frames_data, ignore_index=True)

    fig = px.choropleth(
        all_stats,
        geojson=geojson,
        locations="geo_district",
        featureidkey="properties.DISTRICT",
        color=metric_col,
        hover_name="District",
        animation_frame="Year",
        color_continuous_scale=[
            [0.0, COLORS["low"]],
            [0.5, COLORS["moderate"]],
            [1.0, COLORS["high"]],
        ],
        title=f"Animated {metric_label} — Uttar Pradesh Study Districts (2020–2024)",
        labels={metric_col: metric_label},
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(height=580, margin=dict(l=0, r=0, t=60, b=0))
    return fig


def get_kpi_summary(stats: pd.DataFrame) -> dict[str, Any]:
    """Compute KPI cards for the geographic dashboard header."""
    if stats.empty:
        return {}
    most = stats.loc[stats["LDI"].idxmax()]
    least = stats.loc[stats["LDI"].idxmin()]
    return {
        "most_degraded": most["District"],
        "most_degraded_ldi": float(most["LDI"]),
        "least_degraded": least["District"],
        "least_degraded_ldi": float(least["LDI"]),
        "avg_ldi": float(stats["LDI"].mean()),
        "avg_ndvi": float(stats["NDVI_mean"].mean()),
        "avg_rainfall": float(stats["Rainfall_mean"].mean()),
        "avg_temperature": float(stats["Temperature_mean"].mean()),
    }


def get_district_detail(district: str, year: str | int = "All Years") -> pd.DataFrame:
    """Return grid-level records for a single district."""
    df = load_ldi_dataset()
    df = df[df["District"] == district]
    if year != "All Years":
        df = df[df["Year"] == int(year)]
    return df


def get_district_yearly_trend(district: str) -> pd.DataFrame:
    """Return yearly mean LDI and environmental variables for one district."""
    df = load_ldi_dataset()
    df = df[df["District"] == district]
    return (
        df.groupby("Year")
        .agg(
            LDI=("LDI", "mean"),
            NDVI_mean=("NDVI_mean", "mean"),
            Rainfall_mean=("Rainfall_mean", "mean"),
            Temperature_mean=("Temperature_mean", "mean"),
            SoilMoisture_mean=("SoilMoisture_mean", "mean"),
            BareLand=("BareLand", "mean"),
            TreeCover=("TreeCover", "mean"),
            dominant_class=("Degradation_Class", lambda s: s.mode().iloc[0]),
        )
        .reset_index()
    )


def get_metric_options() -> list[tuple[str, str]]:
    """Return choropleth metric dropdown options."""
    return MAP_METRICS


def get_geojson_source_info() -> str:
    """Return documented GeoJSON source attribution."""
    return GEOJSON_SOURCE_DOC
