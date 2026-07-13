"""
Plotly theming and reusable chart helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from utils.config import CLASS_ORDER, COLORS, PLOTLY_COLOR_SEQUENCE, PLOTLY_TEMPLATE


def apply_plotly_theme() -> None:
    """Register the application Plotly template for all subsequent figures."""
    pio.templates["land_degradation"] = go.layout.Template(
        layout=go.Layout(
            colorway=PLOTLY_COLOR_SEQUENCE,
            paper_bgcolor="rgba(250,250,250,0)",
            plot_bgcolor="rgba(250,250,250,0.5)",
            font=dict(family="Segoe UI, Inter, sans-serif", color=COLORS["text"]),
            title=dict(font=dict(size=18, color=COLORS["primary_dark"])),
            xaxis=dict(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"]),
            yaxis=dict(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"]),
        )
    )
    pio.templates.default = "land_degradation+" + PLOTLY_TEMPLATE


def get_class_color(degradation_class: str) -> str:
    """Return the theme colour associated with a degradation class label."""
    mapping = {
        "Low": COLORS["low"],
        "Moderate": COLORS["moderate"],
        "High": COLORS["high"],
    }
    return mapping.get(degradation_class, COLORS["text_muted"])


def load_static_plot(path: Path) -> bytes | None:
    """Load a pre-generated static plot image from disk."""
    if path.exists():
        return path.read_bytes()
    return None


def create_metric_bar_chart(
    labels: list[str],
    values: list[float],
    title: str = "Model Metrics",
) -> go.Figure:
    """Create a themed horizontal bar chart for metric comparison."""
    apply_plotly_theme()
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=values, colorscale="Viridis"),
            text=[f"{v:.4f}" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Score",
        yaxis_title="Metric",
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def create_histogram(
    df: pd.DataFrame,
    column: str,
    color: str | None = None,
    title: str | None = None,
) -> go.Figure:
    """Interactive histogram for EDA."""
    apply_plotly_theme()
    fig = px.histogram(
        df,
        x=column,
        nbins=40,
        color_discrete_sequence=[color or COLORS["primary"]],
        title=title or f"Distribution of {column}",
    )
    fig.update_layout(height=380, showlegend=False)
    return fig


def create_degradation_count_by_district(
    df: pd.DataFrame,
    district_col: str,
    class_col: str,
) -> go.Figure:
    """Stacked district-wise degradation class counts."""
    apply_plotly_theme()
    counts = (
        df.groupby([district_col, class_col])
        .size()
        .reset_index(name="count")
        .sort_values([district_col, class_col])
    )
    fig = px.bar(
        counts,
        x=district_col,
        y="count",
        color=class_col,
        category_orders={class_col: CLASS_ORDER},
        color_discrete_map={c: get_class_color(c) for c in CLASS_ORDER},
        title="District-wise Degradation Count",
        labels={district_col: "District", "count": "Predictions", class_col: "Class"},
    )
    fig.update_layout(height=440, xaxis_tickangle=-35, legend_title_text="Class")
    return fig


def create_degradation_percentage_pie(df: pd.DataFrame, class_col: str) -> go.Figure:
    """Low, Moderate, High percentage pie chart."""
    apply_plotly_theme()
    counts = df[class_col].value_counts().reindex(CLASS_ORDER, fill_value=0).reset_index()
    counts.columns = [class_col, "count"]
    fig = px.pie(
        counts,
        names=class_col,
        values="count",
        title="Low / Moderate / High Percentage",
        color=class_col,
        category_orders={class_col: CLASS_ORDER},
        color_discrete_map={c: get_class_color(c) for c in CLASS_ORDER},
        hole=0.38,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(height=420, showlegend=True)
    return fig


def create_latest_ndvi_distribution(df: pd.DataFrame, ndvi_col: str) -> go.Figure:
    """Histogram of latest prediction NDVI values."""
    apply_plotly_theme()
    fig = px.histogram(
        df,
        x=ndvi_col,
        nbins=35,
        title="Latest NDVI Distribution",
        color_discrete_sequence=[COLORS["primary"]],
        labels={ndvi_col: "NDVI"},
    )
    fig.update_layout(height=420, showlegend=False, yaxis_title="Records")
    return fig


def create_boxplot_by_class(
    df: pd.DataFrame,
    feature: str,
    class_col: str = "Degradation_Class",
) -> go.Figure:
    """Box plot of a feature grouped by degradation class."""
    apply_plotly_theme()
    palette = {c: get_class_color(c) for c in CLASS_ORDER}
    fig = px.box(
        df,
        x=class_col,
        y=feature,
        color=class_col,
        category_orders={class_col: CLASS_ORDER},
        color_discrete_map=palette,
        title=f"{feature} by Degradation Class",
    )
    fig.update_layout(height=400, showlegend=False)
    return fig


def create_correlation_heatmap(df: pd.DataFrame, columns: list[str]) -> go.Figure:
    """Pearson correlation heatmap."""
    apply_plotly_theme()
    corr = df[columns].corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Feature Correlation Matrix",
    )
    fig.update_layout(height=520)
    return fig


def create_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str | None = None,
    title: str | None = None,
) -> go.Figure:
    """Interactive scatter plot with optional colour grouping."""
    apply_plotly_theme()
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=color,
        color_discrete_map={c: get_class_color(c) for c in CLASS_ORDER} if color else None,
        opacity=0.55,
        title=title or f"{y} vs {x}",
        category_orders={color: CLASS_ORDER} if color else None,
    )
    fig.update_layout(height=420)
    return fig


def create_bar_ranking(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    ascending: bool = False,
    n: int = 10,
) -> go.Figure:
    """Horizontal bar chart for top/bottom district rankings."""
    apply_plotly_theme()
    ranked = df.sort_values(y, ascending=ascending).head(n)
    fig = px.bar(
        ranked,
        x=y,
        y=x,
        orientation="h",
        color=y,
        color_continuous_scale=[
            [0, COLORS["low"]],
            [0.5, COLORS["moderate"]],
            [1, COLORS["high"]],
        ],
        title=title,
    )
    fig.update_layout(height=420, showlegend=False, coloraxis_showscale=False)
    return fig


def create_line_trend(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    color: str | None = None,
) -> go.Figure:
    """Line chart for temporal trends."""
    apply_plotly_theme()
    fig = px.line(
        df,
        x=x,
        y=y,
        markers=True,
        color_discrete_sequence=[color or COLORS["secondary"]],
        title=title,
    )
    fig.update_layout(height=380)
    return fig


def export_figure_png(fig: go.Figure) -> bytes | None:
    """Export a Plotly figure to PNG bytes (requires kaleido)."""
    try:
        return fig.to_image(format="png", scale=2)
    except Exception:
        return None


def create_shap_bar(shap_df: pd.DataFrame, title: str = "SHAP Feature Importance") -> go.Figure:
    """Interactive SHAP importance bar chart."""
    apply_plotly_theme()
    top = shap_df.head(15).sort_values(shap_df.columns[1], ascending=True)
    value_col = top.columns[1]
    fig = px.bar(
        top,
        x=value_col,
        y="feature",
        orientation="h",
        color_discrete_sequence=[COLORS["primary"]],
        title=title,
    )
    fig.update_layout(height=480, showlegend=False)
    return fig
