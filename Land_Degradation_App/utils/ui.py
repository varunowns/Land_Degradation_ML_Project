"""
Streamlit UI helpers: page configuration, custom CSS, and sidebar layout.

Call ``configure_app()`` as the first Streamlit command in every page script.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from utils.config import APP_ICON, APP_TITLE, COLORS, PAGES, STUDY_AREA, STUDY_YEARS


def configure_app() -> None:
    """
    Apply global page configuration and inject custom CSS.

    Must be invoked before any other ``st.*`` call in the current script.
    """
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "About": "Land Degradation Prediction System — B.Tech Major Project",
        },
    )
    _inject_custom_css()


def _inject_custom_css() -> None:
    """Inject a cohesive theme aligned with the earth/remote-sensing palette."""
    st.markdown(
        f"""
        <style>
            /* ---- Global typography ---- */
            html, body, [class*="css"] {{
                font-family: "Segoe UI", "Inter", system-ui, sans-serif;
                color: {COLORS["text"]};
            }}

            /* ---- Main container ---- */
            .block-container {{
                padding-top: 1.5rem;
                padding-bottom: 2rem;
                max-width: 1400px;
            }}

            /* ---- Sidebar ---- */
            section[data-testid="stSidebar"] {{
                background: linear-gradient(
                    180deg,
                    {COLORS["primary_dark"]} 0%,
                    {COLORS["primary"]} 55%,
                    #388E3C 100%
                );
            }}
            section[data-testid="stSidebar"] * {{
                color: #FFFFFF !important;
            }}
            section[data-testid="stSidebar"] .stMarkdown h1,
            section[data-testid="stSidebar"] .stMarkdown h2,
            section[data-testid="stSidebar"] .stMarkdown h3 {{
                color: #FFFFFF !important;
            }}
            section[data-testid="stSidebar"] hr {{
                border-color: rgba(255,255,255,0.25);
            }}

            /* ---- Hero banner ---- */
            .ld-hero {{
                background: linear-gradient(
                    135deg,
                    {COLORS["primary_dark"]} 0%,
                    {COLORS["primary"]} 50%,
                    {COLORS["secondary"]} 100%
                );
                padding: 2rem 2.5rem;
                border-radius: 12px;
                color: white;
                margin-bottom: 1.5rem;
                box-shadow: 0 4px 20px rgba(0,0,0,0.12);
            }}
            .ld-hero h1 {{
                color: white !important;
                margin-bottom: 0.25rem;
                font-size: 2rem;
            }}
            .ld-hero p {{
                color: rgba(255,255,255,0.92);
                margin: 0;
                font-size: 1.05rem;
            }}

            /* ---- Metric cards ---- */
            div[data-testid="stMetric"] {{
                background: {COLORS["surface"]};
                border: 1px solid {COLORS["border"]};
                border-radius: 10px;
                padding: 0.75rem 1rem;
                box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            }}
            div[data-testid="stMetric"] label {{
                color: {COLORS["text_muted"]} !important;
            }}

            /* ---- Info / placeholder panels ---- */
            .ld-placeholder {{
                background: {COLORS["background"]};
                border: 1px dashed {COLORS["border"]};
                border-left: 4px solid {COLORS["accent"]};
                border-radius: 8px;
                padding: 1.25rem 1.5rem;
                margin: 1rem 0;
            }}

            /* ---- Page section headers ---- */
            .ld-section-title {{
                color: {COLORS["primary_dark"]};
                border-bottom: 2px solid {COLORS["primary"]};
                padding-bottom: 0.35rem;
                margin-top: 1.5rem;
                margin-bottom: 1rem;
            }}

            /* ---- Dark mode compatibility ---- */
            @media (prefers-color-scheme: dark) {{
                .block-container {{ background-color: #121212; }}
                div[data-testid="stMetric"] {{
                    background: #1e1e1e;
                    border-color: #333;
                }}
                .ld-placeholder {{
                    background: #1e1e1e;
                    border-color: #444;
                }}
            }}

            /* ---- Hide default Streamlit footer clutter ---- */
            #MainMenu {{ visibility: hidden; }}
            footer {{ visibility: hidden; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_header() -> None:
    """Render branded sidebar header with project context."""
    st.sidebar.markdown(f"## {APP_ICON} {APP_TITLE}")
    st.sidebar.caption(f"**Study area:** {STUDY_AREA}")
    st.sidebar.caption(f"**Period:** {STUDY_YEARS}")
    st.sidebar.divider()


def render_sidebar_navigation() -> None:
    """
    Render a custom quick-navigation block in the sidebar.

    Streamlit's built-in multipage navigation is used as the primary mechanism;
    this block provides descriptive links for clarity.
    """
    st.sidebar.markdown("### Navigate")
    for page in PAGES:
        label = f"{page['icon']} {page['label']}"
        try:
            st.sidebar.page_link(page["file"], label=label)
        except Exception:
            st.sidebar.markdown(f"- {label}")
    st.sidebar.divider()


def render_sidebar_footer() -> None:
    """Render sidebar footer with version and status."""
    st.sidebar.markdown("---")
    st.sidebar.caption("ML pipeline: **Complete**")
    st.sidebar.caption("App shell: **v0.1.0**")
    st.sidebar.caption("Prediction logic: *Pending approval*")


def render_page_header(title: str, subtitle: str = "") -> None:
    """Render a consistent page-level hero banner."""
    subtitle_html = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f"""
        <div class="ld-hero">
            <h1>{title}</h1>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_placeholder(message: str, title: str = "Coming Soon") -> None:
    """Render a styled placeholder panel for unimplemented sections."""
    st.markdown(
        f"""
        <div class="ld-placeholder">
            <strong>{title}</strong><br/>
            {message}
        </div>
        """,
        unsafe_allow_html=True,
    )


def setup_page(title: str, subtitle: str = "", *, show_nav: bool = True) -> None:
    """
    One-call page bootstrap: config, sidebar, and header.

    Parameters
    ----------
    title:
        Page heading displayed in the hero banner.
    subtitle:
        Optional descriptive text below the title.
    show_nav:
        Whether to render the custom sidebar navigation block.
    """
    configure_app()
    render_sidebar_header()
    if show_nav:
        render_sidebar_navigation()
    render_sidebar_footer()
    render_page_header(title, subtitle)
