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

            .stApp {{
                background: {COLORS["background"]};
            }}

            /* ---- Main container ---- */
            .block-container {{
                padding-top: 1.25rem;
                padding-bottom: 2.5rem;
                max-width: 1400px;
            }}

            /* ---- Sidebar ---- */
            section[data-testid="stSidebar"] {{
                background: linear-gradient(180deg, #5D4037 0%, #3E2723 100%);
                border-right: 1px solid {COLORS["border"]};
            }}
            section[data-testid="stSidebar"] * {{
                color: #F5F1E8 !important;
            }}
            section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{
                padding-top: 1rem;
            }}
            section[data-testid="stSidebar"] .stMarkdown h1,
            section[data-testid="stSidebar"] .stMarkdown h2,
            section[data-testid="stSidebar"] .stMarkdown h3 {{
                color: #F5F1E8 !important;
            }}
            section[data-testid="stSidebar"] hr {{
                border-color: rgba(245,241,232,0.25);
            }}
            section[data-testid="stSidebar"] a {{
                border-radius: 8px;
                margin: 0.12rem 0;
                transition: background 160ms ease, transform 160ms ease;
            }}
            section[data-testid="stSidebar"] a:hover {{
                background: rgba(166,124,82,0.18) !important;
                transform: translateX(2px);
            }}
            /* Active Nav Item */
            section[data-testid="stSidebar"] [data-testid="stSidebarNavItems"] li [data-testid="stSidebarNavLink"][aria-current="page"],
            section[data-testid="stSidebar"] [data-testid="stSidebarNavItems"] li div[data-testid="stSidebarNavLink"][aria-current="page"] {{
                background: #8B5E3C !important;
                color: white !important;
            }}
            .ld-sidebar-card {{
                background: rgba(255,255,255,0.05);
                border: 1px solid rgba(255,255,255,0.1);
                border-radius: 10px;
                padding: 0.95rem;
                margin-bottom: 0.9rem;
                box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);
            }}
            .ld-sidebar-card .label {{
                color: rgba(245,241,232,0.72) !important;
                font-size: 0.78rem;
                text-transform: uppercase;
                letter-spacing: 0.04em;
            }}
            .ld-sidebar-card .value {{
                color: #F5F1E8 !important;
                font-weight: 700;
                margin-top: 0.18rem;
            }}

            /* ---- Hero banner ---- */
            .ld-hero {{
                background: linear-gradient(135deg, #5D4037 0%, #3E2723 100%);
                padding: 2.2rem 2.6rem;
                border-radius: 14px;
                color: #F5F1E8;
                margin-bottom: 1.5rem;
                box-shadow: 0 16px 42px rgba(62, 39, 35, 0.30);
                border: 1px solid {COLORS["border"]};
                position: relative;
                overflow: hidden;
            }}
            .ld-hero::after {{
                content: "";
                position: absolute;
                inset: auto -4rem -7rem auto;
                width: 18rem;
                height: 18rem;
                border-radius: 50%;
                background: rgba(255,255,255,0.05);
            }}
            .ld-hero h1 {{
                color: #F5F1E8 !important;
                margin-bottom: 0.25rem;
                font-size: 2.15rem;
                letter-spacing: 0;
            }}
            .ld-hero p {{
                color: rgba(245,241,232,0.92);
                margin: 0;
                font-size: 1.05rem;
            }}

            /* ---- Metric cards ---- */
            div[data-testid="stMetric"] {{
                background: {COLORS["surface"]};
                border: 1px solid {COLORS["border"]};
                border-left: 4px solid {COLORS["primary"]};
                border-radius: 16px;
                padding: 0.9rem 1rem;
                box-shadow: 0 4px 12px rgba(139, 94, 60, 0.1);
            }}
            div[data-testid="stMetric"] label {{
                color: {COLORS["text_muted"]} !important;
                font-size: 0.82rem !important;
            }}
            div[data-testid="stMetricValue"] {{
                color: {COLORS["text"]};
                font-weight: 750;
            }}

            /* ---- Cards and panels ---- */
            div[data-testid="stVerticalBlockBorderWrapper"] {{
                background: {COLORS["surface"]};
                border: 1px solid {COLORS["border"]};
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(139, 94, 60, 0.1);
            }}
            .ld-card {{
                background: {COLORS["surface"]};
                border: 1px solid {COLORS["border"]};
                border-radius: 12px;
                padding: 1.15rem 1.25rem;
                box-shadow: 0 4px 12px rgba(139, 94, 60, 0.1);
                margin-bottom: 1rem;
            }}
            .ld-card-title {{
                color: {COLORS["text"]};
                font-weight: 750;
                font-size: 1.02rem;
                margin-bottom: 0.35rem;
            }}
            .ld-card-muted {{
                color: {COLORS["text_muted"]};
                font-size: 0.92rem;
            }}

            /* ---- Info / placeholder panels ---- */
            .ld-placeholder {{
                background: {COLORS["secondary_surface"]};
                border: 1px dashed {COLORS["border"]};
                border-left: 4px solid {COLORS["primary"]};
                border-radius: 8px;
                padding: 1.25rem 1.5rem;
                margin: 1rem 0;
            }}

            /* ---- Page section headers ---- */
            .ld-section-title {{
                color: {COLORS["text"]};
                border-bottom: 2px solid {COLORS["primary"]};
                padding-bottom: 0.45rem;
                margin-top: 1.35rem;
                margin-bottom: 1rem;
                font-weight: 750;
            }}
            .ld-section-divider {{
                display: flex;
                align-items: center;
                gap: 0.75rem;
                margin: 1.7rem 0 1rem;
                color: {COLORS["text"]};
                font-size: 1.12rem;
                font-weight: 760;
            }}
            .ld-section-divider::after {{
                content: "";
                flex: 1;
                height: 2px;
                background: {COLORS["primary"]};
            }}

            /* ---- Inputs, buttons, tabs ---- */
            .stButton > button,
            .stDownloadButton > button {{
                border-radius: 9px !important;
                border: 1px solid {COLORS["border"]} !important;
                background: #3E342E !important;
                color: #F5F1E8 !important;
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            }}
            .stButton > button:hover,
            .stDownloadButton > button:hover {{
                background: #5D4037 !important;
                color: white !important;
                border-color: {COLORS["primary"]} !important;
            }}

            .stButton > button[kind="primary"],
            .stDownloadButton > button[kind="primary"] {{
                background: #8B5E3C !important;
                border: none !important;
                color: white !important;
            }}
            .stButton > button[kind="primary"]:hover,
            .stDownloadButton > button[kind="primary"]:hover {{
                background: #A67C52 !important;
                color: white !important;
            }}
            
            div[data-baseweb="select"] > div,
            div[data-baseweb="input"] > div,
            div[data-baseweb="textarea"] > div,
            [data-testid="stFileUploader"] section {{
                border-radius: 10px !important;
                border-color: {COLORS["border"]} !important;
                background: {COLORS["surface"]} !important;
                color: {COLORS["text"]} !important;
            }}
            button[data-baseweb="tab"] {{
                border-radius: 999px;
                padding-inline: 1rem;
                color: {COLORS["text"]} !important;
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
    st.sidebar.markdown(
        f"""
        <div class="ld-sidebar-card">
            <div class="label">🌱 Environmental AI</div>
            <div class="value">{APP_ICON} {APP_TITLE}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.caption(f"**Study area:** {STUDY_AREA}")
    st.sidebar.caption(f"**Monitoring period:** {STUDY_YEARS}")
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
    st.sidebar.caption("🌱 ML pipeline: **Operational**")
    st.sidebar.caption("AI monitoring shell: **v0.1.0**")
    st.sidebar.caption("Earth observation stack: **Ready**")


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


def render_section_divider(title: str, icon: str = "🌱") -> None:
    """Render a consistent environmental dashboard section divider."""
    st.markdown(
        f'<div class="ld-section-divider"><span>{icon}</span><span>{title}</span></div>',
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
