"""
Shared Styling Module for Hopper Data Studio
==============================================
Modern, professional CSS styling inspired by shadcn/ui design principles.

This module provides:
- Typography system
- Color palette (zinc-based)
- Component styles (buttons, cards, inputs, badges)
- Layout utilities
- Animation and transitions

Usage:
    import streamlit as st
    from pages._shared_styles import apply_custom_styles

    apply_custom_styles()
"""

import streamlit as st


# Color palette based on shadcn/ui zinc theme
COLORS = {
    # Zinc palette
    'zinc-50': '#fafafa',
    'zinc-100': '#f4f4f5',
    'zinc-200': '#e4e4e7',
    'zinc-300': '#d4d4d8',
    'zinc-400': '#a1a1aa',
    'zinc-500': '#71717a',
    'zinc-600': '#52525b',
    'zinc-700': '#3f3f46',
    'zinc-800': '#27272a',
    'zinc-900': '#18181b',
    'zinc-950': '#09090b',

    # Accent colors
    'primary': '#18181b',       # zinc-900
    'primary-foreground': '#fafafa',  # zinc-50

    # Status colors
    'success': '#16a34a',       # green-600
    'success-bg': '#dcfce7',    # green-100
    'warning': '#ca8a04',       # yellow-600
    'warning-bg': '#fef9c3',    # yellow-100
    'error': '#dc2626',         # red-600
    'error-bg': '#fee2e2',      # red-100
    'info': '#2563eb',          # blue-600
    'info-bg': '#dbeafe',       # blue-100

    # Borders and backgrounds
    'border': '#e4e4e7',        # zinc-200
    'input': '#ffffff',
    'ring': '#18181b',
    'background': '#ffffff',
    'foreground': '#09090b',
    'muted': '#f4f4f5',
    'muted-foreground': '#71717a',
    'card': '#ffffff',
    'card-foreground': '#09090b',
}


def apply_custom_styles():
    """
    Apply custom CSS styles to the Streamlit app.
    Call this at the top of every page for consistent styling.
    """
    st.markdown(f"""
    <style>
    /* ============================================
       GLOBAL RESETS & BASE STYLES
       ============================================ */

    /* Import better fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Root variables */
    :root {{
        --font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        --radius: 0.5rem;
        --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
        --shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1);
        --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
        --shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1);
    }}

    /* Override Streamlit defaults */
    .stApp {{
        font-family: var(--font-family);
        background-color: {COLORS['background']};
    }}

    /* Main content area */
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }}

    /* Remove default padding */
    .main {{
        padding: 0;
    }}

    /* ============================================
       TYPOGRAPHY
       ============================================ */

    h1, h2, h3, h4, h5, h6 {{
        font-family: var(--font-family);
        font-weight: 600;
        letter-spacing: -0.025em;
        color: {COLORS['foreground']};
    }}

    h1 {{
        font-size: 2.25rem;
        line-height: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }}

    h2 {{
        font-size: 1.875rem;
        line-height: 2.25rem;
        margin-bottom: 0.875rem;
    }}

    h3 {{
        font-size: 1.5rem;
        line-height: 2rem;
        margin-bottom: 0.75rem;
    }}

    h4 {{
        font-size: 1.25rem;
        line-height: 1.75rem;
        margin-bottom: 0.625rem;
    }}

    p, .stMarkdown {{
        font-size: 0.875rem;
        line-height: 1.5rem;
        color: {COLORS['muted-foreground']};
    }}

    /* ============================================
       BUTTONS
       ============================================ */

    /* Primary buttons */
    .stButton > button {{
        background-color: {COLORS['primary']};
        color: {COLORS['primary-foreground']};
        border: none;
        border-radius: var(--radius);
        padding: 0.5rem 1rem;
        font-size: 0.875rem;
        font-weight: 500;
        font-family: var(--font-family);
        transition: all 0.2s ease;
        box-shadow: var(--shadow-sm);
        height: 2.5rem;
    }}

    .stButton > button:hover {{
        background-color: {COLORS['zinc-800']};
        box-shadow: var(--shadow);
        transform: translateY(-1px);
    }}

    .stButton > button:active {{
        transform: translateY(0);
        box-shadow: var(--shadow-sm);
    }}

    .stButton > button:focus {{
        outline: none;
        box-shadow: 0 0 0 2px {COLORS['ring']}, var(--shadow);
    }}

    /* Download buttons */
    .stDownloadButton > button {{
        background-color: {COLORS['zinc-100']};
        color: {COLORS['foreground']};
        border: 1px solid {COLORS['border']};
        border-radius: var(--radius);
        padding: 0.5rem 1rem;
        font-size: 0.875rem;
        font-weight: 500;
        transition: all 0.2s ease;
        box-shadow: var(--shadow-sm);
    }}

    .stDownloadButton > button:hover {{
        background-color: {COLORS['zinc-200']};
        border-color: {COLORS['zinc-300']};
        box-shadow: var(--shadow);
    }}

    /* ============================================
       CARDS & CONTAINERS
       ============================================ */

    /* Card base style */
    .card {{
        background-color: {COLORS['card']};
        border: 1px solid {COLORS['border']};
        border-radius: var(--radius);
        padding: 1.5rem;
        box-shadow: var(--shadow-sm);
        transition: box-shadow 0.2s ease;
    }}

    .card:hover {{
        box-shadow: var(--shadow-md);
    }}

    /* Card variants */
    .card-elevated {{
        background-color: {COLORS['card']};
        border: none;
        border-radius: var(--radius);
        padding: 1.5rem;
        box-shadow: var(--shadow-lg);
    }}

    .card-muted {{
        background-color: {COLORS['muted']};
        border: 1px solid {COLORS['border']};
        border-radius: var(--radius);
        padding: 1.5rem;
    }}

    /* Feature card */
    .feature-card {{
        background-color: {COLORS['card']};
        border: 1px solid {COLORS['border']};
        border-radius: var(--radius);
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.2s ease;
        box-shadow: var(--shadow-sm);
    }}

    .feature-card:hover {{
        border-color: {COLORS['zinc-300']};
        box-shadow: var(--shadow-md);
        transform: translateY(-2px);
    }}

    .feature-card h4 {{
        margin: 0 0 0.5rem 0;
        font-size: 1.125rem;
        font-weight: 600;
        color: {COLORS['foreground']};
    }}

    .feature-card p {{
        margin: 0;
        font-size: 0.875rem;
        line-height: 1.5;
        color: {COLORS['muted-foreground']};
    }}

    /* Metric card */
    .metric-card {{
        background: linear-gradient(135deg, {COLORS['zinc-900']} 0%, {COLORS['zinc-700']} 100%);
        border-radius: var(--radius);
        padding: 1.5rem;
        color: white;
        box-shadow: var(--shadow-lg);
        transition: transform 0.2s ease;
    }}

    .metric-card:hover {{
        transform: scale(1.02);
    }}

    .metric-card h3 {{
        margin: 0 0 0.5rem 0;
        font-size: 2rem;
        font-weight: 700;
        color: white;
    }}

    .metric-card p {{
        margin: 0;
        font-size: 0.875rem;
        color: {COLORS['zinc-300']};
    }}

    /* ============================================
       INPUTS & FORMS
       ============================================ */

    /* Text inputs */
    .stTextInput > div > div > input {{
        border: 1px solid {COLORS['border']};
        border-radius: var(--radius);
        padding: 0.5rem 0.75rem;
        font-size: 0.875rem;
        font-family: var(--font-family);
        background-color: {COLORS['input']};
        color: {COLORS['foreground']};
        transition: all 0.2s ease;
    }}

    .stTextInput > div > div > input:focus {{
        outline: none;
        border-color: {COLORS['ring']};
        box-shadow: 0 0 0 2px {COLORS['ring']}33;
    }}

    /* Number inputs */
    .stNumberInput > div > div > input {{
        border: 1px solid {COLORS['border']};
        border-radius: var(--radius);
        padding: 0.5rem 0.75rem;
        font-size: 0.875rem;
        font-family: var(--font-family);
        background-color: {COLORS['input']};
        transition: all 0.2s ease;
    }}

    .stNumberInput > div > div > input:focus {{
        outline: none;
        border-color: {COLORS['ring']};
        box-shadow: 0 0 0 2px {COLORS['ring']}33;
    }}

    /* Select boxes */
    .stSelectbox > div > div {{
        border: 1px solid {COLORS['border']};
        border-radius: var(--radius);
        background-color: {COLORS['input']};
        transition: all 0.2s ease;
    }}

    .stSelectbox > div > div:focus-within {{
        border-color: {COLORS['ring']};
        box-shadow: 0 0 0 2px {COLORS['ring']}33;
    }}

    /* Text areas */
    .stTextArea > div > div > textarea {{
        border: 1px solid {COLORS['border']};
        border-radius: var(--radius);
        padding: 0.75rem;
        font-size: 0.875rem;
        font-family: var(--font-family);
        background-color: {COLORS['input']};
        color: {COLORS['foreground']};
        transition: all 0.2s ease;
    }}

    .stTextArea > div > div > textarea:focus {{
        outline: none;
        border-color: {COLORS['ring']};
        box-shadow: 0 0 0 2px {COLORS['ring']}33;
    }}

    /* ============================================
       BADGES & LABELS
       ============================================ */

    .badge {{
        display: inline-flex;
        align-items: center;
        border-radius: 9999px;
        padding: 0.25rem 0.75rem;
        font-size: 0.75rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }}

    .badge-default {{
        background-color: {COLORS['zinc-100']};
        color: {COLORS['zinc-900']};
        border: 1px solid {COLORS['border']};
    }}

    .badge-success {{
        background-color: {COLORS['success-bg']};
        color: {COLORS['success']};
    }}

    .badge-warning {{
        background-color: {COLORS['warning-bg']};
        color: {COLORS['warning']};
    }}

    .badge-error {{
        background-color: {COLORS['error-bg']};
        color: {COLORS['error']};
    }}

    .badge-info {{
        background-color: {COLORS['info-bg']};
        color: {COLORS['info']};
    }}

    /* ============================================
       ALERTS & NOTIFICATIONS
       ============================================ */

    /* Success messages */
    .stSuccess {{
        background-color: {COLORS['success-bg']};
        border-left: 4px solid {COLORS['success']};
        border-radius: var(--radius);
        padding: 1rem;
        color: {COLORS['success']};
    }}

    /* Warning messages */
    .stWarning {{
        background-color: {COLORS['warning-bg']};
        border-left: 4px solid {COLORS['warning']};
        border-radius: var(--radius);
        padding: 1rem;
        color: {COLORS['warning']};
    }}

    /* Error messages */
    .stError {{
        background-color: {COLORS['error-bg']};
        border-left: 4px solid {COLORS['error']};
        border-radius: var(--radius);
        padding: 1rem;
        color: {COLORS['error']};
    }}

    /* Info messages */
    .stInfo {{
        background-color: {COLORS['info-bg']};
        border-left: 4px solid {COLORS['info']};
        border-radius: var(--radius);
        padding: 1rem;
        color: {COLORS['info']};
    }}

    /* ============================================
       EXPANDERS
       ============================================ */

    .streamlit-expanderHeader {{
        background-color: {COLORS['muted']};
        border: 1px solid {COLORS['border']};
        border-radius: var(--radius);
        padding: 0.75rem 1rem;
        font-weight: 500;
        color: {COLORS['foreground']};
        transition: all 0.2s ease;
    }}

    .streamlit-expanderHeader:hover {{
        background-color: {COLORS['zinc-200']};
        border-color: {COLORS['zinc-300']};
    }}

    .streamlit-expanderContent {{
        border: 1px solid {COLORS['border']};
        border-top: none;
        border-radius: 0 0 var(--radius) var(--radius);
        padding: 1rem;
        background-color: {COLORS['card']};
    }}

    /* ============================================
       DATAFRAMES & TABLES
       ============================================ */

    .dataframe {{
        border: 1px solid {COLORS['border']};
        border-radius: var(--radius);
        overflow: hidden;
        font-size: 0.875rem;
    }}

    .dataframe thead th {{
        background-color: {COLORS['muted']};
        color: {COLORS['foreground']};
        font-weight: 600;
        padding: 0.75rem;
        border-bottom: 2px solid {COLORS['border']};
    }}

    .dataframe tbody td {{
        padding: 0.75rem;
        border-bottom: 1px solid {COLORS['border']};
    }}

    .dataframe tbody tr:hover {{
        background-color: {COLORS['zinc-50']};
    }}

    /* ============================================
       SIDEBAR
       ============================================ */

    .css-1d391kg {{
        background-color: {COLORS['zinc-50']};
        border-right: 1px solid {COLORS['border']};
    }}

    section[data-testid="stSidebar"] {{
        background-color: {COLORS['zinc-50']};
        border-right: 1px solid {COLORS['border']};
    }}

    section[data-testid="stSidebar"] .block-container {{
        padding-top: 2rem;
    }}

    /* Sidebar headers */
    section[data-testid="stSidebar"] h2 {{
        font-size: 0.875rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: {COLORS['zinc-600']};
        margin-bottom: 0.75rem;
    }}

    /* ============================================
       TABS
       ============================================ */

    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.5rem;
        border-bottom: 1px solid {COLORS['border']};
    }}

    .stTabs [data-baseweb="tab"] {{
        border-radius: var(--radius) var(--radius) 0 0;
        padding: 0.5rem 1rem;
        font-weight: 500;
        color: {COLORS['muted-foreground']};
        background-color: transparent;
        border: none;
        transition: all 0.2s ease;
    }}

    .stTabs [data-baseweb="tab"]:hover {{
        color: {COLORS['foreground']};
        background-color: {COLORS['zinc-100']};
    }}

    .stTabs [aria-selected="true"] {{
        color: {COLORS['foreground']};
        background-color: {COLORS['card']};
        border-bottom: 2px solid {COLORS['primary']};
    }}

    /* ============================================
       DIVIDERS
       ============================================ */

    hr {{
        border: none;
        border-top: 1px solid {COLORS['border']};
        margin: 2rem 0;
    }}

    /* ============================================
       UTILITY CLASSES
       ============================================ */

    .text-muted {{
        color: {COLORS['muted-foreground']};
    }}

    .text-sm {{
        font-size: 0.875rem;
        line-height: 1.25rem;
    }}

    .text-xs {{
        font-size: 0.75rem;
        line-height: 1rem;
    }}

    .font-semibold {{
        font-weight: 600;
    }}

    .font-bold {{
        font-weight: 700;
    }}

    /* Spacing utilities */
    .mb-1 {{ margin-bottom: 0.25rem; }}
    .mb-2 {{ margin-bottom: 0.5rem; }}
    .mb-3 {{ margin-bottom: 0.75rem; }}
    .mb-4 {{ margin-bottom: 1rem; }}
    .mb-6 {{ margin-bottom: 1.5rem; }}
    .mb-8 {{ margin-bottom: 2rem; }}

    .mt-1 {{ margin-top: 0.25rem; }}
    .mt-2 {{ margin-top: 0.5rem; }}
    .mt-3 {{ margin-top: 0.75rem; }}
    .mt-4 {{ margin-top: 1rem; }}
    .mt-6 {{ margin-top: 1.5rem; }}
    .mt-8 {{ margin-top: 2rem; }}

    /* ============================================
       ANIMATIONS
       ============================================ */

    @keyframes fadeIn {{
        from {{
            opacity: 0;
            transform: translateY(10px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}

    .animate-fade-in {{
        animation: fadeIn 0.3s ease-out;
    }}

    /* ============================================
       LOADING STATES
       ============================================ */

    .stSpinner > div {{
        border-color: {COLORS['primary']};
    }}

    /* ============================================
       CUSTOM COMPONENTS
       ============================================ */

    /* Header with accent */
    .page-header {{
        border-bottom: 1px solid {COLORS['border']};
        padding-bottom: 1rem;
        margin-bottom: 2rem;
    }}

    .page-header h1 {{
        margin: 0;
        font-size: 2.25rem;
        font-weight: 700;
        color: {COLORS['foreground']};
    }}

    .page-header p {{
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
        color: {COLORS['muted-foreground']};
    }}

    /* Status indicator */
    .status-indicator {{
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }}

    .status-dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
    }}

    .status-dot.success {{ background-color: {COLORS['success']}; }}
    .status-dot.warning {{ background-color: {COLORS['warning']}; }}
    .status-dot.error {{ background-color: {COLORS['error']}; }}
    .status-dot.info {{ background-color: {COLORS['info']}; }}

    /* ============================================
       RESPONSIVE DESIGN
       ============================================ */

    @media (max-width: 768px) {{
        .main .block-container {{
            padding: 1rem;
        }}

        h1 {{
            font-size: 1.875rem;
            line-height: 2.25rem;
        }}

        h2 {{
            font-size: 1.5rem;
            line-height: 2rem;
        }}
    }}

    </style>
    """, unsafe_allow_html=True)


def render_page_header(title: str, description: str = "", badge_text: str = "", badge_type: str = "default"):
    """
    Render a professional page header with optional badge.

    Args:
        title: Page title
        description: Optional description
        badge_text: Optional badge text (e.g., "Beta", "P0")
        badge_type: Badge type (default, success, warning, error, info)
    """
    badge_html = ""
    if badge_text:
        badge_html = f'<span class="badge badge-{badge_type}">{badge_text}</span>'

    st.markdown(f"""
    <div class="page-header animate-fade-in">
        <h1>{title} {badge_html}</h1>
        {f'<p>{description}</p>' if description else ''}
    </div>
    """, unsafe_allow_html=True)


def render_metric_card(title: str, value: str, subtitle: str = ""):
    """
    Render a metric card with gradient background.

    Args:
        title: Metric title
        value: Metric value (formatted string)
        subtitle: Optional subtitle
    """
    st.markdown(f"""
    <div class="metric-card animate-fade-in">
        <p style="margin: 0; font-size: 0.875rem; color: #d4d4d8;">{title}</p>
        <h3>{value}</h3>
        {f'<p style="margin: 0; font-size: 0.75rem; color: #a1a1aa;">{subtitle}</p>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)


def render_status_badge(text: str, status: str = "default"):
    """
    Render a status badge.

    Args:
        text: Badge text
        status: Badge status (default, success, warning, error, info)
    """
    st.markdown(f'<span class="badge badge-{status}">{text}</span>', unsafe_allow_html=True)


def render_feature_card(title: str, description: str, icon: str = ""):
    """
    Render a feature card with hover effects.

    Args:
        title: Feature title
        description: Feature description
        icon: Optional emoji icon
    """
    icon_html = f'<span style="font-size: 1.5rem; margin-right: 0.5rem;">{icon}</span>' if icon else ''

    st.markdown(f"""
    <div class="feature-card animate-fade-in">
        <h4>{icon_html}{title}</h4>
        <p>{description}</p>
    </div>
    """, unsafe_allow_html=True)
