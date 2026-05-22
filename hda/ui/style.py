"""Design tokens and Qt stylesheet strings for the HDA desktop UI.

Single theme: clean white/light-zinc throughout.
No dark sidebar — both the nav bar and content area share the same
light palette so there is no jarring contrast switch.

RULES:
  - Call content_stylesheet() ONCE on the central widget in main_window.py.
    Children inherit automatically; never re-apply it on sub-widgets.
  - The NavBar uses nav_stylesheet() on itself PLUS sets its background via
    QPalette (more reliable than stylesheet for the root background rect).
  - Never use dark colours in the nav bar.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Colour tokens
# ---------------------------------------------------------------------------

# Sidebar / nav bar
SIDEBAR_BG = "#f8fafc"          # slate-50 — barely off-white
SIDEBAR_BORDER = "#e2e8f0"      # subtle right border
SIDEBAR_TEXT = "#3f3f46"        # zinc-700 — readable but not harsh
SIDEBAR_TEXT_MUTED = "#71717a"  # zinc-500
SIDEBAR_HOVER_BG = "#f1f5f9"    # slate-100
SIDEBAR_ACTIVE_BG = "#eff6ff"   # blue-50
SIDEBAR_ACTIVE_TEXT = "#1d4ed8" # blue-700

# Content area
CONTENT_BG = "#ffffff"
CONTENT_SECONDARY_BG = "#f4f4f5"  # zinc-100
BORDER = "#e4e4e7"                # zinc-200
BORDER_DARK = "#d4d4d8"           # zinc-300

TEXT_PRIMARY = "#09090b"          # zinc-950
TEXT_SECONDARY = "#3f3f46"        # zinc-700
TEXT_MUTED = "#71717a"            # zinc-500

ACCENT_BLUE = "#3b82f6"
ACCENT_GREEN = "#16a34a"
ACCENT_AMBER = "#d97706"
ACCENT_RED = "#dc2626"

# ---------------------------------------------------------------------------
# Typography
# ---------------------------------------------------------------------------

FONT_FAMILY = "Inter, Segoe UI, system-ui, sans-serif"
SZ_XS   = "10px"
SZ_SM   = "11px"
SZ_BASE = "13px"
SZ_LG   = "15px"
SZ_XL   = "18px"
SZ_2XL  = "22px"

RADIUS    = "6px"
RADIUS_SM = "4px"


# ---------------------------------------------------------------------------
# Stylesheet builders
# ---------------------------------------------------------------------------

def nav_stylesheet() -> str:
    """Applied once to the NavBar widget. Background rect is handled via
    QPalette in nav_bar.py — this covers only child widget styling."""
    return f"""
    /* ── Labels ─────────────────────────────────────────────────────────── */
    QLabel {{
        background: transparent;
        color: {SIDEBAR_TEXT};
        font-family: {FONT_FAMILY};
        font-size: {SZ_BASE};
    }}
    QLabel#AppTitle {{
        color: {TEXT_PRIMARY};
        font-size: {SZ_LG};
        font-weight: 700;
    }}
    QLabel#AppSubtitle {{
        color: {SIDEBAR_TEXT_MUTED};
        font-size: {SZ_SM};
    }}
    QLabel#NavSectionLabel {{
        color: {SIDEBAR_TEXT_MUTED};
        font-size: {SZ_XS};
        font-weight: 700;
        letter-spacing: 0.08em;
    }}
    /* ── Nav item buttons ────────────────────────────────────────────────── */
    QPushButton#NavItem {{
        background: transparent;
        color: {SIDEBAR_TEXT};
        border: none;
        border-left: 2px solid transparent;
        border-radius: 0px;
        text-align: left;
        padding: 7px 10px 7px 10px;
        font-size: {SZ_BASE};
        font-family: {FONT_FAMILY};
    }}
    QPushButton#NavItem:hover {{
        background: {SIDEBAR_HOVER_BG};
        color: {TEXT_PRIMARY};
    }}
    QPushButton#NavItem[active="true"] {{
        background: {SIDEBAR_ACTIVE_BG};
        color: {SIDEBAR_ACTIVE_TEXT};
        font-weight: 600;
        border-left: 2px solid {ACCENT_BLUE};
    }}
    /* ── Context inputs ──────────────────────────────────────────────────── */
    QLineEdit#NavInput {{
        background: {CONTENT_BG};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER};
        border-radius: {RADIUS_SM};
        padding: 4px 7px;
        font-size: {SZ_SM};
        font-family: {FONT_FAMILY};
        selection-background-color: {ACCENT_BLUE};
    }}
    QLineEdit#NavInput:focus {{
        border-color: {ACCENT_BLUE};
    }}
    QPushButton#NavMicroBtn {{
        background: {CONTENT_SECONDARY_BG};
        color: {TEXT_SECONDARY};
        border: 1px solid {BORDER};
        border-radius: {RADIUS_SM};
        padding: 4px 6px;
        font-size: {SZ_SM};
        font-family: {FONT_FAMILY};
        min-width: 26px;
        max-width: 26px;
    }}
    QPushButton#NavMicroBtn:hover {{
        background: {BORDER};
        color: {TEXT_PRIMARY};
    }}
    QComboBox#NavCombo {{
        background: {CONTENT_BG};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER};
        border-radius: {RADIUS_SM};
        padding: 4px 7px;
        font-size: {SZ_SM};
        font-family: {FONT_FAMILY};
    }}
    QComboBox#NavCombo:focus {{
        border-color: {ACCENT_BLUE};
    }}
    QComboBox#NavCombo::drop-down {{
        border: none;
        width: 16px;
    }}
    QComboBox#NavCombo QAbstractItemView {{
        background: {CONTENT_BG};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER};
        selection-background-color: {CONTENT_SECONDARY_BG};
        selection-color: {TEXT_PRIMARY};
        padding: 2px;
        font-family: {FONT_FAMILY};
        font-size: {SZ_SM};
    }}
    QLabel#NavVersion {{
        color: {SIDEBAR_TEXT_MUTED};
        font-size: {SZ_XS};
    }}
    QFrame#NavDivider {{
        background: {BORDER};
    }}
    """


def content_stylesheet() -> str:
    """Applied once to HDAMainWindow's central widget. All content pages
    inherit this — do NOT re-apply on individual sub-widgets."""
    return f"""
    QWidget {{
        font-family: {FONT_FAMILY};
        font-size: {SZ_BASE};
        color: {TEXT_PRIMARY};
        background: {CONTENT_BG};
    }}
    /* ── Tabs ─────────────────────────────────────────────────────────── */
    QTabWidget::pane {{
        border: 1px solid {BORDER};
        border-top: none;
        background: {CONTENT_BG};
    }}
    QTabBar::tab {{
        background: {CONTENT_SECONDARY_BG};
        color: {TEXT_MUTED};
        border: 1px solid {BORDER};
        border-bottom: none;
        padding: 7px 18px;
        border-top-left-radius: {RADIUS_SM};
        border-top-right-radius: {RADIUS_SM};
        margin-right: 2px;
        font-size: {SZ_BASE};
    }}
    QTabBar::tab:selected {{
        background: {CONTENT_BG};
        color: {TEXT_PRIMARY};
        font-weight: 600;
        border-bottom: 2px solid {ACCENT_BLUE};
    }}
    QTabBar::tab:hover:!selected {{
        color: {TEXT_PRIMARY};
        background: {BORDER};
    }}
    /* ── Buttons ──────────────────────────────────────────────────────── */
    QPushButton {{
        background: {TEXT_PRIMARY};
        color: white;
        border: none;
        border-radius: {RADIUS_SM};
        padding: 7px 16px;
        font-size: {SZ_BASE};
        font-weight: 500;
    }}
    QPushButton:hover {{
        background: {TEXT_SECONDARY};
    }}
    QPushButton:pressed {{
        background: #18181b;
    }}
    QPushButton:disabled {{
        background: {CONTENT_SECONDARY_BG};
        color: {TEXT_MUTED};
    }}
    QPushButton[secondary="true"] {{
        background: {CONTENT_BG};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER};
    }}
    QPushButton[secondary="true"]:hover {{
        background: {CONTENT_SECONDARY_BG};
    }}
    QPushButton[secondary="true"]:disabled {{
        background: {CONTENT_SECONDARY_BG};
        color: {TEXT_MUTED};
        border-color: {BORDER};
    }}
    /* ── Inputs ───────────────────────────────────────────────────────── */
    QLineEdit, QTextEdit, QPlainTextEdit {{
        border: 1px solid {BORDER};
        border-radius: {RADIUS_SM};
        padding: 6px 10px;
        background: {CONTENT_BG};
        color: {TEXT_PRIMARY};
        font-size: {SZ_BASE};
        selection-background-color: {ACCENT_BLUE};
    }}
    QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
        border-color: {ACCENT_BLUE};
    }}
    QLineEdit:read-only {{
        background: {CONTENT_SECONDARY_BG};
        color: {TEXT_MUTED};
    }}
    QSpinBox, QDoubleSpinBox {{
        border: 1px solid {BORDER};
        border-radius: {RADIUS_SM};
        padding: 5px 8px;
        background: {CONTENT_BG};
        color: {TEXT_PRIMARY};
    }}
    QSpinBox:focus, QDoubleSpinBox:focus {{
        border-color: {ACCENT_BLUE};
    }}
    QDateEdit {{
        border: 1px solid {BORDER};
        border-radius: {RADIUS_SM};
        padding: 5px 8px;
        background: {CONTENT_BG};
        color: {TEXT_PRIMARY};
    }}
    QDateEdit:focus {{
        border-color: {ACCENT_BLUE};
    }}
    /* ── Combo boxes ──────────────────────────────────────────────────── */
    QComboBox {{
        border: 1px solid {BORDER};
        border-radius: {RADIUS_SM};
        padding: 6px 10px;
        background: {CONTENT_BG};
        color: {TEXT_PRIMARY};
    }}
    QComboBox:focus {{
        border-color: {ACCENT_BLUE};
    }}
    QComboBox::drop-down {{
        border: none;
        width: 20px;
    }}
    QComboBox QAbstractItemView {{
        background: {CONTENT_BG};
        border: 1px solid {BORDER};
        selection-background-color: {CONTENT_SECONDARY_BG};
        selection-color: {TEXT_PRIMARY};
        padding: 2px;
    }}
    /* ── Lists ────────────────────────────────────────────────────────── */
    QListWidget {{
        border: 1px solid {BORDER};
        border-radius: {RADIUS_SM};
        background: {CONTENT_BG};
        color: {TEXT_PRIMARY};
        outline: none;
    }}
    QListWidget::item {{
        padding: 6px 10px;
    }}
    QListWidget::item:selected {{
        background: {CONTENT_SECONDARY_BG};
        color: {TEXT_PRIMARY};
    }}
    QListWidget::item:hover:!selected {{
        background: #f9f9f9;
    }}
    /* ── Tables ───────────────────────────────────────────────────────── */
    QTableWidget, QTableView {{
        border: 1px solid {BORDER};
        border-radius: {RADIUS_SM};
        background: {CONTENT_BG};
        color: {TEXT_PRIMARY};
        gridline-color: {CONTENT_SECONDARY_BG};
        outline: none;
        alternate-background-color: #fafafa;
    }}
    QTableWidget::item, QTableView::item {{
        padding: 5px 8px;
    }}
    QTableWidget::item:selected, QTableView::item:selected {{
        background: {CONTENT_SECONDARY_BG};
        color: {TEXT_PRIMARY};
    }}
    QHeaderView::section {{
        background: {CONTENT_SECONDARY_BG};
        color: {TEXT_SECONDARY};
        border: none;
        border-right: 1px solid {BORDER};
        border-bottom: 1px solid {BORDER};
        padding: 5px 8px;
        font-weight: 600;
        font-size: {SZ_SM};
    }}
    /* ── Group boxes ──────────────────────────────────────────────────── */
    QGroupBox {{
        border: 1px solid {BORDER};
        border-radius: {RADIUS};
        margin-top: 10px;
        padding-top: 4px;
        font-size: {SZ_BASE};
        font-weight: 600;
        color: {TEXT_SECONDARY};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 4px;
        color: {TEXT_SECONDARY};
    }}
    /* ── Splitters ────────────────────────────────────────────────────── */
    QSplitter::handle {{
        background: {BORDER};
    }}
    QSplitter::handle:horizontal {{
        width: 1px;
    }}
    QSplitter::handle:vertical {{
        height: 1px;
    }}
    /* ── Scrollbars ───────────────────────────────────────────────────── */
    QScrollBar:vertical {{
        width: 7px;
        background: transparent;
        margin: 0;
    }}
    QScrollBar::handle:vertical {{
        background: {BORDER_DARK};
        border-radius: 3px;
        min-height: 20px;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
    QScrollBar:horizontal {{
        height: 7px;
        background: transparent;
        margin: 0;
    }}
    QScrollBar::handle:horizontal {{
        background: {BORDER_DARK};
        border-radius: 3px;
        min-width: 20px;
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
    /* ── Status bar ───────────────────────────────────────────────────── */
    QStatusBar {{
        background: {CONTENT_SECONDARY_BG};
        border-top: 1px solid {BORDER};
        color: {TEXT_MUTED};
        font-size: {SZ_SM};
    }}
    QStatusBar QLabel {{
        background: transparent;
        font-size: {SZ_SM};
    }}
    /* ── Check / radio ────────────────────────────────────────────────── */
    QCheckBox {{
        spacing: 6px;
        color: {TEXT_PRIMARY};
        background: transparent;
    }}
    QCheckBox::indicator {{
        width: 15px;
        height: 15px;
        border: 1px solid {BORDER_DARK};
        border-radius: 3px;
        background: {CONTENT_BG};
    }}
    QCheckBox::indicator:checked {{
        background: {TEXT_PRIMARY};
        border-color: {TEXT_PRIMARY};
    }}
    QRadioButton {{
        spacing: 6px;
        color: {TEXT_PRIMARY};
        background: transparent;
    }}
    /* ── Tool tips ────────────────────────────────────────────────────── */
    QToolTip {{
        background: {TEXT_PRIMARY};
        color: white;
        border: none;
        border-radius: {RADIUS_SM};
        padding: 4px 8px;
        font-size: {SZ_SM};
    }}
    """


def badge_style(kind: str) -> str:
    """Inline style for a status badge label."""
    colours = {
        "info":    (ACCENT_BLUE,  "#eff6ff", "#dbeafe"),
        "success": (ACCENT_GREEN, "#f0fdf4", "#dcfce7"),
        "warning": (ACCENT_AMBER, "#fffbeb", "#fef3c7"),
        "error":   (ACCENT_RED,   "#fef2f2", "#fee2e2"),
        "neutral": (TEXT_MUTED,   CONTENT_SECONDARY_BG, BORDER),
    }
    fg, bg, border = colours.get(kind, colours["neutral"])
    return (
        f"color: {fg}; background: {bg}; border: 1px solid {border}; "
        f"border-radius: {RADIUS_SM}; padding: 2px 8px; "
        f"font-size: {SZ_SM}; font-weight: 600;"
    )
