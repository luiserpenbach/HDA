"""Design tokens and Qt stylesheet strings for the HDA desktop UI.

Palette: shadcn-inspired Zinc — matches the Streamlit app theme exactly.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Colour tokens
# ---------------------------------------------------------------------------

NAV_BG = "#18181b"         # zinc-900
NAV_TEXT = "#fafafa"       # zinc-50
NAV_TEXT_MUTED = "#a1a1aa" # zinc-400
NAV_HOVER = "#27272a"      # zinc-800
NAV_ACTIVE = "#3f3f46"     # zinc-700

CONTENT_BG = "#ffffff"
CONTENT_SECONDARY_BG = "#f4f4f5"  # zinc-100
BORDER = "#e4e4e7"         # zinc-200
BORDER_DARK = "#d4d4d8"   # zinc-300

TEXT_PRIMARY = "#09090b"   # zinc-950
TEXT_SECONDARY = "#3f3f46" # zinc-700
TEXT_MUTED = "#71717a"     # zinc-500

ACCENT_BLUE = "#3b82f6"
ACCENT_GREEN = "#16a34a"
ACCENT_AMBER = "#d97706"
ACCENT_RED = "#dc2626"

# ---------------------------------------------------------------------------
# Typography
# ---------------------------------------------------------------------------

FONT_FAMILY = "Inter, Segoe UI, system-ui, sans-serif"
SZ_XS = "10px"
SZ_SM = "11px"
SZ_BASE = "13px"
SZ_LG = "15px"
SZ_XL = "18px"
SZ_2XL = "22px"

RADIUS = "6px"
RADIUS_SM = "4px"


# ---------------------------------------------------------------------------
# Stylesheet builders
# ---------------------------------------------------------------------------

def nav_stylesheet() -> str:
    return f"""
    QWidget#NavBar {{
        background: {NAV_BG};
        border-right: 1px solid #27272a;
    }}
    QLabel#AppTitle {{
        color: {NAV_TEXT};
        font-size: {SZ_XL};
        font-weight: 700;
    }}
    QLabel#AppSubtitle {{
        color: {NAV_TEXT_MUTED};
        font-size: {SZ_SM};
    }}
    QLabel#NavSectionLabel {{
        color: {NAV_TEXT_MUTED};
        font-size: {SZ_XS};
        font-weight: 600;
        letter-spacing: 0.07em;
        padding: 0px 0px 2px 0px;
    }}
    QPushButton#NavItem {{
        background: transparent;
        color: {NAV_TEXT_MUTED};
        border: none;
        border-radius: {RADIUS_SM};
        text-align: left;
        padding: 7px 12px;
        font-size: {SZ_BASE};
    }}
    QPushButton#NavItem:hover {{
        background: {NAV_HOVER};
        color: {NAV_TEXT};
    }}
    QPushButton#NavItem[active="true"] {{
        background: {NAV_ACTIVE};
        color: {NAV_TEXT};
        font-weight: 600;
    }}
    QFrame#NavDivider {{
        color: #27272a;
        background: #27272a;
    }}
    QLabel#CtxLabel {{
        color: {NAV_TEXT_MUTED};
        font-size: {SZ_SM};
    }}
    QLineEdit#NavInput {{
        background: #27272a;
        color: {NAV_TEXT};
        border: 1px solid #3f3f46;
        border-radius: {RADIUS_SM};
        padding: 4px 6px;
        font-size: {SZ_SM};
        selection-background-color: {ACCENT_BLUE};
    }}
    QLineEdit#NavInput:focus {{
        border-color: {ACCENT_BLUE};
    }}
    QPushButton#NavMicroBtn {{
        background: #3f3f46;
        color: {NAV_TEXT};
        border: none;
        border-radius: {RADIUS_SM};
        padding: 4px 8px;
        font-size: {SZ_SM};
        min-width: 28px;
        max-width: 28px;
    }}
    QPushButton#NavMicroBtn:hover {{
        background: #52525b;
    }}
    QComboBox#NavCombo {{
        background: #27272a;
        color: {NAV_TEXT};
        border: 1px solid #3f3f46;
        border-radius: {RADIUS_SM};
        padding: 4px 6px;
        font-size: {SZ_SM};
    }}
    QComboBox#NavCombo:focus {{
        border-color: {ACCENT_BLUE};
    }}
    QComboBox#NavCombo::drop-down {{
        border: none;
        width: 18px;
    }}
    QComboBox#NavCombo QAbstractItemView {{
        background: #27272a;
        color: {NAV_TEXT};
        border: 1px solid #3f3f46;
        selection-background-color: {NAV_ACTIVE};
        selection-color: {NAV_TEXT};
        padding: 2px;
    }}
    QLabel#NavVersion {{
        color: #52525b;
        font-size: {SZ_XS};
    }}
    """


def content_stylesheet() -> str:
    return f"""
    QWidget {{
        font-family: {FONT_FAMILY};
        font-size: {SZ_BASE};
        color: {TEXT_PRIMARY};
        background: {CONTENT_BG};
    }}
    /* ── Tabs ────────────────────────────────────────────────────────── */
    QTabWidget::pane {{
        border: 1px solid {BORDER};
        border-radius: 0px {RADIUS} {RADIUS} {RADIUS};
        background: {CONTENT_BG};
        top: -1px;
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
    /* ── Buttons ─────────────────────────────────────────────────────── */
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
        background: {NAV_BG};
    }}
    QPushButton:disabled {{
        background: {BORDER};
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
    QPushButton[danger="true"] {{
        background: {ACCENT_RED};
        color: white;
    }}
    QPushButton[danger="true"]:hover {{
        background: #b91c1c;
    }}
    /* ── Inputs ──────────────────────────────────────────────────────── */
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
        outline: none;
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
        font-size: {SZ_BASE};
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
        font-size: {SZ_BASE};
    }}
    QDateEdit:focus {{
        border-color: {ACCENT_BLUE};
    }}
    /* ── Combo boxes ─────────────────────────────────────────────────── */
    QComboBox {{
        border: 1px solid {BORDER};
        border-radius: {RADIUS_SM};
        padding: 6px 10px;
        background: {CONTENT_BG};
        color: {TEXT_PRIMARY};
        font-size: {SZ_BASE};
    }}
    QComboBox:focus {{
        border-color: {ACCENT_BLUE};
    }}
    QComboBox::drop-down {{
        border: none;
        width: 22px;
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
        font-size: {SZ_BASE};
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
        font-size: {SZ_BASE};
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
    /* ── Group boxes ─────────────────────────────────────────────────── */
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
    /* ── Splitters ───────────────────────────────────────────────────── */
    QSplitter::handle {{
        background: {BORDER};
    }}
    QSplitter::handle:horizontal {{
        width: 1px;
    }}
    QSplitter::handle:vertical {{
        height: 1px;
    }}
    /* ── Scrollbars ──────────────────────────────────────────────────── */
    QScrollBar:vertical {{
        width: 8px;
        background: transparent;
        margin: 0;
    }}
    QScrollBar::handle:vertical {{
        background: {BORDER};
        border-radius: 4px;
        min-height: 24px;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}
    QScrollBar:horizontal {{
        height: 8px;
        background: transparent;
        margin: 0;
    }}
    QScrollBar::handle:horizontal {{
        background: {BORDER};
        border-radius: 4px;
        min-width: 24px;
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0;
    }}
    /* ── Status bar ──────────────────────────────────────────────────── */
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
    /* ── Check boxes ─────────────────────────────────────────────────── */
    QCheckBox {{
        spacing: 6px;
        font-size: {SZ_BASE};
        color: {TEXT_PRIMARY};
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
    /* ── Radio buttons ───────────────────────────────────────────────── */
    QRadioButton {{
        spacing: 6px;
        font-size: {SZ_BASE};
        color: {TEXT_PRIMARY};
    }}
    /* ── Frames used as cards ────────────────────────────────────────── */
    QFrame[card="true"] {{
        border: 1px solid {BORDER};
        border-radius: {RADIUS};
        background: {CONTENT_BG};
    }}
    /* ── Tool tips ───────────────────────────────────────────────────── */
    QToolTip {{
        background: {NAV_BG};
        color: {NAV_TEXT};
        border: 1px solid #3f3f46;
        border-radius: {RADIUS_SM};
        padding: 4px 8px;
        font-size: {SZ_SM};
    }}
    """


# ---------------------------------------------------------------------------
# Convenience label helpers (set via setProperty + polish)
# ---------------------------------------------------------------------------

def badge_style(kind: str) -> str:
    """Return inline style for a badge label. kind: info | success | warning | error."""
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
