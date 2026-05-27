"""Design tokens and Qt stylesheet strings for the HDA desktop UI.

Theme: VS Code Dark+ inspired — flat, neutral, no gradients.

RULES:
  - Call content_stylesheet() ONCE on the central widget in main_window.py.
  - NavBar uses nav_stylesheet() + QPalette for its background rect.
  - Call apply_app_font() once in main.py.
  - Call configure_pyqtgraph() before creating plot widgets.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Colour tokens (VS Code Dark+)
# ---------------------------------------------------------------------------

# Sidebar / activity bar
SIDEBAR_BG = "#252526"
SIDEBAR_BORDER = "#3c3c3c"
SIDEBAR_TEXT = "#cccccc"
SIDEBAR_TEXT_MUTED = "#858585"
SIDEBAR_HOVER_BG = "#2a2d2e"
SIDEBAR_ACTIVE_BG = "#37373d"
SIDEBAR_ACTIVE_TEXT = "#ffffff"

# Editor / content
CONTENT_BG = "#1e1e1e"
CONTENT_SECONDARY_BG = "#252526"
INPUT_BG = "#3c3c3c"
BORDER = "#3c3c3c"
BORDER_SUBTLE = "#2b2b2b"

TEXT_PRIMARY = "#cccccc"
TEXT_SECONDARY = "#969696"
TEXT_MUTED = "#858585"

ACCENT_BLUE = "#007acc"
ACCENT_GREEN = "#4ec9b0"
ACCENT_AMBER = "#cca700"
ACCENT_RED = "#f44747"

SELECTION_BG = "#264f78"
BUTTON_PRIMARY = "#0e639c"
BUTTON_PRIMARY_HOVER = "#1177bb"
BUTTON_PRIMARY_PRESSED = "#094771"

# Charts (pyqtgraph)
PLOT_BG = "#1e1e1e"
PLOT_FG = "#cccccc"
PLOT_GRID = "#3c3c3c"

# ---------------------------------------------------------------------------
# Typography — prefer smooth system UI fonts (Segoe UI Variable on Win11)
# ---------------------------------------------------------------------------

APP_FONT_FAMILY = "Segoe UI"
APP_FONT_SIZE = 13
FONT_FAMILY = "'Segoe UI Variable', 'Segoe UI', system-ui, sans-serif"
FONT_MONO = "'Cascadia Mono', 'Cascadia Code', Consolas, 'Courier New', monospace"

SZ_XS = "11px"
SZ_SM = "12px"
SZ_BASE = "13px"
SZ_LG = "15px"
SZ_XL = "18px"
SZ_2XL = "20px"

RADIUS = "4px"
RADIUS_SM = "3px"

_FONT_CANDIDATES = (
    "Segoe UI Variable Display",
    "Segoe UI Variable Text",
    "Segoe UI Variable",
    "Inter",
    "Segoe UI",
    "Roboto",
)


def resolve_app_font_family() -> str:
    """Pick the best available sans-serif UI font on this system."""
    from PySide6.QtGui import QFontDatabase

    families = set(QFontDatabase.families())
    for name in _FONT_CANDIDATES:
        if name in families:
            return name
    return APP_FONT_FAMILY


def apply_app_font(app) -> None:
    """Set the global application font and refresh stylesheet font tokens."""
    global APP_FONT_FAMILY, FONT_FAMILY

    from PySide6.QtGui import QFont

    APP_FONT_FAMILY = resolve_app_font_family()
    FONT_FAMILY = (
        f"'{APP_FONT_FAMILY}', 'Segoe UI Variable', 'Segoe UI', system-ui, sans-serif"
    )

    font = QFont(APP_FONT_FAMILY, APP_FONT_SIZE)
    font.setStyleHint(QFont.StyleHint.SansSerif)
    font.setHintingPreference(QFont.HintingPreference.PreferDefaultHinting)
    font.setStyleStrategy(
        QFont.StyleStrategy.PreferAntialias | QFont.StyleStrategy.PreferQuality
    )
    app.setFont(font)


def configure_pyqtgraph() -> None:
    """Apply dark plot defaults. Safe to call if pyqtgraph is missing."""
    try:
        import pyqtgraph as pg

        pg.setConfigOptions(antialias=True, background=PLOT_BG, foreground=PLOT_FG)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Stylesheet builders
# ---------------------------------------------------------------------------

def nav_stylesheet() -> str:
    return f"""
    QLabel {{
        background: transparent;
        color: {SIDEBAR_TEXT};
        font-family: {FONT_FAMILY};
        font-size: {SZ_BASE};
    }}
    QLabel#AppTitle {{
        color: {SIDEBAR_ACTIVE_TEXT};
        font-size: {SZ_LG};
        font-weight: 600;
    }}
    QLabel#AppSubtitle {{
        color: {SIDEBAR_TEXT_MUTED};
        font-size: {SZ_SM};
    }}
    QLabel#NavSectionLabel {{
        color: {SIDEBAR_TEXT_MUTED};
        font-size: {SZ_XS};
        font-weight: 600;
        letter-spacing: 0.06em;
    }}
    QPushButton#NavItem {{
        background: transparent;
        color: {SIDEBAR_TEXT};
        border: none;
        border-left: 2px solid transparent;
        border-radius: 0px;
        text-align: left;
        padding: 6px 10px;
        font-size: {SZ_BASE};
        font-family: {FONT_FAMILY};
    }}
    QPushButton#NavItem:hover {{
        background: {SIDEBAR_HOVER_BG};
        color: {SIDEBAR_ACTIVE_TEXT};
    }}
    QPushButton#NavItem[active="true"] {{
        background: {SIDEBAR_ACTIVE_BG};
        color: {SIDEBAR_ACTIVE_TEXT};
        font-weight: 600;
        border-left: 2px solid {ACCENT_BLUE};
    }}
    QLineEdit#NavInput {{
        background: {INPUT_BG};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER};
        border-radius: {RADIUS_SM};
        padding: 4px 7px;
        font-size: {SZ_SM};
        font-family: {FONT_FAMILY};
        selection-background-color: {SELECTION_BG};
    }}
    QLineEdit#NavInput:focus {{
        border-color: {ACCENT_BLUE};
    }}
    QPushButton#NavMicroBtn {{
        background: {INPUT_BG};
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
        background: {SIDEBAR_HOVER_BG};
        color: {TEXT_PRIMARY};
        border-color: {ACCENT_BLUE};
    }}
    QComboBox#NavCombo {{
        background: {INPUT_BG};
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
        background: {CONTENT_SECONDARY_BG};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER};
        selection-background-color: {SELECTION_BG};
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
        background: {BORDER_SUBTLE};
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
        padding: 6px 16px;
        border-top-left-radius: {RADIUS_SM};
        border-top-right-radius: {RADIUS_SM};
        margin-right: 1px;
        font-size: {SZ_BASE};
    }}
    QTabBar::tab:selected {{
        background: {CONTENT_BG};
        color: {TEXT_PRIMARY};
        border-bottom: 2px solid {ACCENT_BLUE};
    }}
    QTabBar::tab:hover:!selected {{
        color: {TEXT_PRIMARY};
        background: {SIDEBAR_HOVER_BG};
    }}
    QPushButton {{
        background: {BUTTON_PRIMARY};
        color: #ffffff;
        border: none;
        border-radius: {RADIUS_SM};
        padding: 6px 14px;
        font-size: {SZ_BASE};
        font-weight: 500;
    }}
    QPushButton:hover {{
        background: {BUTTON_PRIMARY_HOVER};
    }}
    QPushButton:pressed {{
        background: {BUTTON_PRIMARY_PRESSED};
    }}
    QPushButton:disabled {{
        background: {INPUT_BG};
        color: {TEXT_MUTED};
    }}
    QPushButton[secondary="true"] {{
        background: {INPUT_BG};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER};
    }}
    QPushButton[secondary="true"]:hover {{
        background: {SIDEBAR_HOVER_BG};
        border-color: {ACCENT_BLUE};
    }}
    QPushButton[secondary="true"]:disabled {{
        background: {CONTENT_SECONDARY_BG};
        color: {TEXT_MUTED};
        border-color: {BORDER_SUBTLE};
    }}
    QLineEdit, QTextEdit, QPlainTextEdit {{
        border: 1px solid {BORDER};
        border-radius: {RADIUS_SM};
        padding: 5px 8px;
        background: {INPUT_BG};
        color: {TEXT_PRIMARY};
        font-size: {SZ_BASE};
        selection-background-color: {SELECTION_BG};
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
        padding: 4px 8px;
        background: {INPUT_BG};
        color: {TEXT_PRIMARY};
    }}
    QSpinBox:focus, QDoubleSpinBox:focus {{
        border-color: {ACCENT_BLUE};
    }}
    QDateEdit {{
        border: 1px solid {BORDER};
        border-radius: {RADIUS_SM};
        padding: 4px 8px;
        background: {INPUT_BG};
        color: {TEXT_PRIMARY};
    }}
    QDateEdit:focus {{
        border-color: {ACCENT_BLUE};
    }}
    QComboBox {{
        border: 1px solid {BORDER};
        border-radius: {RADIUS_SM};
        padding: 5px 8px;
        background: {INPUT_BG};
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
        background: {CONTENT_SECONDARY_BG};
        border: 1px solid {BORDER};
        selection-background-color: {SELECTION_BG};
        selection-color: {TEXT_PRIMARY};
        padding: 2px;
        color: {TEXT_PRIMARY};
    }}
    QListWidget {{
        border: 1px solid {BORDER};
        border-radius: {RADIUS_SM};
        background: {CONTENT_BG};
        color: {TEXT_PRIMARY};
        outline: none;
    }}
    QListWidget::item {{
        padding: 5px 8px;
    }}
    QListWidget::item:selected {{
        background: {SELECTION_BG};
        color: {TEXT_PRIMARY};
    }}
    QListWidget::item:hover:!selected {{
        background: {SIDEBAR_HOVER_BG};
    }}
    QTableWidget, QTableView {{
        border: 1px solid {BORDER};
        border-radius: {RADIUS_SM};
        background: {CONTENT_BG};
        color: {TEXT_PRIMARY};
        gridline-color: {BORDER_SUBTLE};
        outline: none;
        alternate-background-color: {CONTENT_SECONDARY_BG};
    }}
    QTableWidget::item, QTableView::item {{
        padding: 4px 8px;
    }}
    QTableWidget::item:selected, QTableView::item:selected {{
        background: {SELECTION_BG};
        color: {TEXT_PRIMARY};
    }}
    QHeaderView::section {{
        background: {CONTENT_SECONDARY_BG};
        color: {TEXT_SECONDARY};
        border: none;
        border-right: 1px solid {BORDER_SUBTLE};
        border-bottom: 1px solid {BORDER};
        padding: 5px 8px;
        font-weight: 600;
        font-size: {SZ_SM};
    }}
    QLabel#FormFieldLabel {{
        color: {TEXT_PRIMARY};
        font-size: {SZ_SM};
        font-weight: 500;
        background: transparent;
        padding: 0px;
        min-height: 16px;
    }}
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
    QSplitter::handle {{
        background: {BORDER};
    }}
    QSplitter::handle:horizontal {{
        width: 4px;
    }}
    QSplitter::handle:horizontal:hover {{
        background: {ACCENT_BLUE};
    }}
    QSplitter::handle:vertical {{
        height: 4px;
    }}
    QSplitter::handle:vertical:hover {{
        background: {ACCENT_BLUE};
    }}
    QScrollBar:vertical {{
        width: 10px;
        background: {CONTENT_BG};
        margin: 0;
    }}
    QScrollBar::handle:vertical {{
        background: {INPUT_BG};
        border-radius: 4px;
        min-height: 24px;
        margin: 2px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: #4e4e4e;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
    QScrollBar:horizontal {{
        height: 10px;
        background: {CONTENT_BG};
        margin: 0;
    }}
    QScrollBar::handle:horizontal {{
        background: {INPUT_BG};
        border-radius: 4px;
        min-width: 24px;
        margin: 2px;
    }}
    QScrollBar::handle:horizontal:hover {{
        background: #4e4e4e;
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
    QScrollArea {{
        background: transparent;
        border: none;
    }}
    QStatusBar {{
        background: {ACCENT_BLUE};
        border-top: 1px solid {BORDER};
        color: #ffffff;
        font-size: {SZ_SM};
    }}
    QStatusBar QLabel {{
        background: transparent;
        font-size: {SZ_SM};
        color: #ffffff;
    }}
    QCheckBox {{
        spacing: 6px;
        color: {TEXT_PRIMARY};
        background: transparent;
    }}
    QCheckBox::indicator {{
        width: 15px;
        height: 15px;
        border: 1px solid {BORDER};
        border-radius: 3px;
        background: {INPUT_BG};
    }}
    QCheckBox::indicator:checked {{
        background: {ACCENT_BLUE};
        border-color: {ACCENT_BLUE};
    }}
    QRadioButton {{
        spacing: 6px;
        color: {TEXT_PRIMARY};
        background: transparent;
    }}
    QToolTip {{
        background: {CONTENT_SECONDARY_BG};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER};
        border-radius: {RADIUS_SM};
        padding: 4px 8px;
        font-size: {SZ_SM};
    }}
    """


def badge_style(kind: str) -> str:
    colours = {
        "info":    (ACCENT_BLUE,  CONTENT_SECONDARY_BG, BORDER),
        "success": (ACCENT_GREEN, CONTENT_SECONDARY_BG, BORDER),
        "warning": (ACCENT_AMBER, CONTENT_SECONDARY_BG, BORDER),
        "error":   (ACCENT_RED,   CONTENT_SECONDARY_BG, BORDER),
        "neutral": (TEXT_MUTED,   CONTENT_SECONDARY_BG, BORDER),
    }
    fg, bg, border = colours.get(kind, colours["neutral"])
    return (
        f"color: {fg}; background: {bg}; border: 1px solid {border}; "
        f"border-radius: {RADIUS_SM}; padding: 2px 8px; "
        f"font-size: {SZ_SM}; font-weight: 600;"
    )
