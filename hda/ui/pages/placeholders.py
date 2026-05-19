"""Placeholder pages for navigation items not yet implemented.

Each stub shows the page title and a "coming soon" banner so the navigation
is complete and the user can see the full structure.
"""
from __future__ import annotations

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from hda.ui.pages.base import BasePage, InfoBanner
from hda.ui.style import TEXT_MUTED, SZ_BASE


class _PlaceholderPage(BasePage):
    def __init__(self, title: str, description: str, badge: str = "", parent=None):
        super().__init__(title, description, badge_text=badge, badge_kind="neutral", parent=parent)
        banner = InfoBanner(
            f"{title} is not yet implemented in the Qt desktop UI. "
            "Use the Streamlit app (streamlit run app.py) for this functionality.",
            "warning",
        )
        self.content_layout.addWidget(banner)
        self.content_layout.addStretch()


class SingleTestAnalysisPage(_PlaceholderPage):
    def __init__(self, parent=None):
        super().__init__(
            "Single Test Analysis",
            "Analyse an individual test CSV with QC, uncertainty and traceability",
            "P0",
            parent,
        )


class BatchAnalysisPage(_PlaceholderPage):
    def __init__(self, parent=None):
        super().__init__(
            "Batch Analysis",
            "Process multiple test files with a consistent configuration",
            "P1",
            parent,
        )


class CampaignAnalysisPage(_PlaceholderPage):
    def __init__(self, parent=None):
        super().__init__(
            "Campaign Analysis",
            "SPC control charts, capability indices and trend analysis across a campaign",
            "P0",
            parent,
        )


class SystemAnalysisPage(_PlaceholderPage):
    def __init__(self, parent=None):
        super().__init__(
            "System Analysis",
            "Cross-campaign system-level performance analysis",
            "P2",
            parent,
        )


class AnalysisToolsPage(_PlaceholderPage):
    def __init__(self, parent=None):
        super().__init__(
            "Analysis Tools",
            "Advanced anomaly detection, comparison and operating envelope tools",
            "P2",
            parent,
        )


class ConfigurationsPage(_PlaceholderPage):
    def __init__(self, parent=None):
        super().__init__(
            "Configurations",
            "Manage testbench hardware configurations and calibration data",
            "P1",
            parent,
        )
