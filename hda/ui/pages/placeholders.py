"""Placeholder pages for nav items not yet ported to Qt."""
from __future__ import annotations

from hda.ui.pages.base import BasePage, InfoBanner


class _PlaceholderPage(BasePage):
    def __init__(self, title: str, description: str, parent=None):
        super().__init__(title, description, parent=parent)
        banner = InfoBanner(
            f"'{title}' is not yet implemented in the Qt desktop UI. "
            "Use the Streamlit app (streamlit run app.py) for this functionality.",
            "warning",
        )
        self.content_layout.addWidget(banner)
        self.content_layout.addStretch()


class BatchAnalysisPage(_PlaceholderPage):
    def __init__(self, parent=None):
        super().__init__(
            "Batch Analysis",
            "Process multiple test files with a consistent configuration in parallel",
            parent,
        )


class SystemAnalysisPage(_PlaceholderPage):
    def __init__(self, parent=None):
        super().__init__(
            "System Analysis",
            "Cross-campaign system-level performance analysis",
            parent,
        )
