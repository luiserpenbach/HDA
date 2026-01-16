"""
HDA Analysis Plugins

This package contains modular analysis plugins for different test types.

Each plugin implements the AnalysisPlugin protocol and registers itself
with the PluginRegistry during import.

Available plugins are automatically discovered from:
1. This directory (local development)
2. Entry points (pip-installable external plugins)
"""

__all__ = []
