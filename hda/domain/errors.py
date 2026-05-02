"""Typed exception hierarchy.

The Streamlit app mixes raise / return False / silent fallback. The new app
standardizes on this hierarchy: every error path raises a typed exception, and
the UI layer catches HDAError at the boundary to render to the user.
"""

from __future__ import annotations


class HDAError(Exception):
    """Base for all application errors."""


class ConfigError(HDAError):
    """Invalid or incomplete configuration / template / metadata."""


class DBError(HDAError):
    """Persistence layer failure (schema, migration, integrity, lock)."""


class IngestError(HDAError):
    """File parsing or preprocessing failure."""


class AnalysisError(HDAError):
    """Failure during steady-state, QC, or measurement calculation."""


class QCBlockingFailure(AnalysisError):
    """QC reported a blocking failure; analysis must not proceed."""


class IllegalTransition(HDAError):
    """A TestRun state transition was attempted that the state machine forbids."""
