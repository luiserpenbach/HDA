"""Pytest configuration: suppress false-positive collection warnings.

Several domain / repository class names start with ``Test`` (TestRun,
TestState, TestRunRepository) — they are domain types, not test classes.
``__test__ = False`` is the standard pytest hook to declare this.
"""

from __future__ import annotations

from hda.domain.types import TestMetadata, TestRun
from hda.persistence.repositories.test_runs import TestRunRepository


TestRun.__test__ = False  # type: ignore[attr-defined]
TestMetadata.__test__ = False  # type: ignore[attr-defined]
TestRunRepository.__test__ = False  # type: ignore[attr-defined]
