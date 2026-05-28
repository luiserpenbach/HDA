"""Tests for UI input helpers."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from hda.ui.style import resolve_app_font_family


def test_resolve_app_font_family_returns_non_empty_string():
    family = resolve_app_font_family()
    assert isinstance(family, str)
    assert family.strip()
