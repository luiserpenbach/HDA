"""Tests for plot helper utilities."""
from hda.plot_utils import default_steady_window


def test_default_steady_window_from_zero():
    lo, hi = default_steady_window(0.0, 100.0)
    assert lo == 25.0
    assert hi == 75.0


def test_default_steady_window_with_offset():
    lo, hi = default_steady_window(25.0, 75.0)
    assert lo == 37.5
    assert hi == 62.5


def test_default_steady_window_degenerate_span():
    lo, hi = default_steady_window(10.0, 10.0)
    assert lo == 10.0
    assert hi == 10.001
