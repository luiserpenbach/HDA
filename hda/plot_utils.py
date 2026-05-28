"""Shared helpers for time-series plot windows."""
from __future__ import annotations

from typing import Tuple


def default_steady_window(t_min: float, t_max: float) -> Tuple[float, float]:
    """Return the middle 50% of ``[t_min, t_max]`` for steady-state defaults."""
    span = t_max - t_min
    if span <= 0:
        return t_min, max(t_min + 0.001, t_max)
    return t_min + span * 0.25, t_min + span * 0.75
