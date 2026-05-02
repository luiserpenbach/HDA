"""Analysis service interface.

Drives a TestRun through STEADY_DETECTED → QC_RUN → ANALYZED → PERSISTED on a
worker thread. Emits state changes via a callback (the Qt layer adapts these
to signals). Concrete implementation lands with the plugin-aware analysis
pipeline in a follow-up commit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol

from hda.domain.types import SteadyWindow, TestRun


StateChangeCallback = Callable[[TestRun], None]


@dataclass(frozen=True, slots=True)
class AnalysisRequest:
    test_run_id: str
    steady_window: Optional[SteadyWindow] = None
    operator_override: bool = False


class AnalysisService(Protocol):
    def submit(
        self,
        request: AnalysisRequest,
        on_state_change: Optional[StateChangeCallback] = None,
    ) -> None:
        """Schedule analysis on a worker thread."""
        ...
