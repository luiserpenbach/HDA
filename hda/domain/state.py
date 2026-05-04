"""TestRun state machine.

Replaces the implicit, race-prone session_state flow of the Streamlit app with
an explicit DAG of legal transitions. Every state change goes through
``transition()``, which raises ``IllegalTransition`` for forbidden jumps.

Terminal states (PERSISTED, QC_FAILED, ERROR) accept no further transitions.
"""

from __future__ import annotations

from enum import Enum
from typing import Mapping, FrozenSet

from hda.domain.errors import IllegalTransition


class TestState(str, Enum):
    __test__ = False  # tell pytest this is not a test class

    DISCOVERED = "discovered"
    INGESTING = "ingesting"
    AWAITING_METADATA = "awaiting_metadata"
    PREPROCESSED = "preprocessed"
    STEADY_DETECTED = "steady_detected"
    QC_RUN = "qc_run"
    NEEDS_REVIEW = "needs_review"
    ANALYZED = "analyzed"
    PERSISTED = "persisted"
    QC_FAILED = "qc_failed"
    ERROR = "error"


_TERMINAL: FrozenSet[TestState] = frozenset({TestState.ERROR})

# PERSISTED, QC_FAILED, NEEDS_REVIEW are not terminal in the
# operator-action sense: an operator can re-open a finished test by
# choosing a manual steady-state window. The reanalyze entry point on
# AnalysisService is the only place that should drive these "back to
# STEADY_DETECTED" edges; the normal pipeline never reaches them.
ALLOWED_TRANSITIONS: Mapping[TestState, FrozenSet[TestState]] = {
    TestState.DISCOVERED: frozenset({TestState.INGESTING, TestState.ERROR}),
    TestState.INGESTING: frozenset(
        {TestState.AWAITING_METADATA, TestState.PREPROCESSED, TestState.ERROR}
    ),
    TestState.AWAITING_METADATA: frozenset({TestState.PREPROCESSED, TestState.ERROR}),
    TestState.PREPROCESSED: frozenset({TestState.STEADY_DETECTED, TestState.ERROR}),
    TestState.STEADY_DETECTED: frozenset({TestState.QC_RUN, TestState.ERROR}),
    TestState.QC_RUN: frozenset(
        {
            TestState.ANALYZED,
            TestState.NEEDS_REVIEW,
            TestState.QC_FAILED,
            TestState.ERROR,
        }
    ),
    TestState.NEEDS_REVIEW: frozenset(
        {
            TestState.ANALYZED,
            TestState.QC_FAILED,
            TestState.STEADY_DETECTED,
            TestState.ERROR,
        }
    ),
    TestState.ANALYZED: frozenset({TestState.PERSISTED, TestState.ERROR}),
    TestState.PERSISTED: frozenset({TestState.STEADY_DETECTED, TestState.ERROR}),
    TestState.QC_FAILED: frozenset({TestState.STEADY_DETECTED, TestState.ERROR}),
    TestState.ERROR: frozenset(),
}


def is_terminal(state: TestState) -> bool:
    return state in _TERMINAL


def transition(current: TestState, target: TestState) -> TestState:
    """Validate a state transition and return the target on success.

    Raises:
        IllegalTransition: if ``target`` is not in ``ALLOWED_TRANSITIONS[current]``.
    """
    if target not in ALLOWED_TRANSITIONS[current]:
        raise IllegalTransition(
            f"Cannot transition from {current.value} to {target.value}. "
            f"Allowed: {sorted(s.value for s in ALLOWED_TRANSITIONS[current])}"
        )
    return target
