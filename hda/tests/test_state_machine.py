"""State machine guards every TestRun transition. These tests pin the DAG."""

from __future__ import annotations

import pytest

from hda.domain import (
    ALLOWED_TRANSITIONS,
    IllegalTransition,
    TestState,
    is_terminal,
    transition,
)


def test_happy_path_transitions_in_order():
    path = [
        TestState.DISCOVERED,
        TestState.INGESTING,
        TestState.PREPROCESSED,
        TestState.STEADY_DETECTED,
        TestState.QC_RUN,
        TestState.ANALYZED,
        TestState.PERSISTED,
    ]
    for a, b in zip(path, path[1:]):
        assert transition(a, b) == b


def test_awaiting_metadata_branch():
    assert (
        transition(TestState.INGESTING, TestState.AWAITING_METADATA)
        == TestState.AWAITING_METADATA
    )
    assert (
        transition(TestState.AWAITING_METADATA, TestState.PREPROCESSED)
        == TestState.PREPROCESSED
    )


def test_qc_failed_blocks_normal_advance_but_allows_reanalyze():
    """Operator can re-open a QC_FAILED run via a manual steady window;
    normal forward transitions are still blocked."""
    with pytest.raises(IllegalTransition):
        transition(TestState.QC_FAILED, TestState.ANALYZED)
    assert (
        transition(TestState.QC_FAILED, TestState.STEADY_DETECTED)
        == TestState.STEADY_DETECTED
    )


def test_persisted_blocks_normal_advance_but_allows_reanalyze():
    """Same operator-driven escape hatch from PERSISTED."""
    with pytest.raises(IllegalTransition):
        transition(TestState.PERSISTED, TestState.ANALYZED)
    assert (
        transition(TestState.PERSISTED, TestState.STEADY_DETECTED)
        == TestState.STEADY_DETECTED
    )


def test_error_is_terminal():
    assert is_terminal(TestState.ERROR)
    assert ALLOWED_TRANSITIONS[TestState.ERROR] == frozenset()


def test_error_is_reachable_from_every_non_terminal_state():
    for s in TestState:
        if is_terminal(s):
            continue
        assert TestState.ERROR in ALLOWED_TRANSITIONS[s], (
            f"Every non-terminal state must reach ERROR; {s} cannot."
        )


def test_skip_steady_state_is_illegal():
    with pytest.raises(IllegalTransition):
        transition(TestState.PREPROCESSED, TestState.QC_RUN)


def test_backwards_transition_is_illegal():
    with pytest.raises(IllegalTransition):
        transition(TestState.ANALYZED, TestState.STEADY_DETECTED)


def test_only_error_is_truly_terminal():
    assert ALLOWED_TRANSITIONS[TestState.ERROR] == frozenset()
    # PERSISTED and QC_FAILED are operator-reachable via reanalysis,
    # not truly terminal.
    assert TestState.STEADY_DETECTED in ALLOWED_TRANSITIONS[TestState.PERSISTED]
    assert TestState.STEADY_DETECTED in ALLOWED_TRANSITIONS[TestState.QC_FAILED]


def test_review_can_be_resolved_either_way():
    assert (
        transition(TestState.NEEDS_REVIEW, TestState.ANALYZED) == TestState.ANALYZED
    )
    assert (
        transition(TestState.NEEDS_REVIEW, TestState.QC_FAILED)
        == TestState.QC_FAILED
    )
