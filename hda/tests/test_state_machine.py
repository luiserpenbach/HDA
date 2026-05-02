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


def test_qc_failed_is_terminal():
    assert is_terminal(TestState.QC_FAILED)
    with pytest.raises(IllegalTransition):
        transition(TestState.QC_FAILED, TestState.ANALYZED)


def test_persisted_is_terminal():
    assert is_terminal(TestState.PERSISTED)
    with pytest.raises(IllegalTransition):
        transition(TestState.PERSISTED, TestState.ANALYZED)


def test_error_is_terminal_from_every_state():
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


def test_terminal_states_have_no_outgoing():
    for s in (TestState.PERSISTED, TestState.QC_FAILED, TestState.ERROR):
        assert ALLOWED_TRANSITIONS[s] == frozenset()


def test_review_can_be_resolved_either_way():
    assert (
        transition(TestState.NEEDS_REVIEW, TestState.ANALYZED) == TestState.ANALYZED
    )
    assert (
        transition(TestState.NEEDS_REVIEW, TestState.QC_FAILED)
        == TestState.QC_FAILED
    )
