"""Structured logging setup."""

from __future__ import annotations

import logging
from pathlib import Path

from hda.ui.logging_setup import get_logger, setup_logging


def test_setup_creates_log_file_and_writes(tmp_path: Path):
    setup_logging(log_dir=tmp_path)
    log = get_logger("test")
    log.info("hello structured log")

    log_path = tmp_path / "hda.log"
    assert log_path.exists()
    contents = log_path.read_text()
    assert "hello structured log" in contents
    assert "INFO" in contents


def test_get_logger_returns_namespaced_logger():
    log = get_logger("foo")
    assert log.name == "hda.foo"
    log2 = get_logger("hda.bar")
    assert log2.name == "hda.bar"


def test_setup_idempotent_replaces_handlers(tmp_path: Path):
    setup_logging(log_dir=tmp_path)
    setup_logging(log_dir=tmp_path)
    logger = logging.getLogger("hda")
    # stderr + file == 2 handlers; setup must not stack on re-call.
    assert len(logger.handlers) == 2


def test_setup_without_log_dir_returns_none(tmp_path: Path):
    path = setup_logging(log_dir=None)
    assert path is None
