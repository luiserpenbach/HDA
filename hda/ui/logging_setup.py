"""Structured logging.

Writes to ``<log_dir>/hda.log`` (rotating) and stderr. Format is one
line per record so the UI's in-app log console can render it directly.

The legacy app had no structured logging — production bugs were
invisible. Calling ``setup_logging`` once at app startup gives every
module ``logging.getLogger("hda.<area>").info(...)`` and routes it to
the same destination.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional


_FORMAT = "%(asctime)s %(levelname)-7s %(name)s %(message)s"
_DATE_FMT = "%Y-%m-%dT%H:%M:%S"


def setup_logging(
    log_dir: Optional[Path] = None,
    level: int = logging.INFO,
    file_name: str = "hda.log",
    max_bytes: int = 10_000_000,
    backup_count: int = 5,
) -> Path | None:
    """Configure the root ``hda`` logger.

    Returns the path of the active log file (or None when ``log_dir``
    is None — stderr-only mode, used in tests).
    """
    logger = logging.getLogger("hda")
    logger.setLevel(level)
    logger.propagate = False

    for h in list(logger.handlers):
        logger.removeHandler(h)

    formatter = logging.Formatter(_FORMAT, _DATE_FMT)

    stderr = logging.StreamHandler(stream=sys.stderr)
    stderr.setFormatter(formatter)
    logger.addHandler(stderr)

    if log_dir is None:
        return None

    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / file_name
    file_handler = logging.handlers.RotatingFileHandler(
        path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info("logging initialized; file=%s", path)
    return path


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``hda`` namespace."""
    return logging.getLogger(f"hda.{name}" if not name.startswith("hda") else name)
