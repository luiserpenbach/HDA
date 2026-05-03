"""Streaming SHA-256 of files.

Separate from ``canonical_metadata_hash`` because file hashing is bytes-in,
metadata hashing is structured-data-in. Both feed the same traceability
record.

Streaming so a 500 MB hot-fire CSV doesn't get pulled into memory just to
be hashed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from hda.domain.errors import IngestError


_DEFAULT_CHUNK = 1 << 20  # 1 MiB


def hash_file(path: Path, chunk_size: int = _DEFAULT_CHUNK) -> str:
    """Stream-hash a file with SHA-256 and return the hex digest.

    Raises:
        IngestError: file missing or unreadable.
    """
    p = Path(path)
    if not p.exists():
        raise IngestError(f"Cannot hash: file does not exist: {p}")
    if not p.is_file():
        raise IngestError(f"Cannot hash: not a file: {p}")
    h = hashlib.sha256()
    try:
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(chunk_size), b""):
                h.update(chunk)
    except OSError as e:
        raise IngestError(f"Failed to read {p}: {e}") from e
    return h.hexdigest()
