"""Streaming file hash."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from hda.domain.errors import IngestError
from hda.services.hashing import hash_file


def test_hash_matches_reference(tmp_path: Path):
    p = tmp_path / "data.bin"
    payload = b"hopper-data-studio-v3" * 1024
    p.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    assert hash_file(p) == expected


def test_hash_is_stable_across_chunk_sizes(tmp_path: Path):
    p = tmp_path / "data.bin"
    p.write_bytes(b"\x01\x02\x03" * 100_000)
    assert hash_file(p, chunk_size=17) == hash_file(p, chunk_size=1 << 20)


def test_hash_missing_file_raises(tmp_path: Path):
    with pytest.raises(IngestError):
        hash_file(tmp_path / "nope.bin")


def test_hash_directory_raises(tmp_path: Path):
    with pytest.raises(IngestError):
        hash_file(tmp_path)
