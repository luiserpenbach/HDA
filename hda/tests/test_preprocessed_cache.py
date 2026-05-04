"""PreprocessedDataCache: LRU eviction, idempotent put, thread safety."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

from hda.services.preprocessed_cache import (
    CachedPreprocessed,
    PreprocessedDataCache,
)


def _entry(test_run_id: str) -> CachedPreprocessed:
    return CachedPreprocessed(
        test_run_id=test_run_id,
        data=MagicMock(),
        metadata=MagicMock(),
        config_hash="",
        metadata_hash="",
    )


def test_put_get_round_trip():
    cache = PreprocessedDataCache(max_entries=3)
    e = _entry("a")
    cache.put(e)
    assert cache.get("a") is e
    assert cache.has("a") is True


def test_get_missing_returns_none():
    cache = PreprocessedDataCache(max_entries=3)
    assert cache.get("nope") is None
    assert cache.has("nope") is False


def test_lru_eviction_drops_oldest_first():
    cache = PreprocessedDataCache(max_entries=2)
    cache.put(_entry("a"))
    cache.put(_entry("b"))
    cache.put(_entry("c"))  # evicts "a"
    assert cache.has("a") is False
    assert cache.has("b") is True
    assert cache.has("c") is True


def test_get_promotes_to_most_recent():
    cache = PreprocessedDataCache(max_entries=2)
    cache.put(_entry("a"))
    cache.put(_entry("b"))
    cache.get("a")  # promote "a"
    cache.put(_entry("c"))  # should now evict "b", not "a"
    assert cache.has("a") is True
    assert cache.has("b") is False


def test_put_existing_id_replaces_and_promotes():
    cache = PreprocessedDataCache(max_entries=2)
    cache.put(_entry("a"))
    cache.put(_entry("b"))
    new_a = _entry("a")
    cache.put(new_a)  # replaces a + moves to most-recent
    cache.put(_entry("c"))  # evicts "b" (older), keeps "a"
    assert cache.get("a") is new_a
    assert cache.has("b") is False


def test_evict_removes_entry():
    cache = PreprocessedDataCache(max_entries=3)
    cache.put(_entry("a"))
    cache.evict("a")
    assert cache.get("a") is None


def test_clear_empties_cache():
    cache = PreprocessedDataCache(max_entries=3)
    for n in "abc":
        cache.put(_entry(n))
    cache.clear()
    assert len(cache) == 0


def test_max_entries_must_be_positive():
    with pytest.raises(ValueError):
        PreprocessedDataCache(max_entries=0)


def test_concurrent_puts_dont_corrupt():
    cache = PreprocessedDataCache(max_entries=200)

    def worker(prefix: str):
        for i in range(50):
            cache.put(_entry(f"{prefix}-{i}"))

    threads = [
        threading.Thread(target=worker, args=(p,)) for p in ("x", "y", "z")
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(cache) == 150
