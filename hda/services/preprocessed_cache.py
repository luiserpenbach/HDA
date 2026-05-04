"""Preprocessed-data cache.

Re-analysis with a manual steady-state window needs the preprocessed
DataFrame plus the original test metadata back in memory. Re-reading
and re-preprocessing the source CSV from disk would work but costs
seconds per click; for the interactive drag-handle preview we need
microseconds.

Bounded LRU so a long session doesn't exhaust memory; thread-safe so
the QThreadPool worker can populate it while the GUI thread reads.

The cache is *not* a source of truth — it's a performance layer on top
of the DB + the source CSV. Eviction is fine; missing entries trigger
a "re-ingest the source file to enable reanalysis" UI message.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

from hda.domain.types import TestMetadata
from hda.services.preprocessing import PreprocessedData


@dataclass(frozen=True, slots=True)
class CachedPreprocessed:
    test_run_id: str
    data: PreprocessedData
    metadata: TestMetadata
    config_hash: str
    metadata_hash: str


class PreprocessedDataCache:
    def __init__(self, max_entries: int = 32) -> None:
        if max_entries <= 0:
            raise ValueError(f"max_entries must be > 0, got {max_entries}")
        self._max = max_entries
        self._store: "OrderedDict[str, CachedPreprocessed]" = OrderedDict()
        self._lock = threading.Lock()

    def put(self, entry: CachedPreprocessed) -> None:
        with self._lock:
            if entry.test_run_id in self._store:
                self._store.move_to_end(entry.test_run_id)
            self._store[entry.test_run_id] = entry
            while len(self._store) > self._max:
                self._store.popitem(last=False)

    def get(self, test_run_id: str) -> Optional[CachedPreprocessed]:
        with self._lock:
            entry = self._store.get(test_run_id)
            if entry is not None:
                self._store.move_to_end(test_run_id)
            return entry

    def has(self, test_run_id: str) -> bool:
        with self._lock:
            return test_run_id in self._store

    def evict(self, test_run_id: str) -> None:
        with self._lock:
            self._store.pop(test_run_id, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)
