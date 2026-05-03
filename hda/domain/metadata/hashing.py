"""Canonical hashing of metadata for the traceability chain.

The Streamlit app's ``json.dumps(..., default=str)`` made hashes
non-reproducible because numpy/dataclass values stringified inconsistently.
v3 refuses to hash anything that isn't a native JSON-serializable type:
str, int, float (finite, non-NaN), bool, None, dict[str, ...], list[...].

If the caller hands us a numpy scalar, datetime, Path, etc., they must
convert it to a native type first. Conversion is the caller's call to make,
not the hasher's.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping


_ALLOWED_SCALARS = (str, int, float, bool, type(None))


def _check_serializable(value: Any, path: str = "$") -> None:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, str)):
        return
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError(f"Non-finite float at {path}: {value!r}")
        return
    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str):
                raise ValueError(f"Non-string key at {path}: {k!r}")
            _check_serializable(v, f"{path}.{k}")
        return
    if isinstance(value, list):
        for i, v in enumerate(value):
            _check_serializable(v, f"{path}[{i}]")
        return
    raise ValueError(
        f"Non-serializable value at {path}: {type(value).__name__} "
        "(convert to str/int/float/bool/dict/list before hashing)"
    )


def canonical_metadata_hash(values: Mapping[str, Any]) -> str:
    """Return a deterministic SHA-256 hex digest over canonical JSON.

    Properties:
        - sort_keys=True so dict ordering doesn't affect the hash.
        - allow_nan=False rejects NaN/inf; non-finite values would otherwise
          serialize as JS-style ``NaN``/``Infinity`` and silently differ
          between platforms.
        - separators are tight to keep the canonical form unambiguous.
        - any value type other than ``str/int/float/bool/None/dict/list``
          is rejected up-front so the hash is reproducible.
    """
    if not isinstance(values, Mapping):
        raise TypeError(f"Expected Mapping, got {type(values).__name__}")
    _check_serializable(dict(values))
    payload = json.dumps(
        dict(values),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
