"""Schema version constant.

The actual DDL lives in ``persistence/migrations/`` so schema evolution is
described as a sequence of versioned migration steps rather than a single
``CREATE TABLE`` script that drifts from reality.

Current ``SCHEMA_VERSION`` must equal the highest migration number that
``apply_migrations`` reaches when run on an empty database.
"""

from __future__ import annotations

SCHEMA_VERSION = 1
