"""Shared helpers for test-type mapping (no Qt imports)."""

_TEST_TYPE_ALIASES = {
    "cf": "cold_flow",
    "cold_flow": "cold_flow",
    "cold flow": "cold_flow",
    "hf": "hot_fire",
    "hot_fire": "hot_fire",
    "hot fire": "hot_fire",
}


def normalize_test_type(raw: str) -> str:
    """Map metadata test-type codes to analysis config test types."""
    key = (raw or "").strip().lower().replace("-", "_")
    return _TEST_TYPE_ALIASES.get(key, "cold_flow")
