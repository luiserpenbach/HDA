"""Tests for analysis tools helpers."""
import pandas as pd

from hda.analysis_tools_helpers import detect_time_column, metric_columns, numeric_columns


def test_detect_time_column_prefers_time_s():
    df = pd.DataFrame({"time_s": [0, 1], "x": [1, 2]})
    assert detect_time_column(df) == "time_s"


def test_numeric_columns_excludes_time():
    df = pd.DataFrame({"time_s": [0, 1], "pressure": [1.0, 2.0], "note": ["a", "b"]})
    cols = numeric_columns(df, exclude=["time_s"])
    assert cols == ["pressure"]


def test_metric_columns_prefix():
    df = pd.DataFrame({"avg_cd_CALC": [0.6], "raw_x": [1]})
    assert metric_columns(df) == ["avg_cd_CALC"]
