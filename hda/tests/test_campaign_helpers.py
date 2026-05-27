"""Tests for campaign analysis helpers."""

import pandas as pd

from hda.campaign_helpers import (
    campaign_overview_stats,
    filter_campaign_df,
    metric_columns,
    primary_metric_for_type,
    summary_display_columns,
)


def test_primary_metric_for_type():
    assert primary_metric_for_type("cold_flow") == "avg_cd_CALC"
    assert primary_metric_for_type("hot_fire") == "avg_isp_s"


def test_metric_columns_prefers_avg_prefix():
    df = pd.DataFrame({
        "test_id": ["a", "b"],
        "avg_cd_CALC": [0.6, 0.61],
        "row_id": [1, 2],
    })
    cols = metric_columns(df)
    assert "avg_cd_CALC" in cols
    assert "row_id" not in cols


def test_filter_campaign_df_by_part_and_serial():
    df = pd.DataFrame({
        "part": ["P1", "P1", "P2"],
        "serial_num": ["S1", "S2", "S1"],
        "avg_cd_CALC": [0.6, 0.61, 0.62],
    })
    out = filter_campaign_df(df, parts=["P1"], serials=["S1"])
    assert len(out) == 1
    assert out.iloc[0]["serial_num"] == "S1"


def test_campaign_overview_stats_qc():
    df = pd.DataFrame({"qc_passed": [True, False, True]})
    stats = campaign_overview_stats(df, {"type": "cold_flow", "schema_version": 2})
    assert stats["tests"] == "3"
    assert stats["qc_passed"] == "2/3"
    assert stats["type"] == "Cold Flow"


def test_summary_display_columns_intersection():
    df = pd.DataFrame({"test_id": ["x"], "avg_cd_CALC": [0.5], "extra": [1]})
    cols = summary_display_columns(df)
    assert "test_id" in cols
    assert "extra" not in cols
