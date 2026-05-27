"""Pure-Python helpers for campaign analysis (no Qt imports)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

COLD_FLOW_PRIMARY = "avg_cd_CALC"
HOT_FIRE_PRIMARY = "avg_isp_s"

SUMMARY_COLUMNS = [
    "test_id",
    "test_date",
    "part",
    "serial_num",
    "qc_passed",
    "avg_cd_CALC",
    "avg_isp_s",
    "avg_p_up_bar",
    "avg_mf_g_s",
    "avg_p_c_bar",
    "avg_of_ratio",
]


def primary_metric_for_type(campaign_type: str) -> str:
    """Default SPC parameter for a campaign type."""
    if campaign_type == "hot_fire":
        return HOT_FIRE_PRIMARY
    return COLD_FLOW_PRIMARY


def metric_columns(df: pd.DataFrame) -> List[str]:
    """Numeric avg_* columns suitable for SPC."""
    if df is None or df.empty:
        return []
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    preferred = [c for c in numeric if c.startswith("avg_")]
    if preferred:
        return preferred
    return numeric


def filter_campaign_df(
    df: pd.DataFrame,
    parts: Optional[Sequence[str]] = None,
    serials: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Apply optional part / serial filters."""
    if df is None or df.empty:
        return df
    out = df.copy()
    if parts and "part" in out.columns:
        out = out[out["part"].isin(parts)]
    if serials and "serial_num" in out.columns:
        out = out[out["serial_num"].isin(serials)]
    return out.reset_index(drop=True)


def summary_display_columns(df: pd.DataFrame) -> List[str]:
    """Columns to show in the summary table."""
    if df is None or df.empty:
        return []
    return [c for c in SUMMARY_COLUMNS if c in df.columns]


def campaign_type_from_info(info: Dict[str, Any]) -> str:
    """Normalize campaign type from info dict."""
    return str(info.get("type") or info.get("campaign_type") or "cold_flow")


def campaign_overview_stats(
    df: pd.DataFrame,
    info: Dict[str, Any],
) -> Dict[str, str]:
    """Key metrics for the overview row."""
    n = len(df) if df is not None else 0
    qc = "N/A"
    if df is not None and n and "qc_passed" in df.columns:
        passed = int(df["qc_passed"].fillna(False).astype(bool).sum())
        qc = f"{passed}/{n}"
    ctype = campaign_type_from_info(info)
    return {
        "tests": str(n),
        "qc_passed": qc,
        "type": ctype.replace("_", " ").title(),
        "schema": str(info.get("schema_version", "N/A")),
    }
