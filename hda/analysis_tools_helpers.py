"""Shared helpers for the Analysis Tools Qt page."""
from __future__ import annotations

from typing import List, Optional, Sequence

import pandas as pd


def detect_time_column(df: pd.DataFrame) -> Optional[str]:
    for name in ("time_s", "time_ms", "timestamp", "time", "Time", "t"):
        if name in df.columns:
            return name
    return None


def numeric_columns(df: pd.DataFrame, exclude: Optional[Sequence[str]] = None) -> List[str]:
    skip = set(exclude or [])
    out: List[str] = []
    for col in df.columns:
        if col in skip:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            out.append(col)
    return out


def metric_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in numeric_columns(df) if c.startswith("avg_")]


def populate_table(table, df: pd.DataFrame) -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QTableWidgetItem

    table.clear()
    if df is None or df.empty:
        table.setRowCount(0)
        table.setColumnCount(0)
        return
    table.setColumnCount(len(df.columns))
    table.setHorizontalHeaderLabels([str(c) for c in df.columns])
    table.setRowCount(len(df))
    for row_idx, row in enumerate(df.itertuples(index=False)):
        for col_idx, val in enumerate(row):
            text = "" if pd.isna(val) else str(val)
            item = QTableWidgetItem(text)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            table.setItem(row_idx, col_idx, item)
    table.resizeColumnsToContents()
