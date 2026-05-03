"""Basic-means plugin.

For every numeric channel in the steady-state slice, emits a measurement
``avg_<channel>`` whose value is the mean and whose uncertainty is the
quadrature combination of the standard error of the mean and any
calibration uncertainty supplied in ``ctx.sensor_uncertainties``.

This is the substrate plugin: it proves the analysis pipeline end-to-end
(plugin contract, traceability, persistence) before the test-type-specific
plugins (cold-flow, hot-fire) land. It's also genuinely useful as a
fallback for bring-up tests where no plugin has been written yet.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from hda.domain.plugins import AnalysisContext, AnalysisPlugin
from hda.domain.types import MeasurementWithUncertainty, Provenance


class BasicMeansPlugin(AnalysisPlugin):
    name: str = "basic_means"
    version: str = "1.0.0"

    def required_channels(self) -> Sequence[str]:
        return ()

    def compute(
        self, ctx: AnalysisContext
    ) -> Mapping[str, MeasurementWithUncertainty]:
        out: dict[str, MeasurementWithUncertainty] = {}
        ts = ctx.timestamp_column
        for col in ctx.steady_df.columns:
            if col == ts:
                continue
            if not pd.api.types.is_numeric_dtype(ctx.steady_df[col]):
                continue
            arr = ctx.steady_df[col].to_numpy(dtype=float)
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                continue
            mean = float(np.mean(finite))
            sem = (
                float(np.std(finite, ddof=1) / math.sqrt(finite.size))
                if finite.size > 1
                else 0.0
            )
            cal = float(ctx.sensor_uncertainties.get(col, 0.0))
            uncertainty = math.sqrt(sem * sem + cal * cal)
            name = f"avg_{col}"
            out[name] = MeasurementWithUncertainty(
                name=name,
                value=mean,
                uncertainty=uncertainty,
                unit="",
                provenance=Provenance.SENSOR,
            )
        return out
