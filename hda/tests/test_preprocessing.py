"""Preprocessing pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hda.domain.derived import DerivedChannelSpec, UncertaintyMethod
from hda.domain.errors import IngestError
from hda.services.preprocessing import (
    NaNPolicy,
    PreprocessingConfig,
    TimestampUnit,
    preprocess,
)


def _df(timestamps_ms: list[float], **channels: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"timestamp": timestamps_ms, **channels})


def test_rejects_empty_dataframe():
    with pytest.raises(IngestError):
        preprocess(pd.DataFrame(), PreprocessingConfig())


def test_rejects_missing_timestamp_column():
    df = pd.DataFrame({"foo": [1, 2, 3]})
    with pytest.raises(IngestError, match="Timestamp column"):
        preprocess(df, PreprocessingConfig())


def test_converts_milliseconds_to_seconds_and_shifts():
    df = _df([1000.0, 1100.0, 1200.0], pt=[1.0, 2.0, 3.0])
    result = preprocess(
        df,
        PreprocessingConfig(
            resample_freq_hz=None,
            nan_policy=NaNPolicy.LEAVE,
        ),
    )
    np.testing.assert_allclose(
        result.df["timestamp"].to_numpy(), [0.0, 0.1, 0.2]
    )


def test_shift_to_zero_off():
    df = _df([1000.0, 1100.0], pt=[1.0, 2.0])
    result = preprocess(
        df,
        PreprocessingConfig(
            shift_to_zero=False,
            resample_freq_hz=None,
            nan_policy=NaNPolicy.LEAVE,
        ),
    )
    assert result.df["timestamp"].iloc[0] == pytest.approx(1.0)


def test_drops_duplicate_and_nan_timestamps():
    df = _df(
        [0.0, 100.0, 100.0, np.nan, 200.0],
        pt=[1.0, 2.0, 99.0, 5.0, 3.0],
    )
    result = preprocess(
        df,
        PreprocessingConfig(
            resample_freq_hz=None, nan_policy=NaNPolicy.LEAVE
        ),
    )
    assert result.n_samples == 3
    np.testing.assert_allclose(
        result.df["timestamp"].to_numpy(), [0.0, 0.1, 0.2]
    )
    assert result.df["pt"].iloc[1] == pytest.approx(2.0)


def test_unsorted_timestamps_are_sorted():
    df = _df([200.0, 0.0, 100.0], pt=[3.0, 1.0, 2.0])
    result = preprocess(
        df,
        PreprocessingConfig(
            resample_freq_hz=None, nan_policy=NaNPolicy.LEAVE
        ),
    )
    np.testing.assert_allclose(result.df["pt"].to_numpy(), [1.0, 2.0, 3.0])


def test_channel_map_renames_columns():
    df = _df([0.0, 100.0], **{"10001": [1.0, 2.0]})
    result = preprocess(
        df,
        PreprocessingConfig(
            channel_map={"10001": "PT-fuel-up"},
            resample_freq_hz=None,
            nan_policy=NaNPolicy.LEAVE,
        ),
    )
    assert "PT-fuel-up" in result.df.columns
    assert "10001" not in result.df.columns


def test_channel_map_unknown_column_raises():
    df = _df([0.0, 100.0], pt=[1.0, 2.0])
    with pytest.raises(IngestError, match="not in DataFrame"):
        preprocess(
            df,
            PreprocessingConfig(channel_map={"nonexistent": "X"}),
        )


def test_resample_to_uniform_rate():
    # 1 Hz raw, resample to 10 Hz over 1 second
    df = _df([0.0, 1000.0], pt=[0.0, 10.0])
    result = preprocess(
        df,
        PreprocessingConfig(
            resample_freq_hz=10.0, nan_policy=NaNPolicy.LEAVE
        ),
    )
    assert result.n_samples == 11
    assert result.sample_rate_hz == pytest.approx(10.0)
    np.testing.assert_allclose(result.df["pt"].iloc[5], 5.0, rtol=1e-9)


def test_nan_policy_drop():
    df = _df([0.0, 100.0, 200.0], pt=[1.0, np.nan, 3.0])
    result = preprocess(
        df,
        PreprocessingConfig(
            resample_freq_hz=None, nan_policy=NaNPolicy.DROP
        ),
    )
    assert result.n_samples == 2


def test_nan_policy_interpolate():
    df = _df([0.0, 100.0, 200.0], pt=[1.0, np.nan, 3.0])
    result = preprocess(
        df,
        PreprocessingConfig(
            resample_freq_hz=None, nan_policy=NaNPolicy.INTERPOLATE
        ),
    )
    assert result.df["pt"].iloc[1] == pytest.approx(2.0)


def test_derived_channels_appended():
    df = _df(
        [0.0, 100.0, 200.0],
        **{"PT-up": [10.0, 11.0, 12.0], "PT-down": [5.0, 6.0, 7.0]},
    )
    spec = DerivedChannelSpec(
        name="dp_bar",
        unit="bar",
        formula="subtract",
        inputs={"a": "PT-up", "b": "PT-down"},
        uncertainty_method=UncertaintyMethod.NONE,
    )
    result = preprocess(
        df,
        PreprocessingConfig(
            resample_freq_hz=None,
            nan_policy=NaNPolicy.LEAVE,
            derived_channels=(spec,),
        ),
    )
    assert "dp_bar" in result.df.columns
    np.testing.assert_allclose(
        result.df["dp_bar"].to_numpy(), [5.0, 5.0, 5.0]
    )
    assert result.derived_channel_names == ("dp_bar",)


def test_microsecond_timestamps():
    df = _df([0.0, 1000.0], pt=[1.0, 2.0])  # in µs
    result = preprocess(
        df,
        PreprocessingConfig(
            timestamp_unit=TimestampUnit.MICROSECONDS,
            resample_freq_hz=None,
            nan_policy=NaNPolicy.LEAVE,
        ),
    )
    assert result.duration_s == pytest.approx(0.001)
