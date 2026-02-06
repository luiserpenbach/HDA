"""
Tier 3 Integration Tests
========================
Tests for production-ready UI integration features:
- Reporting module Phase 2 sections (transient, frequency, CUSUM/EWMA)
- Core __init__.py exports
- Shared widgets utilities
- Campaign Analysis CUSUM/EWMA integration
- Version consistency
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock


# =============================================================================
# Version Consistency Tests
# =============================================================================

class TestVersionConsistency:
    """Verify version numbers are consistent across the codebase."""

    def test_core_version_is_2_5_0(self):
        from core import __version__
        assert __version__ == "2.5.0", f"Expected 2.5.0, got {__version__}"

    def test_schema_version_is_3(self):
        from core.campaign_manager_v2 import SCHEMA_VERSION
        assert SCHEMA_VERSION == 3


# =============================================================================
# Core Exports Tests
# =============================================================================

class TestCoreExports:
    """Verify all Phase 2 features are properly exported from core/__init__.py."""

    def test_spc_cusum_exports(self):
        from core import CUSUMResult, EWMAResult, create_cusum_chart, create_ewma_chart
        assert CUSUMResult is not None
        assert EWMAResult is not None
        assert callable(create_cusum_chart)
        assert callable(create_ewma_chart)

    def test_transient_analysis_exports(self):
        from core import (TestPhase, PhaseResult, MultiPhaseResult,
                          segment_test_phases, analyze_startup_transient,
                          analyze_shutdown_transient, compute_phase_metrics)
        assert TestPhase is not None
        assert callable(segment_test_phases)
        assert callable(analyze_startup_transient)

    def test_frequency_analysis_exports(self):
        from core import (SpectralResult, HarmonicInfo, CrossSpectralResult,
                          compute_power_spectral_density, compute_spectrogram,
                          detect_harmonics, compute_cross_spectrum,
                          detect_resonance, compute_frequency_bands)
        assert callable(compute_power_spectral_density)
        assert callable(detect_harmonics)

    def test_operating_envelope_exports(self):
        from core import (OperatingEnvelope, calculate_operating_envelope,
                          plot_operating_envelope, create_envelope_report)
        assert callable(calculate_operating_envelope)
        assert callable(plot_operating_envelope)

    def test_control_chart_type_includes_cusum_ewma(self):
        from core import ControlChartType
        assert hasattr(ControlChartType, 'CUSUM')
        assert hasattr(ControlChartType, 'EWMA')


# =============================================================================
# Reporting Module Phase 2 Tests
# =============================================================================

class TestReportingPhase2:
    """Test new reporting module sections for Phase 2 features."""

    def test_generate_transient_section_with_phases(self):
        from core.reporting import generate_transient_section
        from core.transient_analysis import TestPhase, PhaseResult

        phases = [
            PhaseResult(phase=TestPhase.PRETEST, start_ms=0, end_ms=1000,
                        duration_s=1.0, metrics={'mean': 0.5}),
            PhaseResult(phase=TestPhase.STARTUP, start_ms=1000, end_ms=2000,
                        duration_s=1.0, metrics={'mean': 15.0}),
            PhaseResult(phase=TestPhase.STEADY_STATE, start_ms=2000, end_ms=8000,
                        duration_s=6.0, metrics={'mean': 30.0}),
        ]

        html = generate_transient_section(phases)
        assert 'Transient Analysis' in html
        assert 'pretest' in html.lower() or 'PRETEST' in html
        assert 'startup' in html.lower() or 'STARTUP' in html
        assert 'steady_state' in html.lower() or 'steady' in html.lower()

    def test_generate_transient_section_with_startup_metrics(self):
        from core.reporting import generate_transient_section

        startup = {
            'rise_time_s': 0.45,
            'rise_time_10_90_s': 0.38,
            'overshoot_pct': 5.2,
            'settling_time_s': 1.1,
        }
        html = generate_transient_section([], startup_metrics=startup)
        assert 'Rise Time' in html
        assert '0.45' in html or '0.4500' in html
        assert 'Overshoot' in html

    def test_generate_transient_section_with_shutdown_metrics(self):
        from core.reporting import generate_transient_section

        shutdown = {
            'decay_time_s': 0.35,
            'decay_time_90_10_s': 0.28,
            'tail_off_impulse': 12.5,
            'residual_pct': 2.1,
        }
        html = generate_transient_section([], shutdown_metrics=shutdown)
        assert 'Shutdown' in html
        assert 'Decay Time' in html

    def test_generate_frequency_section_with_spectral(self):
        from core.reporting import generate_frequency_section
        from core.frequency_analysis import SpectralResult

        spectral = SpectralResult(
            frequencies=np.array([1, 2, 3, 4, 5]),
            power_spectral_density=np.array([0.1, 0.5, 0.3, 0.05, 0.02]),
            dominant_frequency=2.0,
            dominant_power=0.5,
            total_power=0.97,
            bandwidth=3.0,
            sample_rate_hz=100.0,
            method='welch',
        )

        html = generate_frequency_section(spectral_result=spectral)
        assert 'Frequency Analysis' in html
        assert 'Dominant Frequency' in html
        assert '2.00' in html

    def test_generate_frequency_section_with_harmonics(self):
        from core.reporting import generate_frequency_section
        from core.frequency_analysis import HarmonicInfo

        harmonics = [
            HarmonicInfo(frequency=10.0, power=0.5, harmonic_number=1, relative_power=1.0),
            HarmonicInfo(frequency=20.0, power=0.25, harmonic_number=2, relative_power=0.5),
        ]

        html = generate_frequency_section(harmonics=harmonics)
        assert 'Detected Harmonics' in html
        assert '10.00' in html
        assert '20.00' in html

    def test_generate_frequency_section_with_bands(self):
        from core.reporting import generate_frequency_section

        bands = {'low': 0.3, 'mid': 0.5, 'high': 0.15, 'acoustic': 0.05}
        html = generate_frequency_section(band_powers=bands)
        assert 'Frequency Band Power' in html
        assert 'low' in html
        assert 'mid' in html

    def test_generate_cusum_ewma_section_cusum(self):
        from core.reporting import generate_cusum_ewma_section
        from core.spc import create_cusum_chart

        np.random.seed(42)
        values = np.random.normal(10, 1, 30)
        result = create_cusum_chart(values, parameter_name='test_param')

        html = generate_cusum_ewma_section(cusum_result=result)
        assert 'CUSUM' in html
        assert 'test_param' in html

    def test_generate_cusum_ewma_section_ewma(self):
        from core.reporting import generate_cusum_ewma_section
        from core.spc import create_ewma_chart

        np.random.seed(42)
        values = np.random.normal(10, 1, 30)
        result = create_ewma_chart(values, parameter_name='test_param')

        html = generate_cusum_ewma_section(ewma_result=result)
        assert 'EWMA' in html
        assert 'test_param' in html

    def test_generate_cusum_ewma_section_both(self):
        from core.reporting import generate_cusum_ewma_section
        from core.spc import create_cusum_chart, create_ewma_chart

        np.random.seed(42)
        values = np.random.normal(10, 1, 30)
        cusum = create_cusum_chart(values, parameter_name='pressure')
        ewma = create_ewma_chart(values, parameter_name='pressure')

        html = generate_cusum_ewma_section(cusum_result=cusum, ewma_result=ewma)
        assert 'CUSUM' in html
        assert 'EWMA' in html
        assert 'Advanced SPC' in html


# =============================================================================
# CUSUM/EWMA Integration Tests
# =============================================================================

class TestCUSUMEWMAIntegration:
    """Test CUSUM and EWMA work correctly with campaign data patterns."""

    def test_cusum_detects_shift_in_campaign_data(self):
        """Simulate a campaign where Cd shifts mid-way."""
        from core.spc import create_cusum_chart

        np.random.seed(42)
        # 20 tests at Cd=0.65, then 20 tests at Cd=0.68 (large shift)
        values = np.concatenate([
            np.random.normal(0.65, 0.005, 20),
            np.random.normal(0.68, 0.005, 20)
        ])

        # Use default k and h (auto-scaled to sigma) but set target to first-half mean
        # so the shift in the second half is detectable
        result = create_cusum_chart(values, target=0.65, parameter_name='avg_cd_CALC')

        assert result.n_signals > 0, (
            f"CUSUM should detect the shift from target=0.65. "
            f"C+ max={max(result.c_plus):.3f}, C- max={max(result.c_minus):.3f}"
        )
        assert len(result.c_plus) == len(values)
        assert len(result.c_minus) == len(values)

    def test_ewma_detects_gradual_drift(self):
        """Simulate gradual drift in a campaign."""
        from core.spc import create_ewma_chart

        np.random.seed(42)
        # Linear drift from 0.65 to 0.68 over 30 tests
        trend = np.linspace(0.65, 0.68, 30)
        noise = np.random.normal(0, 0.003, 30)
        values = trend + noise

        result = create_ewma_chart(values, lambda_param=0.2, L=3.0,
                                    parameter_name='avg_cd_CALC')

        assert result.n_signals > 0, "EWMA should detect the gradual drift"
        assert len(result.ewma_values) == len(values)
        assert len(result.ucl) == len(values)
        assert len(result.lcl) == len(values)

    def test_cusum_stable_process_no_signals(self):
        """A stable process should not trigger CUSUM signals."""
        from core.spc import create_cusum_chart

        np.random.seed(42)
        values = np.random.normal(0.65, 0.005, 50)

        result = create_cusum_chart(values, k=0.5, h=5.0)
        assert result.n_signals == 0, "Stable process should have no CUSUM signals"

    def test_ewma_stable_process_no_signals(self):
        """A stable process should not trigger EWMA signals."""
        from core.spc import create_ewma_chart

        np.random.seed(42)
        values = np.random.normal(0.65, 0.005, 50)

        result = create_ewma_chart(values, lambda_param=0.2, L=3.0)
        assert result.n_signals == 0, "Stable process should have no EWMA signals"


# =============================================================================
# Operating Envelope Integration Tests
# =============================================================================

class TestOperatingEnvelopeIntegration:
    """Test operating envelope works with campaign-like data."""

    def _create_hot_fire_campaign_df(self):
        """Create synthetic hot fire campaign data."""
        np.random.seed(42)
        n_tests = 20
        return pd.DataFrame({
            'test_id': [f'HF-{i:03d}' for i in range(n_tests)],
            'avg_of_ratio': np.random.uniform(1.2, 2.5, n_tests),
            'avg_pc_bar': np.random.uniform(10, 30, n_tests),
            'ignition_successful': [True] * 18 + [False] * 2,
        })

    def test_calculate_envelope(self):
        from core.operating_envelope import calculate_operating_envelope

        df = self._create_hot_fire_campaign_df()
        envelope = calculate_operating_envelope(df)

        assert envelope is not None
        assert hasattr(envelope, 'of_min')
        assert hasattr(envelope, 'of_max')
        assert hasattr(envelope, 'pc_min')
        assert hasattr(envelope, 'pc_max')
        assert envelope.of_min < envelope.of_max
        assert envelope.pc_min < envelope.pc_max

    def test_plot_envelope_returns_figure(self):
        from core.operating_envelope import calculate_operating_envelope, plot_operating_envelope
        import plotly.graph_objects as go

        df = self._create_hot_fire_campaign_df()
        envelope = calculate_operating_envelope(df)
        fig = plot_operating_envelope(df, envelope=envelope)

        assert isinstance(fig, go.Figure)

    def test_create_envelope_report_html(self):
        from core.operating_envelope import (
            calculate_operating_envelope, create_envelope_report
        )

        df = self._create_hot_fire_campaign_df()
        envelope = calculate_operating_envelope(df)
        report = create_envelope_report(df, envelope, campaign_name='Test_Campaign')

        assert isinstance(report, str)
        assert 'Test_Campaign' in report


# =============================================================================
# Transient + Frequency End-to-End Tests
# =============================================================================

class TestTransientFrequencyE2E:
    """End-to-end tests combining transient and frequency analysis."""

    def _create_test_signal(self, n=2000, fs=1000):
        """Create a realistic-looking test signal with phases."""
        t = np.arange(n) / fs

        signal = np.zeros(n)
        # Pretest: noise around 0
        signal[:200] = np.random.normal(0, 0.1, 200)
        # Startup ramp
        ramp = np.linspace(0, 30, 300)
        signal[200:500] = ramp + np.random.normal(0, 0.5, 300)
        # Steady state with 50 Hz oscillation
        steady = 30 + 0.5 * np.sin(2 * np.pi * 50 * t[500:1500]) + np.random.normal(0, 0.3, 1000)
        signal[500:1500] = steady
        # Shutdown ramp
        signal[1500:1800] = np.linspace(30, 0, 300) + np.random.normal(0, 0.5, 300)
        # Cooldown
        signal[1800:] = np.random.normal(0, 0.1, 200)

        df = pd.DataFrame({'time_s': t, 'pressure': signal})
        return df, signal, fs

    def test_segment_then_frequency_on_steady(self):
        """Segment test phases, then run frequency analysis on steady state."""
        from core.transient_analysis import segment_test_phases, TestPhase
        from core.frequency_analysis import compute_power_spectral_density, detect_harmonics

        df, signal, fs = self._create_test_signal()

        # Segment phases
        result = segment_test_phases(df, 'pressure', time_col='time_s')
        assert result.n_phases >= 2

        # Find steady state phase
        steady_phases = [p for p in result.phases
                         if p.phase == TestPhase.STEADY_STATE]

        if steady_phases:
            steady = steady_phases[0]
            start_idx = int(steady.start_ms / 1000 * fs)
            end_idx = int(steady.end_ms / 1000 * fs)
            steady_signal = signal[start_idx:end_idx]

            # Run PSD on steady state
            if len(steady_signal) > 64:
                psd = compute_power_spectral_density(steady_signal, sample_rate_hz=fs)
                assert psd.dominant_frequency > 0
                assert psd.total_power > 0

    def test_startup_transient_metrics(self):
        """Verify startup transient metrics are physically reasonable."""
        from core.transient_analysis import analyze_startup_transient

        df, _, _ = self._create_test_signal()
        startup_df = df.iloc[200:600].copy()
        startup_df = startup_df.reset_index(drop=True)

        metrics = analyze_startup_transient(
            startup_df, 'pressure', steady_value=30.0, time_col='time_s'
        )

        assert metrics is not None
        assert isinstance(metrics, dict)
        # Rise time should be positive
        if 'rise_time_s' in metrics and metrics['rise_time_s'] is not None:
            assert metrics['rise_time_s'] > 0

    def test_frequency_bands_computed_successfully(self):
        """Verify frequency band powers are computed and non-negative."""
        from core.frequency_analysis import (
            compute_power_spectral_density, compute_frequency_bands
        )

        np.random.seed(42)
        signal = np.random.normal(0, 1, 1000)
        fs = 1000

        psd = compute_power_spectral_density(signal, sample_rate_hz=fs)
        bands = compute_frequency_bands(psd)

        assert len(bands) > 0, "Should have at least one frequency band"
        for band_name, power in bands.items():
            assert power >= 0, f"Band {band_name} power should be non-negative"
        assert psd.total_power > 0, "Total power should be positive"


# =============================================================================
# Dynamic Schema Tests
# =============================================================================

class TestDynamicSchemaIntegration:
    """Test dynamic schema works with new test types."""

    def test_schema_builder_cold_flow(self):
        from core.campaign_manager_v2 import build_schema_for_test_type
        schema = build_schema_for_test_type('cold_flow')
        assert 'CREATE TABLE' in schema
        assert 'avg_cd_CALC' in schema
        assert 'avg_p_up_bar' in schema

    def test_schema_builder_hot_fire(self):
        from core.campaign_manager_v2 import build_schema_for_test_type
        schema = build_schema_for_test_type('hot_fire')
        assert 'CREATE TABLE' in schema
        assert 'avg_isp_s' in schema or 'avg_pc_bar' in schema

    def test_schema_builder_unknown_type(self):
        from core.campaign_manager_v2 import build_schema_for_test_type
        schema = build_schema_for_test_type('valve_timing')
        assert 'CREATE TABLE' in schema
        assert 'test_id' in schema
        # Should have base columns but not cold_flow/hot_fire specific ones
        assert 'avg_cd_CALC' not in schema

    def test_backward_compat_aliases(self):
        from core.campaign_manager_v2 import COLD_FLOW_SCHEMA_V2, HOT_FIRE_SCHEMA_V2
        assert 'CREATE TABLE' in COLD_FLOW_SCHEMA_V2
        assert 'CREATE TABLE' in HOT_FIRE_SCHEMA_V2
