"""
Phase 2 Feature Tests
=====================
Tests for Phase 2 Engineering Excellence features:
- Transient analysis engine
- FFT/frequency analysis toolkit
- CUSUM and EWMA control charts
- Dynamic database schema

Run with: python -m pytest tests/test_phase2_features.py -v
"""

import sys
import os
import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path
from typing import Dict, Any

# Add core to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# TRANSIENT ANALYSIS TESTS
# =============================================================================

class TestTransientAnalysis:
    """Tests for multi-phase transient analysis."""

    def _create_test_signal(self, n=2000, sample_rate=100):
        """Create a synthetic test signal with known phases."""
        t = np.arange(n) / sample_rate  # seconds

        signal = np.zeros(n)

        # Phase 1: Pretest (0-2s) - baseline noise near 0
        signal[:200] = np.random.normal(0.5, 0.05, 200)

        # Phase 2: Startup (2-4s) - ramp from 0 to 30
        signal[200:400] = np.linspace(0.5, 30.0, 200)

        # Phase 3: Steady state (4-14s) - stable around 30
        signal[400:1400] = 30.0 + np.random.normal(0, 0.3, 1000)

        # Phase 4: Shutdown (14-16s) - ramp from 30 to 0
        signal[1400:1600] = np.linspace(30.0, 0.5, 200)

        # Phase 5: Cooldown (16-20s) - back to baseline
        signal[1600:] = np.random.normal(0.5, 0.05, n - 1600)

        df = pd.DataFrame({
            'time_s': t,
            'pressure': signal,
        })
        return df

    def test_segment_test_phases(self):
        """Test automatic phase segmentation."""
        from core.transient_analysis import segment_test_phases, TestPhase

        df = self._create_test_signal()
        result = segment_test_phases(df, 'pressure', time_col='time_s')

        assert result is not None
        assert len(result.phases) >= 2  # At least startup/steady or similar
        assert result.total_duration_s > 0

        # Check that we found some phases
        phase_types = [p.phase for p in result.phases]
        assert len(phase_types) >= 2

        print(f"[PASS] Segmented into {len(result.phases)} phases: "
              f"{[p.phase.value for p in result.phases]}")

    def test_analyze_startup_transient(self):
        """Test startup transient metrics."""
        from core.transient_analysis import analyze_startup_transient

        df = self._create_test_signal()
        metrics = analyze_startup_transient(
            df, 'pressure', time_col='time_s', steady_value=30.0
        )

        assert metrics is not None
        assert 'rise_time_s' in metrics
        assert metrics['rise_time_s'] > 0
        assert metrics['rise_time_s'] < 5.0  # Should be around 2s

        print(f"[PASS] Startup transient: rise_time={metrics['rise_time_s']:.3f}s")

    def test_analyze_shutdown_transient(self):
        """Test shutdown transient metrics."""
        from core.transient_analysis import analyze_shutdown_transient

        df = self._create_test_signal()
        metrics = analyze_shutdown_transient(
            df, 'pressure', time_col='time_s', steady_value=30.0
        )

        assert metrics is not None
        assert 'decay_time_s' in metrics
        assert metrics['decay_time_s'] > 0

        print(f"[PASS] Shutdown transient: decay_time={metrics['decay_time_s']:.3f}s")

    def test_phase_result_dataclass(self):
        """Test PhaseResult dataclass properties."""
        from core.transient_analysis import PhaseResult, TestPhase

        phase = PhaseResult(
            phase=TestPhase.STEADY_STATE,
            start_ms=4000.0,
            end_ms=14000.0,
            duration_s=10.0,
            metrics={'mean': 30.0, 'std': 0.3},
        )

        assert phase.duration_s == 10.0
        assert phase.phase == TestPhase.STEADY_STATE
        assert phase.metrics['mean'] == 30.0

        print(f"[PASS] PhaseResult: {phase.phase.value}, duration={phase.duration_s}s")

    def test_empty_signal(self):
        """Test handling of empty/short signal raises ValueError."""
        from core.transient_analysis import segment_test_phases

        df = pd.DataFrame({'time_s': [0, 0.01], 'pressure': [1.0, 1.0]})

        # Should raise ValueError for too-short signals (< 10 rows)
        try:
            result = segment_test_phases(df, 'pressure', time_col='time_s')
            assert False, "Expected ValueError for short signal"
        except ValueError as e:
            assert "at least 10 rows" in str(e)

        print("[PASS] Empty signal raises ValueError as expected")


# =============================================================================
# FFT / FREQUENCY ANALYSIS TESTS
# =============================================================================

class TestFrequencyAnalysis:
    """Tests for spectral/frequency analysis toolkit."""

    def _create_test_signal(self, n=4096, sample_rate=1000):
        """Create synthetic signal with known frequency content."""
        t = np.arange(n) / sample_rate
        # 10 Hz fundamental + 30 Hz harmonic + noise
        signal = (5.0 * np.sin(2 * np.pi * 10 * t) +
                  2.0 * np.sin(2 * np.pi * 30 * t) +
                  np.random.normal(0, 0.5, n))
        return signal, sample_rate

    def test_power_spectral_density(self):
        """Test Welch PSD estimation."""
        from core.frequency_analysis import compute_power_spectral_density

        signal, fs = self._create_test_signal()
        result = compute_power_spectral_density(signal, sample_rate_hz=fs)

        assert result.frequencies is not None
        assert result.power_spectral_density is not None
        assert result.dominant_frequency is not None

        # Dominant frequency should be around 10 Hz
        assert abs(result.dominant_frequency - 10.0) < 2.0

        print(f"[PASS] PSD: dominant freq={result.dominant_frequency:.1f} Hz")

    def test_detect_harmonics(self):
        """Test harmonic detection."""
        from core.frequency_analysis import (
            compute_power_spectral_density,
            detect_harmonics,
        )

        signal, fs = self._create_test_signal()
        psd = compute_power_spectral_density(signal, sample_rate_hz=fs)
        harmonics = detect_harmonics(psd, n_harmonics=5)

        assert len(harmonics) >= 1

        # Should detect fundamental around 10 Hz
        fundamental = harmonics[0]
        assert abs(fundamental.frequency - 10.0) < 2.0

        print(f"[PASS] Harmonics: {len(harmonics)} found, "
              f"fundamental={fundamental.frequency:.1f} Hz")

    def test_cross_spectrum(self):
        """Test cross-spectral analysis between two channels."""
        from core.frequency_analysis import compute_cross_spectrum

        n, fs = 4096, 1000
        t = np.arange(n) / fs
        # Two correlated signals (same frequency, different phase)
        sig_a = np.sin(2 * np.pi * 10 * t) + np.random.normal(0, 0.1, n)
        sig_b = np.sin(2 * np.pi * 10 * t + np.pi/4) + np.random.normal(0, 0.1, n)

        result = compute_cross_spectrum(sig_a, sig_b, sample_rate_hz=fs)

        assert result.coherence is not None
        assert result.phase is not None

        # Coherence should be high near 10 Hz
        freq_10_idx = np.argmin(np.abs(result.frequencies - 10.0))
        assert result.coherence[freq_10_idx] > 0.5

        print(f"[PASS] Cross spectrum: coherence at 10Hz = "
              f"{result.coherence[freq_10_idx]:.3f}")

    def test_detect_resonance(self):
        """Test resonance peak detection."""
        from core.frequency_analysis import (
            compute_power_spectral_density,
            detect_resonance,
        )

        signal, fs = self._create_test_signal()
        psd = compute_power_spectral_density(signal, sample_rate_hz=fs)
        resonances = detect_resonance(psd, prominence=2.0)

        assert len(resonances) >= 1
        # First resonance should be near 10 Hz
        assert abs(resonances[0]['frequency'] - 10.0) < 3.0

        print(f"[PASS] Resonance: {len(resonances)} peaks, "
              f"first at {resonances[0]['frequency']:.1f} Hz")

    def test_frequency_bands(self):
        """Test frequency band power integration."""
        from core.frequency_analysis import (
            compute_power_spectral_density,
            compute_frequency_bands,
        )

        signal, fs = self._create_test_signal()
        psd = compute_power_spectral_density(signal, sample_rate_hz=fs)
        bands = compute_frequency_bands(psd)

        assert 'low' in bands
        assert 'mid' in bands
        assert all(v >= 0 for v in bands.values())

        print(f"[PASS] Frequency bands: {bands}")

    def test_spectrogram(self):
        """Test time-frequency spectrogram."""
        from core.frequency_analysis import compute_spectrogram

        signal, fs = self._create_test_signal()
        freqs, times, Sxx = compute_spectrogram(signal, sample_rate_hz=fs)

        assert freqs is not None
        assert times is not None
        assert Sxx.shape[0] == len(freqs)
        assert Sxx.shape[1] == len(times)

        print(f"[PASS] Spectrogram: {Sxx.shape[0]} freq bins x {Sxx.shape[1]} time bins")

    def test_short_signal(self):
        """Test handling of very short signal raises ValueError."""
        from core.frequency_analysis import compute_power_spectral_density

        signal = np.array([1.0, 2.0, 1.0, 2.0])

        # Should raise ValueError for signals shorter than minimum (8 samples)
        try:
            result = compute_power_spectral_density(signal, sample_rate_hz=100)
            assert False, "Expected ValueError for short signal"
        except ValueError as e:
            assert "samples" in str(e).lower()

        print("[PASS] Short signal raises ValueError as expected")


# =============================================================================
# SPC CUSUM/EWMA TESTS
# =============================================================================

class TestCUSUM:
    """Tests for CUSUM control charts."""

    def test_cusum_in_control(self):
        """Test CUSUM on stable process."""
        from core.spc import create_cusum_chart

        np.random.seed(42)
        values = np.random.normal(50, 1, 50)  # Stable process

        result = create_cusum_chart(values, parameter_name='test_param')

        assert result.n_signals == 0 or result.n_signals <= 2  # Minimal signals
        assert result.target is not None
        assert len(result.c_plus) == len(values)
        assert len(result.c_minus) == len(values)

        print(f"[PASS] CUSUM in-control: {result.n_signals} signals")

    def test_cusum_detects_shift(self):
        """Test CUSUM detects mean shift."""
        from core.spc import create_cusum_chart

        np.random.seed(42)
        # Process with shift at point 25
        values = np.concatenate([
            np.random.normal(50, 1, 25),
            np.random.normal(52, 1, 25),  # 2-sigma shift
        ])

        result = create_cusum_chart(values, target=50.0, parameter_name='shift_test')

        # CUSUM should detect the shift
        assert result.n_signals > 0
        # Signals should be after the shift point
        if result.signals_upper:
            assert max(result.signals_upper) >= 25

        print(f"[PASS] CUSUM shift detected: {result.n_signals} signals")

    def test_cusum_custom_parameters(self):
        """Test CUSUM with custom k and h."""
        from core.spc import create_cusum_chart

        values = np.random.normal(100, 2, 30)
        result = create_cusum_chart(
            values, target=100.0, sigma=2.0, k=0.5, h=4.0,
            parameter_name='custom'
        )

        assert result.k == 0.5
        assert result.h == 4.0
        assert result.target == 100.0

        print("[PASS] CUSUM custom parameters")


class TestEWMA:
    """Tests for EWMA control charts."""

    def test_ewma_in_control(self):
        """Test EWMA on stable process."""
        from core.spc import create_ewma_chart

        np.random.seed(42)
        values = np.random.normal(50, 1, 50)

        result = create_ewma_chart(values, parameter_name='test_param')

        assert result.n_signals <= 2  # Should be stable
        assert len(result.ewma_values) == len(values)
        assert len(result.ucl) == len(values)
        assert len(result.lcl) == len(values)

        print(f"[PASS] EWMA in-control: {result.n_signals} signals")

    def test_ewma_detects_drift(self):
        """Test EWMA detects gradual drift."""
        from core.spc import create_ewma_chart

        np.random.seed(42)
        # Gradually drifting process
        n = 50
        drift = np.linspace(0, 4, n)  # 4-sigma drift
        values = np.random.normal(50, 1, n) + drift

        result = create_ewma_chart(values, target=50.0, parameter_name='drift')

        # Should detect drift eventually
        assert result.n_signals > 0

        print(f"[PASS] EWMA drift detected: {result.n_signals} signals")

    def test_ewma_custom_lambda(self):
        """Test EWMA with different lambda values."""
        from core.spc import create_ewma_chart

        np.random.seed(42)
        values = np.random.normal(50, 1, 30)

        # Small lambda (more smoothing)
        r1 = create_ewma_chart(values, lambda_param=0.1, parameter_name='low_lambda')
        # Large lambda (less smoothing, closer to raw data)
        r2 = create_ewma_chart(values, lambda_param=0.4, parameter_name='high_lambda')

        assert r1.lambda_param == 0.1
        assert r2.lambda_param == 0.4

        # Low lambda should smooth more -> smaller range in EWMA values
        r1_range = np.ptp(r1.ewma_values)
        r2_range = np.ptp(r2.ewma_values)
        assert r1_range <= r2_range * 1.1  # Allow small margin

        print(f"[PASS] EWMA lambda: low_range={r1_range:.3f}, high_range={r2_range:.3f}")

    def test_ewma_time_varying_limits(self):
        """Test that EWMA limits converge to steady-state."""
        from core.spc import create_ewma_chart

        values = np.random.normal(50, 1, 100)
        result = create_ewma_chart(values, parameter_name='limits')

        # Limits should widen from zero at start to steady state
        assert result.ucl[0] < result.ucl[-1] or abs(result.ucl[0] - result.ucl[-1]) < 1
        # UCL should be above center, LCL below
        assert all(result.ucl >= result.center_line - 1e-10)
        assert all(result.lcl <= result.center_line + 1e-10)

        print("[PASS] EWMA time-varying limits")


# =============================================================================
# DYNAMIC DATABASE SCHEMA TESTS
# =============================================================================

class TestDynamicSchema:
    """Tests for dynamic database schema generation."""

    def test_build_schema_cold_flow(self):
        """Test schema generation for cold flow."""
        from core.campaign_manager_v2 import build_schema_for_test_type

        sql = build_schema_for_test_type('cold_flow')

        assert 'CREATE TABLE test_results' in sql
        assert 'test_id TEXT PRIMARY KEY' in sql
        assert 'avg_cd_CALC REAL' in sql
        assert 'raw_data_hash TEXT' in sql

        print("[PASS] Cold flow schema generated correctly")

    def test_build_schema_hot_fire(self):
        """Test schema generation for hot fire."""
        from core.campaign_manager_v2 import build_schema_for_test_type

        sql = build_schema_for_test_type('hot_fire')

        assert 'CREATE TABLE test_results' in sql
        assert 'avg_pc_bar REAL' in sql
        assert 'avg_isp_s REAL' in sql
        assert 'raw_data_hash TEXT' in sql

        print("[PASS] Hot fire schema generated correctly")

    def test_build_schema_unknown_type(self):
        """Test schema generation for unknown type (base columns only)."""
        from core.campaign_manager_v2 import build_schema_for_test_type

        sql = build_schema_for_test_type('valve_timing')

        assert 'CREATE TABLE test_results' in sql
        assert 'test_id TEXT PRIMARY KEY' in sql
        assert 'raw_data_hash TEXT' in sql
        # Should NOT have cold_flow or hot_fire specific columns
        assert 'avg_cd_CALC' not in sql
        assert 'avg_pc_bar' not in sql

        print("[PASS] Unknown type gets base schema only")

    def test_schema_version_3(self):
        """Test schema version is now 3."""
        from core.campaign_manager_v2 import SCHEMA_VERSION
        assert SCHEMA_VERSION == 3

        print(f"[PASS] Schema version: {SCHEMA_VERSION}")

    def test_create_campaign_with_dynamic_schema(self):
        """Test creating campaign uses dynamic schema."""
        import tempfile
        from core.campaign_manager_v2 import (
            create_campaign,
            CAMPAIGN_DIR,
        )

        # Use temp directory
        import core.campaign_manager_v2 as cm
        old_dir = cm.CAMPAIGN_DIR
        try:
            cm.CAMPAIGN_DIR = tempfile.mkdtemp()

            db_path = create_campaign('dynamic_test', 'cold_flow')
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            c.execute("PRAGMA table_info(test_results)")
            columns = {row[1] for row in c.fetchall()}
            conn.close()

            # Should have both base and cold_flow columns
            assert 'test_id' in columns
            assert 'raw_data_hash' in columns
            assert 'avg_cd_CALC' in columns
            assert 'avg_p_up_bar' in columns

            print(f"[PASS] Dynamic schema campaign: {len(columns)} columns")
        finally:
            cm.CAMPAIGN_DIR = old_dir

    def test_backward_compat_aliases(self):
        """Test backward-compatible schema aliases still work."""
        from core.campaign_manager_v2 import COLD_FLOW_SCHEMA_V2, HOT_FIRE_SCHEMA_V2

        assert 'test_id' in COLD_FLOW_SCHEMA_V2
        assert 'avg_cd_CALC' in COLD_FLOW_SCHEMA_V2
        assert 'test_id' in HOT_FIRE_SCHEMA_V2
        assert 'avg_pc_bar' in HOT_FIRE_SCHEMA_V2

        print("[PASS] Backward-compatible aliases work")


# =============================================================================
# SPC CHART TYPE ENUM TESTS
# =============================================================================

class TestSPCChartTypes:
    """Test that new chart types are in the enum."""

    def test_cusum_in_enum(self):
        """Test CUSUM chart type exists."""
        from core.spc import ControlChartType
        assert hasattr(ControlChartType, 'CUSUM')
        print("[PASS] CUSUM in ControlChartType")

    def test_ewma_in_enum(self):
        """Test EWMA chart type exists."""
        from core.spc import ControlChartType
        assert hasattr(ControlChartType, 'EWMA')
        print("[PASS] EWMA in ControlChartType")
