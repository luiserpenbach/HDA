"""
Extended QC Check Tests (P0 Critical)
=====================================
Tests for QC functions not covered by test_p0_components.py

Run with: python -m pytest tests/test_qc_extended.py -v
Or:       python tests/test_qc_extended.py
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Add core to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.qc_checks import (
    QCStatus,
    QCCheckResult,
    QCReport,
    check_timestamp_monotonic,
    check_timestamp_gaps,
    check_sampling_rate_stability,
    check_sensor_range,
    check_sensor_ranges_from_config,
    check_flatline,
    check_saturation,
    check_nan_ratio,
    check_pressure_flow_correlation,
    run_qc_checks,
    run_quick_qc,
    assert_qc_passed,
    format_qc_for_display,
)


def create_clean_dataframe(n_samples=1000):
    """Create a clean test dataframe with no issues."""
    np.random.seed(42)
    t = np.arange(n_samples) * 10  # 10ms intervals

    # Pressure: steady with small noise
    pressure = 25 + np.random.normal(0, 0.2, n_samples)

    # Flow: correlated with pressure
    flow = pressure * 0.5 + np.random.normal(0, 0.1, n_samples)

    return pd.DataFrame({
        'timestamp': t,
        'PT-01': pressure,
        'FM-01': flow,
    })


class TestSamplingRateStability:
    """Tests for check_sampling_rate_stability function."""

    def test_stable_sampling_rate_pass(self):
        """Test that stable sampling rate passes."""
        df = pd.DataFrame({
            'timestamp': np.arange(0, 10000, 10)  # Perfect 10ms intervals
        })

        result = check_sampling_rate_stability(df, 'timestamp', max_cv_percent=10.0)

        assert result.status == QCStatus.PASS
        assert result.details['cv_percent'] < 1.0
        assert abs(result.details['estimated_rate_hz'] - 100) < 1

        print(f"[PASS] Stable sampling: {result.message}")

    def test_unstable_sampling_rate_warns(self):
        """Test that unstable sampling rate warns."""
        np.random.seed(42)
        # Varying intervals
        intervals = 10 + np.random.normal(0, 3, 999)  # Mean 10, std 3 (30% CV)
        timestamps = np.cumsum(np.concatenate([[0], intervals]))

        df = pd.DataFrame({'timestamp': timestamps})

        result = check_sampling_rate_stability(df, 'timestamp', max_cv_percent=10.0)

        assert result.status == QCStatus.WARN
        assert result.details['cv_percent'] > 10.0

        print(f"[PASS] Unstable sampling detected: CV={result.details['cv_percent']:.1f}%")

    def test_sampling_rate_missing_column(self):
        """Test handling of missing time column."""
        df = pd.DataFrame({'other_col': [1, 2, 3]})

        result = check_sampling_rate_stability(df, 'timestamp')

        assert result.status == QCStatus.SKIP
        assert 'not found' in result.message

        print("[PASS] Missing column handled correctly")

    def test_sampling_rate_insufficient_data(self):
        """Test with too few samples."""
        df = pd.DataFrame({'timestamp': [0, 10, 20]})  # Only 3 samples

        result = check_sampling_rate_stability(df, 'timestamp')

        assert result.status == QCStatus.SKIP
        assert 'Insufficient' in result.message

        print("[PASS] Insufficient data handled correctly")


class TestSensorRange:
    """Tests for check_sensor_range function."""

    def test_sensor_range_pass(self):
        """Test sensor within range passes."""
        df = pd.DataFrame({'PT-01': np.linspace(10, 30, 100)})

        result = check_sensor_range(df, 'PT-01', min_val=0, max_val=50)

        assert result.status == QCStatus.PASS
        assert result.details['min'] >= 10
        assert result.details['max'] <= 30

        print(f"[PASS] Sensor in range: {result.message}")

    def test_sensor_range_below_min_fails(self):
        """Test sensor below min fails."""
        df = pd.DataFrame({'PT-01': np.linspace(-5, 30, 100)})

        result = check_sensor_range(df, 'PT-01', min_val=0, max_val=50)

        assert result.status == QCStatus.FAIL
        assert 'below min' in result.message.lower()

        print(f"[PASS] Below min detected: {result.message}")

    def test_sensor_range_above_max_fails(self):
        """Test sensor above max fails."""
        df = pd.DataFrame({'PT-01': np.linspace(10, 100, 100)})

        result = check_sensor_range(df, 'PT-01', min_val=0, max_val=50)

        assert result.status == QCStatus.FAIL
        assert 'above max' in result.message.lower()

        print(f"[PASS] Above max detected: {result.message}")

    def test_sensor_range_negative_not_allowed(self):
        """Test negative values when not allowed."""
        df = pd.DataFrame({'PT-01': np.linspace(-5, 30, 100)})

        result = check_sensor_range(df, 'PT-01', allow_negative=False)

        assert result.status == QCStatus.FAIL
        assert 'negative' in result.message.lower()

        print(f"[PASS] Negative values detected: {result.message}")

    def test_sensor_range_missing_column(self):
        """Test missing column handling."""
        df = pd.DataFrame({'other': [1, 2, 3]})

        result = check_sensor_range(df, 'PT-01', min_val=0, max_val=50)

        assert result.status == QCStatus.SKIP

        print("[PASS] Missing column handled correctly")

    def test_sensor_range_all_nan(self):
        """Test all NaN data."""
        df = pd.DataFrame({'PT-01': [np.nan, np.nan, np.nan]})

        result = check_sensor_range(df, 'PT-01', min_val=0, max_val=50)

        assert result.status == QCStatus.FAIL
        assert 'no valid data' in result.message.lower()

        print("[PASS] All NaN handled correctly")


class TestSensorRangesFromConfig:
    """Tests for check_sensor_ranges_from_config function."""

    def test_ranges_from_config_multiple_sensors(self):
        """Test checking multiple sensors from config."""
        df = pd.DataFrame({
            'PT-01': np.linspace(10, 30, 100),
            'FM-01': np.linspace(5, 15, 100),
        })

        config = {
            'sensor_limits': {
                'PT-01': {'min': 0, 'max': 50, 'unit': 'bar'},
                'FM-01': {'min': 0, 'max': 100, 'unit': 'g/s'},
            }
        }

        results = check_sensor_ranges_from_config(df, config)

        assert len(results) == 2
        assert all(r.status == QCStatus.PASS for r in results)

        print(f"[PASS] Multiple sensors from config: {len(results)} checks")

    def test_ranges_from_config_some_missing(self):
        """Test config with sensors not in dataframe."""
        df = pd.DataFrame({
            'PT-01': np.linspace(10, 30, 100),
        })

        config = {
            'sensor_limits': {
                'PT-01': {'min': 0, 'max': 50},
                'PT-02': {'min': 0, 'max': 50},  # Not in df
            }
        }

        results = check_sensor_ranges_from_config(df, config)

        # Should only have 1 result (PT-02 skipped)
        assert len(results) == 1
        assert results[0].name == 'sensor_range_PT-01'

        print("[PASS] Missing sensors skipped correctly")

    def test_ranges_from_config_empty(self):
        """Test empty config."""
        df = create_clean_dataframe()
        config = {}

        results = check_sensor_ranges_from_config(df, config)

        assert results == []

        print("[PASS] Empty config handled")


class TestSaturation:
    """Tests for check_saturation function."""

    def test_saturation_pass(self):
        """Test no saturation detected."""
        np.random.seed(42)
        df = pd.DataFrame({
            'PT-01': np.random.normal(50, 5, 1000)  # Normal distribution
        })

        result = check_saturation(df, 'PT-01')

        assert result.status == QCStatus.PASS

        print(f"[PASS] No saturation: {result.message}")

    def test_saturation_at_max(self):
        """Test saturation at maximum value."""
        data = np.ones(1000) * 50
        data[:100] = np.linspace(0, 50, 100)  # 10% at max

        df = pd.DataFrame({'PT-01': data})

        result = check_saturation(df, 'PT-01', saturation_threshold=0.05)

        assert result.status in (QCStatus.WARN, QCStatus.FAIL)
        assert 'max' in result.message.lower()

        print(f"[PASS] Max saturation detected: {result.message}")

    def test_saturation_at_min(self):
        """Test saturation at minimum value."""
        data = np.ones(1000) * 0
        data[:100] = np.linspace(0, 50, 100)  # 90% at min

        df = pd.DataFrame({'PT-01': data})

        result = check_saturation(df, 'PT-01', saturation_threshold=0.05)

        assert result.status in (QCStatus.WARN, QCStatus.FAIL)
        assert 'min' in result.message.lower()

        print(f"[PASS] Min saturation detected: {result.message}")

    def test_saturation_missing_column(self):
        """Test missing column handling."""
        df = pd.DataFrame({'other': [1, 2, 3]})

        result = check_saturation(df, 'PT-01')

        assert result.status == QCStatus.SKIP

        print("[PASS] Missing column handled")


class TestPressureFlowCorrelation:
    """Tests for check_pressure_flow_correlation function."""

    def test_correlation_positive_pass(self):
        """Test positive correlation passes."""
        np.random.seed(42)
        pressure = np.linspace(10, 50, 100) + np.random.normal(0, 1, 100)
        flow = pressure * 0.5 + np.random.normal(0, 1, 100)

        df = pd.DataFrame({'PT-01': pressure, 'FM-01': flow})

        result = check_pressure_flow_correlation(df, 'PT-01', 'FM-01')

        assert result.status == QCStatus.PASS
        assert result.details['correlation'] > 0.9

        print(f"[PASS] Positive correlation: {result.details['correlation']:.3f}")

    def test_correlation_negative_fails(self):
        """Test negative correlation fails."""
        pressure = np.linspace(10, 50, 100)
        flow = -pressure + 60  # Negative correlation

        df = pd.DataFrame({'PT-01': pressure, 'FM-01': flow})

        result = check_pressure_flow_correlation(df, 'PT-01', 'FM-01')

        assert result.status == QCStatus.FAIL
        assert result.details['correlation'] < 0
        assert 'NEGATIVE' in result.message

        print(f"[PASS] Negative correlation detected: {result.details['correlation']:.3f}")

    def test_correlation_weak_warns(self):
        """Test weak correlation warns."""
        np.random.seed(42)
        pressure = np.linspace(10, 50, 100)
        flow = np.random.normal(25, 10, 100)  # No correlation with pressure

        df = pd.DataFrame({'PT-01': pressure, 'FM-01': flow})

        result = check_pressure_flow_correlation(df, 'PT-01', 'FM-01', min_correlation=0.3)

        assert result.status in (QCStatus.WARN, QCStatus.FAIL)
        assert abs(result.details['correlation']) < 0.3

        print(f"[PASS] Weak correlation detected: {result.details['correlation']:.3f}")

    def test_correlation_missing_columns(self):
        """Test missing columns handling."""
        df = pd.DataFrame({'PT-01': [1, 2, 3]})

        result = check_pressure_flow_correlation(df, 'PT-01', 'FM-01')

        assert result.status == QCStatus.SKIP
        assert 'Missing' in result.message

        print("[PASS] Missing columns handled")

    def test_correlation_insufficient_data(self):
        """Test with too few data points."""
        df = pd.DataFrame({'PT-01': [1, 2], 'FM-01': [3, 4]})

        result = check_pressure_flow_correlation(df, 'PT-01', 'FM-01')

        assert result.status == QCStatus.SKIP
        assert 'Insufficient' in result.message

        print("[PASS] Insufficient data handled")


class TestRunQuickQC:
    """Tests for run_quick_qc function."""

    def test_quick_qc_clean_data(self):
        """Test quick QC on clean data."""
        df = create_clean_dataframe()

        report = run_quick_qc(df, time_col='timestamp')

        assert report.passed
        assert len(report.checks) > 0

        print(f"[PASS] Quick QC passed: {len(report.checks)} checks")

    def test_quick_qc_non_monotonic_timestamp(self):
        """Test quick QC catches non-monotonic timestamps."""
        df = pd.DataFrame({
            'timestamp': [0, 10, 5, 20, 30],  # Non-monotonic
            'PT-01': [1, 2, 3, 4, 5],
        })

        report = run_quick_qc(df, time_col='timestamp')

        assert not report.passed
        assert any('monotonic' in c.name for c in report.blocking_failures)

        print("[PASS] Quick QC catches non-monotonic timestamps")

    def test_quick_qc_high_nan(self):
        """Test quick QC catches high NaN ratio."""
        df = pd.DataFrame({
            'timestamp': range(100),
            'PT-01': [np.nan] * 60 + list(range(40)),  # 60% NaN
        })

        report = run_quick_qc(df, time_col='timestamp')

        assert not report.passed

        print("[PASS] Quick QC catches high NaN ratio")

    def test_quick_qc_constant_column(self):
        """Test quick QC catches constant columns."""
        df = pd.DataFrame({
            'timestamp': range(100),
            'PT-01': [50.0] * 100,  # All constant
        })

        report = run_quick_qc(df, time_col='timestamp')

        assert not report.passed
        assert any('constant' in c.name for c in report.blocking_failures)

        print("[PASS] Quick QC catches constant columns")


class TestFormatQCForDisplay:
    """Tests for format_qc_for_display function."""

    def test_format_passed_report(self):
        """Test formatting a passed QC report."""
        report = QCReport()
        report.add_check(QCCheckResult(
            name='test_check',
            status=QCStatus.PASS,
            message='Test passed'
        ))

        formatted = format_qc_for_display(report)

        assert 'PASSED' in formatted
        assert 'test_check' in formatted

        print("[PASS] Formatted passed report")

    def test_format_failed_report(self):
        """Test formatting a failed QC report."""
        report = QCReport()
        report.add_check(QCCheckResult(
            name='failing_check',
            status=QCStatus.FAIL,
            message='Test failed',
            blocking=True
        ))

        formatted = format_qc_for_display(report)

        assert 'FAILED' in formatted
        assert 'Blocking Failures' in formatted
        assert 'failing_check' in formatted

        print("[PASS] Formatted failed report")

    def test_format_report_with_warnings(self):
        """Test formatting report with warnings."""
        report = QCReport()
        report.add_check(QCCheckResult(
            name='pass_check',
            status=QCStatus.PASS,
            message='Passed'
        ))
        report.add_check(QCCheckResult(
            name='warn_check',
            status=QCStatus.WARN,
            message='Warning issued',
            blocking=False
        ))

        formatted = format_qc_for_display(report)

        assert 'Warnings' in formatted
        assert 'warn_check' in formatted

        print("[PASS] Formatted report with warnings")


class TestQCReportClass:
    """Tests for QCReport class properties."""

    def test_qc_report_passed(self):
        """Test QCReport.passed property."""
        report = QCReport()
        report.add_check(QCCheckResult('a', QCStatus.PASS, 'ok'))
        report.add_check(QCCheckResult('b', QCStatus.PASS, 'ok'))

        assert report.passed == True

        print("[PASS] Report correctly shows passed")

    def test_qc_report_failed_blocking(self):
        """Test QCReport.passed with blocking failure."""
        report = QCReport()
        report.add_check(QCCheckResult('a', QCStatus.PASS, 'ok'))
        report.add_check(QCCheckResult('b', QCStatus.FAIL, 'failed', blocking=True))

        assert report.passed == False

        print("[PASS] Report correctly shows failed (blocking)")

    def test_qc_report_non_blocking_failure(self):
        """Test QCReport with non-blocking failure still passes."""
        report = QCReport()
        report.add_check(QCCheckResult('a', QCStatus.PASS, 'ok'))
        report.add_check(QCCheckResult('b', QCStatus.FAIL, 'failed', blocking=False))

        assert report.passed == True  # Non-blocking doesn't fail report

        print("[PASS] Non-blocking failure doesn't fail report")

    def test_qc_report_summary(self):
        """Test QCReport.summary property."""
        report = QCReport()
        report.add_check(QCCheckResult('a', QCStatus.PASS, 'ok'))
        report.add_check(QCCheckResult('b', QCStatus.WARN, 'warn'))
        report.add_check(QCCheckResult('c', QCStatus.FAIL, 'fail'))
        report.add_check(QCCheckResult('d', QCStatus.SKIP, 'skip'))

        summary = report.summary

        assert summary['total'] == 4
        assert summary['passed'] == 1
        assert summary['warnings'] == 1
        assert summary['failed'] == 1
        assert summary['skipped'] == 1

        print(f"[PASS] Summary: {summary}")

    def test_qc_report_to_dict(self):
        """Test QCReport serialization."""
        report = QCReport()
        report.add_check(QCCheckResult('a', QCStatus.PASS, 'ok'))

        d = report.to_dict()

        assert 'passed' in d
        assert 'summary' in d
        assert 'checks' in d
        assert 'timestamp' in d

        print("[PASS] Report to_dict works")


class TestAssertQCPassed:
    """Tests for assert_qc_passed function."""

    def test_assert_passed_no_raise(self):
        """Test assert_qc_passed with passing report."""
        report = QCReport()
        report.add_check(QCCheckResult('a', QCStatus.PASS, 'ok'))

        result = assert_qc_passed(report, raise_on_fail=True)
        assert result == True

        print("[PASS] Assert passed correctly")

    def test_assert_failed_raises(self):
        """Test assert_qc_passed raises on failure."""
        report = QCReport()
        report.add_check(QCCheckResult('a', QCStatus.FAIL, 'failed', blocking=True))

        try:
            assert_qc_passed(report, raise_on_fail=True)
            assert False, "Should have raised"
        except ValueError as e:
            assert 'FAILED' in str(e)

        print("[PASS] Assert raises on failure")

    def test_assert_failed_no_raise(self):
        """Test assert_qc_passed without raising."""
        report = QCReport()
        report.add_check(QCCheckResult('a', QCStatus.FAIL, 'failed', blocking=True))

        result = assert_qc_passed(report, raise_on_fail=False)
        assert result == False

        print("[PASS] Assert returns False without raising")


def run_all_tests():
    """Run all extended QC tests."""
    print("=" * 60)
    print("Extended QC Check Tests (P0 Critical)")
    print("=" * 60)

    test_classes = [
        TestSamplingRateStability,
        TestSensorRange,
        TestSensorRangesFromConfig,
        TestSaturation,
        TestPressureFlowCorrelation,
        TestRunQuickQC,
        TestFormatQCForDisplay,
        TestQCReportClass,
        TestAssertQCPassed,
    ]

    passed = 0
    failed = 0

    for test_class in test_classes:
        print(f"\n{test_class.__name__}")
        print("-" * 40)

        instance = test_class()
        methods = [m for m in dir(instance) if m.startswith('test_')]

        for method_name in methods:
            method = getattr(instance, method_name)
            try:
                method()
                passed += 1
            except Exception as e:
                print(f"[FAIL] {method_name}: {e}")
                import traceback
                traceback.print_exc()
                failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
