"""
Test Suite for P1 Components
============================
Tests for SPC, Reporting, Batch Analysis, and Export modules.

Run with: python tests/test_p1_components.py
"""

import sys
import os
import tempfile
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add core to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.spc import (
    calculate_imr_limits,
    check_western_electric_rules,
    calculate_capability,
    detect_trend,
    create_imr_chart,
    ControlLimits,
    ProcessCapability,
    ViolationType,
)

from core.reporting import (
    generate_test_report,
    generate_campaign_report,
    generate_measurement_table,
    save_report,
    generate_key_charts_section,
    generate_appendix_charts_section,
    _resolve_sensor_column,
    _create_time_series_chart,
    _create_bar_chart,
    PLOTLY_AVAILABLE,
)

from core.batch_analysis import (
    BatchTestResult,
    BatchAnalysisReport,
    discover_test_files,
    extract_test_id_from_path,
    load_csv_with_timestamp,
)

from core.export import (
    export_campaign_csv,
    export_campaign_json,
    create_traceability_report,
)


def create_campaign_dataframe():
    """Create a realistic campaign DataFrame for testing."""
    np.random.seed(42)
    n_tests = 20
    
    # Simulate Cd values with slight drift
    base_cd = 0.65
    drift = np.linspace(0, 0.02, n_tests)
    noise = np.random.normal(0, 0.01, n_tests)
    cd_values = base_cd + drift + noise
    
    # Add one outlier
    cd_values[15] = 0.72  # Out of control point
    
    return pd.DataFrame({
        'test_id': [f'CF-{i+1:03d}' for i in range(n_tests)],
        'test_timestamp': pd.date_range('2024-01-01', periods=n_tests, freq='D'),
        'part': ['INJ-V1'] * n_tests,
        'serial_num': [f'SN-{i+1:03d}' for i in range(n_tests)],
        'avg_cd_CALC': cd_values,
        'u_cd_CALC': np.random.uniform(0.005, 0.015, n_tests),
        'avg_p_up_bar': np.random.uniform(24, 26, n_tests),
        'avg_mf_g_s': np.random.uniform(12, 13, n_tests),
        'qc_passed': [1] * n_tests,
        'raw_data_hash': [f'sha256:abc{i:03d}' for i in range(n_tests)],
        'config_hash': ['sha256:config123'] * n_tests,
        'analyst_username': ['test_user'] * n_tests,
        'processing_version': ['2.0.0'] * n_tests,
    })


class TestSPC:
    """Test Statistical Process Control functions."""
    
    def test_calculate_imr_limits(self):
        """Test I-MR control limit calculation."""
        values = np.array([10.0, 10.2, 9.8, 10.1, 9.9, 10.0, 10.3, 9.7])
        
        i_limits, mr_limits = calculate_imr_limits(values)
        
        assert i_limits.center_line == np.mean(values)
        assert i_limits.ucl > i_limits.center_line
        assert i_limits.lcl < i_limits.center_line
        assert mr_limits.center_line > 0
        
        print(f"[PASS] I-MR limits: CL={i_limits.center_line:.3f}, UCL={i_limits.ucl:.3f}, LCL={i_limits.lcl:.3f}")
    
    def test_western_electric_rules_no_violation(self):
        """Test Western Electric rules with in-control data."""
        np.random.seed(42)
        values = np.random.normal(10, 0.1, 20)
        
        limits = ControlLimits(
            center_line=10.0,
            ucl=10.3,
            lcl=9.7,
        )
        limits.calculate_zones()
        
        violations = check_western_electric_rules(values, limits)
        
        # Most points should have no violations
        n_violations = sum(1 for v in violations if len(v) > 0)
        
        print(f"[PASS] Western Electric rules: {n_violations} points with violations out of {len(values)}")
    
    def test_western_electric_beyond_3sigma(self):
        """Test detection of point beyond 3 sigma."""
        values = np.array([10.0, 10.0, 10.0, 10.0, 15.0])  # Last point way out
        
        limits = ControlLimits(
            center_line=10.0,
            ucl=10.5,
            lcl=9.5,
        )
        limits.calculate_zones()
        
        violations = check_western_electric_rules(values, limits)
        
        assert ViolationType.BEYOND_3SIGMA in violations[-1]
        print(f"[PASS] Beyond 3σ violation detected")
    
    def test_calculate_capability(self):
        """Test process capability calculation."""
        np.random.seed(42)
        values = np.random.normal(10, 0.1, 50)
        
        capability = calculate_capability(values, usl=10.5, lsl=9.5, target=10.0)
        
        assert capability.cp is not None
        assert capability.cpk is not None
        assert capability.cp > 0
        
        print(f"[PASS] Capability: Cp={capability.cp:.2f}, Cpk={capability.cpk:.2f}")
        print(f"  {capability.summary()}")
    
    def test_detect_trend_no_trend(self):
        """Test trend detection with stable data."""
        np.random.seed(42)
        values = np.random.normal(10, 0.1, 20)
        
        has_trend, direction, slope = detect_trend(values)
        
        # Random data shouldn't have strong trend
        print(f"[PASS] Trend detection (random): has_trend={has_trend}, direction={direction}")
    
    def test_detect_trend_with_trend(self):
        """Test trend detection with trending data."""
        values = np.linspace(10, 12, 20) + np.random.normal(0, 0.05, 20)
        
        has_trend, direction, slope = detect_trend(values)
        
        assert has_trend == True
        assert direction == 'increasing'
        
        print(f"[PASS] Trend detection (increasing): direction={direction}, slope={slope:.4f}")
    
    def test_create_imr_chart(self):
        """Test full I-MR chart creation."""
        df = create_campaign_dataframe()
        
        analysis = create_imr_chart(
            df, 
            parameter='avg_cd_CALC',
            test_id_col='test_id',
            usl=0.70,
            lsl=0.60,
        )
        
        assert analysis.n_points == len(df)
        assert analysis.limits.center_line > 0
        
        print(f"[PASS] I-MR Chart: {analysis.n_points} points, {analysis.n_violations} violations")
        print(f"  Limits: CL={analysis.limits.center_line:.4f}, UCL={analysis.limits.ucl:.4f}")
        
        if analysis.capability:
            print(f"  Cpk={analysis.capability.cpk:.2f}")


class TestReporting:
    """Test reporting functions."""
    
    def test_generate_measurement_table(self):
        """Test measurement table HTML generation."""
        from core.uncertainty import MeasurementWithUncertainty
        
        measurements = {
            'Cd': MeasurementWithUncertainty(0.654, 0.012, '-', 'Cd'),
            'pressure': MeasurementWithUncertainty(25.0, 0.5, 'bar', 'pressure'),
        }
        
        html = generate_measurement_table(measurements)
        
        assert '<table' in html
        assert 'Cd' in html
        assert '0.654' in html or '0.6540' in html
        
        print(f"[PASS] Measurement table generated ({len(html)} chars)")
    
    def test_generate_test_report(self):
        """Test single test report generation."""
        from core.uncertainty import MeasurementWithUncertainty
        
        measurements = {
            'Cd': MeasurementWithUncertainty(0.654, 0.012, '-', 'Cd'),
            'pressure_upstream': MeasurementWithUncertainty(25.0, 0.5, 'bar', 'pressure'),
            'mass_flow': MeasurementWithUncertainty(12.5, 0.25, 'g/s', 'mass_flow'),
        }
        
        traceability = {
            'raw_data_hash': 'sha256:abc123',
            'config_name': 'Test Config',
            'analyst_username': 'test_user',
            'analysis_timestamp_utc': datetime.now().isoformat(),
        }
        
        qc_report = {
            'passed': True,
            'summary': {'passed': 5, 'warnings': 0, 'failed': 0},
            'checks': [
                {'name': 'timestamp_check', 'status': 'PASS', 'message': 'OK'},
            ]
        }
        
        html = generate_test_report(
            test_id='CF-001',
            test_type='cold_flow',
            measurements=measurements,
            traceability=traceability,
            qc_report=qc_report,
        )
        
        assert '<!DOCTYPE html>' in html
        assert 'CF-001' in html
        assert 'Quality Control' in html
        assert 'Traceability' in html
        
        print(f"[PASS] Test report generated ({len(html)} chars)")
    
    def test_generate_campaign_report(self):
        """Test campaign report generation."""
        df = create_campaign_dataframe()
        
        html = generate_campaign_report(
            campaign_name='Test Campaign',
            df=df,
            parameters=['avg_cd_CALC', 'avg_p_up_bar'],
        )
        
        assert '<!DOCTYPE html>' in html
        assert 'Test Campaign' in html
        assert 'Summary Statistics' in html
        
        print(f"[PASS] Campaign report generated ({len(html)} chars)")
    
    def test_save_report(self):
        """Test saving report to file."""
        html = "<html><body>Test Report</body></html>"
        
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            temp_path = f.name
        
        try:
            result_path = save_report(html, temp_path)
            
            assert result_path.exists()
            
            with open(result_path) as f:
                content = f.read()
            
            assert 'Test Report' in content
            
            print(f"[PASS] Report saved to file")
        finally:
            os.unlink(temp_path)


class TestReportCharts:
    """Test report chart generation functions."""

    def _make_test_df(self):
        """Create a synthetic time-series DataFrame."""
        np.random.seed(42)
        n = 500
        t = np.linspace(0, 5, n)
        return pd.DataFrame({
            'time_s': t,
            'IG-PT-01': 25.0 + np.random.normal(0, 0.1, n),
            'IG-PT-02': 1.0 + np.random.normal(0, 0.05, n),
            'FM-01': 12.5 + np.random.normal(0, 0.2, n),
            'TC-01': 295.0 + np.random.normal(0, 0.5, n),
        })

    def _make_test_config(self):
        """Create a config with sensor_roles."""
        return {
            'sensor_roles': {
                'upstream_pressure': 'IG-PT-01',
                'downstream_pressure': 'IG-PT-02',
                'mass_flow': 'FM-01',
                'fluid_temperature': 'TC-01',
            },
        }

    def _make_test_measurements(self):
        """Create test measurements with uncertainties."""
        from core.uncertainty import MeasurementWithUncertainty
        return {
            'Cd': MeasurementWithUncertainty(0.654, 0.012, '-', 'Cd'),
            'pressure_upstream': MeasurementWithUncertainty(25.0, 0.5, 'bar', 'pressure'),
            'mass_flow': MeasurementWithUncertainty(12.5, 0.25, 'g/s', 'mass_flow'),
            'delta_p': MeasurementWithUncertainty(24.0, 0.55, 'bar', 'delta_p'),
        }

    def test_resolve_sensor_column(self):
        """Test sensor role resolution."""
        config = self._make_test_config()
        assert _resolve_sensor_column('upstream_pressure', config) == 'IG-PT-01'
        assert _resolve_sensor_column('mass_flow', config) == 'FM-01'
        assert _resolve_sensor_column('nonexistent', config) is None

        # Legacy columns fallback
        legacy_config = {'columns': {'upstream_pressure': 'PT-01'}}
        assert _resolve_sensor_column('upstream_pressure', legacy_config) == 'PT-01'

        print("[PASS] Sensor column resolution works")

    def test_create_time_series_chart(self):
        """Test single time-series chart generation."""
        if not PLOTLY_AVAILABLE:
            print("[SKIP] Plotly not available")
            return

        df = self._make_test_df()
        html = _create_time_series_chart(
            df, 'time_s', 'IG-PT-01',
            title='Upstream Pressure', y_label='Pressure (bar)',
            steady_window=(1.0, 4.0), color='#2563eb',
            is_first_chart=True,
        )
        assert 'plotly' in html.lower(), "Should contain plotly reference"
        assert len(html) > 100, "Should produce substantial HTML"

        # Second chart should not include plotly.js
        html2 = _create_time_series_chart(
            df, 'time_s', 'FM-01',
            title='Mass Flow', y_label='Flow (g/s)',
            is_first_chart=False,
        )
        assert 'cdn.plot.ly' not in html2, "Non-first chart should not include CDN"

        print(f"[PASS] Time series chart generated ({len(html)} chars)")

    def test_create_bar_chart(self):
        """Test bar chart generation for computed metrics."""
        if not PLOTLY_AVAILABLE:
            print("[SKIP] Plotly not available")
            return

        measurements = self._make_test_measurements()
        html = _create_bar_chart(
            measurements, ['Cd', 'delta_p'],
            title='Key Metrics', is_first_chart=True,
        )
        assert len(html) > 100, "Should produce substantial HTML"
        assert 'Key Metrics' in html

        # Empty metrics
        html_empty = _create_bar_chart(measurements, ['nonexistent'], title='Empty')
        assert html_empty == '', "Should return empty for missing metrics"

        print(f"[PASS] Bar chart generated ({len(html)} chars)")

    def test_key_charts_cold_flow(self):
        """Test key charts section for cold flow."""
        if not PLOTLY_AVAILABLE:
            print("[SKIP] Plotly not available")
            return

        df = self._make_test_df()
        config = self._make_test_config()
        measurements = self._make_test_measurements()

        html = generate_key_charts_section(
            df=df, test_type='cold_flow', config=config,
            measurements=measurements, steady_window=(1.0, 4.0),
        )

        assert 'chart-grid' in html, "Should contain chart grid"
        assert 'Key Charts' in html
        assert 'chart-cell' in html

        print(f"[PASS] Cold flow key charts generated ({len(html)} chars)")

    def test_key_charts_hot_fire(self):
        """Test key charts section for hot fire."""
        if not PLOTLY_AVAILABLE:
            print("[SKIP] Plotly not available")
            return

        from core.uncertainty import MeasurementWithUncertainty

        df = pd.DataFrame({
            'time_s': np.linspace(0, 5, 300),
            'PC-01': 15.0 + np.random.normal(0, 0.1, 300),
            'LC-01': 200.0 + np.random.normal(0, 2, 300),
            'OX-FM-01': 5.0 + np.random.normal(0, 0.1, 300),
        })
        config = {
            'sensor_roles': {
                'chamber_pressure': 'PC-01',
                'thrust': 'LC-01',
                'mass_flow_ox': 'OX-FM-01',
            },
        }
        measurements = {
            'Isp': MeasurementWithUncertainty(220.0, 5.0, 's', 'Isp'),
            'c_star': MeasurementWithUncertainty(1450.0, 30.0, 'm/s', 'c_star'),
            'of_ratio': MeasurementWithUncertainty(2.5, 0.1, '-', 'of_ratio'),
        }

        html = generate_key_charts_section(
            df=df, test_type='hot_fire', config=config,
            measurements=measurements,
        )

        assert 'chart-grid' in html
        assert 'Key Charts' in html

        print(f"[PASS] Hot fire key charts generated ({len(html)} chars)")

    def test_key_charts_missing_sensor(self):
        """Test graceful degradation with missing sensor."""
        if not PLOTLY_AVAILABLE:
            print("[SKIP] Plotly not available")
            return

        df = self._make_test_df()
        config = {'sensor_roles': {'upstream_pressure': 'IG-PT-01'}}  # Missing others
        measurements = self._make_test_measurements()

        html = generate_key_charts_section(
            df=df, test_type='cold_flow', config=config,
            measurements=measurements,
        )

        assert 'Sensor not available' in html, "Should show placeholder for missing sensors"
        assert 'chart-grid' in html, "Grid should still render"

        print("[PASS] Missing sensor handled gracefully")

    def test_appendix_charts(self):
        """Test appendix section generation."""
        if not PLOTLY_AVAILABLE:
            print("[SKIP] Plotly not available")
            return

        df = self._make_test_df()
        config = self._make_test_config()

        # Key columns should be excluded from appendix
        key_columns = ['IG-PT-01', 'IG-PT-02', 'FM-01']

        html = generate_appendix_charts_section(
            df=df, config=config, steady_window=(1.0, 4.0),
            key_columns=key_columns,
        )

        assert 'Appendix' in html
        assert 'TC-01' in html, "Non-key sensor should appear in appendix"

        print(f"[PASS] Appendix charts generated ({len(html)} chars)")

    def test_report_with_charts(self):
        """Test full report generation with embedded charts."""
        if not PLOTLY_AVAILABLE:
            print("[SKIP] Plotly not available")
            return

        df = self._make_test_df()
        config = self._make_test_config()
        measurements = self._make_test_measurements()

        traceability = {
            'raw_data_hash': 'sha256:abc123',
            'analyst_username': 'test_user',
            'analysis_timestamp_utc': datetime.now().isoformat(),
        }

        html = generate_test_report(
            test_id='CF-001', test_type='cold_flow',
            measurements=measurements, traceability=traceability,
            config=config, df=df, steady_window=(1.0, 4.0),
        )

        assert '<!DOCTYPE html>' in html
        assert 'chart-grid' in html, "Should contain chart grid"
        assert 'Appendix' in html, "Should contain appendix"
        assert 'Key Charts' in html

        print(f"[PASS] Full report with charts generated ({len(html)} chars)")

    def test_report_without_df_backward_compat(self):
        """Test that report without df is backward compatible (no charts)."""
        measurements = self._make_test_measurements()
        traceability = {
            'raw_data_hash': 'sha256:abc123',
            'analyst_username': 'test_user',
            'analysis_timestamp_utc': datetime.now().isoformat(),
        }

        html = generate_test_report(
            test_id='CF-001', test_type='cold_flow',
            measurements=measurements, traceability=traceability,
        )

        assert '<!DOCTYPE html>' in html
        assert 'Key Charts' not in html, "No chart section without df"
        assert 'Appendix' not in html, "No appendix without df"
        assert 'Key Results' in html, "Key results should still be present"
        assert 'Traceability' in html

        print("[PASS] Backward-compatible report (no charts) works")


class TestBatchAnalysis:
    """Test batch analysis functions."""
    
    def test_batch_test_result(self):
        """Test BatchTestResult dataclass."""
        result = BatchTestResult(
            file_path='/path/to/test.csv',
            test_id='CF-001',
            success=True,
            qc_passed=True,
            processing_time_s=1.5,
        )
        
        d = result.to_dict()
        
        assert d['test_id'] == 'CF-001'
        assert d['success'] == True
        
        print(f"[PASS] BatchTestResult: {result.test_id}")
    
    def test_batch_analysis_report(self):
        """Test BatchAnalysisReport."""
        results = [
            BatchTestResult('f1.csv', 'CF-001', True, qc_passed=True),
            BatchTestResult('f2.csv', 'CF-002', True, qc_passed=False),
            BatchTestResult('f3.csv', 'CF-003', False, error_message='Test error'),
        ]
        
        report = BatchAnalysisReport(
            batch_id='test_batch',
            config_name='test_config',
            start_time=datetime.now(),
            results=results,
        )
        report.update_summary()
        
        assert report.total_files == 3
        assert report.successful == 2
        assert report.failed == 1
        assert report.qc_failed == 1
        
        print(f"[PASS] BatchAnalysisReport: {report.successful}/{report.total_files} successful")
        print(f"  {report.summary()}")
    
    def test_extract_test_id_from_path(self):
        """Test test ID extraction from path."""
        path = Path('/data/tests/CF-001_20240115.csv')
        
        test_id = extract_test_id_from_path(path)
        
        assert test_id == 'CF-001_20240115'
        
        print(f"[PASS] Test ID extracted: {test_id}")
    
    def test_discover_test_files(self):
        """Test file discovery."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            for i in range(3):
                Path(tmpdir, f'test_{i}.csv').touch()
            Path(tmpdir, 'other.txt').touch()
            
            files = discover_test_files(tmpdir, '*.csv')
            
            assert len(files) == 3
            assert all(f.suffix == '.csv' for f in files)
            
            print(f"[PASS] Discovered {len(files)} test files")
    
    def test_load_csv_with_timestamp(self):
        """Test CSV loading with timestamp handling."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("time,value\n")
            f.write("0,1.0\n")
            f.write("10,1.1\n")
            f.write("20,1.2\n")
            temp_path = f.name
        
        try:
            df = load_csv_with_timestamp(Path(temp_path))
            
            assert 'timestamp' in df.columns
            assert len(df) == 3
            
            print(f"[PASS] CSV loaded with timestamp column")
        finally:
            os.unlink(temp_path)


class TestExport:
    """Test export functions."""
    
    def test_export_campaign_csv(self):
        """Test CSV export."""
        df = create_campaign_dataframe()
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            result_path = export_campaign_csv(
                df, temp_path,
                metadata={'campaign': 'Test', 'exported_by': 'test_user'}
            )
            
            assert result_path.exists()
            
            # Read back
            with open(result_path) as f:
                content = f.read()
            
            assert 'Hopper Data Studio' in content
            assert 'test_id' in content
            
            print(f"[PASS] CSV export successful")
        finally:
            os.unlink(temp_path)
    
    def test_export_campaign_json(self):
        """Test JSON export."""
        df = create_campaign_dataframe()
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            result_path = export_campaign_json(
                df, temp_path,
                campaign_info={'name': 'Test Campaign', 'type': 'cold_flow'}
            )
            
            assert result_path.exists()
            
            with open(result_path) as f:
                data = json.load(f)
            
            assert 'export_info' in data
            assert 'statistics' in data
            assert 'tests' in data
            assert len(data['tests']) == len(df)
            
            print(f"[PASS] JSON export successful ({len(data['tests'])} tests)")
        finally:
            os.unlink(temp_path)
    
    def test_create_traceability_report(self):
        """Test traceability report generation."""
        df = create_campaign_dataframe()
        
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            temp_path = f.name
        
        try:
            result_path = create_traceability_report(df, temp_path)
            
            assert result_path.exists()
            
            with open(result_path) as f:
                content = f.read()
            
            assert 'TRACEABILITY REPORT' in content
            assert 'CF-001' in content
            assert 'sha256' in content
            
            print(f"[PASS] Traceability report created")
        finally:
            os.unlink(temp_path)


def run_all_tests():
    """Run all P1 tests and report results."""
    print("=" * 60)
    print("P1 Component Tests")
    print("=" * 60)
    
    test_classes = [
        TestSPC,
        TestReporting,
        TestReportCharts,
        TestBatchAnalysis,
        TestExport,
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
                failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
