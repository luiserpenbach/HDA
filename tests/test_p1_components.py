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
        
        print(f"✓ I-MR limits: CL={i_limits.center_line:.3f}, UCL={i_limits.ucl:.3f}, LCL={i_limits.lcl:.3f}")
    
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
        
        print(f"✓ Western Electric rules: {n_violations} points with violations out of {len(values)}")
    
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
        print(f"✓ Beyond 3σ violation detected")
    
    def test_calculate_capability(self):
        """Test process capability calculation."""
        np.random.seed(42)
        values = np.random.normal(10, 0.1, 50)
        
        capability = calculate_capability(values, usl=10.5, lsl=9.5, target=10.0)
        
        assert capability.cp is not None
        assert capability.cpk is not None
        assert capability.cp > 0
        
        print(f"✓ Capability: Cp={capability.cp:.2f}, Cpk={capability.cpk:.2f}")
        print(f"  {capability.summary()}")
    
    def test_detect_trend_no_trend(self):
        """Test trend detection with stable data."""
        np.random.seed(42)
        values = np.random.normal(10, 0.1, 20)
        
        has_trend, direction, slope = detect_trend(values)
        
        # Random data shouldn't have strong trend
        print(f"✓ Trend detection (random): has_trend={has_trend}, direction={direction}")
    
    def test_detect_trend_with_trend(self):
        """Test trend detection with trending data."""
        values = np.linspace(10, 12, 20) + np.random.normal(0, 0.05, 20)
        
        has_trend, direction, slope = detect_trend(values)
        
        assert has_trend == True
        assert direction == 'increasing'
        
        print(f"✓ Trend detection (increasing): direction={direction}, slope={slope:.4f}")
    
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
        
        print(f"✓ I-MR Chart: {analysis.n_points} points, {analysis.n_violations} violations")
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
        
        print(f"✓ Measurement table generated ({len(html)} chars)")
    
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
        
        print(f"✓ Test report generated ({len(html)} chars)")
    
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
        
        print(f"✓ Campaign report generated ({len(html)} chars)")
    
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
            
            print(f"✓ Report saved to file")
        finally:
            os.unlink(temp_path)


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
        
        print(f"✓ BatchTestResult: {result.test_id}")
    
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
        
        print(f"✓ BatchAnalysisReport: {report.successful}/{report.total_files} successful")
        print(f"  {report.summary()}")
    
    def test_extract_test_id_from_path(self):
        """Test test ID extraction from path."""
        path = Path('/data/tests/CF-001_20240115.csv')
        
        test_id = extract_test_id_from_path(path)
        
        assert test_id == 'CF-001_20240115'
        
        print(f"✓ Test ID extracted: {test_id}")
    
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
            
            print(f"✓ Discovered {len(files)} test files")
    
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
            
            print(f"✓ CSV loaded with timestamp column")
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
            
            print(f"✓ CSV export successful")
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
            
            print(f"✓ JSON export successful ({len(data['tests'])} tests)")
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
            
            print(f"✓ Traceability report created")
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
                print(f"✗ {method_name}: {e}")
                failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
