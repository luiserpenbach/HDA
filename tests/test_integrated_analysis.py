"""
Integrated Analysis Tests (P0 Critical)
=======================================
Tests for AnalysisResult serialization and integrated analysis functions.

Run with: python -m pytest tests/test_integrated_analysis.py -v
Or:       python tests/test_integrated_analysis.py
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add core to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.integrated_analysis import (
    AnalysisResult,
    analyze_test,
    analyze_cold_flow_test,
    analyze_hot_fire_test,
    quick_analyze,
    format_measurement_table,
)
from core.uncertainty import MeasurementWithUncertainty
from core.qc_checks import QCReport, QCCheckResult, QCStatus


def create_test_dataframe():
    """Create a realistic test dataframe for cold flow analysis."""
    np.random.seed(42)
    n_samples = 1000

    t = np.arange(n_samples) * 10  # 10ms intervals (timestamps in ms)

    # Pressure: ramp up, steady, ramp down
    pressure = np.zeros(n_samples)
    pressure[:100] = np.linspace(0, 25, 100)
    pressure[100:800] = 25 + np.random.normal(0, 0.2, 700)
    pressure[800:] = np.linspace(25, 0, 200)

    # Flow: follows pressure
    flow = pressure * 0.5 + np.random.normal(0, 0.1, n_samples)
    flow = np.maximum(flow, 0)

    # Temperature
    temp = 293 + np.random.normal(0, 0.5, n_samples)

    return pd.DataFrame({
        'timestamp': t,
        'PT-01': pressure,
        'FM-01': flow,
        'TC-01': temp,
    })


def create_test_config():
    """Create a valid cold flow configuration."""
    return {
        'config_name': 'Test_Cold_Flow_Config',
        'test_type': 'cold_flow',
        'fluid': {
            'name': 'water',
            'density_kg_m3': 1000,
        },
        'geometry': {
            'orifice_area_mm2': 1.0,
            'orifice_area_uncertainty_mm2': 0.01,
        },
        'columns': {
            'timestamp': 'timestamp',
            'upstream_pressure': 'PT-01',
            'mass_flow': 'FM-01',
        },
        'sensor_roles': {
            'upstream_pressure': 'PT-01',
            'mass_flow': 'FM-01',
        },
        'uncertainties': {
            'PT-01': {'type': 'rel', 'value': 0.005},
            'FM-01': {'type': 'rel', 'value': 0.01},
        },
        'settings': {
            'resample_freq_ms': 10,
        }
    }


def create_mock_qc_report(passed=True):
    """Create a mock QC report."""
    report = QCReport()
    report.add_check(QCCheckResult(
        name='timestamp_monotonic',
        status=QCStatus.PASS,
        message='Timestamps monotonic'
    ))
    if not passed:
        report.add_check(QCCheckResult(
            name='blocking_check',
            status=QCStatus.FAIL,
            message='Intentional failure',
            blocking=True
        ))
    return report


def create_mock_measurements():
    """Create mock measurements with uncertainties."""
    return {
        'pressure_upstream': MeasurementWithUncertainty(
            value=25.0, uncertainty=0.125, unit='bar', name='pressure_upstream'
        ),
        'mass_flow': MeasurementWithUncertainty(
            value=12.5, uncertainty=0.125, unit='g/s', name='mass_flow'
        ),
        'delta_p': MeasurementWithUncertainty(
            value=25.0, uncertainty=0.125, unit='bar', name='delta_p'
        ),
        'Cd': MeasurementWithUncertainty(
            value=0.654, uncertainty=0.018, unit='-', name='Cd'
        ),
    }


class TestAnalysisResult:
    """Tests for AnalysisResult class."""

    def test_analysis_result_creation(self):
        """Test basic AnalysisResult creation."""
        result = AnalysisResult(
            test_id='TEST-001',
            qc_report=create_mock_qc_report(passed=True),
            measurements=create_mock_measurements(),
            raw_values={'PT-01': 25.0, 'FM-01': 12.5},
            traceability={'raw_data_hash': 'sha256:abc123'},
            config=create_test_config(),
            steady_window=(1500, 5000),
            metadata={'part': 'INJ-01', 'serial_num': 'SN-001'},
        )

        assert result.test_id == 'TEST-001'
        assert result.passed_qc == True
        assert result.has_warnings == False
        assert 'Cd' in result.measurements

        print(f"[PASS] Created AnalysisResult: {result.test_id}")

    def test_analysis_result_passed_qc_property(self):
        """Test passed_qc property."""
        result_passed = AnalysisResult(
            test_id='PASS-001',
            qc_report=create_mock_qc_report(passed=True),
            measurements={},
            raw_values={},
            traceability={},
            config={},
            steady_window=(0, 1000),
        )
        assert result_passed.passed_qc == True

        result_failed = AnalysisResult(
            test_id='FAIL-001',
            qc_report=create_mock_qc_report(passed=False),
            measurements={},
            raw_values={},
            traceability={},
            config={},
            steady_window=(0, 1000),
        )
        assert result_failed.passed_qc == False

        print("[PASS] passed_qc property works correctly")

    def test_analysis_result_get_measurement(self):
        """Test get_measurement method."""
        result = AnalysisResult(
            test_id='TEST-001',
            qc_report=create_mock_qc_report(),
            measurements=create_mock_measurements(),
            raw_values={},
            traceability={},
            config={},
            steady_window=(0, 1000),
        )

        cd = result.get_measurement('Cd')
        assert cd is not None
        assert cd.value == 0.654
        assert cd.uncertainty == 0.018

        missing = result.get_measurement('nonexistent')
        assert missing is None

        print("[PASS] get_measurement works correctly")

    def test_analysis_result_get_value(self):
        """Test get_value method."""
        result = AnalysisResult(
            test_id='TEST-001',
            qc_report=create_mock_qc_report(),
            measurements=create_mock_measurements(),
            raw_values={},
            traceability={},
            config={},
            steady_window=(0, 1000),
        )

        cd_value = result.get_value('Cd')
        assert cd_value == 0.654

        missing = result.get_value('nonexistent')
        assert missing is None

        print("[PASS] get_value works correctly")

    def test_analysis_result_get_uncertainty(self):
        """Test get_uncertainty method."""
        result = AnalysisResult(
            test_id='TEST-001',
            qc_report=create_mock_qc_report(),
            measurements=create_mock_measurements(),
            raw_values={},
            traceability={},
            config={},
            steady_window=(0, 1000),
        )

        cd_uncertainty = result.get_uncertainty('Cd')
        assert cd_uncertainty == 0.018

        missing = result.get_uncertainty('nonexistent')
        assert missing is None

        print("[PASS] get_uncertainty works correctly")

    def test_analysis_result_format_summary(self):
        """Test format_summary method."""
        result = AnalysisResult(
            test_id='TEST-001',
            qc_report=create_mock_qc_report(),
            measurements=create_mock_measurements(),
            raw_values={},
            traceability={},
            config={},
            steady_window=(0, 1000),
        )

        summary = result.format_summary()

        assert 'TEST-001' in summary
        assert 'PASSED' in summary
        assert 'Cd' in summary

        print(f"[PASS] Format summary:\n{summary[:200]}...")


class TestAnalysisResultSerialization:
    """Tests for AnalysisResult serialization to database record."""

    def test_to_database_record_cold_flow(self):
        """Test cold flow database record generation."""
        result = AnalysisResult(
            test_id='CF-001',
            qc_report=create_mock_qc_report(),
            measurements=create_mock_measurements(),
            raw_values={'PT-01': 25.0, 'FM-01': 12.5},
            traceability={
                'raw_data_hash': 'sha256:abc123',
                'config_hash': 'sha256:config456',
                'analyst_username': 'test_user',
            },
            config={'config_name': 'test'},
            steady_window=(1500, 5000),
            metadata={
                'part': 'INJ-01',
                'serial_num': 'SN-001',
                'fluid': 'water',
            },
        )

        record = result.to_database_record(campaign_type='cold_flow')

        # Check basic fields
        assert record['test_id'] == 'CF-001'
        assert record['qc_passed'] == 1
        assert 'test_timestamp' in record

        # Check metadata
        assert record['part'] == 'INJ-01'
        assert record['serial_num'] == 'SN-001'

        # Check traceability
        assert record['raw_data_hash'] == 'sha256:abc123'
        assert record['analyst_username'] == 'test_user'

        # Check measurements
        assert record['avg_p_up_bar'] == 25.0
        assert record['u_p_up_bar'] == 0.125
        assert record['avg_mf_g_s'] == 12.5
        assert record['avg_cd_CALC'] == 0.654
        assert record['u_cd_CALC'] == 0.018

        print(f"[PASS] Cold flow record has {len(record)} fields")

    def test_to_database_record_hot_fire(self):
        """Test hot fire database record generation."""
        hot_fire_measurements = {
            'chamber_pressure': MeasurementWithUncertainty(
                value=50.0, uncertainty=0.5, unit='bar', name='chamber_pressure'
            ),
            'thrust': MeasurementWithUncertainty(
                value=1000.0, uncertainty=10.0, unit='N', name='thrust'
            ),
            'mass_flow_total': MeasurementWithUncertainty(
                value=425.0, uncertainty=5.0, unit='g/s', name='mass_flow_total'
            ),
            'of_ratio': MeasurementWithUncertainty(
                value=2.4, uncertainty=0.05, unit='-', name='of_ratio'
            ),
            'Isp': MeasurementWithUncertainty(
                value=280.0, uncertainty=3.0, unit='s', name='Isp'
            ),
            'c_star': MeasurementWithUncertainty(
                value=1500.0, uncertainty=20.0, unit='m/s', name='c_star'
            ),
        }

        result = AnalysisResult(
            test_id='HF-001',
            qc_report=create_mock_qc_report(),
            measurements=hot_fire_measurements,
            raw_values={},
            traceability={'raw_data_hash': 'sha256:hf123'},
            config={},
            steady_window=(1000, 3000),
            metadata={'propellants': 'LOX/RP-1'},
        )

        record = result.to_database_record(campaign_type='hot_fire')

        # Check hot fire specific fields
        assert record['avg_pc_bar'] == 50.0
        assert record['u_pc_bar'] == 0.5
        assert record['avg_thrust_n'] == 1000.0
        assert record['u_thrust_n'] == 10.0
        assert record['avg_mf_total_g_s'] == 425.0
        assert record['avg_of_ratio'] == 2.4
        assert record['avg_isp_s'] == 280.0
        assert record['avg_c_star_m_s'] == 1500.0

        print(f"[PASS] Hot fire record has {len(record)} fields")

    def test_to_database_record_missing_measurements(self):
        """Test record generation with missing measurements."""
        result = AnalysisResult(
            test_id='PARTIAL-001',
            qc_report=create_mock_qc_report(),
            measurements={
                'pressure_upstream': MeasurementWithUncertainty(
                    value=25.0, uncertainty=0.125, unit='bar', name='pressure_upstream'
                ),
                # Missing mass_flow, Cd, etc.
            },
            raw_values={},
            traceability={},
            config={},
            steady_window=(0, 1000),
        )

        record = result.to_database_record(campaign_type='cold_flow')

        # Should have pressure but not Cd
        assert record['avg_p_up_bar'] == 25.0
        assert 'avg_cd_CALC' not in record or record.get('avg_cd_CALC') is None

        print("[PASS] Partial measurements handled correctly")

    def test_to_database_record_qc_summary(self):
        """Test QC summary serialization."""
        result = AnalysisResult(
            test_id='QC-001',
            qc_report=create_mock_qc_report(),
            measurements={},
            raw_values={},
            traceability={},
            config={},
            steady_window=(0, 1000),
        )

        record = result.to_database_record(campaign_type='cold_flow')

        # QC summary should be JSON serialized
        qc_summary = json.loads(record['qc_summary'])
        assert 'total' in qc_summary
        assert 'passed' in qc_summary

        print(f"[PASS] QC summary: {qc_summary}")


class TestQuickAnalyze:
    """Tests for quick_analyze function."""

    def test_quick_analyze_cold_flow(self):
        """Test quick analysis for cold flow."""
        df = create_test_dataframe()
        # Extract steady state portion
        steady_df = df[(df['timestamp'] >= 1500) & (df['timestamp'] <= 5000)]

        config = create_test_config()

        results = quick_analyze(steady_df, config, test_type='cold_flow')

        # Should have measurements
        assert len(results) > 0

        # Check for key metrics
        if 'pressure_upstream' in results:
            assert results['pressure_upstream'].uncertainty >= 0

        print(f"[PASS] Quick analyze returned {len(results)} measurements")


class TestFormatMeasurementTable:
    """Tests for format_measurement_table function."""

    def test_format_measurement_table_basic(self):
        """Test basic measurement table formatting."""
        measurements = create_mock_measurements()

        table = format_measurement_table(measurements)

        # Should be markdown table
        assert '|' in table
        assert 'Parameter' in table
        assert 'Value' in table
        assert 'Uncertainty' in table

        # Should contain our measurements
        assert 'Cd' in table
        assert 'pressure_upstream' in table

        print(f"[PASS] Formatted table:\n{table}")

    def test_format_measurement_table_empty(self):
        """Test formatting empty measurements."""
        table = format_measurement_table({})

        # Should still have header
        assert 'Parameter' in table
        assert 'Value' in table

        print("[PASS] Empty table formatted")


class TestAnalyzeTestFunction:
    """Tests for the main analyze_test function (plugin-based)."""

    def test_analyze_test_cold_flow_skip_qc(self):
        """Test analyze_test with skip_qc=True."""
        df = create_test_dataframe()
        config = create_test_config()

        result = analyze_cold_flow_test(
            df=df,
            config=config,
            steady_window=(1500, 5000),
            test_id='SKIP-QC-001',
            skip_qc=True,  # Skip QC for speed
        )

        assert result is not None
        assert result.test_id == 'SKIP-QC-001'
        assert result.traceability is not None

        print(f"[PASS] analyze_cold_flow_test with skip_qc: {result.test_id}")

    def test_analyze_test_creates_traceability(self):
        """Test that analyze_test creates traceability record."""
        df = create_test_dataframe()
        config = create_test_config()

        result = analyze_cold_flow_test(
            df=df,
            config=config,
            steady_window=(1500, 5000),
            test_id='TRACE-001',
            detection_method='CV-based',
            skip_qc=True,
        )

        # Check traceability exists
        assert result.traceability is not None
        assert 'raw_data_hash' in result.traceability or 'data_hash' in result.traceability
        assert 'analyst_username' in result.traceability
        assert 'processing_version' in result.traceability

        print(f"[PASS] Traceability created: {list(result.traceability.keys())[:5]}...")

    def test_analyze_test_with_metadata(self):
        """Test analyze_test with metadata."""
        df = create_test_dataframe()
        config = create_test_config()

        metadata = {
            'part': 'INJ-B1-03',
            'serial_num': 'SN-2024-001',
            'operator': 'Test Operator',
            'test_fluid': 'water',
        }

        result = analyze_cold_flow_test(
            df=df,
            config=config,
            steady_window=(1500, 5000),
            test_id='META-001',
            metadata=metadata,
            skip_qc=True,
        )

        assert result.metadata['part'] == 'INJ-B1-03'
        assert result.metadata['serial_num'] == 'SN-2024-001'

        print(f"[PASS] Metadata preserved: {result.metadata}")


class TestAnalyzeHotFire:
    """Tests for hot fire analysis."""

    def test_analyze_hot_fire_skip_qc(self):
        """Test analyze_hot_fire_test with skip_qc=True."""
        # Create hot fire test data
        np.random.seed(42)
        n = 1000
        t = np.arange(n) * 10

        pc = np.zeros(n)
        pc[:100] = np.linspace(0, 50, 100)
        pc[100:800] = 50 + np.random.normal(0, 0.5, 700)
        pc[800:] = np.linspace(50, 0, 200)

        thrust = pc * 20 + np.random.normal(0, 10, n)
        mf_ox = pc * 6 + np.random.normal(0, 2, n)
        mf_fuel = pc * 2.5 + np.random.normal(0, 1, n)

        df = pd.DataFrame({
            'timestamp': t,
            'PC-01': pc,
            'LC-01': thrust,
            'FM-OX': mf_ox,
            'FM-FUEL': mf_fuel,
        })

        config = {
            'config_name': 'Test_Hot_Fire_Config',
            'test_type': 'hot_fire',
            'fluid': {
                'name': 'LOX/RP-1',
                'density_kg_m3': 1000,  # Approximate for testing
                'ox_density_kg_m3': 1141,
                'fuel_density_kg_m3': 820,
            },
            'columns': {
                'timestamp': 'timestamp',
                'chamber_pressure': 'PC-01',
                'thrust': 'LC-01',
            },
            'sensor_roles': {
                'chamber_pressure': 'PC-01',
                'thrust': 'LC-01',
                'mass_flow_ox': 'FM-OX',
                'mass_flow_fuel': 'FM-FUEL',
            },
            'geometry': {
                'throat_area_mm2': 100.0,
                'throat_area_uncertainty_mm2': 1.0,
            },
            'uncertainties': {
                'PC-01': {'type': 'rel', 'value': 0.01},
                'LC-01': {'type': 'rel', 'value': 0.01},
                'FM-OX': {'type': 'rel', 'value': 0.01},
                'FM-FUEL': {'type': 'rel', 'value': 0.01},
            },
        }

        result = analyze_hot_fire_test(
            df=df,
            config=config,
            steady_window=(1500, 5000),
            test_id='HF-TEST-001',
            skip_qc=True,
        )

        assert result is not None
        assert result.test_id == 'HF-TEST-001'

        print(f"[PASS] Hot fire analysis: {result.test_id}")


def run_all_tests():
    """Run all integrated analysis tests."""
    print("=" * 60)
    print("Integrated Analysis Tests (P0 Critical)")
    print("=" * 60)

    test_classes = [
        TestAnalysisResult,
        TestAnalysisResultSerialization,
        TestQuickAnalyze,
        TestFormatMeasurementTable,
        TestAnalyzeTestFunction,
        TestAnalyzeHotFire,
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
