"""
Test Suite for Plugin System (Phase 1)

Tests the plugin architecture including:
- PluginRegistry discovery and registration
- Plugin protocol validation
- ColdFlowPlugin functionality
- analyze_test() routing
- Backward compatibility with analyze_cold_flow_test()

Version: 1.0.0
Created: 2026-01-16
"""

import unittest
import pandas as pd
import numpy as np
from typing import Dict, Any

# Plugin system
from core.plugins import (
    PluginRegistry,
    AnalysisPlugin,
    PluginMetadata,
    validate_plugin,
)

# Core analysis
from core.integrated_analysis import analyze_test, analyze_cold_flow_test, AnalysisResult
from core.uncertainty import MeasurementWithUncertainty
from core.qc_checks import QCReport


class TestPluginRegistry(unittest.TestCase):
    """Test the PluginRegistry discovery and management."""

    def setUp(self):
        """Clear registry before each test."""
        PluginRegistry.clear()

    def test_plugin_discovery(self):
        """Test that plugins are discovered on first access."""
        plugins = PluginRegistry.get_plugins()

        # Should have discovered at least the ColdFlowPlugin
        self.assertGreater(len(plugins), 0, "Should discover at least one plugin")

        # Check that cold_flow plugin exists
        plugin_slugs = [p.metadata.slug for p in plugins]
        self.assertIn('cold_flow', plugin_slugs, "ColdFlowPlugin should be discovered")

    def test_get_plugin_by_slug(self):
        """Test retrieving a plugin by slug."""
        plugin = PluginRegistry.get_plugin('cold_flow')

        self.assertIsNotNone(plugin)
        self.assertEqual(plugin.metadata.slug, 'cold_flow')
        self.assertEqual(plugin.metadata.test_type, 'cold_flow')
        self.assertEqual(plugin.metadata.name, 'Cold Flow Injector Analysis')

    def test_get_nonexistent_plugin(self):
        """Test that getting a nonexistent plugin raises KeyError."""
        with self.assertRaises(KeyError) as context:
            PluginRegistry.get_plugin('nonexistent_plugin')

        self.assertIn('not found', str(context.exception))

    def test_list_available_plugins(self):
        """Test listing all available plugins."""
        plugins_info = PluginRegistry.list_available_plugins()

        self.assertIsInstance(plugins_info, list)
        self.assertGreater(len(plugins_info), 0)

        # Check structure
        for info in plugins_info:
            self.assertIn('name', info)
            self.assertIn('slug', info)
            self.assertIn('version', info)
            self.assertIn('test_type', info)
            self.assertIn('description', info)

    def test_get_plugins_by_test_type(self):
        """Test filtering plugins by test type."""
        cold_flow_plugins = PluginRegistry.get_plugins_by_test_type('cold_flow')

        self.assertGreater(len(cold_flow_plugins), 0)
        for plugin in cold_flow_plugins:
            self.assertEqual(plugin.metadata.test_type, 'cold_flow')


class TestColdFlowPlugin(unittest.TestCase):
    """Test the ColdFlowPlugin implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.plugin = PluginRegistry.get_plugin('cold_flow')
        self.test_config = self._create_test_config()
        self.test_df = self._create_test_dataframe()

    def _create_test_config(self) -> Dict[str, Any]:
        """Create a valid test configuration."""
        return {
            'config_name': 'test_config',
            'test_type': 'cold_flow',
            'columns': {
                'upstream_pressure': 'PT_UP',
                'downstream_pressure': 'PT_DOWN',
                'mass_flow': 'FM_01',
            },
            'geometry': {
                'orifice_area_mm2': 3.14159,
                'orifice_area_uncertainty_mm2': 0.01,
            },
            'uncertainties': {
                'PT_UP': {'type': 'relative', 'value': 0.005},
                'PT_DOWN': {'type': 'relative', 'value': 0.005},
                'FM_01': {'type': 'absolute', 'value': 0.5, 'unit': 'g/s'},
            },
            'fluid': {
                'name': 'nitrogen',
                'temperature_k': 300,
                'density_kg_m3': 1.165,  # N2 at 300K, 1 bar
                'density_uncertainty_kg_m3': 0.01,
            },
        }

    def _create_test_dataframe(self) -> pd.DataFrame:
        """Create synthetic test data."""
        np.random.seed(42)
        n_samples = 1000

        # Time from 0 to 10 seconds
        time_ms = np.linspace(0, 10000, n_samples)

        # Pressures (bar)
        p_up = 10.0 + np.random.normal(0, 0.05, n_samples)
        p_down = 1.0 + np.random.normal(0, 0.01, n_samples)

        # Mass flow (g/s)
        mass_flow = 125.0 + np.random.normal(0, 1.0, n_samples)

        return pd.DataFrame({
            'timestamp': time_ms,
            'PT_UP': p_up,
            'PT_DOWN': p_down,
            'FM_01': mass_flow,
        })

    def test_plugin_metadata(self):
        """Test that plugin metadata is properly defined."""
        metadata = self.plugin.metadata

        self.assertIsInstance(metadata, PluginMetadata)
        self.assertEqual(metadata.slug, 'cold_flow')
        self.assertEqual(metadata.test_type, 'cold_flow')
        self.assertGreater(len(metadata.database_columns), 0)
        self.assertGreater(len(metadata.uncertainty_specs), 0)

    def test_validate_config_valid(self):
        """Test config validation with valid config."""
        # Should not raise
        try:
            self.plugin.validate_config(self.test_config)
        except Exception as e:
            self.fail(f"validate_config raised exception with valid config: {e}")

    def test_validate_config_missing_pressure(self):
        """Test config validation catches missing pressure sensor."""
        bad_config = self.test_config.copy()
        bad_config['columns'] = {'mass_flow': 'FM_01'}  # No pressure

        with self.assertRaises(ValueError) as context:
            self.plugin.validate_config(bad_config)

        self.assertIn('pressure', str(context.exception).lower())

    def test_validate_config_missing_flow(self):
        """Test config validation catches missing flow sensor."""
        bad_config = self.test_config.copy()
        bad_config['columns'] = {'upstream_pressure': 'PT_UP'}  # No flow

        with self.assertRaises(ValueError) as context:
            self.plugin.validate_config(bad_config)

        self.assertIn('flow', str(context.exception).lower())

    def test_run_qc_checks(self):
        """Test QC checks return a valid report."""
        qc_report = self.plugin.run_qc_checks(self.test_df, self.test_config, quick=False)

        self.assertIsInstance(qc_report, QCReport)
        self.assertIsInstance(qc_report.passed, bool)
        self.assertIsInstance(qc_report.checks, list)

    def test_extract_steady_state(self):
        """Test steady state extraction."""
        steady_window = (2000, 8000)  # 2s to 8s
        steady_df = self.plugin.extract_steady_state(
            self.test_df, steady_window, self.test_config
        )

        # Check that we got the right subset
        self.assertLess(len(steady_df), len(self.test_df))
        self.assertGreater(len(steady_df), 0)

        # Check time bounds
        self.assertGreaterEqual(steady_df['timestamp'].min(), steady_window[0])
        self.assertLessEqual(steady_df['timestamp'].max(), steady_window[1])

    def test_compute_raw_metrics(self):
        """Test raw metrics computation."""
        steady_window = (2000, 8000)
        steady_df = self.plugin.extract_steady_state(
            self.test_df, steady_window, self.test_config
        )

        raw_metrics = self.plugin.compute_raw_metrics(
            steady_df, self.test_config, metadata=None
        )

        # Check that we got averages
        self.assertIsInstance(raw_metrics, dict)
        self.assertIn('PT_UP', raw_metrics)
        self.assertIn('PT_DOWN', raw_metrics)
        self.assertIn('FM_01', raw_metrics)

        # Check values are reasonable
        self.assertAlmostEqual(raw_metrics['PT_UP'], 10.0, delta=0.5)
        self.assertAlmostEqual(raw_metrics['PT_DOWN'], 1.0, delta=0.2)
        self.assertAlmostEqual(raw_metrics['FM_01'], 125.0, delta=5.0)

    def test_calculate_measurements_with_uncertainties(self):
        """Test uncertainty calculation."""
        steady_window = (2000, 8000)
        steady_df = self.plugin.extract_steady_state(
            self.test_df, steady_window, self.test_config
        )
        raw_metrics = self.plugin.compute_raw_metrics(
            steady_df, self.test_config, metadata=None
        )

        measurements = self.plugin.calculate_measurements_with_uncertainties(
            raw_metrics, self.test_config, metadata=None
        )

        # Check structure
        self.assertIsInstance(measurements, dict)

        # Check key measurements
        self.assertIn('pressure_upstream', measurements)
        self.assertIn('mass_flow', measurements)
        self.assertIn('delta_p', measurements)
        self.assertIn('Cd', measurements)

        # Check that uncertainties are present
        for name, meas in measurements.items():
            self.assertIsInstance(meas, MeasurementWithUncertainty)
            self.assertGreater(meas.uncertainty, 0, f"{name} should have uncertainty > 0")

    def test_get_display_order(self):
        """Test display order specification."""
        order = self.plugin.get_display_order()

        self.assertIsInstance(order, list)
        self.assertGreater(len(order), 0)
        self.assertIn('Cd', order)


class TestAnalyzeTestFunction(unittest.TestCase):
    """Test the new analyze_test() function with plugin routing."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_config = self._create_test_config()
        self.test_df = self._create_test_dataframe()

    def _create_test_config(self) -> Dict[str, Any]:
        """Create a valid test configuration."""
        return {
            'config_name': 'test_config',
            'test_type': 'cold_flow',
            'columns': {
                'upstream_pressure': 'PT_UP',
                'downstream_pressure': 'PT_DOWN',
                'mass_flow': 'FM_01',
            },
            'geometry': {
                'orifice_area_mm2': 3.14159,
                'orifice_area_uncertainty_mm2': 0.01,
            },
            'uncertainties': {
                'PT_UP': {'type': 'relative', 'value': 0.005},
                'PT_DOWN': {'type': 'relative', 'value': 0.005},
                'FM_01': {'type': 'absolute', 'value': 0.5, 'unit': 'g/s'},
            },
            'fluid': {
                'name': 'nitrogen',
                'temperature_k': 300,
                'density_kg_m3': 1.165,  # N2 at 300K, 1 bar
                'density_uncertainty_kg_m3': 0.01,
            },
        }

    def _create_test_dataframe(self) -> pd.DataFrame:
        """Create synthetic test data."""
        np.random.seed(42)
        n_samples = 1000

        time_ms = np.linspace(0, 10000, n_samples)
        p_up = 10.0 + np.random.normal(0, 0.05, n_samples)
        p_down = 1.0 + np.random.normal(0, 0.01, n_samples)
        mass_flow = 125.0 + np.random.normal(0, 1.0, n_samples)

        return pd.DataFrame({
            'timestamp': time_ms,
            'PT_UP': p_up,
            'PT_DOWN': p_down,
            'FM_01': mass_flow,
        })

    def test_analyze_test_cold_flow(self):
        """Test analyze_test with cold_flow plugin."""
        result = analyze_test(
            df=self.test_df,
            config=self.test_config,
            steady_window=(2000, 8000),
            test_id="TEST-001",
            plugin_slug="cold_flow",
            skip_qc=True,  # Skip QC for faster test
        )

        # Check result structure
        self.assertIsInstance(result, AnalysisResult)
        self.assertEqual(result.test_id, "TEST-001")
        self.assertTrue(result.passed_qc)

        # Check measurements
        self.assertIn('Cd', result.measurements)
        self.assertIn('pressure_upstream', result.measurements)
        self.assertIn('mass_flow', result.measurements)

        # Check traceability
        self.assertIsNotNone(result.traceability)
        self.assertIn('config_hash', result.traceability)

    def test_analyze_test_invalid_plugin(self):
        """Test analyze_test with invalid plugin slug."""
        with self.assertRaises(KeyError):
            analyze_test(
                df=self.test_df,
                config=self.test_config,
                steady_window=(2000, 8000),
                test_id="TEST-002",
                plugin_slug="nonexistent_plugin",
            )


class TestBackwardCompatibility(unittest.TestCase):
    """Test that existing analyze_cold_flow_test() still works."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_config = self._create_test_config()
        self.test_df = self._create_test_dataframe()

    def _create_test_config(self) -> Dict[str, Any]:
        """Create a valid test configuration."""
        return {
            'config_name': 'test_config',
            'test_type': 'cold_flow',
            'columns': {
                'upstream_pressure': 'PT_UP',
                'downstream_pressure': 'PT_DOWN',
                'mass_flow': 'FM_01',
            },
            'geometry': {
                'orifice_area_mm2': 3.14159,
                'orifice_area_uncertainty_mm2': 0.01,
            },
            'uncertainties': {
                'PT_UP': {'type': 'relative', 'value': 0.005},
                'PT_DOWN': {'type': 'relative', 'value': 0.005},
                'FM_01': {'type': 'absolute', 'value': 0.5, 'unit': 'g/s'},
            },
            'fluid': {
                'name': 'nitrogen',
                'temperature_k': 300,
                'density_kg_m3': 1.165,  # N2 at 300K, 1 bar
                'density_uncertainty_kg_m3': 0.01,
            },
        }

    def _create_test_dataframe(self) -> pd.DataFrame:
        """Create synthetic test data."""
        np.random.seed(42)
        n_samples = 1000

        time_ms = np.linspace(0, 10000, n_samples)
        p_up = 10.0 + np.random.normal(0, 0.05, n_samples)
        p_down = 1.0 + np.random.normal(0, 0.01, n_samples)
        mass_flow = 125.0 + np.random.normal(0, 1.0, n_samples)

        return pd.DataFrame({
            'timestamp': time_ms,
            'PT_UP': p_up,
            'PT_DOWN': p_down,
            'FM_01': mass_flow,
        })

    def test_analyze_cold_flow_test_still_works(self):
        """Test that the old API still works (backward compatibility)."""
        result = analyze_cold_flow_test(
            df=self.test_df,
            config=self.test_config,
            steady_window=(2000, 8000),
            test_id="TEST-LEGACY-001",
            skip_qc=True,
        )

        # Should produce identical result to new API
        self.assertIsInstance(result, AnalysisResult)
        self.assertEqual(result.test_id, "TEST-LEGACY-001")
        self.assertTrue(result.passed_qc)
        self.assertIn('Cd', result.measurements)

    def test_results_identical_old_vs_new_api(self):
        """Test that old and new APIs produce identical results."""
        # Old API
        result_old = analyze_cold_flow_test(
            df=self.test_df,
            config=self.test_config,
            steady_window=(2000, 8000),
            test_id="TEST-COMPARE",
            skip_qc=True,
        )

        # New API
        result_new = analyze_test(
            df=self.test_df,
            config=self.test_config,
            steady_window=(2000, 8000),
            test_id="TEST-COMPARE",
            plugin_slug="cold_flow",
            skip_qc=True,
        )

        # Compare key measurements
        for key in ['Cd', 'pressure_upstream', 'mass_flow']:
            if key in result_old.measurements and key in result_new.measurements:
                old_val = result_old.measurements[key].value
                new_val = result_new.measurements[key].value
                self.assertAlmostEqual(
                    old_val, new_val, places=10,
                    msg=f"Mismatch in {key}: old={old_val}, new={new_val}"
                )


class TestPluginValidation(unittest.TestCase):
    """Test plugin validation utilities."""

    def test_validate_plugin(self):
        """Test plugin validation function."""
        plugin = PluginRegistry.get_plugin('cold_flow')
        errors = validate_plugin(plugin)

        self.assertEqual(len(errors), 0, f"Valid plugin should have no errors: {errors}")


if __name__ == '__main__':
    unittest.main()
