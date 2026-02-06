"""
Extended Uncertainty Tests (P0 Critical)
========================================
Tests for hot fire uncertainties and edge cases not covered by test_p0_components.py

Run with: python -m pytest tests/test_uncertainty_extended.py -v
Or:       python tests/test_uncertainty_extended.py
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Add core to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.uncertainty import (
    UncertaintyType,
    SensorUncertainty,
    GeometryUncertainty,
    MeasurementWithUncertainty,
    parse_uncertainty_config,
    parse_geometry_uncertainties,
    calculate_isp_uncertainty,
    calculate_c_star_uncertainty,
    calculate_of_ratio_uncertainty,
    calculate_hot_fire_uncertainties,
    calculate_statistical_uncertainty,
    combine_uncertainties,
    get_fluid_density_and_uncertainty,
    CorrelationMatrix,
    propagate_with_correlation,
    monte_carlo_propagation,
    MonteCarloResult,
)


class TestHotFireUncertainties:
    """Test hot fire uncertainty calculations."""

    def test_isp_uncertainty_basic(self):
        """Test basic Isp uncertainty calculation."""
        result = calculate_isp_uncertainty(
            thrust_n=1000.0,
            u_thrust_n=10.0,  # 1%
            mass_flow_kg_s=0.5,
            u_mass_flow_kg_s=0.005,  # 1%
        )

        assert result.name == 'Isp'
        assert result.unit == 's'

        # Isp = F / (m_dot * g0) = 1000 / (0.5 * 9.80665) ≈ 204 s
        expected_isp = 1000.0 / (0.5 * 9.80665)
        assert abs(result.value - expected_isp) < 0.1

        # Relative uncertainty should be RSS of thrust and mass flow
        # sqrt(0.01^2 + 0.01^2) ≈ 1.4%
        assert result.relative_uncertainty_percent < 2.0
        assert result.relative_uncertainty_percent > 1.0

        print(f"[PASS] Isp uncertainty: {result}")

    def test_isp_uncertainty_zero_mass_flow(self):
        """Test Isp with zero mass flow returns inf uncertainty."""
        result = calculate_isp_uncertainty(
            thrust_n=1000.0,
            u_thrust_n=10.0,
            mass_flow_kg_s=0.0,
            u_mass_flow_kg_s=0.0,
        )

        assert result.value == 0.0
        assert np.isinf(result.uncertainty)

        print("[PASS] Isp with zero mass flow handled correctly")

    def test_isp_uncertainty_zero_thrust(self):
        """Test Isp with zero thrust returns inf uncertainty."""
        result = calculate_isp_uncertainty(
            thrust_n=0.0,
            u_thrust_n=0.0,
            mass_flow_kg_s=0.5,
            u_mass_flow_kg_s=0.005,
        )

        assert result.value == 0.0
        assert np.isinf(result.uncertainty)

        print("[PASS] Isp with zero thrust handled correctly")

    def test_c_star_uncertainty_basic(self):
        """Test basic C* uncertainty calculation."""
        result = calculate_c_star_uncertainty(
            chamber_pressure_pa=5e6,  # 50 bar
            u_chamber_pressure_pa=50000,  # 1%
            throat_area_m2=1e-4,  # 100 mm^2
            u_throat_area_m2=1e-6,  # 1%
            mass_flow_kg_s=0.5,
            u_mass_flow_kg_s=0.005,  # 1%
        )

        assert result.name == 'c_star'
        assert result.unit == 'm/s'

        # C* = Pc * At / m_dot = 5e6 * 1e-4 / 0.5 = 1000 m/s
        expected_c_star = 5e6 * 1e-4 / 0.5
        assert abs(result.value - expected_c_star) < 1.0

        # Relative uncertainty should be RSS of three 1% uncertainties ≈ 1.7%
        assert result.relative_uncertainty_percent < 2.5
        assert result.relative_uncertainty_percent > 1.0

        print(f"[PASS] C* uncertainty: {result}")

    def test_c_star_uncertainty_zero_values(self):
        """Test C* with zero values."""
        result = calculate_c_star_uncertainty(
            chamber_pressure_pa=0.0,
            u_chamber_pressure_pa=0.0,
            throat_area_m2=1e-4,
            u_throat_area_m2=1e-6,
            mass_flow_kg_s=0.5,
            u_mass_flow_kg_s=0.005,
        )

        assert result.value == 0.0
        assert np.isinf(result.uncertainty)

        print("[PASS] C* with zero pressure handled correctly")

    def test_of_ratio_uncertainty_basic(self):
        """Test basic O/F ratio uncertainty calculation."""
        result = calculate_of_ratio_uncertainty(
            m_ox_kg_s=0.3,
            u_m_ox_kg_s=0.003,  # 1%
            m_fuel_kg_s=0.125,
            u_m_fuel_kg_s=0.00125,  # 1%
        )

        assert result.name == 'of_ratio'
        assert result.unit == '-'

        # O/F = 0.3 / 0.125 = 2.4
        assert abs(result.value - 2.4) < 0.01

        # Relative uncertainty RSS of two 1% ≈ 1.4%
        assert result.relative_uncertainty_percent < 2.0
        assert result.relative_uncertainty_percent > 1.0

        print(f"[PASS] O/F ratio uncertainty: {result}")

    def test_of_ratio_zero_fuel(self):
        """Test O/F with zero fuel returns inf."""
        result = calculate_of_ratio_uncertainty(
            m_ox_kg_s=0.3,
            u_m_ox_kg_s=0.003,
            m_fuel_kg_s=0.0,
            u_m_fuel_kg_s=0.0,
        )

        assert result.value == 0.0
        assert np.isinf(result.uncertainty)

        print("[PASS] O/F with zero fuel handled correctly")

    def test_calculate_hot_fire_uncertainties_full(self):
        """Test full hot fire uncertainty calculation."""
        config = {
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
            }
        }

        avg_values = {
            'PC-01': 50.0,  # bar
            'LC-01': 1000.0,  # N
            'FM-OX': 300.0,  # g/s
            'FM-FUEL': 125.0,  # g/s
        }

        results = calculate_hot_fire_uncertainties(avg_values, config)

        # Check all expected measurements exist
        assert 'chamber_pressure' in results
        assert 'thrust' in results
        assert 'mass_flow_total' in results
        assert 'of_ratio' in results
        assert 'Isp' in results
        assert 'c_star' in results

        # Check values are reasonable
        assert results['of_ratio'].value > 2.0  # Should be around 2.4
        assert results['Isp'].value > 0
        assert results['c_star'].value > 0

        # All should have uncertainties
        for name, meas in results.items():
            assert meas.uncertainty > 0 or np.isinf(meas.uncertainty), f"{name} missing uncertainty"

        print("[PASS] Full hot fire uncertainties calculated")
        for name, meas in results.items():
            print(f"    {name}: {meas}")

    def test_calculate_hot_fire_uncertainties_missing_sensors(self):
        """Test hot fire uncertainties with missing sensors."""
        config = {
            'sensor_roles': {
                'chamber_pressure': 'PC-01',
                # Missing thrust, ox, fuel
            },
            'geometry': {},
            'uncertainties': {
                'PC-01': {'type': 'rel', 'value': 0.01},
            }
        }

        avg_values = {
            'PC-01': 50.0,
        }

        results = calculate_hot_fire_uncertainties(avg_values, config)

        # Should still have chamber pressure
        assert 'chamber_pressure' in results
        assert results['chamber_pressure'].value == 50.0

        # Should NOT have derived metrics that require missing data
        assert 'Isp' not in results or results.get('Isp', {}).value == 0

        print("[PASS] Hot fire handles missing sensors gracefully")


class TestStatisticalUncertainty:
    """Test statistical uncertainty calculations."""

    def test_calculate_statistical_uncertainty_basic(self):
        """Test basic statistical uncertainty."""
        np.random.seed(42)
        data = np.random.normal(100, 5, 100)
        df = pd.DataFrame({'value': data})

        mean, std, uncertainty = calculate_statistical_uncertainty(df, 'value')

        assert abs(mean - 100) < 2  # Should be close to 100
        assert abs(std - 5) < 2  # Should be close to 5
        assert uncertainty > 0
        assert uncertainty < std  # Uncertainty of mean < std

        print(f"[PASS] Statistical uncertainty: {mean:.2f} ± {uncertainty:.2f}")

    def test_calculate_statistical_uncertainty_small_sample(self):
        """Test statistical uncertainty with very small sample."""
        df = pd.DataFrame({'value': [100.0]})

        mean, std, uncertainty = calculate_statistical_uncertainty(df, 'value')

        assert mean == 100.0
        assert std == 0.0
        assert np.isinf(uncertainty)

        print("[PASS] Small sample handled correctly")

    def test_calculate_statistical_uncertainty_confidence(self):
        """Test statistical uncertainty with different confidence levels."""
        np.random.seed(42)
        data = np.random.normal(100, 5, 100)
        df = pd.DataFrame({'value': data})

        mean_95, std_95, u_95 = calculate_statistical_uncertainty(df, 'value', confidence=0.95)
        mean_99, std_99, u_99 = calculate_statistical_uncertainty(df, 'value', confidence=0.99)

        # 99% confidence should have larger uncertainty
        assert u_99 > u_95

        print(f"[PASS] 95% CI: ±{u_95:.3f}, 99% CI: ±{u_99:.3f}")


class TestCombineUncertainties:
    """Test uncertainty combination."""

    def test_combine_uncertainties_basic(self):
        """Test RSS combination of uncertainties."""
        systematic = 1.0
        statistical = 1.0

        combined = combine_uncertainties(systematic, statistical)

        # RSS: sqrt(1^2 + 1^2) = sqrt(2) ≈ 1.414
        assert abs(combined - np.sqrt(2)) < 0.001

        print(f"[PASS] Combined uncertainty: {combined:.4f}")

    def test_combine_uncertainties_zero(self):
        """Test combination with zero uncertainties."""
        combined = combine_uncertainties(1.0, 0.0)
        assert combined == 1.0

        combined = combine_uncertainties(0.0, 0.0)
        assert combined == 0.0

        print("[PASS] Zero uncertainty combination works")


class TestParseUncertaintyConfig:
    """Test uncertainty config parsing."""

    def test_parse_uncertainty_config_relative(self):
        """Test parsing relative uncertainties."""
        config = {
            'uncertainties': {
                'PT-01': {'type': 'rel', 'value': 0.01},
                'PT-02': {'type': 'pct_rd', 'value': 0.02},
            }
        }

        uncertainties = parse_uncertainty_config(config)

        assert 'PT-01' in uncertainties
        assert uncertainties['PT-01'].u_type == UncertaintyType.RELATIVE
        assert uncertainties['PT-01'].value == 0.01

        assert uncertainties['PT-02'].u_type == UncertaintyType.PERCENT_READING

        print("[PASS] Relative uncertainties parsed")

    def test_parse_uncertainty_config_absolute(self):
        """Test parsing absolute uncertainties."""
        config = {
            'uncertainties': {
                'LC-01': {'type': 'abs', 'value': 5.0, 'unit': 'N'},
            }
        }

        uncertainties = parse_uncertainty_config(config)

        assert uncertainties['LC-01'].u_type == UncertaintyType.ABSOLUTE
        assert uncertainties['LC-01'].value == 5.0
        assert uncertainties['LC-01'].unit == 'N'

        print("[PASS] Absolute uncertainties parsed")

    def test_parse_uncertainty_config_percent_fs(self):
        """Test parsing percent of full scale uncertainties."""
        config = {
            'uncertainties': {
                'PT-01': {'type': 'pct_fs', 'value': 0.005, 'full_scale': 100},
            }
        }

        uncertainties = parse_uncertainty_config(config)

        assert uncertainties['PT-01'].u_type == UncertaintyType.PERCENT_FS
        assert uncertainties['PT-01'].full_scale == 100

        # Test absolute uncertainty calculation
        u_abs = uncertainties['PT-01'].get_absolute_uncertainty(50.0)
        assert u_abs == 0.5  # 0.5% of 100 full scale

        print("[PASS] Percent full scale uncertainties parsed")

    def test_parse_uncertainty_config_empty(self):
        """Test parsing empty config."""
        config = {}

        uncertainties = parse_uncertainty_config(config)
        assert uncertainties == {}

        print("[PASS] Empty config handled")


class TestParseGeometryUncertainties:
    """Test geometry uncertainty parsing."""

    def test_parse_geometry_uncertainties_basic(self):
        """Test basic geometry parsing."""
        config = {
            'geometry': {
                'orifice_area_mm2': 0.785,
                'orifice_area_uncertainty_mm2': 0.005,
                'throat_area_mm2': 78.5,
                'throat_area_uncertainty_mm2': 0.5,
            }
        }

        geom = parse_geometry_uncertainties(config)

        assert 'orifice_area_mm2' in geom
        assert geom['orifice_area_mm2'].nominal_value == 0.785
        assert geom['orifice_area_mm2'].uncertainty == 0.005
        assert geom['orifice_area_mm2'].unit == 'mm²'

        print("[PASS] Geometry uncertainties parsed")

    def test_parse_geometry_uncertainties_relative(self):
        """Test relative geometry uncertainty calculation."""
        config = {
            'geometry': {
                'orifice_area_mm2': 1.0,
                'orifice_area_uncertainty_mm2': 0.01,
            }
        }

        geom = parse_geometry_uncertainties(config)
        rel_u = geom['orifice_area_mm2'].relative_uncertainty

        assert abs(rel_u - 0.01) < 0.001  # 1%

        print("[PASS] Relative geometry uncertainty correct")


class TestSensorUncertaintyMethods:
    """Test SensorUncertainty class methods."""

    def test_sensor_uncertainty_relative_calculation(self):
        """Test relative uncertainty calculation."""
        sensor = SensorUncertainty(
            sensor_id='PT-01',
            u_type=UncertaintyType.RELATIVE,
            value=0.01,  # 1%
        )

        # At reading = 100, absolute uncertainty should be 1.0
        u_abs = sensor.get_absolute_uncertainty(100.0)
        assert u_abs == 1.0

        # Relative should be 0.01
        u_rel = sensor.get_relative_uncertainty(100.0)
        assert u_rel == 0.01

        print("[PASS] Relative sensor uncertainty calculation")

    def test_sensor_uncertainty_absolute_calculation(self):
        """Test absolute uncertainty calculation."""
        sensor = SensorUncertainty(
            sensor_id='LC-01',
            u_type=UncertaintyType.ABSOLUTE,
            value=5.0,
        )

        # Absolute uncertainty is always 5.0
        u_abs = sensor.get_absolute_uncertainty(100.0)
        assert u_abs == 5.0

        u_abs_2 = sensor.get_absolute_uncertainty(1000.0)
        assert u_abs_2 == 5.0

        print("[PASS] Absolute sensor uncertainty calculation")

    def test_sensor_uncertainty_near_zero_reading(self):
        """Test uncertainty calculation near zero reading."""
        sensor = SensorUncertainty(
            sensor_id='PT-01',
            u_type=UncertaintyType.RELATIVE,
            value=0.01,
        )

        # Near zero, relative uncertainty should be inf
        u_rel = sensor.get_relative_uncertainty(1e-15)
        assert np.isinf(u_rel)

        print("[PASS] Near-zero reading handled correctly")

    def test_sensor_uncertainty_percent_fs_missing_full_scale(self):
        """Test percent FS without full scale raises error."""
        sensor = SensorUncertainty(
            sensor_id='PT-01',
            u_type=UncertaintyType.PERCENT_FS,
            value=0.005,
            full_scale=None,  # Missing!
        )

        try:
            sensor.get_absolute_uncertainty(50.0)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert 'Full scale required' in str(e)

        print("[PASS] Missing full scale raises error")


class TestMeasurementWithUncertainty:
    """Test MeasurementWithUncertainty class."""

    def test_measurement_properties(self):
        """Test measurement property calculations."""
        meas = MeasurementWithUncertainty(
            value=100.0,
            uncertainty=5.0,
            unit='bar',
            name='pressure'
        )

        assert meas.relative_uncertainty == 0.05
        assert meas.relative_uncertainty_percent == 5.0

        print(f"[PASS] Measurement: {meas}")

    def test_measurement_to_dict(self):
        """Test measurement serialization."""
        meas = MeasurementWithUncertainty(
            value=100.0,
            uncertainty=5.0,
            unit='bar',
            name='pressure'
        )

        d = meas.to_dict()

        assert 'pressure_value' in d
        assert 'pressure_uncertainty' in d
        assert 'pressure_rel_uncertainty_pct' in d
        assert d['pressure_value'] == 100.0
        assert d['pressure_uncertainty'] == 5.0

        print("[PASS] Measurement to_dict works")

    def test_measurement_near_zero_value(self):
        """Test measurement with near-zero value."""
        meas = MeasurementWithUncertainty(
            value=1e-15,
            uncertainty=1.0,
            unit='bar',
            name='pressure'
        )

        assert np.isinf(meas.relative_uncertainty)
        assert np.isinf(meas.relative_uncertainty_percent)

        print("[PASS] Near-zero measurement handled")


class TestGetFluidDensityAndUncertainty:
    """Test fluid property lookup."""

    def test_get_fluid_density_from_config(self):
        """Test getting density from config fallback."""
        config = {
            'fluid': {
                'density_kg_m3': 1000.0,
                'density_uncertainty_kg_m3': 5.0,
            }
        }

        density, uncertainty = get_fluid_density_and_uncertainty(config)

        assert density == 1000.0
        assert uncertainty == 5.0

        print(f"[PASS] Density from config: {density} ± {uncertainty}")

    def test_get_fluid_density_default_uncertainty(self):
        """Test default uncertainty calculation."""
        config = {
            'fluid': {
                'density_kg_m3': 1000.0,
                # No uncertainty specified
            }
        }

        density, uncertainty = get_fluid_density_and_uncertainty(config)

        assert density == 1000.0
        assert uncertainty == 5.0  # Default 0.5%

        print(f"[PASS] Default uncertainty: {uncertainty}")

    def test_get_fluid_density_empty_config(self):
        """Test with empty config uses defaults."""
        config = {}

        density, uncertainty = get_fluid_density_and_uncertainty(config)

        assert density == 1000.0  # Default water
        assert uncertainty > 0

        print(f"[PASS] Default density: {density}")


class TestCoverageFactor:
    """Test GUM-aligned coverage factor support."""

    def test_default_coverage_factor(self):
        """Test default coverage factor is k=1."""
        meas = MeasurementWithUncertainty(value=100.0, uncertainty=5.0)
        assert meas.coverage_factor == 1.0
        assert meas.expanded_uncertainty == 5.0

        print("[PASS] Default coverage factor k=1")

    def test_expanded_uncertainty(self):
        """Test expanded uncertainty with k=2."""
        meas = MeasurementWithUncertainty(
            value=100.0, uncertainty=5.0, coverage_factor=2.0
        )
        assert meas.expanded_uncertainty == 10.0

        print("[PASS] Expanded uncertainty U = k*u = 10.0")

    def test_at_confidence_95(self):
        """Test conversion to 95% confidence level."""
        meas = MeasurementWithUncertainty(value=100.0, uncertainty=5.0)
        meas_95 = meas.at_confidence(0.95)

        # k should be approximately 1.96 for normal distribution
        assert abs(meas_95.coverage_factor - 1.96) < 0.01
        assert meas_95.confidence_level == 0.95
        assert abs(meas_95.expanded_uncertainty - 5.0 * 1.96) < 0.1

        print(f"[PASS] 95% confidence: k={meas_95.coverage_factor:.3f}, U={meas_95.expanded_uncertainty:.2f}")

    def test_at_confidence_99(self):
        """Test conversion to 99% confidence level."""
        meas = MeasurementWithUncertainty(value=100.0, uncertainty=5.0)
        meas_99 = meas.at_confidence(0.99)

        # k should be approximately 2.576 for normal distribution
        assert abs(meas_99.coverage_factor - 2.576) < 0.01
        assert meas_99.expanded_uncertainty > meas.at_confidence(0.95).expanded_uncertainty

        print(f"[PASS] 99% confidence: k={meas_99.coverage_factor:.3f}")

    def test_at_confidence_with_dof(self):
        """Test confidence with small degrees of freedom (t-distribution)."""
        meas = MeasurementWithUncertainty(
            value=100.0, uncertainty=5.0, degrees_of_freedom=5
        )
        meas_95 = meas.at_confidence(0.95)

        # For DoF=5, t-distribution k should be ~2.571 (> 1.96)
        assert meas_95.coverage_factor > 2.0
        assert meas_95.coverage_factor > 1.96  # Wider than normal

        print(f"[PASS] Small DoF (5): k={meas_95.coverage_factor:.3f} (> 1.96)")

    def test_to_dict_with_coverage(self):
        """Test serialization includes coverage factor when non-default."""
        meas = MeasurementWithUncertainty(
            value=100.0, uncertainty=5.0, name='test',
            coverage_factor=2.0
        )

        d = meas.to_dict()
        assert 'test_coverage_factor' in d
        assert 'test_expanded_uncertainty' in d
        assert d['test_coverage_factor'] == 2.0
        assert d['test_expanded_uncertainty'] == 10.0

        print("[PASS] Serialization includes coverage factor")


class TestCorrelationMatrix:
    """Test correlation-aware uncertainty propagation."""

    def test_identity_correlation(self):
        """Test identity (uncorrelated) matrix."""
        corr = CorrelationMatrix.identity(['a', 'b', 'c'])
        assert corr.get_correlation('a', 'a') == 1.0
        assert corr.get_correlation('a', 'b') == 0.0

        print("[PASS] Identity correlation matrix")

    def test_propagation_uncorrelated(self):
        """Test propagation matches RSS for uncorrelated inputs."""
        # f = a + b, df/da = 1, df/db = 1
        partials = {'a': 1.0, 'b': 1.0}
        uncertainties = {'a': 3.0, 'b': 4.0}

        u = propagate_with_correlation(partials, uncertainties)
        expected = np.sqrt(3**2 + 4**2)  # 5.0

        assert abs(u - expected) < 1e-10

        print(f"[PASS] Uncorrelated propagation: {u:.4f} (expected {expected:.4f})")

    def test_propagation_fully_correlated(self):
        """Test propagation with fully correlated inputs."""
        # f = a - b, df/da = 1, df/db = -1
        # If a and b are perfectly correlated (r=1):
        # u_c^2 = u_a^2 + u_b^2 + 2*1*(-1)*u_a*u_b*1 = u_a^2 + u_b^2 - 2*u_a*u_b = (u_a - u_b)^2
        partials = {'a': 1.0, 'b': -1.0}
        uncertainties = {'a': 3.0, 'b': 3.0}
        corr = CorrelationMatrix(
            variables=['a', 'b'],
            matrix=np.array([[1.0, 1.0], [1.0, 1.0]])
        )

        u = propagate_with_correlation(partials, uncertainties, corr)
        # (3)^2 + (3)^2 + 2*(1)*(-1)*(3)*(3)*(1) = 9 + 9 - 18 = 0
        assert abs(u) < 1e-10

        print(f"[PASS] Fully correlated cancellation: u={u:.6f}")

    def test_propagation_partial_correlation(self):
        """Test propagation with partial correlation."""
        partials = {'a': 1.0, 'b': 1.0}
        uncertainties = {'a': 1.0, 'b': 1.0}
        corr = CorrelationMatrix(
            variables=['a', 'b'],
            matrix=np.array([[1.0, 0.5], [0.5, 1.0]])
        )

        u = propagate_with_correlation(partials, uncertainties, corr)

        # u^2 = 1 + 1 + 2*1*1*1*1*0.5 = 3, u = sqrt(3)
        expected = np.sqrt(3.0)
        assert abs(u - expected) < 1e-10

        # Should be larger than uncorrelated case
        u_uncorr = propagate_with_correlation(partials, uncertainties)
        assert u > u_uncorr

        print(f"[PASS] Partial correlation: {u:.4f} > uncorrelated {u_uncorr:.4f}")


class TestMonteCarloEngine:
    """Test Monte Carlo uncertainty propagation."""

    def test_mc_linear_function(self):
        """Test MC on simple linear function f = 2*a + 3*b."""
        def func(a, b):
            return 2 * a + 3 * b

        result = monte_carlo_propagation(
            func=func,
            input_means={'a': 10.0, 'b': 5.0},
            input_uncertainties={'a': 1.0, 'b': 0.5},
            n_samples=50000,
            seed=42,
        )

        # Expected: mean = 2*10 + 3*5 = 35
        # Expected std: sqrt((2*1)^2 + (3*0.5)^2) = sqrt(4+2.25) = sqrt(6.25) = 2.5
        assert abs(result.mean - 35.0) < 0.1
        assert abs(result.std - 2.5) < 0.1
        assert result.n_samples > 49000

        print(f"[PASS] MC linear: mean={result.mean:.2f}, std={result.std:.3f}")

    def test_mc_nonlinear_function(self):
        """Test MC on nonlinear function (Cd formula)."""
        def cd_func(mass_flow, area, delta_p, density):
            return mass_flow / (area * np.sqrt(2 * density * delta_p))

        result = monte_carlo_propagation(
            func=cd_func,
            input_means={'mass_flow': 0.01, 'area': 1e-6, 'delta_p': 1e5, 'density': 1000},
            input_uncertainties={'mass_flow': 1e-4, 'area': 1e-8, 'delta_p': 1e3, 'density': 5},
            n_samples=20000,
            seed=42,
        )

        assert result.mean > 0
        assert result.std > 0
        assert 2.5 in result.percentiles or 97.5 in result.percentiles

        print(f"[PASS] MC nonlinear (Cd): {result.mean:.4f} ± {result.std:.4f}")

    def test_mc_to_measurement(self):
        """Test conversion from MC result to MeasurementWithUncertainty."""
        def func(a):
            return a * 2

        result = monte_carlo_propagation(
            func=func,
            input_means={'a': 50.0},
            input_uncertainties={'a': 5.0},
            n_samples=10000,
            seed=42,
        )

        meas = result.to_measurement(unit='bar', name='test', confidence=0.95)

        assert meas.name == 'test'
        assert meas.unit == 'bar'
        assert abs(meas.value - 100.0) < 1.0
        assert meas.uncertainty > 0
        assert meas.confidence_level == 0.95

        print(f"[PASS] MC to measurement: {meas}")

    def test_mc_with_correlation(self):
        """Test MC with correlated inputs."""
        def func(a, b):
            return a - b

        # With perfect correlation, a-b should have very small spread
        corr = CorrelationMatrix(
            variables=['a', 'b'],
            matrix=np.array([[1.0, 0.99], [0.99, 1.0]])
        )

        result_corr = monte_carlo_propagation(
            func=func,
            input_means={'a': 100.0, 'b': 100.0},
            input_uncertainties={'a': 5.0, 'b': 5.0},
            correlation=corr,
            n_samples=20000,
            seed=42,
        )

        result_uncorr = monte_carlo_propagation(
            func=func,
            input_means={'a': 100.0, 'b': 100.0},
            input_uncertainties={'a': 5.0, 'b': 5.0},
            n_samples=20000,
            seed=42,
        )

        # Correlated case should have MUCH smaller spread
        assert result_corr.std < result_uncorr.std * 0.3

        print(f"[PASS] MC correlated std={result_corr.std:.3f} << uncorrelated std={result_uncorr.std:.3f}")

    def test_mc_reproducible_with_seed(self):
        """Test MC reproducibility with seed."""
        def func(a):
            return a ** 2

        r1 = monte_carlo_propagation(
            func=func, input_means={'a': 10.0},
            input_uncertainties={'a': 1.0}, seed=123,
        )

        r2 = monte_carlo_propagation(
            func=func, input_means={'a': 10.0},
            input_uncertainties={'a': 1.0}, seed=123,
        )

        assert r1.mean == r2.mean
        assert r1.std == r2.std

        print("[PASS] MC reproducible with seed")


def run_all_tests():
    """Run all extended uncertainty tests."""
    print("=" * 60)
    print("Extended Uncertainty Tests (P0 Critical)")
    print("=" * 60)

    test_classes = [
        TestHotFireUncertainties,
        TestStatisticalUncertainty,
        TestCombineUncertainties,
        TestParseUncertaintyConfig,
        TestParseGeometryUncertainties,
        TestSensorUncertaintyMethods,
        TestMeasurementWithUncertainty,
        TestGetFluidDensityAndUncertainty,
        TestCoverageFactor,
        TestCorrelationMatrix,
        TestMonteCarloEngine,
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
