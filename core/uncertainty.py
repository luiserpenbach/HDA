"""
Uncertainty Quantification Module
=================================
Provides rigorous error propagation for all propulsion metrics.

Key principle: Every metric must have error bars. You don't make 
hardware decisions on point estimates.

Uncertainty Sources:
- Sensor uncertainties (from calibration)
- Geometry tolerances (from manufacturing)
- Fluid property uncertainties (from temperature, etc.)
- Statistical uncertainty (from measurement variation)

Propagation Methods:
- Analytical (for simple formulas)
- Monte Carlo (for complex calculations)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings

try:
    from . import fluid_properties
    FLUID_PROPERTIES_AVAILABLE = True
except ImportError:
    FLUID_PROPERTIES_AVAILABLE = False
    fluid_properties = None


class UncertaintyType(Enum):
    """Types of uncertainty specification."""
    ABSOLUTE = "abs"           # Fixed value (e.g., ±0.5 bar)
    RELATIVE = "rel"           # Fraction of reading (e.g., ±1%)
    PERCENT_FS = "pct_fs"      # Percent of full scale
    PERCENT_READING = "pct_rd" # Percent of reading (same as relative)


@dataclass
class SensorUncertainty:
    """
    Uncertainty specification for a single sensor.
    
    Attributes:
        sensor_id: Sensor identifier (e.g., 'PT-01')
        u_type: Type of uncertainty specification
        value: Uncertainty value
        full_scale: Full scale value (required for PERCENT_FS type)
        unit: Physical unit of the measurement
    """
    sensor_id: str
    u_type: UncertaintyType
    value: float
    full_scale: Optional[float] = None
    unit: Optional[str] = None
    
    def get_absolute_uncertainty(self, reading: float) -> float:
        """
        Calculate absolute uncertainty for a given reading.
        
        Args:
            reading: The measured value
            
        Returns:
            Absolute uncertainty in same units as reading
        """
        if self.u_type == UncertaintyType.ABSOLUTE:
            return self.value
        
        elif self.u_type in (UncertaintyType.RELATIVE, UncertaintyType.PERCENT_READING):
            return abs(reading * self.value)
        
        elif self.u_type == UncertaintyType.PERCENT_FS:
            if self.full_scale is None:
                raise ValueError(f"Full scale required for PERCENT_FS type: {self.sensor_id}")
            return self.full_scale * self.value
        
        else:
            raise ValueError(f"Unknown uncertainty type: {self.u_type}")
    
    def get_relative_uncertainty(self, reading: float) -> float:
        """
        Calculate relative uncertainty (fraction) for a given reading.
        
        Args:
            reading: The measured value
            
        Returns:
            Relative uncertainty as fraction (not percent)
        """
        if abs(reading) < 1e-10:
            return float('inf')
        
        return self.get_absolute_uncertainty(reading) / abs(reading)


@dataclass
class GeometryUncertainty:
    """
    Uncertainty specification for geometric parameters.
    
    Attributes:
        parameter: Parameter name (e.g., 'orifice_area')
        nominal_value: Nominal value
        uncertainty: Absolute uncertainty
        unit: Physical unit
    """
    parameter: str
    nominal_value: float
    uncertainty: float
    unit: str
    
    @property
    def relative_uncertainty(self) -> float:
        """Relative uncertainty as fraction."""
        if self.nominal_value == 0:
            return float('inf')
        return self.uncertainty / self.nominal_value


@dataclass
class MeasurementWithUncertainty:
    """
    A measurement value with its associated uncertainty (GUM-aligned).

    Attributes:
        value: Central/mean value
        uncertainty: Standard uncertainty u(y) (1-sigma equivalent)
        unit: Physical unit
        name: Parameter name for display
        coverage_factor: k-factor for expanded uncertainty (default k=1)
        confidence_level: Confidence level associated with coverage factor
        degrees_of_freedom: Effective degrees of freedom (for Welch-Satterthwaite)

    The `uncertainty` field stores the standard uncertainty u(y).
    Expanded uncertainty U = k * u(y) is accessible via `expanded_uncertainty`.
    """
    value: float
    uncertainty: float
    unit: str = ""
    name: str = ""
    coverage_factor: float = 1.0
    confidence_level: float = 0.6827
    degrees_of_freedom: Optional[float] = None

    @property
    def relative_uncertainty(self) -> float:
        """Relative standard uncertainty as fraction."""
        if abs(self.value) < 1e-10:
            return float('inf')
        return self.uncertainty / abs(self.value)

    @property
    def relative_uncertainty_percent(self) -> float:
        """Relative standard uncertainty as percentage."""
        return self.relative_uncertainty * 100

    @property
    def expanded_uncertainty(self) -> float:
        """Expanded uncertainty U = k * u(y)."""
        return self.coverage_factor * self.uncertainty

    def at_confidence(self, confidence: float = 0.95) -> 'MeasurementWithUncertainty':
        """
        Return a copy with expanded uncertainty at specified confidence level.

        Uses normal distribution k-factors:
        - 68.27%: k=1.0
        - 90%:    k=1.645
        - 95%:    k=1.96 (approximate, exact for large DoF)
        - 95.45%: k=2.0
        - 99%:    k=2.576
        - 99.73%: k=3.0

        For small degrees of freedom, uses t-distribution.
        """
        if self.degrees_of_freedom is not None and self.degrees_of_freedom < 30:
            from scipy import stats
            k = stats.t.ppf((1 + confidence) / 2, df=self.degrees_of_freedom)
        else:
            from scipy import stats
            k = stats.norm.ppf((1 + confidence) / 2)

        return MeasurementWithUncertainty(
            value=self.value,
            uncertainty=self.uncertainty,
            unit=self.unit,
            name=self.name,
            coverage_factor=k,
            confidence_level=confidence,
            degrees_of_freedom=self.degrees_of_freedom,
        )

    def __str__(self) -> str:
        if self.coverage_factor != 1.0:
            return (f"{self.value:.4g} ± {self.expanded_uncertainty:.4g} {self.unit} "
                    f"(k={self.coverage_factor:.2f}, {self.confidence_level*100:.0f}%)")
        return f"{self.value:.4g} ± {self.uncertainty:.4g} {self.unit} ({self.relative_uncertainty_percent:.1f}%)"

    def to_dict(self) -> Dict[str, Any]:
        d = {
            f"{self.name}_value": self.value,
            f"{self.name}_uncertainty": self.uncertainty,
            f"{self.name}_rel_uncertainty_pct": self.relative_uncertainty_percent,
        }
        if self.coverage_factor != 1.0:
            d[f"{self.name}_coverage_factor"] = self.coverage_factor
            d[f"{self.name}_expanded_uncertainty"] = self.expanded_uncertainty
        return d


def parse_uncertainty_config(config: Dict[str, Any]) -> Dict[str, SensorUncertainty]:
    """
    Parse uncertainty specifications from configuration.
    
    Expected config format:
    {
        "uncertainties": {
            "PT-01": {"type": "rel", "value": 0.005},
            "FM-01": {"type": "pct_fs", "value": 0.01, "full_scale": 100},
            "LC-01": {"type": "abs", "value": 5.0}
        }
    }
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary mapping sensor IDs to SensorUncertainty objects
    """
    uncertainties = {}
    u_config = config.get('uncertainties', {})
    
    for sensor_id, spec in u_config.items():
        u_type_str = spec.get('type', 'rel')
        
        # Map string to enum (support both short and full names)
        type_map = {
            'abs': UncertaintyType.ABSOLUTE,
            'absolute': UncertaintyType.ABSOLUTE,
            'rel': UncertaintyType.RELATIVE,
            'relative': UncertaintyType.RELATIVE,
            'pct_fs': UncertaintyType.PERCENT_FS,
            'pct_rd': UncertaintyType.PERCENT_READING,
            'percent_fs': UncertaintyType.PERCENT_FS,
            'percent_reading': UncertaintyType.PERCENT_READING,
        }
        
        u_type = type_map.get(u_type_str, UncertaintyType.RELATIVE)
        
        uncertainties[sensor_id] = SensorUncertainty(
            sensor_id=sensor_id,
            u_type=u_type,
            value=spec.get('value', 0.01),
            full_scale=spec.get('full_scale'),
            unit=spec.get('unit')
        )
    
    return uncertainties


def parse_geometry_uncertainties(config: Dict[str, Any]) -> Dict[str, GeometryUncertainty]:
    """
    Parse geometry uncertainty specifications from configuration.
    
    Expected config format:
    {
        "geometry": {
            "orifice_area_mm2": 0.785,
            "orifice_area_uncertainty_mm2": 0.005,
            "throat_area_mm2": 78.5,
            "throat_area_uncertainty_mm2": 0.5
        }
    }
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary mapping parameter names to GeometryUncertainty objects
    """
    uncertainties = {}
    geom = config.get('geometry', {})
    
    # Find pairs of value and uncertainty
    for key, value in geom.items():
        if '_uncertainty_' in key:
            continue  # Skip uncertainty entries, process with value
        
        # Look for corresponding uncertainty
        u_key = key.replace('_mm2', '_uncertainty_mm2').replace('_mm', '_uncertainty_mm')
        if u_key not in geom:
            # Try alternative pattern
            parts = key.rsplit('_', 1)
            if len(parts) == 2:
                u_key = f"{parts[0]}_uncertainty_{parts[1]}"
        
        uncertainty = geom.get(u_key, 0.0)
        
        # Determine unit from key
        if 'mm2' in key:
            unit = 'mm²'
        elif 'mm' in key:
            unit = 'mm'
        else:
            unit = ''
        
        uncertainties[key] = GeometryUncertainty(
            parameter=key,
            nominal_value=value,
            uncertainty=uncertainty,
            unit=unit
        )
    
    return uncertainties


def get_fluid_density_and_uncertainty(
    config: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None
) -> Tuple[float, float]:
    """
    Get fluid density and uncertainty using the fluid_properties module.

    Tries to use CoolProp via fluid_properties module first, falls back to
    hardcoded config values if unavailable.

    Args:
        config: Test configuration
        metadata: Optional test metadata with fluid conditions

    Returns:
        Tuple of (density_kg_m3, uncertainty_kg_m3)
    """
    fluid = config.get('fluid', {})

    # Try fluid_properties module if available
    if FLUID_PROPERTIES_AVAILABLE and fluid_properties:
        try:
            # Try to get from metadata first
            if metadata:
                try:
                    density, uncertainty = fluid_properties.get_density_from_metadata(metadata)
                    return density, uncertainty
                except (ValueError, KeyError):
                    pass

            # Try to construct from config
            fluid_name = (
                metadata.get('test_fluid', '') if metadata else '' or
                fluid.get('name', '') or
                fluid.get('fluid', '')
            )

            if fluid_name:
                T_K = (
                    metadata.get('fluid_temperature_K', 293.15) if metadata else
                    fluid.get('temperature_K', 293.15)
                )
                P_Pa = (
                    metadata.get('fluid_pressure_Pa', 101325.0) if metadata else
                    fluid.get('pressure_Pa', 101325.0)
                )

                density, uncertainty = fluid_properties.get_density(
                    fluid=fluid_name,
                    T_K=T_K,
                    P_Pa=P_Pa
                )
                return density, uncertainty

        except Exception as e:
            warnings.warn(f"Failed to get fluid properties via fluid_properties module: {e}")

    # Fallback to config values
    density = (
        fluid.get('density_kg_m3') or
        fluid.get('ox_density_kg_m3') or
        fluid.get('water_density_kg_m3') or
        1000  # Default to water
    )
    uncertainty = fluid.get('density_uncertainty_kg_m3', density * 0.005)  # Default 0.5%

    return density, uncertainty


# =============================================================================
# COLD FLOW UNCERTAINTY PROPAGATION
# =============================================================================

def calculate_cd_uncertainty(
    mass_flow_gs: float,
    u_mass_flow_gs: float,
    area_mm2: float,
    u_area_mm2: float,
    delta_p_bar: float,
    u_delta_p_bar: float,
    density_kg_m3: float,
    u_density_kg_m3: float = 0.0
) -> MeasurementWithUncertainty:
    """
    Calculate discharge coefficient with uncertainty propagation.
    
    Cd = ṁ / (A × √(2ρΔP))
    
    Relative uncertainty (RSS):
    (u_Cd/Cd)² = (u_ṁ/ṁ)² + (u_A/A)² + 0.25×(u_ρ/ρ)² + 0.25×(u_ΔP/ΔP)²
    
    Args:
        mass_flow_gs: Mass flow rate in g/s
        u_mass_flow_gs: Uncertainty in mass flow (g/s)
        area_mm2: Orifice area in mm²
        u_area_mm2: Uncertainty in area (mm²)
        delta_p_bar: Pressure drop in bar
        u_delta_p_bar: Uncertainty in pressure drop (bar)
        density_kg_m3: Fluid density in kg/m³
        u_density_kg_m3: Uncertainty in density (kg/m³)
        
    Returns:
        MeasurementWithUncertainty for Cd
    """
    # Convert units
    m_dot_kg_s = mass_flow_gs * 1e-3
    u_m_dot_kg_s = u_mass_flow_gs * 1e-3
    area_m2 = area_mm2 * 1e-6
    u_area_m2 = u_area_mm2 * 1e-6
    delta_p_pa = delta_p_bar * 1e5
    u_delta_p_pa = u_delta_p_bar * 1e5
    
    # Calculate Cd
    if delta_p_pa <= 0 or area_m2 <= 0 or density_kg_m3 <= 0:
        return MeasurementWithUncertainty(
            value=0.0, uncertainty=float('inf'), unit='-', name='Cd'
        )
    
    cd = m_dot_kg_s / (area_m2 * np.sqrt(2 * density_kg_m3 * delta_p_pa))
    
    # Calculate relative uncertainties
    rel_u_m = u_m_dot_kg_s / m_dot_kg_s if m_dot_kg_s > 0 else 0
    rel_u_a = u_area_m2 / area_m2 if area_m2 > 0 else 0
    rel_u_p = u_delta_p_pa / delta_p_pa if delta_p_pa > 0 else 0
    rel_u_rho = u_density_kg_m3 / density_kg_m3 if density_kg_m3 > 0 else 0
    
    # RSS propagation
    # Note: ΔP and ρ have 0.5 exponent in formula, so factor of 0.25 on variance
    rel_u_cd_squared = (
        rel_u_m**2 + 
        rel_u_a**2 + 
        0.25 * rel_u_rho**2 + 
        0.25 * rel_u_p**2
    )
    
    rel_u_cd = np.sqrt(rel_u_cd_squared)
    u_cd = cd * rel_u_cd
    
    return MeasurementWithUncertainty(
        value=cd,
        uncertainty=u_cd,
        unit='-',
        name='Cd'
    )


def calculate_cold_flow_uncertainties(
    avg_values: Dict[str, float],
    config: Dict[str, Any],
    sensor_uncertainties: Optional[Dict[str, SensorUncertainty]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, MeasurementWithUncertainty]:
    """
    Calculate all cold flow metrics with uncertainties.

    Args:
        avg_values: Dictionary of averaged sensor values
        config: Test configuration
        sensor_uncertainties: Pre-parsed sensor uncertainties (optional)
        metadata: Test metadata for fluid property lookup (optional)

    Returns:
        Dictionary of measurements with uncertainties
    """
    results = {}

    # Parse uncertainties if not provided
    if sensor_uncertainties is None:
        sensor_uncertainties = parse_uncertainty_config(config)

    geom_uncertainties = parse_geometry_uncertainties(config)

    # Get sensor role mappings (v2.4.0+ - sensor assignments moved to metadata)
    # Fall back to legacy 'columns' if 'sensor_roles' not present
    sensor_roles = config.get('sensor_roles', {})
    if not sensor_roles:
        sensor_roles = config.get('columns', {})
    geom = config.get('geometry', {})

    # Extract values and uncertainties (using sensor_roles or legacy columns)
    p_up_col = sensor_roles.get('upstream_pressure') or sensor_roles.get('inlet_pressure')
    p_down_col = sensor_roles.get('downstream_pressure')
    mf_col = sensor_roles.get('mass_flow') or sensor_roles.get('mf')

    p_up = avg_values.get(p_up_col, 0)
    p_down = avg_values.get(p_down_col, 0) if p_down_col else 0
    mass_flow = avg_values.get(mf_col, 0)

    # Get sensor uncertainties
    u_p_up = 0.0
    u_p_down = 0.0
    u_mf = 0.0

    if p_up_col and p_up_col in sensor_uncertainties:
        u_p_up = sensor_uncertainties[p_up_col].get_absolute_uncertainty(p_up)

    if p_down_col and p_down_col in sensor_uncertainties:
        u_p_down = sensor_uncertainties[p_down_col].get_absolute_uncertainty(p_down)

    if mf_col and mf_col in sensor_uncertainties:
        u_mf = sensor_uncertainties[mf_col].get_absolute_uncertainty(mass_flow)

    # Delta P and uncertainty
    delta_p = p_up - p_down
    u_delta_p = np.sqrt(u_p_up**2 + u_p_down**2)

    # Store pressure measurements
    results['pressure_upstream'] = MeasurementWithUncertainty(
        value=p_up, uncertainty=u_p_up, unit='bar', name='pressure_upstream'
    )

    results['mass_flow'] = MeasurementWithUncertainty(
        value=mass_flow, uncertainty=u_mf, unit='g/s', name='mass_flow'
    )

    results['delta_p'] = MeasurementWithUncertainty(
        value=delta_p, uncertainty=u_delta_p, unit='bar', name='delta_p'
    )

    # Get geometry
    area_key = 'orifice_area_mm2'
    area = geom.get(area_key, 0)
    u_area = geom.get('orifice_area_uncertainty_mm2', 0)

    if area_key in geom_uncertainties:
        u_area = geom_uncertainties[area_key].uncertainty

    # Get density using fluid_properties module
    density, u_density = get_fluid_density_and_uncertainty(config, metadata)

    # Calculate Cd with uncertainty
    if area > 0 and delta_p > 0 and mass_flow > 0:
        results['Cd'] = calculate_cd_uncertainty(
            mass_flow_gs=mass_flow,
            u_mass_flow_gs=u_mf,
            area_mm2=area,
            u_area_mm2=u_area,
            delta_p_bar=delta_p,
            u_delta_p_bar=u_delta_p,
            density_kg_m3=density,
            u_density_kg_m3=u_density
        )

    return results


# =============================================================================
# HOT FIRE UNCERTAINTY PROPAGATION
# =============================================================================

def calculate_isp_uncertainty(
    thrust_n: float,
    u_thrust_n: float,
    mass_flow_kg_s: float,
    u_mass_flow_kg_s: float,
    g0: float = 9.80665
) -> MeasurementWithUncertainty:
    """
    Calculate specific impulse with uncertainty.
    
    Isp = F / (ṁ × g₀)
    
    Relative uncertainty:
    (u_Isp/Isp)² = (u_F/F)² + (u_ṁ/ṁ)²
    
    Args:
        thrust_n: Thrust in Newtons
        u_thrust_n: Uncertainty in thrust (N)
        mass_flow_kg_s: Total mass flow in kg/s
        u_mass_flow_kg_s: Uncertainty in mass flow (kg/s)
        g0: Standard gravity (m/s²)
        
    Returns:
        MeasurementWithUncertainty for Isp
    """
    if mass_flow_kg_s <= 0 or thrust_n <= 0:
        return MeasurementWithUncertainty(
            value=0.0, uncertainty=float('inf'), unit='s', name='Isp'
        )
    
    isp = thrust_n / (mass_flow_kg_s * g0)
    
    rel_u_f = u_thrust_n / thrust_n
    rel_u_m = u_mass_flow_kg_s / mass_flow_kg_s
    
    rel_u_isp = np.sqrt(rel_u_f**2 + rel_u_m**2)
    u_isp = isp * rel_u_isp
    
    return MeasurementWithUncertainty(
        value=isp, uncertainty=u_isp, unit='s', name='Isp'
    )


def calculate_c_star_uncertainty(
    chamber_pressure_pa: float,
    u_chamber_pressure_pa: float,
    throat_area_m2: float,
    u_throat_area_m2: float,
    mass_flow_kg_s: float,
    u_mass_flow_kg_s: float
) -> MeasurementWithUncertainty:
    """
    Calculate characteristic velocity with uncertainty.
    
    C* = Pc × At / ṁ
    
    Relative uncertainty:
    (u_C*/C*)² = (u_Pc/Pc)² + (u_At/At)² + (u_ṁ/ṁ)²
    
    Args:
        chamber_pressure_pa: Chamber pressure in Pa
        u_chamber_pressure_pa: Uncertainty in pressure (Pa)
        throat_area_m2: Throat area in m²
        u_throat_area_m2: Uncertainty in throat area (m²)
        mass_flow_kg_s: Total mass flow in kg/s
        u_mass_flow_kg_s: Uncertainty in mass flow (kg/s)
        
    Returns:
        MeasurementWithUncertainty for C*
    """
    if mass_flow_kg_s <= 0 or chamber_pressure_pa <= 0 or throat_area_m2 <= 0:
        return MeasurementWithUncertainty(
            value=0.0, uncertainty=float('inf'), unit='m/s', name='c_star'
        )
    
    c_star = (chamber_pressure_pa * throat_area_m2) / mass_flow_kg_s
    
    rel_u_p = u_chamber_pressure_pa / chamber_pressure_pa
    rel_u_a = u_throat_area_m2 / throat_area_m2
    rel_u_m = u_mass_flow_kg_s / mass_flow_kg_s
    
    rel_u_c_star = np.sqrt(rel_u_p**2 + rel_u_a**2 + rel_u_m**2)
    u_c_star = c_star * rel_u_c_star
    
    return MeasurementWithUncertainty(
        value=c_star, uncertainty=u_c_star, unit='m/s', name='c_star'
    )


def calculate_of_ratio_uncertainty(
    m_ox_kg_s: float,
    u_m_ox_kg_s: float,
    m_fuel_kg_s: float,
    u_m_fuel_kg_s: float
) -> MeasurementWithUncertainty:
    """
    Calculate O/F ratio with uncertainty.
    
    O/F = ṁ_ox / ṁ_fuel
    
    Relative uncertainty:
    (u_OF/OF)² = (u_ṁox/ṁox)² + (u_ṁfuel/ṁfuel)²
    """
    if m_fuel_kg_s <= 0 or m_ox_kg_s <= 0:
        return MeasurementWithUncertainty(
            value=0.0, uncertainty=float('inf'), unit='-', name='of_ratio'
        )
    
    of_ratio = m_ox_kg_s / m_fuel_kg_s
    
    rel_u_ox = u_m_ox_kg_s / m_ox_kg_s
    rel_u_fuel = u_m_fuel_kg_s / m_fuel_kg_s
    
    rel_u_of = np.sqrt(rel_u_ox**2 + rel_u_fuel**2)
    u_of = of_ratio * rel_u_of
    
    return MeasurementWithUncertainty(
        value=of_ratio, uncertainty=u_of, unit='-', name='of_ratio'
    )


def calculate_hot_fire_uncertainties(
    avg_values: Dict[str, float],
    config: Dict[str, Any],
    sensor_uncertainties: Optional[Dict[str, SensorUncertainty]] = None
) -> Dict[str, MeasurementWithUncertainty]:
    """
    Calculate all hot fire metrics with uncertainties.
    
    Args:
        avg_values: Dictionary of averaged sensor values
        config: Test configuration
        sensor_uncertainties: Pre-parsed sensor uncertainties (optional)
        
    Returns:
        Dictionary of measurements with uncertainties
    """
    results = {}
    g0 = 9.80665
    
    # Parse uncertainties if not provided
    if sensor_uncertainties is None:
        sensor_uncertainties = parse_uncertainty_config(config)
    
    geom_uncertainties = parse_geometry_uncertainties(config)

    # Get sensor role mappings (v2.4.0+ - sensor assignments moved to metadata)
    # Fall back to legacy 'columns' if 'sensor_roles' not present
    sensor_roles = config.get('sensor_roles', {})
    if not sensor_roles:
        sensor_roles = config.get('columns', {})
    geom = config.get('geometry', {})

    # Extract values (using sensor_roles or legacy columns)
    pc_col = sensor_roles.get('chamber_pressure')
    thrust_col = sensor_roles.get('thrust')
    mf_ox_col = sensor_roles.get('mass_flow_ox')
    mf_fuel_col = sensor_roles.get('mass_flow_fuel')
    
    pc = avg_values.get(pc_col, 0) if pc_col else 0
    thrust = avg_values.get(thrust_col, 0) if thrust_col else 0
    mf_ox = avg_values.get(mf_ox_col, 0) if mf_ox_col else 0
    mf_fuel = avg_values.get(mf_fuel_col, 0) if mf_fuel_col else 0
    mf_total = mf_ox + mf_fuel
    
    # Get sensor uncertainties
    u_pc = 0.0
    u_thrust = 0.0
    u_mf_ox = 0.0
    u_mf_fuel = 0.0
    
    if pc_col and pc_col in sensor_uncertainties:
        u_pc = sensor_uncertainties[pc_col].get_absolute_uncertainty(pc)
    
    if thrust_col and thrust_col in sensor_uncertainties:
        u_thrust = sensor_uncertainties[thrust_col].get_absolute_uncertainty(thrust)
    
    if mf_ox_col and mf_ox_col in sensor_uncertainties:
        u_mf_ox = sensor_uncertainties[mf_ox_col].get_absolute_uncertainty(mf_ox)
    
    if mf_fuel_col and mf_fuel_col in sensor_uncertainties:
        u_mf_fuel = sensor_uncertainties[mf_fuel_col].get_absolute_uncertainty(mf_fuel)
    
    # Total mass flow uncertainty
    u_mf_total = np.sqrt(u_mf_ox**2 + u_mf_fuel**2)
    
    # Store base measurements
    results['chamber_pressure'] = MeasurementWithUncertainty(
        value=pc, uncertainty=u_pc, unit='bar', name='chamber_pressure'
    )
    
    results['thrust'] = MeasurementWithUncertainty(
        value=thrust, uncertainty=u_thrust, unit='N', name='thrust'
    )
    
    results['mass_flow_total'] = MeasurementWithUncertainty(
        value=mf_total, uncertainty=u_mf_total, unit='g/s', name='mass_flow_total'
    )
    
    # O/F Ratio
    if mf_ox > 0 and mf_fuel > 0:
        results['of_ratio'] = calculate_of_ratio_uncertainty(
            m_ox_kg_s=mf_ox * 1e-3,
            u_m_ox_kg_s=u_mf_ox * 1e-3,
            m_fuel_kg_s=mf_fuel * 1e-3,
            u_m_fuel_kg_s=u_mf_fuel * 1e-3
        )
    
    # Isp
    if thrust > 0 and mf_total > 0:
        results['Isp'] = calculate_isp_uncertainty(
            thrust_n=thrust,
            u_thrust_n=u_thrust,
            mass_flow_kg_s=mf_total * 1e-3,
            u_mass_flow_kg_s=u_mf_total * 1e-3
        )
    
    # C*
    throat_area_mm2 = geom.get('throat_area_mm2', 0)
    u_throat_area_mm2 = geom.get('throat_area_uncertainty_mm2', 0)
    
    if pc > 0 and mf_total > 0 and throat_area_mm2 > 0:
        results['c_star'] = calculate_c_star_uncertainty(
            chamber_pressure_pa=pc * 1e5,
            u_chamber_pressure_pa=u_pc * 1e5,
            throat_area_m2=throat_area_mm2 * 1e-6,
            u_throat_area_m2=u_throat_area_mm2 * 1e-6,
            mass_flow_kg_s=mf_total * 1e-3,
            u_mass_flow_kg_s=u_mf_total * 1e-3
        )
    
    return results


# =============================================================================
# STATISTICAL UNCERTAINTY FROM DATA
# =============================================================================

def calculate_statistical_uncertainty(
    df: pd.DataFrame,
    column: str,
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Calculate statistical uncertainty from measurement variation.
    
    Uses standard error of the mean with t-distribution correction.
    
    Args:
        df: DataFrame containing data
        column: Column name to analyze
        confidence: Confidence level (default 95%)
        
    Returns:
        Tuple of (mean, std, uncertainty_of_mean)
    """
    from scipy import stats
    
    data = df[column].dropna()
    n = len(data)
    
    if n < 2:
        return data.mean() if n > 0 else 0.0, 0.0, float('inf')
    
    mean = data.mean()
    std = data.std(ddof=1)
    sem = std / np.sqrt(n)
    
    # t-distribution critical value
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n-1)
    
    uncertainty = t_crit * sem
    
    return mean, std, uncertainty


def combine_uncertainties(
    systematic_uncertainty: float,
    statistical_uncertainty: float
) -> float:
    """
    Combine systematic and statistical uncertainties.
    
    Uses root-sum-square (RSS) combination.
    
    Args:
        systematic_uncertainty: From sensor specs, geometry tolerances, etc.
        statistical_uncertainty: From data variation
        
    Returns:
        Combined uncertainty
    """
    return np.sqrt(systematic_uncertainty**2 + statistical_uncertainty**2)


def format_with_uncertainty(
    value: float,
    uncertainty: float,
    significant_figures: int = 2
) -> str:
    """
    Format a value with uncertainty using proper significant figures.
    
    The uncertainty determines the precision of the value.
    
    Args:
        value: Central value
        uncertainty: Absolute uncertainty
        significant_figures: Significant figures for uncertainty
        
    Returns:
        Formatted string like "0.654 ± 0.018"
    """
    if uncertainty == 0 or np.isinf(uncertainty) or np.isnan(uncertainty):
        return f"{value:.4g} ± ?"
    
    # Determine decimal places from uncertainty
    if uncertainty >= 1:
        u_decimals = max(0, significant_figures - int(np.floor(np.log10(uncertainty))) - 1)
    else:
        u_decimals = -int(np.floor(np.log10(uncertainty))) + significant_figures - 1
    
    u_decimals = max(0, min(6, u_decimals))

    return f"{value:.{u_decimals}f} ± {uncertainty:.{u_decimals}f}"


# =============================================================================
# CORRELATION-AWARE UNCERTAINTY PROPAGATION
# =============================================================================

@dataclass
class CorrelationMatrix:
    """
    Correlation matrix for input quantities.

    Used when input uncertainties are not independent (e.g., two pressure
    sensors sharing the same calibration reference).
    """
    variables: List[str]
    matrix: np.ndarray

    def __post_init__(self):
        n = len(self.variables)
        if self.matrix.shape != (n, n):
            raise ValueError(f"Matrix shape {self.matrix.shape} doesn't match {n} variables")

    def get_correlation(self, var_a: str, var_b: str) -> float:
        """Get correlation coefficient between two variables."""
        i = self.variables.index(var_a)
        j = self.variables.index(var_b)
        return float(self.matrix[i, j])

    @classmethod
    def identity(cls, variables: List[str]) -> 'CorrelationMatrix':
        """Create uncorrelated (identity) matrix."""
        n = len(variables)
        return cls(variables=variables, matrix=np.eye(n))


def propagate_with_correlation(
    partial_derivatives: Dict[str, float],
    uncertainties: Dict[str, float],
    correlation: Optional[CorrelationMatrix] = None,
) -> float:
    """
    Propagate uncertainties with correlation using the GUM law of propagation.

    u_c^2 = sum_i (df/dx_i)^2 * u(x_i)^2
           + 2 * sum_{i<j} (df/dx_i)(df/dx_j) * u(x_i) * u(x_j) * r(x_i, x_j)

    Args:
        partial_derivatives: {variable_name: df/dx_i} sensitivity coefficients
        uncertainties: {variable_name: u(x_i)} standard uncertainties
        correlation: Optional correlation matrix between variables

    Returns:
        Combined standard uncertainty u_c
    """
    variables = list(partial_derivatives.keys())
    n = len(variables)

    # Build arrays
    c = np.array([partial_derivatives[v] for v in variables])
    u = np.array([uncertainties.get(v, 0.0) for v in variables])

    # Uncorrelated terms
    u_c_sq = np.sum((c * u) ** 2)

    # Correlated terms (only if correlation matrix provided)
    if correlation is not None:
        for i in range(n):
            for j in range(i + 1, n):
                vi, vj = variables[i], variables[j]
                if vi in correlation.variables and vj in correlation.variables:
                    r_ij = correlation.get_correlation(vi, vj)
                    u_c_sq += 2 * c[i] * c[j] * u[i] * u[j] * r_ij

    return np.sqrt(max(0.0, u_c_sq))


# =============================================================================
# MONTE CARLO UNCERTAINTY ENGINE
# =============================================================================

@dataclass
class MonteCarloResult:
    """
    Result of Monte Carlo uncertainty propagation.

    Attributes:
        mean: Mean of MC samples
        std: Standard deviation of MC samples
        percentiles: Dictionary of percentile values (e.g., {2.5: val, 97.5: val})
        n_samples: Number of MC samples used
        samples: Raw MC output samples (optional, for diagnostics)
    """
    mean: float
    std: float
    percentiles: Dict[float, float]
    n_samples: int
    samples: Optional[np.ndarray] = None

    def to_measurement(self, unit: str = "", name: str = "",
                       confidence: float = 0.95) -> MeasurementWithUncertainty:
        """Convert to MeasurementWithUncertainty using MC statistics."""
        # Estimate coverage factor from MC distribution
        lower_p = (1 - confidence) / 2 * 100
        upper_p = (1 + confidence) / 2 * 100
        half_interval = (self.percentiles.get(upper_p, self.mean + 2 * self.std)
                        - self.percentiles.get(lower_p, self.mean - 2 * self.std)) / 2
        k = half_interval / self.std if self.std > 0 else 2.0

        return MeasurementWithUncertainty(
            value=self.mean,
            uncertainty=self.std,
            unit=unit,
            name=name,
            coverage_factor=k,
            confidence_level=confidence,
        )


def monte_carlo_propagation(
    func,
    input_means: Dict[str, float],
    input_uncertainties: Dict[str, float],
    correlation: Optional[CorrelationMatrix] = None,
    n_samples: int = 10000,
    seed: Optional[int] = None,
    return_samples: bool = False,
) -> MonteCarloResult:
    """
    Monte Carlo uncertainty propagation for arbitrary functions.

    Samples input distributions (assumed normal), evaluates the function,
    and computes output statistics.

    Args:
        func: Callable that takes **kwargs of input variables and returns a float
        input_means: {variable_name: mean_value}
        input_uncertainties: {variable_name: standard_uncertainty}
        correlation: Optional correlation matrix between inputs
        n_samples: Number of Monte Carlo samples
        seed: Random seed for reproducibility
        return_samples: If True, include raw samples in result

    Returns:
        MonteCarloResult with output statistics
    """
    rng = np.random.default_rng(seed)
    variables = list(input_means.keys())
    n_vars = len(variables)

    means = np.array([input_means[v] for v in variables])
    stds = np.array([input_uncertainties.get(v, 0.0) for v in variables])

    # Generate correlated or uncorrelated samples
    if correlation is not None:
        # Build covariance matrix from correlation and standard deviations
        cov = np.outer(stds, stds) * correlation.matrix
        samples = rng.multivariate_normal(means, cov, size=n_samples)
    else:
        # Independent normal samples
        samples = rng.normal(
            loc=means,
            scale=stds,
            size=(n_samples, n_vars)
        )

    # Evaluate function for each sample
    outputs = np.zeros(n_samples)
    for i in range(n_samples):
        kwargs = {variables[j]: samples[i, j] for j in range(n_vars)}
        try:
            outputs[i] = func(**kwargs)
        except (ValueError, ZeroDivisionError, FloatingPointError):
            outputs[i] = np.nan

    # Remove failed evaluations
    valid = ~np.isnan(outputs) & ~np.isinf(outputs)
    valid_outputs = outputs[valid]

    if len(valid_outputs) < 100:
        warnings.warn(
            f"Monte Carlo: only {len(valid_outputs)}/{n_samples} valid samples. "
            f"Results may be unreliable."
        )

    if len(valid_outputs) == 0:
        return MonteCarloResult(
            mean=0.0, std=float('inf'),
            percentiles={}, n_samples=0,
        )

    # Compute statistics
    mean = float(np.mean(valid_outputs))
    std = float(np.std(valid_outputs, ddof=1))

    percentile_levels = [0.5, 2.5, 5.0, 16.0, 25.0, 50.0, 75.0, 84.0, 95.0, 97.5, 99.5]
    percentiles = {
        p: float(np.percentile(valid_outputs, p))
        for p in percentile_levels
    }

    return MonteCarloResult(
        mean=mean,
        std=std,
        percentiles=percentiles,
        n_samples=len(valid_outputs),
        samples=valid_outputs if return_samples else None,
    )
