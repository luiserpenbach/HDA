"""
Fluid Properties Module
=======================
Calculate fluid properties using CoolProp for test analysis.

This module provides:
- Fluid property lookup (density, viscosity, etc.)
- Common fluid aliases for convenience
- Fallback values when CoolProp is unavailable
- Uncertainty estimation for fluid properties

Usage:
    from core.fluid_properties import get_fluid_properties, FluidState
    
    # Get properties at specific conditions
    props = get_fluid_properties("Water", T_K=293.15, P_Pa=101325)
    print(f"Density: {props.density_kg_m3} kg/m3")
    
    # Or use FluidState dataclass
    state = FluidState(fluid="Nitrogen", temperature_K=300, pressure_Pa=500000)
    props = state.get_properties()
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
import warnings

# Try to import CoolProp
try:
    import CoolProp.CoolProp as CP
    from CoolProp.CoolProp import PropsSI
    COOLPROP_AVAILABLE = True
except ImportError:
    COOLPROP_AVAILABLE = False
    PropsSI = None


# =============================================================================
# FLUID NAME ALIASES
# =============================================================================

# Map common names to CoolProp fluid names
FLUID_ALIASES: Dict[str, str] = {
    # Water
    'water': 'Water',
    'h2o': 'Water',
    'deionized water': 'Water',
    'di water': 'Water',
    
    # Nitrogen
    'nitrogen': 'Nitrogen',
    'n2': 'Nitrogen',
    'gn2': 'Nitrogen',
    'ln2': 'Nitrogen',
    
    # Oxygen
    'oxygen': 'Oxygen',
    'o2': 'Oxygen',
    'gox': 'Oxygen',
    'lox': 'Oxygen',
    
    # Helium
    'helium': 'Helium',
    'he': 'Helium',
    
    # Air
    'air': 'Air',
    
    # Argon
    'argon': 'Argon',
    'ar': 'Argon',
    
    # CO2
    'carbon dioxide': 'CarbonDioxide',
    'co2': 'CarbonDioxide',
    
    # Methane
    'methane': 'Methane',
    'ch4': 'Methane',
    'lng': 'Methane',
    
    # Ethanol
    'ethanol': 'Ethanol',
    'ethyl alcohol': 'Ethanol',
    
    # Isopropanol
    'isopropanol': 'Isopropanol',
    'ipa': 'Isopropanol',
    'isopropyl alcohol': 'Isopropanol',
    '2-propanol': 'Isopropanol',
    
    # Propane
    'propane': 'Propane',
    'c3h8': 'Propane',
    
    # Hydrogen
    'hydrogen': 'Hydrogen',
    'h2': 'Hydrogen',
    'lh2': 'Hydrogen',
    
    # n-Dodecane (RP-1 surrogate)
    'rp-1': 'n-Dodecane',
    'rp1': 'n-Dodecane',
    'kerosene': 'n-Dodecane',
    'jet-a': 'n-Dodecane',
    'dodecane': 'n-Dodecane',
    'n-dodecane': 'n-Dodecane',
    
    # Nitrous Oxide
    'nitrous oxide': 'NitrousOxide',
    'n2o': 'NitrousOxide',
    'nitrous': 'NitrousOxide',
    
    # Ammonia
    'ammonia': 'Ammonia',
    'nh3': 'Ammonia',
}


# Fallback properties when CoolProp is not available (at 20C, 1 atm)
FALLBACK_PROPERTIES: Dict[str, Dict[str, float]] = {
    'Water': {
        'density_kg_m3': 998.2,
        'viscosity_Pa_s': 0.001002,
        'specific_heat_J_kg_K': 4182,
        'thermal_conductivity_W_m_K': 0.598,
        'phase': 'liquid',
    },
    'Nitrogen': {
        'density_kg_m3': 1.165,
        'viscosity_Pa_s': 1.76e-5,
        'specific_heat_J_kg_K': 1040,
        'thermal_conductivity_W_m_K': 0.0258,
        'phase': 'gas',
        'gamma': 1.4,
        'molar_mass_kg_mol': 0.028014,
    },
    'Oxygen': {
        'density_kg_m3': 1.331,
        'viscosity_Pa_s': 2.04e-5,
        'specific_heat_J_kg_K': 918,
        'thermal_conductivity_W_m_K': 0.0263,
        'phase': 'gas',
        'gamma': 1.4,
        'molar_mass_kg_mol': 0.032,
    },
    'Helium': {
        'density_kg_m3': 0.166,
        'viscosity_Pa_s': 1.96e-5,
        'specific_heat_J_kg_K': 5193,
        'thermal_conductivity_W_m_K': 0.152,
        'phase': 'gas',
        'gamma': 1.667,
        'molar_mass_kg_mol': 0.004003,
    },
    'Air': {
        'density_kg_m3': 1.204,
        'viscosity_Pa_s': 1.81e-5,
        'specific_heat_J_kg_K': 1006,
        'thermal_conductivity_W_m_K': 0.0257,
        'phase': 'gas',
        'gamma': 1.4,
        'molar_mass_kg_mol': 0.02897,
    },
    'Ethanol': {
        'density_kg_m3': 789.0,
        'viscosity_Pa_s': 0.00109,
        'specific_heat_J_kg_K': 2440,
        'thermal_conductivity_W_m_K': 0.171,
        'phase': 'liquid',
    },
    'Isopropanol': {
        'density_kg_m3': 786.0,
        'viscosity_Pa_s': 0.00237,
        'specific_heat_J_kg_K': 2600,
        'thermal_conductivity_W_m_K': 0.140,
        'phase': 'liquid',
    },
    'n-Dodecane': {
        'density_kg_m3': 750.0,
        'viscosity_Pa_s': 0.00134,
        'specific_heat_J_kg_K': 2210,
        'thermal_conductivity_W_m_K': 0.140,
        'phase': 'liquid',
    },
    'NitrousOxide': {
        'density_kg_m3': 1.83,  # Gas at 1 atm
        'viscosity_Pa_s': 1.46e-5,
        'specific_heat_J_kg_K': 880,
        'thermal_conductivity_W_m_K': 0.017,
        'phase': 'gas',
        'gamma': 1.27,
        'molar_mass_kg_mol': 0.044013,
    },
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FluidProperties:
    """
    Container for fluid properties at a specific state.
    
    All properties use SI units.
    """
    fluid_name: str
    temperature_K: float
    pressure_Pa: float
    
    # Core properties
    density_kg_m3: float = 0.0
    viscosity_Pa_s: float = 0.0
    specific_heat_J_kg_K: float = 0.0
    thermal_conductivity_W_m_K: float = 0.0
    
    # Gas-specific properties
    gamma: Optional[float] = None  # Cp/Cv ratio
    speed_of_sound_m_s: Optional[float] = None
    molar_mass_kg_mol: Optional[float] = None
    compressibility_factor: Optional[float] = None
    
    # Phase info
    phase: str = "unknown"  # 'liquid', 'gas', 'supercritical', 'two-phase'
    quality: Optional[float] = None  # Vapor quality (0-1) for two-phase
    
    # Uncertainty estimates (relative, 0-1)
    density_uncertainty: float = 0.01  # Default 1%
    viscosity_uncertainty: float = 0.02  # Default 2%
    
    # Source info
    source: str = "unknown"  # 'coolprop', 'fallback', 'user'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'fluid_name': self.fluid_name,
            'temperature_K': self.temperature_K,
            'pressure_Pa': self.pressure_Pa,
            'density_kg_m3': self.density_kg_m3,
            'viscosity_Pa_s': self.viscosity_Pa_s,
            'specific_heat_J_kg_K': self.specific_heat_J_kg_K,
            'thermal_conductivity_W_m_K': self.thermal_conductivity_W_m_K,
            'gamma': self.gamma,
            'speed_of_sound_m_s': self.speed_of_sound_m_s,
            'molar_mass_kg_mol': self.molar_mass_kg_mol,
            'compressibility_factor': self.compressibility_factor,
            'phase': self.phase,
            'quality': self.quality,
            'density_uncertainty': self.density_uncertainty,
            'viscosity_uncertainty': self.viscosity_uncertainty,
            'source': self.source,
        }


@dataclass
class FluidState:
    """
    Define a fluid state for property lookup.
    
    Usage:
        state = FluidState(fluid="Water", temperature_K=300, pressure_Pa=101325)
        props = state.get_properties()
    """
    fluid: str
    temperature_K: float = 293.15  # Default 20C
    pressure_Pa: float = 101325.0  # Default 1 atm
    
    def get_properties(self) -> FluidProperties:
        """Get fluid properties at this state."""
        return get_fluid_properties(
            self.fluid,
            T_K=self.temperature_K,
            P_Pa=self.pressure_Pa
        )
    
    @classmethod
    def from_metadata(cls, metadata: Dict[str, Any]) -> 'FluidState':
        """
        Create FluidState from test metadata.
        
        Args:
            metadata: Test metadata dict with test_fluid, fluid_temperature_K, fluid_pressure_Pa
            
        Returns:
            FluidState object
        """
        return cls(
            fluid=metadata.get('test_fluid', ''),
            temperature_K=metadata.get('fluid_temperature_K', 293.15),
            pressure_Pa=metadata.get('fluid_pressure_Pa', 101325.0),
        )


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def normalize_fluid_name(name: str) -> str:
    """
    Normalize a fluid name to CoolProp format.
    
    Args:
        name: Fluid name (can be alias)
        
    Returns:
        CoolProp fluid name
        
    Raises:
        ValueError: If fluid name not recognized
    """
    if not name:
        raise ValueError("Fluid name cannot be empty")
    
    # Check if already a valid CoolProp name
    name_lower = name.lower().strip()
    
    # Check aliases
    if name_lower in FLUID_ALIASES:
        return FLUID_ALIASES[name_lower]
    
    # Check if it's already a CoolProp name (case-insensitive)
    for alias, coolprop_name in FLUID_ALIASES.items():
        if coolprop_name.lower() == name_lower:
            return coolprop_name
    
    # Return as-is, hope CoolProp recognizes it
    return name


def get_fluid_properties(
    fluid: str,
    T_K: float = 293.15,
    P_Pa: float = 101325.0,
    use_fallback: bool = True,
) -> FluidProperties:
    """
    Get fluid properties at specified conditions.
    
    Args:
        fluid: Fluid name (can use aliases like 'water', 'n2', 'ipa')
        T_K: Temperature in Kelvin
        P_Pa: Pressure in Pascals
        use_fallback: If True, use fallback values when CoolProp unavailable
        
    Returns:
        FluidProperties object
        
    Raises:
        ValueError: If fluid not found and no fallback available
    """
    try:
        fluid_name = normalize_fluid_name(fluid)
    except ValueError:
        fluid_name = fluid
    
    # Try CoolProp first
    if COOLPROP_AVAILABLE:
        try:
            return _get_properties_coolprop(fluid_name, T_K, P_Pa)
        except Exception as e:
            if use_fallback:
                warnings.warn(f"CoolProp failed for {fluid_name}: {e}. Using fallback.")
            else:
                raise ValueError(f"CoolProp failed for {fluid_name}: {e}")
    
    # Use fallback
    if use_fallback:
        return _get_properties_fallback(fluid_name, T_K, P_Pa)
    else:
        raise ValueError(f"CoolProp not available and fallback disabled for {fluid_name}")


def _get_properties_coolprop(fluid_name: str, T_K: float, P_Pa: float) -> FluidProperties:
    """Get properties using CoolProp."""
    props = FluidProperties(
        fluid_name=fluid_name,
        temperature_K=T_K,
        pressure_Pa=P_Pa,
        source='coolprop',
    )
    
    try:
        # Core properties (convert NumPy types to native Python floats for SQLite compatibility)
        props.density_kg_m3 = float(PropsSI('D', 'T', T_K, 'P', P_Pa, fluid_name))
        props.viscosity_Pa_s = float(PropsSI('V', 'T', T_K, 'P', P_Pa, fluid_name))
        props.specific_heat_J_kg_K = float(PropsSI('C', 'T', T_K, 'P', P_Pa, fluid_name))
        props.thermal_conductivity_W_m_K = float(PropsSI('L', 'T', T_K, 'P', P_Pa, fluid_name))

        # Phase determination
        phase_index = PropsSI('Phase', 'T', T_K, 'P', P_Pa, fluid_name)
        phase_map = {
            0: 'liquid',
            1: 'supercritical',
            2: 'supercritical_gas',
            3: 'supercritical_liquid',
            5: 'gas',
            6: 'two-phase',
        }
        props.phase = phase_map.get(int(phase_index), 'unknown')

        # Gas properties (convert NumPy types to native Python floats)
        if props.phase in ['gas', 'supercritical', 'supercritical_gas']:
            cp = float(PropsSI('C', 'T', T_K, 'P', P_Pa, fluid_name))
            cv = float(PropsSI('O', 'T', T_K, 'P', P_Pa, fluid_name))  # O = Cv
            if cv > 0:
                props.gamma = float(cp / cv)
            props.speed_of_sound_m_s = float(PropsSI('A', 'T', T_K, 'P', P_Pa, fluid_name))
            props.compressibility_factor = float(PropsSI('Z', 'T', T_K, 'P', P_Pa, fluid_name))

        # Molar mass (always available, convert to native Python float)
        props.molar_mass_kg_mol = float(PropsSI('M', 'T', T_K, 'P', P_Pa, fluid_name))
        
        # CoolProp uncertainty estimates (rough)
        props.density_uncertainty = 0.001  # CoolProp typically <0.1% for common fluids
        props.viscosity_uncertainty = 0.01  # ~1% for viscosity
        
    except Exception as e:
        raise ValueError(f"CoolProp property calculation failed: {e}")
    
    return props


def _get_properties_fallback(fluid_name: str, T_K: float, P_Pa: float) -> FluidProperties:
    """Get properties from fallback tables."""
    props = FluidProperties(
        fluid_name=fluid_name,
        temperature_K=T_K,
        pressure_Pa=P_Pa,
        source='fallback',
    )
    
    if fluid_name not in FALLBACK_PROPERTIES:
        raise ValueError(
            f"No fallback properties for '{fluid_name}'. "
            f"Available: {list(FALLBACK_PROPERTIES.keys())}"
        )
    
    fallback = FALLBACK_PROPERTIES[fluid_name]
    
    props.density_kg_m3 = fallback.get('density_kg_m3', 0)
    props.viscosity_Pa_s = fallback.get('viscosity_Pa_s', 0)
    props.specific_heat_J_kg_K = fallback.get('specific_heat_J_kg_K', 0)
    props.thermal_conductivity_W_m_K = fallback.get('thermal_conductivity_W_m_K', 0)
    props.phase = fallback.get('phase', 'unknown')
    props.gamma = fallback.get('gamma')
    props.molar_mass_kg_mol = fallback.get('molar_mass_kg_mol')
    
    # Correct for pressure/temperature if gas (ideal gas approximation)
    if props.phase == 'gas':
        # Reference conditions: 293.15 K, 101325 Pa
        T_ref = 293.15
        P_ref = 101325.0
        props.density_kg_m3 *= (P_Pa / P_ref) * (T_ref / T_K)
        
        # Speed of sound for ideal gas
        if props.gamma and props.molar_mass_kg_mol:
            R = 8.314  # J/(mol*K)
            props.speed_of_sound_m_s = (
                props.gamma * R * T_K / props.molar_mass_kg_mol
            ) ** 0.5
    
    # Higher uncertainty for fallback values
    props.density_uncertainty = 0.02  # 2%
    props.viscosity_uncertainty = 0.05  # 5%
    
    warnings.warn(
        f"Using fallback properties for {fluid_name}. "
        "Install CoolProp for accurate properties: pip install CoolProp"
    )
    
    return props


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_density(
    fluid: str,
    T_K: float = 293.15,
    P_Pa: float = 101325.0,
) -> Tuple[float, float]:
    """
    Convenience function to get just density and uncertainty.
    
    Args:
        fluid: Fluid name
        T_K: Temperature [K]
        P_Pa: Pressure [Pa]
        
    Returns:
        Tuple of (density_kg_m3, uncertainty_kg_m3)
    """
    props = get_fluid_properties(fluid, T_K, P_Pa)
    uncertainty = props.density_kg_m3 * props.density_uncertainty
    return props.density_kg_m3, uncertainty


def get_density_from_metadata(metadata: Dict[str, Any]) -> Tuple[float, float]:
    """
    Get fluid density from test metadata.
    
    Args:
        metadata: Test metadata dict
        
    Returns:
        Tuple of (density_kg_m3, uncertainty_kg_m3)
        
    Raises:
        ValueError: If test_fluid not specified in metadata
    """
    fluid = metadata.get('test_fluid', '')
    if not fluid:
        raise ValueError("test_fluid not specified in metadata")
    
    T_K = metadata.get('fluid_temperature_K', 293.15)
    P_Pa = metadata.get('fluid_pressure_Pa', 101325.0)
    
    return get_density(fluid, T_K, P_Pa)


def list_available_fluids() -> Dict[str, str]:
    """
    List all available fluid aliases.
    
    Returns:
        Dict mapping alias -> CoolProp name
    """
    return FLUID_ALIASES.copy()


def is_compressible(fluid: str, T_K: float = 293.15, P_Pa: float = 101325.0) -> bool:
    """
    Determine if fluid should be treated as compressible.
    
    Args:
        fluid: Fluid name
        T_K: Temperature [K]
        P_Pa: Pressure [Pa]
        
    Returns:
        True if gas/compressible, False if liquid/incompressible
    """
    props = get_fluid_properties(fluid, T_K, P_Pa)
    return props.phase in ['gas', 'supercritical', 'supercritical_gas']


def check_coolprop_available() -> bool:
    """Check if CoolProp is installed."""
    return COOLPROP_AVAILABLE


# =============================================================================
# BIPROPELLANT SUPPORT
# =============================================================================

@dataclass
class BipropellantState:
    """
    Define oxidizer and fuel states for hot fire analysis.
    """
    oxidizer: str
    fuel: str
    ox_temperature_K: float = 90.0  # LOX default
    fuel_temperature_K: float = 293.15
    ox_pressure_Pa: float = 101325.0
    fuel_pressure_Pa: float = 101325.0
    
    def get_ox_properties(self) -> FluidProperties:
        """Get oxidizer properties."""
        return get_fluid_properties(
            self.oxidizer,
            T_K=self.ox_temperature_K,
            P_Pa=self.ox_pressure_Pa
        )
    
    def get_fuel_properties(self) -> FluidProperties:
        """Get fuel properties."""
        return get_fluid_properties(
            self.fuel,
            T_K=self.fuel_temperature_K,
            P_Pa=self.fuel_pressure_Pa
        )
    
    @classmethod
    def from_metadata(cls, metadata: Dict[str, Any]) -> 'BipropellantState':
        """
        Create from test metadata.
        
        Expected metadata fields:
            - oxidizer, fuel
            - ox_temperature_K, fuel_temperature_K
            - ox_pressure_Pa, fuel_pressure_Pa
        """
        return cls(
            oxidizer=metadata.get('oxidizer', ''),
            fuel=metadata.get('fuel', ''),
            ox_temperature_K=metadata.get('ox_temperature_K', 90.0),
            fuel_temperature_K=metadata.get('fuel_temperature_K', 293.15),
            ox_pressure_Pa=metadata.get('ox_pressure_Pa', 101325.0),
            fuel_pressure_Pa=metadata.get('fuel_pressure_Pa', 101325.0),
        )
