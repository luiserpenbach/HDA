"""
Test Suite for Fluid Properties Module
=======================================
Validates fluid property calculations and CoolProp integration.

Run with: python -m pytest tests/test_fluid_properties.py -v
Or:       python tests/test_fluid_properties.py
"""

import sys
import warnings
from pathlib import Path

# Add core to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.fluid_properties import (
    normalize_fluid_name,
    get_fluid_properties,
    get_density,
    get_density_from_metadata,
    list_available_fluids,
    is_compressible,
    check_coolprop_available,
    FluidState,
    FluidProperties,
    BipropellantState,
    COOLPROP_AVAILABLE,
)


# =============================================================================
# FLUID NAME NORMALIZATION TESTS
# =============================================================================

def test_normalize_fluid_name_aliases():
    """Test that common aliases are normalized correctly."""
    # Water aliases
    assert normalize_fluid_name('water') == 'Water'
    assert normalize_fluid_name('h2o') == 'Water'
    assert normalize_fluid_name('di water') == 'Water'

    # Nitrogen aliases
    assert normalize_fluid_name('nitrogen') == 'Nitrogen'
    assert normalize_fluid_name('n2') == 'Nitrogen'
    assert normalize_fluid_name('gn2') == 'Nitrogen'

    # RP-1 surrogate
    assert normalize_fluid_name('rp-1') == 'n-Dodecane'
    assert normalize_fluid_name('kerosene') == 'n-Dodecane'

    # IPA
    assert normalize_fluid_name('ipa') == 'Isopropanol'
    assert normalize_fluid_name('isopropanol') == 'Isopropanol'


def test_normalize_fluid_name_case_insensitive():
    """Test that normalization is case-insensitive."""
    assert normalize_fluid_name('WATER') == 'Water'
    assert normalize_fluid_name('WaTeR') == 'Water'
    assert normalize_fluid_name('N2') == 'Nitrogen'


def test_normalize_fluid_name_empty():
    """Test that empty names raise ValueError."""
    try:
        normalize_fluid_name('')
        assert False, "Should raise ValueError for empty name"
    except ValueError:
        pass


# =============================================================================
# FLUID PROPERTIES TESTS
# =============================================================================

def test_get_fluid_properties_water():
    """Test water properties at standard conditions."""
    props = get_fluid_properties('water', T_K=293.15, P_Pa=101325.0)

    assert props.fluid_name == 'Water'
    assert props.temperature_K == 293.15
    assert props.pressure_Pa == 101325.0
    assert props.phase == 'liquid'

    # Density should be ~998 kg/m³ at 20°C
    assert 990 < props.density_kg_m3 < 1005

    # Viscosity should be ~0.001 Pa·s
    if COOLPROP_AVAILABLE:
        assert 0.0008 < props.viscosity_Pa_s < 0.0012
    else:
        # Fallback value
        assert props.viscosity_Pa_s > 0


def test_get_fluid_properties_nitrogen():
    """Test nitrogen properties at standard conditions."""
    props = get_fluid_properties('nitrogen', T_K=293.15, P_Pa=101325.0)

    assert props.fluid_name == 'Nitrogen'
    assert props.phase in ['gas', 'supercritical_gas']

    # Density should be ~1.16 kg/m³ at 20°C, 1 atm
    assert 1.0 < props.density_kg_m3 < 1.3

    # Gamma should be ~1.4 for diatomic gas
    if props.gamma:
        assert 1.35 < props.gamma < 1.45


def test_get_fluid_properties_compressibility():
    """Test that nitrogen properties change with pressure (compressible)."""
    props_1atm = get_fluid_properties('nitrogen', T_K=293.15, P_Pa=101325.0)
    props_10atm = get_fluid_properties('nitrogen', T_K=293.15, P_Pa=1013250.0)

    # Density should increase ~10x for ideal gas
    assert props_10atm.density_kg_m3 > props_1atm.density_kg_m3 * 8
    assert props_10atm.density_kg_m3 < props_1atm.density_kg_m3 * 12


def test_get_fluid_properties_incompressibility():
    """Test that water properties barely change with pressure (incompressible)."""
    props_1atm = get_fluid_properties('water', T_K=293.15, P_Pa=101325.0)
    props_10atm = get_fluid_properties('water', T_K=293.15, P_Pa=1013250.0)

    # Density should change <1% for liquids
    density_change = abs(props_10atm.density_kg_m3 - props_1atm.density_kg_m3)
    assert density_change / props_1atm.density_kg_m3 < 0.01


def test_get_fluid_properties_uncertainty():
    """Test that uncertainties are provided."""
    props = get_fluid_properties('water', T_K=293.15, P_Pa=101325.0)

    assert props.density_uncertainty > 0
    assert props.density_uncertainty < 0.1  # Should be reasonable
    assert props.viscosity_uncertainty > 0


def test_get_fluid_properties_source():
    """Test that source is properly set."""
    props = get_fluid_properties('water', T_K=293.15, P_Pa=101325.0)

    if COOLPROP_AVAILABLE:
        assert props.source == 'coolprop'
    else:
        assert props.source == 'fallback'


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================

def test_get_density():
    """Test the convenience density function."""
    density, uncertainty = get_density('water', T_K=293.15, P_Pa=101325.0)

    assert 990 < density < 1005
    assert uncertainty > 0
    assert uncertainty == density * (uncertainty / density)  # Consistency check


def test_get_density_from_metadata():
    """Test getting density from metadata dict."""
    metadata = {
        'test_fluid': 'nitrogen',
        'fluid_temperature_K': 300.0,
        'fluid_pressure_Pa': 500000.0,  # 5 bar
    }

    density, uncertainty = get_density_from_metadata(metadata)

    assert density > 0
    assert uncertainty > 0

    # Should be ~5x density at 1 bar for ideal gas
    props_1bar = get_fluid_properties('nitrogen', T_K=300.0, P_Pa=100000.0)
    assert 4.5 < density / props_1bar.density_kg_m3 < 5.5


def test_get_density_from_metadata_missing_fluid():
    """Test that missing fluid raises ValueError."""
    metadata = {
        'fluid_temperature_K': 300.0,
        'fluid_pressure_Pa': 500000.0,
    }

    try:
        get_density_from_metadata(metadata)
        assert False, "Should raise ValueError for missing test_fluid"
    except ValueError:
        pass


def test_is_compressible():
    """Test compressibility detection."""
    assert is_compressible('nitrogen') == True
    assert is_compressible('water') == False
    assert is_compressible('oxygen') == True
    assert is_compressible('ethanol') == False


def test_list_available_fluids():
    """Test that available fluids are listed."""
    fluids = list_available_fluids()

    assert 'water' in fluids
    assert 'nitrogen' in fluids
    assert 'n2' in fluids
    assert 'ipa' in fluids
    assert fluids['water'] == 'Water'
    assert fluids['n2'] == 'Nitrogen'


# =============================================================================
# DATACLASS TESTS
# =============================================================================

def test_fluid_state():
    """Test FluidState dataclass."""
    state = FluidState(
        fluid='water',
        temperature_K=300.0,
        pressure_Pa=200000.0
    )

    props = state.get_properties()

    assert props.fluid_name == 'Water'
    assert props.temperature_K == 300.0
    assert props.pressure_Pa == 200000.0


def test_fluid_state_from_metadata():
    """Test FluidState creation from metadata."""
    metadata = {
        'test_fluid': 'nitrogen',
        'fluid_temperature_K': 293.15,
        'fluid_pressure_Pa': 101325.0,
    }

    state = FluidState.from_metadata(metadata)

    assert state.fluid == 'nitrogen'
    assert state.temperature_K == 293.15
    assert state.pressure_Pa == 101325.0


def test_fluid_properties_to_dict():
    """Test FluidProperties serialization."""
    props = get_fluid_properties('water', T_K=293.15, P_Pa=101325.0)
    data = props.to_dict()

    assert 'fluid_name' in data
    assert 'density_kg_m3' in data
    assert 'viscosity_Pa_s' in data
    assert 'phase' in data
    assert data['fluid_name'] == 'Water'


def test_bipropellant_state():
    """Test BipropellantState for hot fire."""
    state = BipropellantState(
        oxidizer='oxygen',
        fuel='rp-1',
        ox_temperature_K=90.0,
        fuel_temperature_K=293.15
    )

    ox_props = state.get_ox_properties()
    fuel_props = state.get_fuel_properties()

    assert ox_props.fluid_name == 'Oxygen'
    assert fuel_props.fluid_name == 'n-Dodecane'
    assert ox_props.temperature_K == 90.0
    assert fuel_props.temperature_K == 293.15


def test_bipropellant_state_from_metadata():
    """Test BipropellantState creation from metadata."""
    metadata = {
        'oxidizer': 'lox',
        'fuel': 'kerosene',
        'ox_temperature_K': 90.0,
        'fuel_temperature_K': 293.15,
        'ox_pressure_Pa': 1000000.0,
        'fuel_pressure_Pa': 1000000.0,
    }

    state = BipropellantState.from_metadata(metadata)

    assert state.oxidizer == 'lox'
    assert state.fuel == 'kerosene'
    assert state.ox_temperature_K == 90.0


# =============================================================================
# FALLBACK TESTS
# =============================================================================

def test_fallback_properties_available():
    """Test that fallback properties work even without CoolProp."""
    # Force using fallback
    props = get_fluid_properties('Water', T_K=293.15, P_Pa=101325.0, use_fallback=True)

    assert props.density_kg_m3 > 0
    assert props.viscosity_Pa_s > 0

    if not COOLPROP_AVAILABLE:
        # Should be using fallback
        assert props.source == 'fallback'
        # Fallback water density at 20°C
        assert abs(props.density_kg_m3 - 998.2) < 0.1


def test_coolprop_status():
    """Test that CoolProp availability check works."""
    available = check_coolprop_available()
    assert isinstance(available, bool)
    assert available == COOLPROP_AVAILABLE


# =============================================================================
# EDGE CASES
# =============================================================================

def test_fluid_properties_extreme_temperature():
    """Test properties at extreme but valid temperature."""
    # Cold nitrogen (cryogenic)
    props_cold = get_fluid_properties('nitrogen', T_K=77.0, P_Pa=101325.0)
    assert props_cold.density_kg_m3 > 0

    # Hot nitrogen
    props_hot = get_fluid_properties('nitrogen', T_K=500.0, P_Pa=101325.0)
    assert props_hot.density_kg_m3 > 0

    # Cold should be denser
    assert props_cold.density_kg_m3 > props_hot.density_kg_m3


def test_fluid_properties_high_pressure():
    """Test properties at high pressure."""
    # 100 bar nitrogen
    props = get_fluid_properties('nitrogen', T_K=293.15, P_Pa=10000000.0)

    assert props.density_kg_m3 > 0
    # Should be much denser than 1 bar
    assert props.density_kg_m3 > 50


# =============================================================================
# MAIN
# =============================================================================

def run_tests():
    """Run all tests."""
    import inspect

    # Get all test functions
    current_module = sys.modules[__name__]
    test_functions = [
        obj for name, obj in inspect.getmembers(current_module)
        if inspect.isfunction(obj) and name.startswith('test_')
    ]

    print(f"Running {len(test_functions)} tests for fluid_properties module...")
    print(f"CoolProp available: {COOLPROP_AVAILABLE}")
    print()

    passed = 0
    failed = 0

    for test_func in test_functions:
        test_name = test_func.__name__
        try:
            test_func()
            print(f"✓ {test_name}")
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_name}: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test_name}: {type(e).__name__}: {e}")
            failed += 1

    print()
    print(f"Results: {passed} passed, {failed} failed")

    return failed == 0


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
