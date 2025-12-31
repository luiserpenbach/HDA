"""
Configuration Validation Module
===============================
Schema validation for test configurations using dataclasses.

Key Principle: Fail fast on bad configs. A typo in a config should
raise an immediate, clear error - not silently produce wrong results.

Features:
- Strict type validation
- Required vs optional fields
- Value range constraints
- Cross-field validation
- Helpful error messages

Note: Uses dataclasses for compatibility. Install pydantic for enhanced validation.
"""

from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
from pathlib import Path

# Try to import pydantic, fall back to dataclass-based validation
try:
    from pydantic import BaseModel, Field, validator, root_validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    # Create stub classes for compatibility
    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
        
        def dict(self, exclude_none=False):
            result = {}
            for key, value in self.__dict__.items():
                if exclude_none and value is None:
                    continue
                if hasattr(value, 'dict'):
                    result[key] = value.dict(exclude_none=exclude_none)
                elif isinstance(value, dict):
                    result[key] = value
                else:
                    result[key] = value
            return result
        
        class Config:
            extra = 'allow'
    
    def Field(default=None, **kwargs):
        return default
    
    def validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def root_validator(func=None, **kwargs):
        if func is None:
            def decorator(f):
                return f
            return decorator
        return func


# =============================================================================
# DATACLASS-BASED VALIDATION (Works without pydantic)
# =============================================================================

@dataclass
class FluidConfigDC:
    """Fluid properties configuration (dataclass version)."""
    name: Optional[str] = None
    density_kg_m3: Optional[float] = None
    density_uncertainty_kg_m3: Optional[float] = None
    temperature_K: Optional[float] = None
    ox_density_kg_m3: Optional[float] = None
    fuel_density_kg_m3: Optional[float] = None
    water_density_kg_m3: Optional[float] = None
    
    def get_density(self) -> Optional[float]:
        return (
            self.density_kg_m3 or 
            self.ox_density_kg_m3 or 
            self.fuel_density_kg_m3 or
            self.water_density_kg_m3
        )
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FluidConfigDC':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class GeometryConfigDC:
    """Geometry configuration (dataclass version)."""
    orifice_area_mm2: Optional[float] = None
    orifice_area_uncertainty_mm2: Optional[float] = None
    orifice_diameter_mm: Optional[float] = None
    throat_area_mm2: Optional[float] = None
    throat_area_uncertainty_mm2: Optional[float] = None
    throat_diameter_mm: Optional[float] = None
    expansion_ratio: Optional[float] = None
    exit_area_mm2: Optional[float] = None
    ox_injector_area_mm2: Optional[float] = None
    fuel_injector_area_mm2: Optional[float] = None
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'GeometryConfigDC':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ColumnsConfigDC:
    """Column mapping (dataclass version)."""
    timestamp: str = "timestamp"
    # Cold flow columns
    mass_flow: Optional[str] = None
    mf: Optional[str] = None
    upstream_pressure: Optional[str] = None
    inlet_pressure: Optional[str] = None
    downstream_pressure: Optional[str] = None
    upstream_temperature: Optional[str] = None
    # Hot fire columns
    chamber_pressure: Optional[str] = None
    thrust: Optional[str] = None
    mass_flow_ox: Optional[str] = None
    mass_flow_fuel: Optional[str] = None
    inlet_pressure_ox: Optional[str] = None
    inlet_pressure_fuel: Optional[str] = None
    
    def get_pressure_column(self) -> Optional[str]:
        return self.upstream_pressure or self.inlet_pressure or self.chamber_pressure
    
    def get_flow_column(self) -> Optional[str]:
        return self.mass_flow or self.mf
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ColumnsConfigDC':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class SettingsConfigDC:
    """Processing settings (dataclass version)."""
    resample_freq_ms: float = 10
    steady_window_ms: float = 500
    cv_threshold: float = 1.5
    target_c_star: Optional[float] = None
    target_cf: Optional[float] = None
    target_isp: Optional[float] = None
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SettingsConfigDC':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class TestConfigDC:
    """Complete test configuration (dataclass version)."""
    config_name: str
    fluid: FluidConfigDC
    geometry: GeometryConfigDC
    columns: ColumnsConfigDC
    description: Optional[str] = None
    channel_config: Optional[Dict[str, str]] = None
    uncertainties: Optional[Dict[str, Any]] = None
    sensor_limits: Optional[Dict[str, Any]] = None
    settings: Optional[SettingsConfigDC] = None
    reference_values: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            'config_name': self.config_name,
            'description': self.description,
            'fluid': asdict(self.fluid),
            'geometry': asdict(self.geometry),
            'columns': asdict(self.columns),
            'channel_config': self.channel_config,
            'uncertainties': self.uncertainties,
            'sensor_limits': self.sensor_limits,
            'reference_values': self.reference_values,
        }
        if self.settings:
            result['settings'] = asdict(self.settings)
        return {k: v for k, v in result.items() if v is not None}
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TestConfigDC':
        fluid = FluidConfigDC.from_dict(data.get('fluid', {}))
        geometry = GeometryConfigDC.from_dict(data.get('geometry', {}))
        columns = ColumnsConfigDC.from_dict(data.get('columns', {}))
        settings = SettingsConfigDC.from_dict(data.get('settings', {})) if data.get('settings') else None
        
        return cls(
            config_name=data.get('config_name', 'unnamed'),
            description=data.get('description'),
            fluid=fluid,
            geometry=geometry,
            columns=columns,
            channel_config=data.get('channel_config'),
            uncertainties=data.get('uncertainties'),
            sensor_limits=data.get('sensor_limits'),
            settings=settings,
            reference_values=data.get('reference_values'),
        )


def validate_config_simple(config: Dict[str, Any], config_type: str = 'auto') -> TestConfigDC:
    """
    Simple validation using dataclasses (no pydantic required).
    
    Args:
        config: Configuration dictionary
        config_type: 'cold_flow', 'hot_fire', or 'auto'
        
    Returns:
        Validated TestConfigDC object
        
    Raises:
        ValueError: If validation fails
    """
    errors = []
    
    # Check required top-level fields
    if not config.get('config_name'):
        errors.append("config_name is required")
    
    # Check fluid
    fluid = config.get('fluid', {})
    density = (
        fluid.get('density_kg_m3') or 
        fluid.get('ox_density_kg_m3') or 
        fluid.get('fuel_density_kg_m3') or
        fluid.get('water_density_kg_m3')
    )
    if not density:
        errors.append("Fluid density is required (set density_kg_m3)")
    elif density <= 0:
        errors.append(f"Fluid density must be positive, got {density}")
    
    # Check geometry based on type
    geometry = config.get('geometry', {})
    columns = config.get('columns', {})
    
    # Auto-detect type
    if config_type == 'auto':
        if 'chamber_pressure' in columns:
            config_type = 'hot_fire'
        else:
            config_type = 'cold_flow'
    
    if config_type == 'cold_flow':
        if not geometry.get('orifice_area_mm2'):
            errors.append("orifice_area_mm2 is required in geometry for cold flow")
        
        has_pressure = columns.get('upstream_pressure') or columns.get('inlet_pressure')
        has_flow = columns.get('mass_flow') or columns.get('mf')
        
        if not has_pressure:
            errors.append("upstream_pressure or inlet_pressure required in columns")
        if not has_flow:
            errors.append("mass_flow or mf required in columns")
    
    elif config_type == 'hot_fire':
        if not geometry.get('throat_area_mm2'):
            errors.append("throat_area_mm2 is required in geometry for hot fire")
        
        if not columns.get('chamber_pressure'):
            errors.append("chamber_pressure required in columns for hot fire")
    
    if errors:
        raise ValueError("Configuration validation failed:\n  - " + "\n  - ".join(errors))
    
    return TestConfigDC.from_dict(config)


class UncertaintySpec(BaseModel):
    """Sensor uncertainty specification."""
    type: str = Field(..., description="Uncertainty type: 'abs', 'rel', 'pct_fs'")
    value: float = Field(..., ge=0, description="Uncertainty value")
    full_scale: Optional[float] = Field(None, ge=0, description="Full scale for pct_fs type")
    unit: Optional[str] = Field(None, description="Physical unit")
    
    @validator('type')
    def validate_type(cls, v):
        allowed = ['abs', 'rel', 'pct_fs', 'pct_rd', 'percent_fs', 'percent_reading']
        if v not in allowed:
            raise ValueError(f"Uncertainty type must be one of {allowed}, got '{v}'")
        return v
    
    @root_validator(skip_on_failure=True)
    def check_full_scale_required(cls, values):
        u_type = values.get('type')
        full_scale = values.get('full_scale')

        if u_type in ('pct_fs', 'percent_fs') and full_scale is None:
            raise ValueError(f"full_scale is required for uncertainty type '{u_type}'")

        return values


class SensorLimit(BaseModel):
    """Physical limits for a sensor."""
    min: Optional[float] = Field(None, description="Minimum valid value")
    max: Optional[float] = Field(None, description="Maximum valid value")
    unit: Optional[str] = Field(None, description="Physical unit")
    allow_negative: bool = Field(True, description="Whether negative values are valid")
    
    @root_validator(skip_on_failure=True)
    def check_min_max(cls, values):
        min_val = values.get('min')
        max_val = values.get('max')
        
        if min_val is not None and max_val is not None and min_val >= max_val:
            raise ValueError(f"min ({min_val}) must be less than max ({max_val})")
        
        return values


class FluidConfig(BaseModel):
    """Fluid properties configuration."""
    name: Optional[str] = Field(None, description="Fluid name (e.g., 'Nitrogen', 'Water')")
    density_kg_m3: Optional[float] = Field(None, gt=0, description="Fluid density in kg/m³")
    density_uncertainty_kg_m3: Optional[float] = Field(None, ge=0, description="Density uncertainty")
    temperature_K: Optional[float] = Field(None, gt=0, description="Fluid temperature in Kelvin")
    
    # Alternative field names (for backward compatibility)
    ox_density_kg_m3: Optional[float] = Field(None, gt=0)
    fuel_density_kg_m3: Optional[float] = Field(None, gt=0)
    water_density_kg_m3: Optional[float] = Field(None, gt=0)
    
    class Config:
        extra = 'allow'  # Allow extra fields for backward compatibility
    
    def get_density(self) -> Optional[float]:
        """Get the applicable density value."""
        return (
            self.density_kg_m3 or 
            self.ox_density_kg_m3 or 
            self.fuel_density_kg_m3 or
            self.water_density_kg_m3
        )


class GeometryConfig(BaseModel):
    """Geometry configuration for flow calculations."""
    # Cold flow geometry
    orifice_area_mm2: Optional[float] = Field(None, gt=0, description="Orifice area in mm²")
    orifice_area_uncertainty_mm2: Optional[float] = Field(None, ge=0)
    orifice_diameter_mm: Optional[float] = Field(None, gt=0, description="Orifice diameter in mm")
    
    # Hot fire geometry
    throat_area_mm2: Optional[float] = Field(None, gt=0, description="Nozzle throat area in mm²")
    throat_area_uncertainty_mm2: Optional[float] = Field(None, ge=0)
    throat_diameter_mm: Optional[float] = Field(None, gt=0)
    expansion_ratio: Optional[float] = Field(None, ge=1.0, description="Nozzle expansion ratio")
    exit_area_mm2: Optional[float] = Field(None, gt=0)
    
    # Injector geometry
    ox_injector_area_mm2: Optional[float] = Field(None, gt=0)
    fuel_injector_area_mm2: Optional[float] = Field(None, gt=0)
    
    class Config:
        extra = 'allow'


class ColdFlowColumns(BaseModel):
    """Column mapping for cold flow tests."""
    timestamp: str = Field("timestamp", min_length=1, description="Timestamp column name")
    mass_flow: Optional[str] = Field(None, description="Mass flow column")
    mf: Optional[str] = Field(None, description="Mass flow column (alternate)")
    upstream_pressure: Optional[str] = Field(None, description="Upstream pressure column")
    inlet_pressure: Optional[str] = Field(None, description="Inlet pressure (alternate)")
    downstream_pressure: Optional[str] = Field(None, description="Downstream pressure")
    upstream_temperature: Optional[str] = Field(None, description="Upstream temperature")
    
    class Config:
        extra = 'allow'
    
    @root_validator(skip_on_failure=True)
    def check_required_columns(cls, values):
        # Must have at least one pressure column
        has_pressure = any([
            values.get('upstream_pressure'),
            values.get('inlet_pressure')
        ])
        
        # Must have at least one flow column
        has_flow = any([
            values.get('mass_flow'),
            values.get('mf')
        ])
        
        if not has_pressure:
            raise ValueError("At least one pressure column required (upstream_pressure or inlet_pressure)")
        
        if not has_flow:
            raise ValueError("At least one flow column required (mass_flow or mf)")
        
        return values
    
    def get_pressure_column(self) -> str:
        """Get the applicable pressure column name."""
        return self.upstream_pressure or self.inlet_pressure
    
    def get_flow_column(self) -> str:
        """Get the applicable flow column name."""
        return self.mass_flow or self.mf


class HotFireColumns(BaseModel):
    """Column mapping for hot fire tests."""
    timestamp: str = Field("timestamp", min_length=1, description="Timestamp column name")
    chamber_pressure: str = Field(..., min_length=1, description="Chamber pressure column")
    thrust: Optional[str] = Field(None, description="Thrust column")
    mass_flow_ox: Optional[str] = Field(None, description="Oxidizer mass flow")
    mass_flow_fuel: Optional[str] = Field(None, description="Fuel mass flow")
    inlet_pressure_ox: Optional[str] = Field(None, description="Oxidizer inlet pressure")
    inlet_pressure_fuel: Optional[str] = Field(None, description="Fuel inlet pressure")
    
    class Config:
        extra = 'allow'


class SettingsConfig(BaseModel):
    """Processing settings configuration."""
    resample_freq_ms: float = Field(10, gt=0, le=1000, description="Resampling frequency in ms")
    steady_window_ms: float = Field(500, gt=0, description="Steady state window size in ms")
    cv_threshold: float = Field(1.5, gt=0, le=100, description="CV threshold for steady state (%)")
    
    # Theoretical targets (for efficiency calculations)
    target_c_star: Optional[float] = Field(None, gt=0, description="Target C* in m/s")
    target_cf: Optional[float] = Field(None, gt=0, description="Target thrust coefficient")
    target_isp: Optional[float] = Field(None, gt=0, description="Target Isp in seconds")
    
    class Config:
        extra = 'allow'


class ReferenceValues(BaseModel):
    """Reference/nominal values for comparison."""
    target_c_star: Optional[float] = Field(None, gt=0)
    target_isp: Optional[float] = Field(None, gt=0)
    nominal_of: Optional[float] = Field(None, gt=0, description="Nominal O/F ratio")
    nominal_mf: Optional[float] = Field(None, gt=0, description="Nominal mass flow")
    
    class Config:
        extra = 'allow'


class ColdFlowConfig(BaseModel):
    """Complete configuration for cold flow testing."""
    config_name: str = Field(..., min_length=1, description="Configuration name")
    description: Optional[str] = Field(None, description="Configuration description")
    
    fluid: FluidConfig
    geometry: GeometryConfig
    columns: ColdFlowColumns
    
    channel_config: Optional[Dict[str, str]] = Field(
        None, 
        description="Raw channel ID to sensor name mapping"
    )
    uncertainties: Optional[Dict[str, Any]] = Field(
        None,
        description="Sensor uncertainty specifications"
    )
    sensor_limits: Optional[Dict[str, Any]] = Field(
        None,
        description="Physical limits for each sensor"
    )
    settings: Optional[SettingsConfig] = Field(
        default_factory=SettingsConfig,
        description="Processing settings"
    )
    reference_values: Optional[ReferenceValues] = None
    
    class Config:
        extra = 'allow'
    
    @root_validator(skip_on_failure=True)
    def check_geometry_for_cd(cls, values):
        """Verify we have geometry needed for Cd calculation."""
        geom = values.get('geometry')
        if geom and geom.orifice_area_mm2 is None:
            raise ValueError("orifice_area_mm2 is required in geometry for Cd calculation")
        return values
    
    @root_validator(skip_on_failure=True)
    def check_fluid_for_cd(cls, values):
        """Verify we have fluid properties needed for Cd calculation."""
        fluid = values.get('fluid')
        if fluid and fluid.get_density() is None:
            raise ValueError("Fluid density is required for Cd calculation (set density_kg_m3)")
        return values
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.dict(exclude_none=True)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ColdFlowConfig':
        """Create from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> 'ColdFlowConfig':
        """Load from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


class HotFireConfig(BaseModel):
    """Complete configuration for hot fire testing."""
    config_name: str = Field(..., min_length=1, description="Configuration name")
    description: Optional[str] = Field(None, description="Configuration description")
    
    geometry: GeometryConfig
    columns: HotFireColumns
    
    fluid: Optional[FluidConfig] = None
    
    channel_config: Optional[Dict[str, str]] = Field(
        None,
        description="Raw channel ID to sensor name mapping"
    )
    uncertainties: Optional[Dict[str, Any]] = Field(
        None,
        description="Sensor uncertainty specifications"
    )
    sensor_limits: Optional[Dict[str, Any]] = Field(
        None,
        description="Physical limits for each sensor"
    )
    settings: Optional[SettingsConfig] = Field(
        default_factory=SettingsConfig,
        description="Processing settings"
    )
    reference_values: Optional[ReferenceValues] = None
    
    class Config:
        extra = 'allow'
    
    @root_validator(skip_on_failure=True)
    def check_geometry_for_performance(cls, values):
        """Verify we have geometry needed for C* calculation."""
        geom = values.get('geometry')
        if geom and geom.throat_area_mm2 is None:
            raise ValueError("throat_area_mm2 is required in geometry for C* calculation")
        return values
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.dict(exclude_none=True)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HotFireConfig':
        """Create from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> 'HotFireConfig':
        """Load from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_config(
    config: Dict[str, Any],
    config_type: str = 'auto'
) -> Union[ColdFlowConfig, HotFireConfig]:
    """
    Validate a configuration dictionary.
    
    Args:
        config: Configuration dictionary to validate
        config_type: 'cold_flow', 'hot_fire', or 'auto' (detect from content)
        
    Returns:
        Validated configuration object
        
    Raises:
        ValueError: If validation fails with detailed error message
    """
    # Use simple validation if pydantic is not available
    if not PYDANTIC_AVAILABLE:
        return validate_config_simple(config, config_type)
    
    # Auto-detect type if needed
    if config_type == 'auto':
        columns = config.get('columns', {})
        if 'chamber_pressure' in columns:
            config_type = 'hot_fire'
        elif 'mass_flow' in columns or 'mf' in columns or 'upstream_pressure' in columns:
            config_type = 'cold_flow'
        else:
            raise ValueError(
                "Cannot auto-detect config type. "
                "Specify config_type='cold_flow' or 'hot_fire'"
            )
    
    try:
        if config_type == 'cold_flow':
            return ColdFlowConfig(**config)
        elif config_type == 'hot_fire':
            return HotFireConfig(**config)
        else:
            raise ValueError(f"Unknown config_type: {config_type}")
            
    except Exception as e:
        raise ValueError(f"Configuration validation failed:\n{str(e)}")


def validate_config_file(
    path: Union[str, Path],
    config_type: str = 'auto'
) -> Union[ColdFlowConfig, HotFireConfig]:
    """
    Load and validate a configuration file.
    
    Args:
        path: Path to JSON configuration file
        config_type: 'cold_flow', 'hot_fire', or 'auto'
        
    Returns:
        Validated configuration object
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If validation fails
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    try:
        with open(path, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {str(e)}")
    
    return validate_config(config, config_type)


def check_columns_exist(
    df,
    config: Union[ColdFlowConfig, HotFireConfig, Dict[str, Any]]
) -> List[str]:
    """
    Check if required columns from config exist in DataFrame.
    
    Args:
        df: DataFrame to check
        config: Configuration object or dictionary
        
    Returns:
        List of missing column names (empty if all present)
    """
    if isinstance(config, (ColdFlowConfig, HotFireConfig)):
        columns = config.columns
        col_dict = columns.dict(exclude_none=True)
    else:
        col_dict = config.get('columns', {})
    
    missing = []
    for key, col_name in col_dict.items():
        if key == 'timestamp':
            continue
        if col_name and col_name not in df.columns:
            missing.append(f"{key}: '{col_name}'")
    
    return missing


def get_config_hash(config: Union[ColdFlowConfig, HotFireConfig, Dict[str, Any]]) -> str:
    """
    Get hash of configuration for traceability.
    
    Args:
        config: Configuration object or dictionary
        
    Returns:
        SHA-256 hash string
    """
    import hashlib
    
    if isinstance(config, (ColdFlowConfig, HotFireConfig)):
        config_dict = config.to_dict()
    else:
        config_dict = config
    
    config_str = json.dumps(config_dict, sort_keys=True, default=str)
    sha256_hash = hashlib.sha256(config_str.encode('utf-8'))
    return f"sha256:{sha256_hash.hexdigest()}"


def load_and_validate_config(
    config_name: str,
    config_dir: str = "configs"
) -> Dict[str, Any]:
    """
    Load config by name with validation, returning dict for backward compatibility.

    Args:
        config_name: Name of config (without .json extension)
        config_dir: Directory containing config files

    Returns:
        Validated configuration as dictionary
    """
    path = Path(config_dir) / f"{config_name}.json"

    validated = validate_config_file(path)
    return validated.to_dict()


# =============================================================================
# V2.3.0: ACTIVE CONFIGURATION AND TEST METADATA SEPARATION
# =============================================================================

class ChannelMapping(BaseModel):
    """
    Maps DAQ channel IDs to P&ID sensor names.

    Example:
        {
            "10001": "FU-PT-01",  # Channel 10001 → Pressure Transducer 01
            "10002": "FU-PT-02",  # Channel 10002 → Pressure Transducer 02
            "10003": "FU-TT-01"   # Channel 10003 → Temperature Transducer 01
        }
    """
    class Config:
        extra = 'allow'


class SensorMapping(BaseModel):
    """
    Maps sensor roles to column names (from CSV or P&ID names).

    This is the analysis-level mapping that tells the system which columns
    to use for each type of measurement.
    """
    timestamp: str = Field("timestamp", min_length=1, description="Timestamp column")
    pressure_upstream: Optional[str] = Field(None, description="Upstream pressure sensor")
    pressure_downstream: Optional[str] = Field(None, description="Downstream pressure sensor")
    temperature: Optional[str] = Field(None, description="Temperature sensor")
    mass_flow: Optional[str] = Field(None, description="Mass flow sensor")
    chamber_pressure: Optional[str] = Field(None, description="Chamber pressure (hot fire)")
    thrust: Optional[str] = Field(None, description="Thrust sensor (hot fire)")

    class Config:
        extra = 'allow'


class SensorUncertainty(BaseModel):
    """Individual sensor uncertainty specification."""
    value: float = Field(..., ge=0, description="Uncertainty magnitude")
    unit: str = Field(..., description="Physical unit (psi, K, g/s, etc.)")
    type: str = Field("absolute", description="Type: absolute, relative, percent_fs")
    full_scale: Optional[float] = Field(None, ge=0, description="Full scale for percent_fs type")

    @validator('type')
    def validate_type(cls, v):
        allowed = ['absolute', 'relative', 'percent_fs', 'percent_reading']
        if v not in allowed:
            raise ValueError(f"Uncertainty type must be one of {allowed}, got '{v}'")
        return v


class ProcessingSettings(BaseModel):
    """Processing and analysis settings."""
    steady_state_method: Optional[str] = Field("cv", description="Detection method: cv, ml, derivative, manual")
    cv_threshold: Optional[float] = Field(0.02, gt=0, description="CV threshold for steady state")
    window_size: Optional[int] = Field(50, gt=0, description="Window size for detection")
    resample_freq_ms: Optional[float] = Field(10, gt=0, description="Resampling frequency in ms")

    class Config:
        extra = 'allow'


class ActiveConfiguration(BaseModel):
    """
    Active Configuration (Testbench Hardware)
    ==========================================
    Describes the test hardware setup: sensors, uncertainties, channel mappings.

    This configuration changes only when testbench is modified or recalibrated.
    It does NOT include test article properties (geometry, fluid) - those go in metadata.

    Version: 2.3.0
    """
    config_name: str = Field(..., min_length=1, description="Configuration name")
    config_version: str = Field("2.3.0", description="Config schema version")
    test_type: str = Field(..., description="Test type: cold_flow or hot_fire")
    description: Optional[str] = Field(None, description="Configuration description")

    channel_mapping: Optional[Dict[str, str]] = Field(
        None,
        description="Maps DAQ channel IDs (e.g. '10001') to P&ID sensor names (e.g. 'FU-PT-01')"
    )

    sensor_mapping: Dict[str, str] = Field(
        ...,
        description="Maps sensor roles to column names for analysis"
    )

    sensor_uncertainties: Dict[str, Dict[str, Any]] = Field(
        ...,
        description="Uncertainty specifications for each sensor"
    )

    processing: Optional[ProcessingSettings] = Field(
        default_factory=ProcessingSettings,
        description="Processing and analysis settings"
    )

    sensor_limits: Optional[Dict[str, Any]] = Field(
        None,
        description="Physical limits for each sensor (optional)"
    )

    class Config:
        extra = 'allow'

    @validator('test_type')
    def validate_test_type(cls, v):
        if v not in ['cold_flow', 'hot_fire']:
            raise ValueError(f"test_type must be 'cold_flow' or 'hot_fire', got '{v}'")
        return v

    @root_validator(skip_on_failure=True)
    def check_no_geometry_or_fluid(cls, values):
        """Ensure geometry and fluid are NOT in active configuration."""
        if 'geometry' in values:
            raise ValueError(
                "geometry belongs in Test Metadata, not Active Configuration. "
                "Use metadata.json or UI entry for test article properties."
            )
        if 'fluid' in values:
            raise ValueError(
                "fluid belongs in Test Metadata, not Active Configuration. "
                "Use metadata.json or UI entry for test article properties."
            )
        return values

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.dict(exclude_none=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ActiveConfiguration':
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> 'ActiveConfiguration':
        """Load from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


class TestMetadata(BaseModel):
    """
    Test Metadata (Per Test Article)
    =================================
    Describes the test article properties and conditions.

    This metadata changes for every test - different injector element, fluid, etc.
    It does NOT include testbench hardware info - that goes in Active Configuration.

    Source: metadata.json in test folder, or UI entry.
    Required for campaign save, optional for analysis/reporting.

    Version: 2.3.0
    """
    part_number: Optional[str] = Field(None, description="Part number of test article")
    serial_number: Optional[str] = Field(None, description="Serial number of test article")
    test_datetime: Optional[str] = Field(None, description="Test date and time (ISO format)")
    analyst: Optional[str] = Field(None, description="Engineer performing the test")
    test_type: Optional[str] = Field(None, description="Test type: cold_flow or hot_fire")
    test_id: Optional[str] = Field(None, description="Unique test identifier")

    geometry: Optional[Dict[str, Any]] = Field(
        None,
        description="Test article geometry (orifice_area_mm2, throat_area_mm2, etc.)"
    )

    fluid: Optional[Dict[str, Any]] = Field(
        None,
        description="Working fluid properties (name, gamma, density, etc.)"
    )

    test_conditions: Optional[Dict[str, Any]] = Field(
        None,
        description="Test conditions (ambient_pressure, target_pressure, notes, etc.)"
    )

    class Config:
        extra = 'allow'  # Allow additional metadata fields

    @validator('test_type')
    def validate_test_type(cls, v):
        if v and v not in ['cold_flow', 'hot_fire']:
            raise ValueError(f"test_type must be 'cold_flow' or 'hot_fire', got '{v}'")
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.dict(exclude_none=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestMetadata':
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> 'TestMetadata':
        """Load from JSON file (typically metadata.json in test folder)."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_folder(cls, folder_path: Union[str, Path]) -> Optional['TestMetadata']:
        """
        Load metadata.json from test folder if it exists.

        Args:
            folder_path: Path to test data folder

        Returns:
            TestMetadata object if metadata.json exists, None otherwise
        """
        folder = Path(folder_path)
        metadata_file = folder / "metadata.json"

        if metadata_file.exists():
            return cls.from_file(metadata_file)
        return None

    def is_complete_for_campaign(self) -> bool:
        """Check if metadata has minimum required fields for campaign save."""
        required = [
            self.part_number or self.test_id,  # Need at least one identifier
            self.geometry,
            self.fluid
        ]
        return all(required)

    def get_missing_for_campaign(self) -> List[str]:
        """Get list of fields missing for campaign save."""
        missing = []

        if not (self.part_number or self.test_id):
            missing.append("part_number or test_id")
        if not self.geometry:
            missing.append("geometry")
        if not self.fluid:
            missing.append("fluid")

        return missing


def detect_config_format(config: Dict[str, Any]) -> str:
    """
    Detect if config is old format (v2.0-v2.2) or new format (v2.3+).

    Args:
        config: Configuration dictionary

    Returns:
        'old' (has geometry/fluid), 'new' (no geometry/fluid), or 'active_config'
    """
    has_geometry = 'geometry' in config
    has_fluid = 'fluid' in config
    has_version = 'config_version' in config

    # New format explicitly declares version and has no geometry/fluid
    if has_version and not has_geometry and not has_fluid:
        return 'active_config'

    # Old format has geometry and/or fluid
    if has_geometry or has_fluid:
        return 'old'

    # Ambiguous - treat as new format
    return 'active_config'


def split_old_config(old_config: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Split old-format config into Active Configuration + Test Metadata.

    Args:
        old_config: Old format config (v2.0-v2.2) with geometry/fluid mixed in

    Returns:
        Tuple of (active_config_dict, metadata_dict)
    """
    # Active Configuration: everything EXCEPT geometry, fluid, reference_values
    active_config = {
        'config_name': old_config.get('config_name', 'Migrated Config'),
        'config_version': '2.3.0',
        'test_type': old_config.get('test_type', 'cold_flow'),  # May need auto-detection
        'description': old_config.get('description'),
        'channel_mapping': old_config.get('channel_config'),  # Rename channel_config → channel_mapping
        'sensor_mapping': old_config.get('columns', {}),  # Old 'columns' becomes 'sensor_mapping'
        'sensor_uncertainties': old_config.get('sensor_uncertainties', {}),
        'processing': old_config.get('settings'),
        'sensor_limits': old_config.get('sensor_limits'),
    }

    # Remove None values
    active_config = {k: v for k, v in active_config.items() if v is not None}

    # Test Metadata: geometry, fluid, and any metadata fields
    metadata = {
        'test_type': old_config.get('test_type', 'cold_flow'),
        'geometry': old_config.get('geometry'),
        'fluid': old_config.get('fluid'),
        'test_conditions': {},
    }

    # Add reference values to test_conditions if present
    if 'reference_values' in old_config:
        metadata['test_conditions']['reference_values'] = old_config['reference_values']

    # Remove None values
    metadata = {k: v for k, v in metadata.items() if v is not None}

    return active_config, metadata


def validate_active_configuration(
    config: Dict[str, Any],
    auto_migrate: bool = True
) -> ActiveConfiguration:
    """
    Validate Active Configuration with automatic migration from old format.

    Args:
        config: Configuration dictionary
        auto_migrate: If True, automatically migrate old-format configs

    Returns:
        Validated ActiveConfiguration object

    Raises:
        ValueError: If validation fails
    """
    format_type = detect_config_format(config)

    if format_type == 'old':
        if not auto_migrate:
            raise ValueError(
                "Old config format detected (has geometry/fluid). "
                "Set auto_migrate=True to automatically split into config + metadata."
            )

        # Auto-migrate
        active_config, _ = split_old_config(config)
        config = active_config

    try:
        return ActiveConfiguration(**config)
    except Exception as e:
        raise ValueError(f"Active Configuration validation failed:\n{str(e)}")


def validate_test_metadata(
    metadata: Dict[str, Any],
    require_complete: bool = False
) -> TestMetadata:
    """
    Validate Test Metadata.

    Args:
        metadata: Metadata dictionary
        require_complete: If True, require all fields needed for campaign save

    Returns:
        Validated TestMetadata object

    Raises:
        ValueError: If validation fails or required fields missing
    """
    try:
        validated = TestMetadata(**metadata)

        if require_complete and not validated.is_complete_for_campaign():
            missing = validated.get_missing_for_campaign()
            raise ValueError(
                f"Metadata incomplete for campaign save. Missing: {', '.join(missing)}"
            )

        return validated

    except Exception as e:
        raise ValueError(f"Test Metadata validation failed:\n{str(e)}")


def merge_config_and_metadata(
    config: Union[ActiveConfiguration, Dict[str, Any]],
    metadata: Union[TestMetadata, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Merge Active Configuration and Test Metadata for analysis.

    This creates a combined dictionary compatible with existing analysis functions.

    Args:
        config: Active Configuration object or dict
        metadata: Test Metadata object or dict

    Returns:
        Merged configuration dictionary
    """
    if isinstance(config, ActiveConfiguration):
        config_dict = config.to_dict()
    else:
        config_dict = config

    if isinstance(metadata, TestMetadata):
        metadata_dict = metadata.to_dict()
    else:
        metadata_dict = metadata

    # Start with config
    merged = config_dict.copy()

    # Add geometry and fluid from metadata
    if 'geometry' in metadata_dict:
        merged['geometry'] = metadata_dict['geometry']
    if 'fluid' in metadata_dict:
        merged['fluid'] = metadata_dict['fluid']

    # Add test conditions if present
    if 'test_conditions' in metadata_dict:
        merged.setdefault('test_conditions', {}).update(metadata_dict['test_conditions'])

    return merged
