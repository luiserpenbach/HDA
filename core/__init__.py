"""
Hopper Data Studio - Core Module
=================================
Engineering integrity foundations for propulsion test data analysis.

P0 Components (Non-negotiable):
- traceability: Data hashing, audit trails, reproducibility
- uncertainty: Error propagation for all metrics
- qc_checks: Pre-analysis data quality validation
- config_validation: Pydantic-based config schemas
- campaign_manager_v2: Enhanced database with full traceability
- integrated_analysis: High-level analysis API

P1 Components (High Priority):
- spc: Statistical Process Control, control charts, capability indices
- reporting: HTML reports with full traceability
- batch_analysis: Multi-file processing
- export: Enhanced data export with metadata

Shared Utilities:
- config_manager: Unified configuration management (eliminates duplication)
- steady_state_detection: Consolidated steady-state detection methods
- metadata_manager: Test metadata loading and validation (v2.3.0+)

Usage:
    from core.traceability import create_full_traceability_record
    from core.uncertainty import calculate_cold_flow_uncertainties
    from core.qc_checks import run_qc_checks, assert_qc_passed
    from core.config_validation import validate_config, ActiveConfiguration, TestMetadata
    from core.metadata_manager import load_metadata_from_folder
    from core.campaign_manager_v2 import save_cold_flow_result
    from core.spc import create_imr_chart, analyze_campaign_spc
    from core.reporting import generate_test_report
    from core.batch_analysis import run_batch_analysis
    from core.export import export_campaign_excel
    from core.config_manager import ConfigManager
    from core.steady_state_detection import detect_steady_state_auto
"""

from .traceability import (
    compute_file_hash,
    compute_dataframe_hash,
    compute_config_hash,
    create_config_snapshot,
    DataTraceability,
    ProcessingRecord,
    AnalysisContext,
    create_full_traceability_record,
    verify_data_integrity,
    PROCESSING_VERSION,
)

from .uncertainty import (
    SensorUncertainty,
    GeometryUncertainty,
    MeasurementWithUncertainty,
    UncertaintyType,
    parse_uncertainty_config,
    parse_geometry_uncertainties,
    calculate_cd_uncertainty,
    calculate_cold_flow_uncertainties,
    calculate_isp_uncertainty,
    calculate_c_star_uncertainty,
    calculate_of_ratio_uncertainty,
    calculate_hot_fire_uncertainties,
    calculate_statistical_uncertainty,
    combine_uncertainties,
    format_with_uncertainty,
)

from .qc_checks import (
    QCStatus,
    QCCheckResult,
    QCReport,
    run_qc_checks,
    run_quick_qc,
    assert_qc_passed,
    format_qc_for_display,
    check_timestamp_monotonic,
    check_timestamp_gaps,
    check_sampling_rate_stability,
    check_sensor_range,
    check_flatline,
    check_saturation,
    check_nan_ratio,
    check_pressure_flow_correlation,
)

from .config_validation import (
    # Legacy validation (v2.0-v2.2)
    validate_config,
    validate_config_simple,
    validate_config_file,
    check_columns_exist,
    get_config_hash,
    load_and_validate_config,
    TestConfigDC,
    # V2.3.0: Active Configuration and Test Metadata
    ActiveConfiguration,
    TestMetadata,
    validate_active_configuration,
    validate_test_metadata,
    merge_config_and_metadata,
    detect_config_format,
    split_old_config,
)

from .campaign_manager_v2 import (
    get_available_campaigns,
    get_campaign_names,
    create_campaign,
    get_campaign_info,
    get_campaign_data,
    save_to_campaign,
    save_cold_flow_result,
    save_hot_fire_result,
    verify_test_data_integrity,
    get_test_traceability,
    migrate_database,
    SCHEMA_VERSION,
)

from .integrated_analysis import (
    AnalysisResult,
    analyze_cold_flow_test,
    analyze_hot_fire_test,
    quick_analyze,
    format_measurement_table,
)

# P1 Components
from .spc import (
    ControlChartType,
    ViolationType,
    ControlLimits,
    ControlChartPoint,
    ProcessCapability,
    SPCAnalysis,
    calculate_imr_limits,
    calculate_xbar_r_limits,
    check_western_electric_rules,
    calculate_capability,
    detect_trend,
    create_imr_chart,
    analyze_campaign_spc,
    format_spc_summary,
    CHART_CONSTANTS,
)

from .reporting import (
    generate_test_report,
    generate_campaign_report,
    generate_measurement_table as generate_measurement_table_html,
    generate_qc_section,
    generate_traceability_section,
    generate_summary_cards,
    save_report,
)

from .batch_analysis import (
    BatchTestResult,
    BatchAnalysisReport,
    discover_test_files,
    extract_test_id_from_path,
    process_single_file,
    run_batch_analysis,
    batch_cold_flow_analysis,
    batch_hot_fire_analysis,
    save_batch_to_campaign,
    export_batch_results,
    load_csv_with_timestamp,
)

from .export import (
    export_campaign_csv,
    export_campaign_excel,
    export_campaign_json,
    export_campaign_parquet,
    export_test_data_with_context,
    create_traceability_report,
    export_for_qualification,
)

# Configuration Management
from .config_manager import (
    ConfigManager,
    ConfigInfo,
)

# Steady-State Detection
from .steady_state_detection import (
    detect_steady_state_cv,
    detect_steady_state_ml,
    detect_steady_state_derivative,
    detect_steady_state_simple,
    detect_steady_state_auto,
    validate_steady_window,
)

# Metadata Management (v2.3.0+)
from .metadata_manager import (
    MetadataManager,
    MetadataSource,
    load_metadata_from_folder,
    create_metadata_template,
    save_metadata_template,
)

# Saved Configurations (v2.3.0+, formerly "templates")
from .saved_configs import (
    SavedConfig,
    SavedConfigManager,
    load_saved_config,
    # Backward compatibility aliases
    ConfigTemplate,  # Alias for SavedConfig
    TemplateManager,  # Alias for SavedConfigManager
    create_config_from_template,  # Alias for load_saved_config
)

__version__ = "2.3.0"