"""
Enhanced Campaign Manager
=========================
Updated campaign database management with:
- Full traceability fields
- Uncertainty storage
- Schema versioning and migrations
- QC result storage
"""

import sqlite3
import pandas as pd
from datetime import datetime
import json
import os
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

CAMPAIGN_DIR = "campaigns"
SCHEMA_VERSION = 2  # Increment when schema changes


# =============================================================================
# SCHEMA DEFINITIONS
# =============================================================================

CAMPAIGN_INFO_SCHEMA = """
    CREATE TABLE campaign_info (
        campaign_name TEXT PRIMARY KEY,
        campaign_type TEXT,
        created_date TEXT,
        description TEXT,
        part_family TEXT,
        test_article TEXT,
        schema_version INTEGER DEFAULT 1
    )
"""

COLD_FLOW_SCHEMA_V2 = """
    CREATE TABLE test_results (
        test_id TEXT PRIMARY KEY,
        
        -- Basic metadata
        test_path TEXT,
        part TEXT,
        serial_num TEXT,
        test_timestamp TEXT,
        operator TEXT,
        fluid TEXT,
        
        -- Primary measurements with uncertainties
        avg_p_up_bar REAL,
        u_p_up_bar REAL,
        avg_T_up_K REAL,
        avg_p_down_bar REAL,
        avg_mf_g_s REAL,
        u_mf_g_s REAL,
        
        -- Geometry
        orifice_area_mm2 REAL,
        u_orifice_area_mm2 REAL,
        avg_rho_kg_m3 REAL,
        
        -- Derived metrics with uncertainties
        avg_cd_CALC REAL,
        u_cd_CALC REAL,
        cd_rel_uncertainty_pct REAL,
        dp_bar REAL,
        u_dp_bar REAL,
        
        -- Traceability
        raw_data_path TEXT,
        raw_data_hash TEXT,
        raw_data_filename TEXT,
        config_name TEXT,
        config_hash TEXT,
        config_snapshot TEXT,
        
        -- Analysis context
        analyst_username TEXT,
        analyst_hostname TEXT,
        analysis_timestamp_utc TEXT,
        processing_version TEXT,
        
        -- Processing record
        steady_window_start_ms REAL,
        steady_window_end_ms REAL,
        steady_window_duration_ms REAL,
        detection_method TEXT,
        detection_parameters TEXT,
        resample_freq_ms REAL,
        stability_channels TEXT,
        
        -- QC results
        qc_passed INTEGER,
        qc_summary TEXT,
        
        -- Additional
        comments TEXT,
        config_used TEXT
    )
"""

HOT_FIRE_SCHEMA_V2 = """
    CREATE TABLE test_results (
        test_id TEXT PRIMARY KEY,
        
        -- Basic metadata
        test_path TEXT,
        part TEXT,
        serial_num TEXT,
        test_timestamp TEXT,
        operator TEXT,
        propellants TEXT,
        
        -- Test conditions
        duration_s REAL,
        ambient_pressure_bar REAL,
        
        -- Primary measurements with uncertainties
        avg_pc_bar REAL,
        u_pc_bar REAL,
        avg_thrust_n REAL,
        u_thrust_n REAL,
        avg_mf_total_g_s REAL,
        u_mf_total_g_s REAL,
        avg_mf_ox_g_s REAL,
        u_mf_ox_g_s REAL,
        avg_mf_fuel_g_s REAL,
        u_mf_fuel_g_s REAL,
        
        -- Derived metrics with uncertainties
        avg_of_ratio REAL,
        u_of_ratio REAL,
        avg_isp_s REAL,
        u_isp_s REAL,
        avg_c_star_m_s REAL,
        u_c_star_m_s REAL,
        avg_cf REAL,
        
        -- Efficiency metrics
        eta_c_star_pct REAL,
        eta_isp_pct REAL,
        total_impulse_ns REAL,
        
        -- Traceability
        raw_data_path TEXT,
        raw_data_hash TEXT,
        raw_data_filename TEXT,
        config_name TEXT,
        config_hash TEXT,
        config_snapshot TEXT,
        
        -- Analysis context
        analyst_username TEXT,
        analyst_hostname TEXT,
        analysis_timestamp_utc TEXT,
        processing_version TEXT,
        
        -- Processing record
        steady_window_start_ms REAL,
        steady_window_end_ms REAL,
        steady_window_duration_ms REAL,
        detection_method TEXT,
        detection_parameters TEXT,
        resample_freq_ms REAL,
        stability_channels TEXT,
        
        -- QC results
        qc_passed INTEGER,
        qc_summary TEXT,
        
        -- Additional
        comments TEXT,
        config_used TEXT
    )
"""

# Schema migrations
MIGRATIONS = {
    2: {
        'cold_flow': [
            # Add uncertainty columns
            "ALTER TABLE test_results ADD COLUMN u_p_up_bar REAL",
            "ALTER TABLE test_results ADD COLUMN u_mf_g_s REAL",
            "ALTER TABLE test_results ADD COLUMN u_orifice_area_mm2 REAL",
            "ALTER TABLE test_results ADD COLUMN u_cd_CALC REAL",
            "ALTER TABLE test_results ADD COLUMN cd_rel_uncertainty_pct REAL",
            "ALTER TABLE test_results ADD COLUMN u_dp_bar REAL",
            # Add traceability columns
            "ALTER TABLE test_results ADD COLUMN raw_data_path TEXT",
            "ALTER TABLE test_results ADD COLUMN raw_data_hash TEXT",
            "ALTER TABLE test_results ADD COLUMN raw_data_filename TEXT",
            "ALTER TABLE test_results ADD COLUMN config_hash TEXT",
            "ALTER TABLE test_results ADD COLUMN config_snapshot TEXT",
            "ALTER TABLE test_results ADD COLUMN analyst_username TEXT",
            "ALTER TABLE test_results ADD COLUMN analyst_hostname TEXT",
            "ALTER TABLE test_results ADD COLUMN analysis_timestamp_utc TEXT",
            "ALTER TABLE test_results ADD COLUMN processing_version TEXT",
            # Add processing record columns
            "ALTER TABLE test_results ADD COLUMN steady_window_start_ms REAL",
            "ALTER TABLE test_results ADD COLUMN steady_window_end_ms REAL",
            "ALTER TABLE test_results ADD COLUMN steady_window_duration_ms REAL",
            "ALTER TABLE test_results ADD COLUMN detection_method TEXT",
            "ALTER TABLE test_results ADD COLUMN detection_parameters TEXT",
            "ALTER TABLE test_results ADD COLUMN resample_freq_ms REAL",
            "ALTER TABLE test_results ADD COLUMN stability_channels TEXT",
            # Add QC columns
            "ALTER TABLE test_results ADD COLUMN qc_passed INTEGER",
            "ALTER TABLE test_results ADD COLUMN qc_summary TEXT",
        ],
        'hot_fire': [
            # Add uncertainty columns
            "ALTER TABLE test_results ADD COLUMN u_pc_bar REAL",
            "ALTER TABLE test_results ADD COLUMN u_thrust_n REAL",
            "ALTER TABLE test_results ADD COLUMN u_mf_total_g_s REAL",
            "ALTER TABLE test_results ADD COLUMN u_mf_ox_g_s REAL",
            "ALTER TABLE test_results ADD COLUMN u_mf_fuel_g_s REAL",
            "ALTER TABLE test_results ADD COLUMN u_of_ratio REAL",
            "ALTER TABLE test_results ADD COLUMN u_isp_s REAL",
            "ALTER TABLE test_results ADD COLUMN u_c_star_m_s REAL",
            "ALTER TABLE test_results ADD COLUMN ambient_pressure_bar REAL",
            # Add traceability columns
            "ALTER TABLE test_results ADD COLUMN raw_data_path TEXT",
            "ALTER TABLE test_results ADD COLUMN raw_data_hash TEXT",
            "ALTER TABLE test_results ADD COLUMN raw_data_filename TEXT",
            "ALTER TABLE test_results ADD COLUMN config_hash TEXT",
            "ALTER TABLE test_results ADD COLUMN config_snapshot TEXT",
            "ALTER TABLE test_results ADD COLUMN analyst_username TEXT",
            "ALTER TABLE test_results ADD COLUMN analyst_hostname TEXT",
            "ALTER TABLE test_results ADD COLUMN analysis_timestamp_utc TEXT",
            "ALTER TABLE test_results ADD COLUMN processing_version TEXT",
            # Add processing record columns
            "ALTER TABLE test_results ADD COLUMN steady_window_start_ms REAL",
            "ALTER TABLE test_results ADD COLUMN steady_window_end_ms REAL",
            "ALTER TABLE test_results ADD COLUMN steady_window_duration_ms REAL",
            "ALTER TABLE test_results ADD COLUMN detection_method TEXT",
            "ALTER TABLE test_results ADD COLUMN detection_parameters TEXT",
            "ALTER TABLE test_results ADD COLUMN resample_freq_ms REAL",
            "ALTER TABLE test_results ADD COLUMN stability_channels TEXT",
            # Add QC columns
            "ALTER TABLE test_results ADD COLUMN qc_passed INTEGER",
            "ALTER TABLE test_results ADD COLUMN qc_summary TEXT",
        ]
    }
}


# =============================================================================
# SCHEMA VERSION MANAGEMENT
# =============================================================================

def get_schema_version(db_path: str) -> int:
    """Get current schema version from database."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    try:
        c.execute("SELECT schema_version FROM campaign_info LIMIT 1")
        result = c.fetchone()
        version = result[0] if result and result[0] else 1
    except sqlite3.OperationalError:
        # Column doesn't exist, it's version 1
        version = 1
    
    conn.close()
    return version


def set_schema_version(db_path: str, version: int):
    """Update schema version in database."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    try:
        # Try to update
        c.execute("UPDATE campaign_info SET schema_version = ?", (version,))
    except sqlite3.OperationalError:
        # Column doesn't exist, add it
        c.execute("ALTER TABLE campaign_info ADD COLUMN schema_version INTEGER DEFAULT 1")
        c.execute("UPDATE campaign_info SET schema_version = ?", (version,))
    
    conn.commit()
    conn.close()


def migrate_database(db_path: str) -> Tuple[int, int]:
    """
    Migrate database to latest schema version.
    
    Returns:
        Tuple of (old_version, new_version)
    """
    current_version = get_schema_version(db_path)
    
    if current_version >= SCHEMA_VERSION:
        return current_version, current_version
    
    # Get campaign type
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT campaign_type FROM campaign_info LIMIT 1")
    result = c.fetchone()
    campaign_type = result[0] if result else 'cold_flow'
    
    # Apply migrations
    for version in range(current_version + 1, SCHEMA_VERSION + 1):
        if version in MIGRATIONS:
            migrations = MIGRATIONS[version].get(campaign_type, [])
            for sql in migrations:
                try:
                    c.execute(sql)
                except sqlite3.OperationalError as e:
                    # Column might already exist
                    if "duplicate column" not in str(e).lower():
                        print(f"Migration warning: {e}")
    
    conn.commit()
    conn.close()
    
    set_schema_version(db_path, SCHEMA_VERSION)
    
    return current_version, SCHEMA_VERSION


def check_column_exists(db_path: str, table: str, column: str) -> bool:
    """Check if a column exists in a table."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(f"PRAGMA table_info({table})")
    columns = [row[1] for row in c.fetchall()]
    conn.close()
    return column in columns


# =============================================================================
# CAMPAIGN MANAGEMENT
# =============================================================================

def get_available_campaigns() -> List[Dict[str, Any]]:
    """
    List all campaign databases with their metadata.

    Returns:
        List of dictionaries with campaign info including:
        - name: campaign name
        - type: 'cold_flow' or 'hot_fire'
        - test_count: number of tests
        - created_date: creation date
    """
    if not os.path.exists(CAMPAIGN_DIR):
        os.makedirs(CAMPAIGN_DIR)
        return []

    dbs = [f for f in os.listdir(CAMPAIGN_DIR) if f.endswith('.db')]
    campaigns = []

    for db_file in dbs:
        campaign_name = os.path.splitext(db_file)[0]
        db_path = os.path.join(CAMPAIGN_DIR, db_file)

        try:
            conn = sqlite3.connect(db_path)
            c = conn.cursor()

            # Get campaign info
            c.execute("SELECT campaign_type, created_date, schema_version FROM campaign_info LIMIT 1")
            info = c.fetchone()

            # Get test count
            c.execute("SELECT COUNT(*) FROM test_results")
            count = c.fetchone()[0]

            conn.close()

            campaigns.append({
                'name': campaign_name,
                'type': info[0] if info else 'unknown',
                'created_date': info[1] if info else None,
                'schema_version': info[2] if info and len(info) > 2 else 1,
                'test_count': count,
            })
        except Exception as e:
            # If there's an error reading, still include it with minimal info
            campaigns.append({
                'name': campaign_name,
                'type': 'unknown',
                'created_date': None,
                'test_count': 0,
                'error': str(e),
            })

    return campaigns


def get_campaign_names() -> List[str]:
    """List all campaign names (simple string list)."""
    if not os.path.exists(CAMPAIGN_DIR):
        os.makedirs(CAMPAIGN_DIR)
        return []

    dbs = [f for f in os.listdir(CAMPAIGN_DIR) if f.endswith('.db')]
    return [os.path.splitext(f)[0] for f in dbs]


def check_campaign_exists(campaign_name: str) -> bool:
    """
    Check if campaign database file exists.

    Args:
        campaign_name: Campaign name (without .db extension)

    Returns:
        True if campaign database exists, False otherwise
    """
    if not os.path.exists(CAMPAIGN_DIR):
        return False

    campaign_file = os.path.join(CAMPAIGN_DIR, f"{campaign_name}.db")
    return os.path.exists(campaign_file)


def suggest_campaign_name(test_id: str, test_type: str) -> Optional[str]:
    """
    Suggest database campaign name based on test_id and test_type.

    This creates a suggested campaign name that links the folder-based campaign
    (from test_id) to a database campaign name for time-series analysis.

    Format: SYSTEM-TYPE-CAMPAIGN
    Example: test_id="RCS-C01-CF-001", test_type="cold_flow" → "RCS-CF-C01"

    Args:
        test_id: Test identifier following SYSTEM-CAMPAIGN-TYPE-RUN format
        test_type: Test type ('cold_flow' or 'hot_fire')

    Returns:
        Suggested campaign name or None if cannot parse test_id

    Note:
        This bridges the gap between folder-based campaigns (organizational)
        and database campaigns (analytical). See CLAUDE.md for more details.
    """
    from .test_metadata import parse_campaign_from_test_id

    parsed = parse_campaign_from_test_id(test_id)
    if not parsed:
        return None

    # Map test_type to short code
    type_map = {
        'cold_flow': 'CF',
        'hot_fire': 'HF',
    }
    type_code = type_map.get(test_type, parsed.get('type', 'TEST'))

    return f"{parsed['system']}-{type_code}-{parsed['campaign']}"


def create_campaign(campaign_name: str, campaign_type: str = 'cold_flow', description: str = '') -> str:
    """
    Create a new campaign database with v2 schema.

    Args:
        campaign_name: Name of campaign (e.g., "INJ_C1_Acceptance")
        campaign_type: 'cold_flow' or 'hot_fire'
        description: Optional description of the campaign

    Returns:
        Path to created database
    """
    if not os.path.exists(CAMPAIGN_DIR):
        os.makedirs(CAMPAIGN_DIR)

    db_path = os.path.join(CAMPAIGN_DIR, f"{campaign_name}.db")

    if os.path.exists(db_path):
        raise ValueError(f"Campaign '{campaign_name}' already exists!")

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Create campaign info table with schema version
    c.execute("""
        CREATE TABLE campaign_info (
            campaign_name TEXT PRIMARY KEY,
            campaign_type TEXT,
            created_date TEXT,
            description TEXT,
            part_family TEXT,
            test_article TEXT,
            schema_version INTEGER DEFAULT 2
        )
    """)

    c.execute("""
        INSERT INTO campaign_info VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (campaign_name, campaign_type, datetime.now().isoformat(), description, "", "", SCHEMA_VERSION))

    # Create test results table with new schema
    if campaign_type == 'cold_flow':
        c.execute(COLD_FLOW_SCHEMA_V2)
    elif campaign_type == 'hot_fire':
        c.execute(HOT_FIRE_SCHEMA_V2)
    else:
        raise ValueError(f"Unknown campaign type: {campaign_type}")

    conn.commit()
    conn.close()

    return db_path


def get_campaign_info(campaign_name: str) -> Optional[Dict[str, Any]]:
    """Get campaign metadata."""
    db_path = os.path.join(CAMPAIGN_DIR, f"{campaign_name}.db")

    if not os.path.exists(db_path):
        return None

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT * FROM campaign_info")
    info = c.fetchone()
    conn.close()

    if info:
        return {
            'campaign_name': info[0],
            'campaign_type': info[1],
            'created_date': info[2],
            'description': info[3],
            'part_family': info[4],
            'test_article': info[5],
            'schema_version': info[6] if len(info) > 6 else 1
        }
    return None


def get_campaign_data(campaign_name: str, migrate: bool = True) -> pd.DataFrame:
    """
    Retrieve all test results from a campaign.

    Args:
        campaign_name: Name of the campaign
        migrate: If True, run migrations if needed

    Returns:
        DataFrame with all test results
    """
    db_path = os.path.join(CAMPAIGN_DIR, f"{campaign_name}.db")

    if not os.path.exists(db_path):
        raise ValueError(f"Campaign '{campaign_name}' not found!")

    # Run migrations if needed
    if migrate:
        old_v, new_v = migrate_database(db_path)
        if old_v != new_v:
            print(f"Migrated {campaign_name} from schema v{old_v} to v{new_v}")

    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM test_results ORDER BY test_timestamp DESC", conn)
    conn.close()

    return df


# =============================================================================
# SAVE FUNCTIONS WITH TRACEABILITY
# =============================================================================

def save_to_campaign(
    campaign_name: str,
    data: Dict[str, Any],
    migrate: bool = True
) -> bool:
    """
    Save test result to campaign with full traceability.

    Args:
        campaign_name: Target campaign name
        data: Dictionary containing all test data including:
              - Basic metadata (test_id, part, serial_num, etc.)
              - Measurements (avg_p_up_bar, avg_mf_g_s, etc.)
              - Uncertainties (u_p_up_bar, u_mf_g_s, u_cd_CALC, etc.)
              - Traceability (raw_data_hash, config_hash, etc.)
              - Processing record (steady_window_start_ms, etc.)
              - QC results (qc_passed, qc_summary)
        migrate: If True, run migrations if needed

    Returns:
        True if saved successfully
    """
    db_path = os.path.join(CAMPAIGN_DIR, f"{campaign_name}.db")

    if not os.path.exists(db_path):
        raise ValueError(f"Campaign '{campaign_name}' not found!")

    # Run migrations if needed
    if migrate:
        migrate_database(db_path)

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Get campaign type
    c.execute("SELECT campaign_type FROM campaign_info")
    campaign_type = c.fetchone()[0]

    # Get columns from database
    c.execute("PRAGMA table_info(test_results)")
    db_columns = [row[1] for row in c.fetchall()]

    # Populate test_path if not already present (for traceability)
    if 'test_path' not in data or data['test_path'] is None:
        # Try to extract from test_folder in metadata
        if 'test_folder' in data:
            data['test_path'] = str(data['test_folder'])

    # Filter data to only include existing columns
    filtered_data = {k: v for k, v in data.items() if k in db_columns}

    # Helper function to sanitize values for SQLite compatibility
    def sanitize_for_sqlite(value):
        """Convert value to SQLite-compatible type (handles NumPy types)."""
        import numpy as np
        if isinstance(value, (np.integer, np.floating)):
            return float(value)
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, (list, tuple)):
            return [sanitize_for_sqlite(v) for v in value]
        else:
            return value

    # Sanitize all values to prevent NumPy type binding errors
    filtered_data = {k: sanitize_for_sqlite(v) for k, v in filtered_data.items()}

    # Check if exists
    c.execute("SELECT 1 FROM test_results WHERE test_id = ?", (data.get('test_id'),))
    exists = c.fetchone()

    if exists:
        # Update
        set_pairs = ', '.join([f"{k}=?" for k in filtered_data.keys() if k != 'test_id'])
        vals = [v for k, v in filtered_data.items() if k != 'test_id']
        vals.append(filtered_data['test_id'])

        query = f"UPDATE test_results SET {set_pairs} WHERE test_id=?"
        c.execute(query, vals)
    else:
        # Insert
        cols = list(filtered_data.keys())
        vals = list(filtered_data.values())
        placeholders = ','.join(['?'] * len(cols))

        query = f"INSERT INTO test_results ({','.join(cols)}) VALUES ({placeholders})"
        c.execute(query, vals)

    conn.commit()
    conn.close()

    return True


def save_cold_flow_result(
    campaign_name: str,
    test_id: str,
    metadata: Dict[str, Any],
    measurements: Dict[str, float],
    uncertainties: Dict[str, float],
    traceability: Dict[str, Any],
    qc_result: Optional[Dict[str, Any]] = None,
    comments: str = ""
) -> bool:
    """
    Convenience function to save cold flow result with all components.

    Args:
        campaign_name: Target campaign
        test_id: Unique test identifier
        metadata: Part, serial, operator, fluid, etc.
        measurements: avg_p_up_bar, avg_mf_g_s, avg_cd_CALC, etc.
        uncertainties: u_p_up_bar, u_mf_g_s, u_cd_CALC, etc.
        traceability: raw_data_hash, config_hash, steady_window, etc.
        qc_result: QC report summary
        comments: User comments

    Returns:
        True if saved successfully
    """
    record = {
        'test_id': test_id,
        'test_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'comments': comments,
        **metadata,
        **measurements,
        **uncertainties,
        **traceability,
    }

    if qc_result:
        record['qc_passed'] = 1 if qc_result.get('passed', False) else 0
        record['qc_summary'] = json.dumps(qc_result.get('summary', {}))

    return save_to_campaign(campaign_name, record)


def save_hot_fire_result(
    campaign_name: str,
    test_id: str,
    metadata: Dict[str, Any],
    measurements: Dict[str, float],
    uncertainties: Dict[str, float],
    traceability: Dict[str, Any],
    qc_result: Optional[Dict[str, Any]] = None,
    comments: str = ""
) -> bool:
    """
    Convenience function to save hot fire result with all components.

    Args:
        campaign_name: Target campaign
        test_id: Unique test identifier
        metadata: Part, serial, operator, propellants, etc.
        measurements: avg_pc_bar, avg_thrust_n, avg_isp_s, etc.
        uncertainties: u_pc_bar, u_thrust_n, u_isp_s, etc.
        traceability: raw_data_hash, config_hash, steady_window, etc.
        qc_result: QC report summary
        comments: User comments

    Returns:
        True if saved successfully
    """
    record = {
        'test_id': test_id,
        'test_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'comments': comments,
        **metadata,
        **measurements,
        **uncertainties,
        **traceability,
    }

    if qc_result:
        record['qc_passed'] = 1 if qc_result.get('passed', False) else 0
        record['qc_summary'] = json.dumps(qc_result.get('summary', {}))

    return save_to_campaign(campaign_name, record)


# =============================================================================
# VERIFICATION FUNCTIONS
# =============================================================================

def verify_test_data_integrity(
    campaign_name: str,
    test_id: str
) -> Tuple[bool, str]:
    """
    Verify that stored test data matches its recorded hash.

    Args:
        campaign_name: Campaign containing the test
        test_id: Test to verify

    Returns:
        Tuple of (is_valid, message)
    """
    from .traceability import verify_data_integrity

    db_path = os.path.join(CAMPAIGN_DIR, f"{campaign_name}.db")

    if not os.path.exists(db_path):
        return False, f"Campaign not found: {campaign_name}"

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute(
        "SELECT raw_data_path, raw_data_hash FROM test_results WHERE test_id = ?",
        (test_id,)
    )
    result = c.fetchone()
    conn.close()

    if not result:
        return False, f"Test not found: {test_id}"

    raw_path, expected_hash = result

    if not raw_path:
        return False, "No raw data path stored (test may have been uploaded directly)"

    if not expected_hash:
        return False, "No hash stored for this test"

    return verify_data_integrity(raw_path, expected_hash)


def get_test_traceability(
    campaign_name: str,
    test_id: str
) -> Optional[Dict[str, Any]]:
    """
    Get full traceability record for a test.

    Args:
        campaign_name: Campaign containing the test
        test_id: Test to retrieve

    Returns:
        Dictionary with all traceability fields, or None if not found
    """
    db_path = os.path.join(CAMPAIGN_DIR, f"{campaign_name}.db")

    if not os.path.exists(db_path):
        return None

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    traceability_cols = [
        'raw_data_path', 'raw_data_hash', 'raw_data_filename',
        'config_name', 'config_hash', 'config_snapshot',
        'analyst_username', 'analyst_hostname', 'analysis_timestamp_utc',
        'processing_version', 'steady_window_start_ms', 'steady_window_end_ms',
        'detection_method', 'detection_parameters', 'resample_freq_ms',
        'qc_passed', 'qc_summary'
    ]

    # Check which columns exist
    c.execute("PRAGMA table_info(test_results)")
    existing_cols = [row[1] for row in c.fetchall()]

    available_cols = [col for col in traceability_cols if col in existing_cols]

    if not available_cols:
        conn.close()
        return None

    query = f"SELECT {','.join(available_cols)} FROM test_results WHERE test_id = ?"
    c.execute(query, (test_id,))
    result = c.fetchone()
    conn.close()

    if not result:
        return None

    return dict(zip(available_cols, result))