# Create new file: data_lib/campaign_manager.py

import sqlite3
import pandas as pd
from datetime import datetime
import json
import os

CAMPAIGN_DIR = "campaigns"


def get_available_campaigns():
    """List all campaign databases."""
    if not os.path.exists(CAMPAIGN_DIR):
        os.makedirs(CAMPAIGN_DIR)

    dbs = [f for f in os.listdir(CAMPAIGN_DIR) if f.endswith('.db')]
    return [os.path.splitext(f)[0] for f in dbs]


def create_campaign(campaign_name, campaign_type='cold_flow'):
    """
    Create a new campaign database.

    Args:
        campaign_name: Name of campaign (e.g., "INJ_C1_Acceptance")
        campaign_type: 'cold_flow' or 'hot_fire'
    """
    db_path = os.path.join(CAMPAIGN_DIR, f"{campaign_name}.db")

    if os.path.exists(db_path):
        raise ValueError(f"Campaign '{campaign_name}' already exists!")

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Campaign metadata table
    c.execute('''
        CREATE TABLE campaign_info (
            campaign_name TEXT PRIMARY KEY,
            campaign_type TEXT,
            created_date TEXT,
            description TEXT,
            part_family TEXT,
            test_article TEXT
        )
    ''')

    c.execute('''
        INSERT INTO campaign_info VALUES (?, ?, ?, ?, ?, ?)
    ''', (campaign_name, campaign_type, datetime.now().isoformat(), "", "", ""))

    if campaign_type == 'cold_flow':
        c.execute('''
            CREATE TABLE test_results (
                test_id TEXT PRIMARY KEY,
                test_path TEXT,
                part TEXT,
                serial_num TEXT,
                test_timestamp TEXT,
                operator TEXT,
                fluid TEXT,
                avg_p_up_bar REAL,
                avg_T_up_K REAL,
                avg_p_down_bar REAL,
                avg_mf_g_s REAL,
                orifice_area_mm2 REAL,
                avg_rho_kg_m3 REAL,
                avg_cd_CALC REAL,
                dp_bar REAL,
                comments TEXT,
                config_used TEXT
            )
        ''')

    elif campaign_type == 'hot_fire':
        c.execute('''
            CREATE TABLE test_results (
                test_id TEXT PRIMARY KEY,
                test_path TEXT,
                part TEXT,
                serial_num TEXT,
                test_timestamp TEXT,
                operator TEXT,
                propellants TEXT,
                duration_s REAL,
                avg_pc_bar REAL,
                avg_thrust_n REAL,
                avg_mf_total_g_s REAL,
                avg_mf_ox_g_s REAL,
                avg_mf_fuel_g_s REAL,
                avg_of_ratio REAL,
                avg_isp_s REAL,
                avg_c_star_m_s REAL,
                avg_cf REAL,
                eta_c_star_pct REAL,
                eta_isp_pct REAL,
                total_impulse_ns REAL,
                comments TEXT,
                config_used TEXT
            )
        ''')

    conn.commit()
    conn.close()

    return db_path


def save_to_campaign(campaign_name, data):
    """Save test result to specific campaign."""
    db_path = os.path.join(CAMPAIGN_DIR, f"{campaign_name}.db")

    if not os.path.exists(db_path):
        raise ValueError(f"Campaign '{campaign_name}' not found!")

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Get campaign type
    c.execute("SELECT campaign_type FROM campaign_info")
    campaign_type = c.fetchone()[0]

    # Prepare columns based on type
    if campaign_type == 'cold_flow':
        cols = [
            'test_id', 'test_path', 'part', 'serial_num', 'test_timestamp',
            'operator', 'fluid', 'avg_p_up_bar', 'avg_T_up_K', 'avg_p_down_bar',
            'avg_mf_g_s', 'orifice_area_mm2', 'avg_rho_kg_m3', 'avg_cd_CALC',
            'dp_bar', 'comments', 'config_used'
        ]
    else:  # hot_fire
        cols = [
            'test_id', 'test_path', 'part', 'serial_num', 'test_timestamp',
            'operator', 'propellants', 'duration_s', 'avg_pc_bar', 'avg_thrust_n',
            'avg_mf_total_g_s', 'avg_mf_ox_g_s', 'avg_mf_fuel_g_s', 'avg_of_ratio',
            'avg_isp_s', 'avg_c_star_m_s', 'avg_cf', 'eta_c_star_pct', 'eta_isp_pct',
            'total_impulse_ns', 'comments', 'config_used'
        ]

    vals = [data.get(k) for k in cols]

    # Check if exists
    c.execute("SELECT 1 FROM test_results WHERE test_id = ?", (data.get('test_id'),))
    exists = c.fetchone()

    if exists:
        # Update
        set_str = ', '.join([f"{k}=?" for k in cols if k != 'test_id'])
        query = f"UPDATE test_results SET {set_str} WHERE test_id=?"
        update_vals = vals[1:] + [vals[0]]
        c.execute(query, update_vals)
    else:
        # Insert
        placeholders = ','.join(['?'] * len(cols))
        query = f"INSERT INTO test_results ({','.join(cols)}) VALUES ({placeholders})"
        c.execute(query, vals)

    conn.commit()
    conn.close()


def get_campaign_data(campaign_name):
    """Retrieve all test results from a campaign."""
    db_path = os.path.join(CAMPAIGN_DIR, f"{campaign_name}.db")

    if not os.path.exists(db_path):
        raise ValueError(f"Campaign '{campaign_name}' not found!")

    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM test_results ORDER BY test_timestamp DESC", conn)
    conn.close()

    return df


def get_campaign_info(campaign_name):
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
            'test_article': info[5]
        }
    return None


def link_test_to_campaign(campaign_name, test_metadata_path):
    """
    Import a test from the file system into a campaign.

    Args:
        campaign_name: Target campaign
        test_metadata_path: Path to metadata.json from your TM system
    """
    with open(test_metadata_path, 'r') as f:
        metadata = json.load(f)

    # Build record from metadata
    # This will need custom logic based on test type
    # For now, return the metadata for manual processing
    return metadata