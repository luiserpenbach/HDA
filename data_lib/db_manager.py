import sqlite3
import pandas as pd
from datetime import datetime
import json

DB_NAME = 'test_campaign.db'


def init_db():
    """Initialize the database with required tables."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    # 1. Generic Test Results (Existing)
    c.execute('''
        CREATE TABLE IF NOT EXISTS test_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            config_name TEXT,
            timestamp TEXT,
            duration REAL,
            avg_cv REAL,
            rise_time REAL,
            avg_pressure REAL,
            avg_thrust REAL,
            mass_flow_total REAL,
            isp REAL,
            c_star REAL,
            derived_metrics TEXT,
            comments TEXT
        )
    ''')

    # 2. Cold Flow Campaign (Strict Schema)
    # Matches columns from: CAMPAIGN_DB-INJ-C1-CF.csv
    c.execute('''
        CREATE TABLE IF NOT EXISTS cold_flow_campaign (
            test_id TEXT PRIMARY KEY,
            part TEXT,
            serial_num TEXT,
            test_timestamp TEXT,
            fluid TEXT,
            avg_p_up_bar REAL,
            avg_T_up_K REAL,
            avg_p_down_bar REAL,
            avg_mf_g_s REAL,
            orfifice_area_mm2 REAL,
            avg_rho_CALC REAL,
            avg_cd_CALC REAL,
            comments TEXT
        )
    ''')

    conn.commit()
    conn.close()


# --- COLD FLOW FUNCTIONS ---

def save_cold_flow_record(data):
    """
    Saves or Updates a record in the cold_flow_campaign table.
    data: dict containing keys matching the schema.
    """
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    # Check if record exists
    c.execute("SELECT 1 FROM cold_flow_campaign WHERE test_id = ?", (data.get('test_id'),))
    exists = c.fetchone()

    cols = [
        'test_id', 'part', 'serial_num', 'test_timestamp', 'fluid',
        'avg_p_up_bar', 'avg_T_up_K', 'avg_p_down_bar', 'avg_mf_g_s',
        'orfifice_area_mm2', 'avg_rho_CALC', 'avg_cd_CALC', 'comments'
    ]

    vals = [data.get(k) for k in cols]

    if exists:
        # Update existing record
        # (Construct "part=?, serial_num=?..." string)
        set_str = ', '.join([f"{k}=?" for k in cols if k != 'test_id'])
        query = f"UPDATE cold_flow_campaign SET {set_str} WHERE test_id=?"
        # Move test_id to the end of vals for the WHERE clause
        update_vals = vals[1:] + [vals[0]]
        c.execute(query, update_vals)
    else:
        # Insert new
        placeholders = ','.join(['?'] * len(cols))
        query = f"INSERT INTO cold_flow_campaign ({','.join(cols)}) VALUES ({placeholders})"
        c.execute(query, vals)

    conn.commit()
    conn.close()


def get_cold_flow_history():
    conn = sqlite3.connect(DB_NAME)
    try:
        df = pd.read_sql("SELECT * FROM cold_flow_campaign ORDER BY test_timestamp DESC", conn)
    except:
        df = pd.DataFrame()
    conn.close()
    return df


# --- EXISTING GENERIC FUNCTIONS ---

def save_test_result(filename, config_name, stats, derived, comments=""):
    init_db()
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    # Convert derived dict to JSON string for storage
    derived_json = json.dumps(derived)

    c.execute('''
        INSERT INTO test_results (
            filename, config_name, timestamp, duration, avg_cv, rise_time,
            avg_pressure, avg_thrust, mass_flow_total, isp, c_star,
            derived_metrics, comments
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        filename, config_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        stats.get('duration'), stats.get('avg_cv'), stats.get('rise_time'),
        stats.get('avg_pressure'), stats.get('avg_thrust'), stats.get('mass_flow_total'),
        stats.get('isp'), stats.get('c_star'),
        derived_json, comments
    ))
    conn.commit()
    conn.close()


def get_campaign_history():
    init_db()
    conn = sqlite3.connect(DB_NAME)
    try:
        df = pd.read_sql("SELECT * FROM test_results ORDER BY timestamp DESC", conn)
    except:
        df = pd.DataFrame()
    conn.close()
    return df