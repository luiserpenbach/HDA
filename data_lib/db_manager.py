import sqlite3
import pandas as pd
from datetime import datetime
import os

DB_PATH = "test_campaign.db"


def init_db():
    """Creates the tests table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # We store metadata (filename, date) and key metrics (stats)
    # Flexible schema: We define core columns, but you can add more later
    c.execute('''
        CREATE TABLE IF NOT EXISTS test_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            filename TEXT UNIQUE,
            config_name TEXT,

            -- Primary Metrics
            duration_s REAL,
            avg_pressure_bar REAL,
            avg_thrust_n REAL,
            total_flow_gs REAL,

            -- Performance Metrics
            isp_s REAL,
            c_star_ms REAL,
            eta_c_star_pct REAL,
            cd_ox REAL,
            cd_fuel REAL,

            -- Quality Metrics
            stability_cv_pct REAL,
            rise_time_s REAL,

            comments TEXT
        )
    ''')
    conn.commit()
    conn.close()


def save_test_result(filename, config_name, stats, derived, comments=""):
    """
    Saves a single test result to the DB. Updates if filename already exists.
    """
    init_db()  # Ensure DB exists
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Combine stats and derived metrics into one flat structure
    # We use .get() to safely handle missing metrics (e.g. Cold Flow has no Isp)
    data = {
        'filename': filename,
        'config_name': config_name,
        'duration_s': stats.get('duration', 0),
        'avg_pressure_bar': stats.get('avg_chamber_pressure', 0) or stats.get('avg_pressure', 0),
        'avg_thrust_n': stats.get('avg_thrust', 0),
        'total_flow_gs': stats.get('mass_flow_total', 0) or stats.get('avg_flow', 0),

        'isp_s': stats.get('isp', 0),
        'c_star_ms': stats.get('c_star', 0),

        'eta_c_star_pct': derived.get('η C* (%)', None),
        'cd_ox': derived.get('Cd (Ox)', None),
        'cd_fuel': derived.get('Cd (Fuel)', None),

        'stability_cv_pct': stats.get('avg_cv', 0),
        'rise_time_s': stats.get('rise_time', 0),
        'comments': comments
    }

    # UPSERT Logic (Insert or Replace based on filename)
    # This is clean: if you re-analyze a file with better cuts, it updates the record.
    cols = ', '.join(data.keys())
    placeholders = ', '.join(['?'] * len(data))

    # SQLite UPSERT syntax (requires newer SQLite, fallback to REPLACE usually fine for this scale)
    query = f'''
        INSERT OR REPLACE INTO test_history ({cols}) 
        VALUES ({placeholders})
    '''

    c.execute(query, list(data.values()))
    conn.commit()
    conn.close()


def get_campaign_history():
    """Returns the entire history as a Pandas DataFrame."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM test_history ORDER BY timestamp DESC", conn)
    conn.close()
    return df