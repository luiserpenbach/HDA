import CoolProp.CoolProp as CP
import numpy as np
import pandas as pd


# --- CONSTANTS ---
g0 = 9.80665  # Standard Gravity (m/s^2)

def calc_cd(fluid, mf_g_s, A_mm2, p1_bar, T1_C, p2_bar=1.013):
    mf = mf_g_s / 1000.0    # mass flow,kg/s
    p1 = p1_bar * 1e5       # upstream pressure, Pa
    T1 = T1_C + 273.15      # upstream temp., K
    dp = p1-p2_bar*1e5      # Pressure difference,
    A = A_mm2*1e-6

    rho = CP.PropsSI("D", "P", p1, "T", T1, fluid)

    cd = mf / (A*np.sqrt(2*rho*dp))
    return cd


def calculate_performance(df, config):
    """
    Enriches the DataFrame with propulsion metrics based on the provided config.

    Args:
        df (pd.DataFrame): The dataframe containing sensor data (resampled/smoothed).
        config (dict): Configuration dictionary mapping standard names to column names
                       and defining engine geometry.

    Structure of config expected:
    {
        "columns": {
            "chamber_pressure": "col_name_1",  # Units: Bar
            "thrust": "col_name_2",            # Units: Newtons
            "mass_flow_ox": "col_name_3",      # Units: g/s
            "mass_flow_fuel": "col_name_4"     # Units: g/s
        },
        "geometry": {
            "throat_area_mm2": 15.5  # throat area in square millimeters
        }
    }

    Returns:
        pd.DataFrame: A copy of df with added columns:
                      ['mass_flow_total', 'of_ratio', 'c_star', 'isp', 'cf']
    """
    df_calc = df.copy()
    cols = config.get('columns', {})
    geom = config.get('geometry', {})

    # 1. RETRIEVE COLUMN NAMES
    col_pc = cols.get('chamber_pressure')
    col_thrust = cols.get('thrust')
    col_m_ox = cols.get('mass_flow_ox')
    col_m_fuel = cols.get('mass_flow_fuel')

    # 2. MASS FLOW & MIXTURE RATIO
    # We need total mass flow for almost everything.
    m_dot_total = None

    if col_m_ox and col_m_fuel and col_m_ox in df and col_m_fuel in df:
        # Calculate Total Flow (g/s)
        df_calc['mass_flow_total'] = df[col_m_ox] + df[col_m_fuel]
        m_dot_total = df_calc['mass_flow_total']

        # Calculate O/F Ratio
        # Handle division by zero (fuel flow = 0) gracefully
        with np.errstate(divide='ignore', invalid='ignore'):
            df_calc['of_ratio'] = df[col_m_ox] / df[col_m_fuel]
            # Clean up infinite values (e.g. before flow starts)
            df_calc['of_ratio'] = df_calc['of_ratio'].replace([np.inf, -np.inf], np.nan)

    elif col_m_ox and col_m_ox in df:
        # Fallback: Maybe a monopropellant or cold gas test?
        m_dot_total = df[col_m_ox]
        df_calc['mass_flow_total'] = m_dot_total

    # 3. C* (CHARACTERISTIC VELOCITY)
    # C* = (Pc * At) / m_dot
    # Units: Pc [Bar], At [mm^2], m_dot [g/s]
    # Conversion:
    #   Pc (Bar -> Pa): * 1e5
    #   At (mm^2 -> m^2): * 1e-6
    #   m_dot (g/s -> kg/s): * 1e-3
    #   Result m/s

    ath_mm2 = geom.get('throat_area_mm2')

    if col_pc and m_dot_total is not None and ath_mm2 and col_pc in df:
        ath_m2 = ath_mm2 * 1e-6
        pc_pa = df[col_pc] * 1e5
        m_dot_kg = m_dot_total * 1e-3

        with np.errstate(divide='ignore', invalid='ignore'):
            df_calc['c_star'] = (pc_pa * ath_m2) / m_dot_kg
            # Filter noise: C* is physically impossible if flow is near zero
            # We mask values where mass flow is < 0.1 g/s to avoid garbage spikes
            df_calc.loc[m_dot_total < 0.1, 'c_star'] = np.nan

    # 4. Isp (SPECIFIC IMPULSE)
    # Isp = F / (m_dot * g0)
    # Units: F [N], m_dot [g/s] -> kg/s
    if col_thrust and m_dot_total is not None and col_thrust in df:
        thrust_n = df[col_thrust]
        m_dot_kg = m_dot_total * 1e-3

        with np.errstate(divide='ignore', invalid='ignore'):
            df_calc['isp'] = thrust_n / (m_dot_kg * g0)
            df_calc.loc[m_dot_total < 0.1, 'isp'] = np.nan

    # 5. Cf (THRUST COEFFICIENT)
    # Cf = F / (Pc * At)
    # Units: F [N], Pc [Bar] -> Pa, At [mm^2] -> m^2
    if col_thrust and col_pc and ath_mm2 and col_thrust in df and col_pc in df:
        ath_m2 = ath_mm2 * 1e-6
        thrust_n = df[col_thrust]
        pc_pa = df[col_pc] * 1e5

        with np.errstate(divide='ignore', invalid='ignore'):
            df_calc['cf'] = thrust_n / (pc_pa * ath_m2)
            # Filter: If pressure is near vacuum/zero, Cf explodes
            df_calc.loc[df[col_pc] < 0.1, 'cf'] = np.nan

    return df_calc


def calculate_combustion_efficiency(df, target_c_star):
    """
    Calculates eta_c_star (Efficiency) based on a theoretical target.
    Expects 'c_star' column to exist.
    """
    if 'c_star' not in df:
        return None

    return (df['c_star'] / target_c_star) * 100.0


def calculate_performance(df, config):
    """
    Enriches the DataFrame with propulsion metrics based on the provided config.

    Args:
        df (pd.DataFrame): The dataframe containing sensor data (resampled/smoothed).
        config (dict): Configuration dictionary mapping standard names to column names
                       and defining engine geometry.

    Structure of config expected:
    {
        "columns": {
            "chamber_pressure": "col_name_1",  # Units: Bar
            "thrust": "col_name_2",            # Units: Newtons
            "mass_flow_ox": "col_name_3",      # Units: g/s
            "mass_flow_fuel": "col_name_4"     # Units: g/s
        },
        "geometry": {
            "throat_area_mm2": 15.5  # throat area in square millimeters
        }
    }

    Returns:
        pd.DataFrame: A copy of df with added columns:
                      ['mass_flow_total', 'of_ratio', 'c_star', 'isp', 'cf']
    """
    df_calc = df.copy()
    cols = config.get('columns', {})
    geom = config.get('geometry', {})

    # 1. RETRIEVE IMPORTANT COLUMN NAMES
    col_pc = cols.get('chamber_pressure')
    col_thrust = cols.get('thrust')
    col_m_ox = cols.get('mass_flow_ox')
    col_m_fuel = cols.get('mass_flow_fuel')

    # 2. MASS FLOW & MIXTURE RATIO
    # We need total mass flow for almost everything.
    m_dot_total = None

    if col_m_ox and col_m_fuel and col_m_ox in df and col_m_fuel in df:
        # Calculate Total Flow (g/s)
        df_calc['mass_flow_total'] = df[col_m_ox] + df[col_m_fuel]
        m_dot_total = df_calc['mass_flow_total']

        # Calculate O/F Ratio
        # Handle division by zero (fuel flow = 0) gracefully
        with np.errstate(divide='ignore', invalid='ignore'):
            df_calc['of_ratio'] = df[col_m_ox] / df[col_m_fuel]
            # Clean up infinite values (e.g. before flow starts)
            df_calc['of_ratio'] = df_calc['of_ratio'].replace([np.inf, -np.inf], np.nan)

    elif col_m_ox and col_m_ox in df:
        # Fallback: Maybe a monopropellant or cold gas test?
        m_dot_total = df[col_m_ox]
        df_calc['mass_flow_total'] = m_dot_total

    # 3. C* (CHARACTERISTIC VELOCITY)
    # C* = (Pc * At) / m_dot
    # Units: Pc [Bar], At [mm^2], m_dot [g/s]
    # Conversion:
    #   Pc (Bar -> Pa): * 1e5
    #   At (mm^2 -> m^2): * 1e-6
    #   m_dot (g/s -> kg/s): * 1e-3
    #   Result m/s

    ath_mm2 = geom.get('throat_area_mm2')

    if col_pc and m_dot_total is not None and ath_mm2 and col_pc in df:
        ath_m2 = ath_mm2 * 1e-6
        pc_pa = df[col_pc] * 1e5
        m_dot_kg = m_dot_total * 1e-3

        with np.errstate(divide='ignore', invalid='ignore'):
            df_calc['c_star'] = (pc_pa * ath_m2) / m_dot_kg
            # Filter noise: C* is physically impossible if flow is near zero
            # We mask values where mass flow is < 0.1 g/s to avoid garbage spikes
            df_calc.loc[m_dot_total < 0.1, 'c_star'] = np.nan

    # 4. Isp (SPECIFIC IMPULSE)
    # Isp = F / (m_dot * g0)
    # Units: F [N], m_dot [g/s] -> kg/s
    if col_thrust and m_dot_total is not None and col_thrust in df:
        thrust_n = df[col_thrust]
        m_dot_kg = m_dot_total * 1e-3

        with np.errstate(divide='ignore', invalid='ignore'):
            df_calc['isp'] = thrust_n / (m_dot_kg * g0)
            df_calc.loc[m_dot_total < 0.1, 'isp'] = np.nan

    # 5. Cf (THRUST COEFFICIENT)
    # Cf = F / (Pc * At)
    # Units: F [N], Pc [Bar] -> Pa, At [mm^2] -> m^2
    if col_thrust and col_pc and ath_mm2 and col_thrust in df and col_pc in df:
        ath_m2 = ath_mm2 * 1e-6
        thrust_n = df[col_thrust]
        pc_pa = df[col_pc] * 1e5

        with np.errstate(divide='ignore', invalid='ignore'):
            df_calc['cf'] = thrust_n / (pc_pa * ath_m2)
            # Filter: If pressure is near vacuum/zero, Cf explodes
            df_calc.loc[df[col_pc] < 0.1, 'cf'] = np.nan

    return df_calc


def calculate_combustion_efficiency(df, target_c_star):
    """
    Calculates eta_c_star (Efficiency) based on a theoretical target.
    Expects 'c_star' column to exist.
    """
    if 'c_star' not in df:
        return None

    return (df['c_star'] / target_c_star) * 100.0


def analyze_efficiency_stats(df, t_start, t_end, target_c_star=None, target_isp=None):
    """
    Returns a dictionary of averaged performance metrics for the steady window.
    """
    mask = (df['timestamp'] >= t_start) & (df['timestamp'] <= t_end)
    steady = df[mask]

    stats = {}

    if steady.empty:
        return stats

    # Calculate simple averages of the calculated columns
    for metric in ['c_star', 'isp', 'cf', 'of_ratio', 'mass_flow_total']:
        if metric in steady:
            stats[f'avg_{metric}'] = steady[metric].mean()

    # Efficiency Calculations
    if target_c_star and 'avg_c_star' in stats:
        stats['eta_c_star'] = (stats['avg_c_star'] / target_c_star) * 100.0

    if target_isp and 'avg_isp' in stats:
        stats['eta_isp'] = (stats['avg_isp'] / target_isp) * 100.0

    return stats
