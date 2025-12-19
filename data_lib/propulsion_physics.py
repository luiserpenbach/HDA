import CoolProp.CoolProp as CP
import numpy as np
import pandas as pd


# --- CONSTANTS ---
g0 = 9.80665  # Standard Gravity (m/s^2)



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



def calculate_derived_metrics(avg_stats, config):
    """
    Calculates metrics that rely on STEADY STATE AVERAGES (single values),
    rather than time-series data.

    Args:
        avg_stats (dict): Dictionary of averaged values (e.g., {'mass_flow_ox': 0.5, 'chamber_pressure': 20.0})
        config (dict): The loaded test configuration.

    Returns:
        dict: New metrics to display (e.g., {'Cd_Ox': 0.45, 'Eta_C_Star': 95.5})
    """
    derived = {}
    geom = config.get('geometry', {})
    fluid = config.get('fluid', {})  # New config section for densities
    targets = config.get('settings', {})  # For target C* / Isp

    # --- HOT FIRE METRICS ---
    # 1. Combustion Efficiency (Eta C*)
    if 'c_star' in avg_stats and targets.get('target_c_star'):
        target = targets['target_c_star']
        if target > 0:
            derived['η C* (%)'] = (avg_stats['c_star'] / target) * 100.0

    # 2. Isp Efficiency
    if 'isp' in avg_stats and targets.get('target_isp'):
        target = targets['target_isp']
        if target > 0:
            derived['η Isp (%)'] = (avg_stats['isp'] / target) * 100.0

    # --- COLD FLOW METRICS (Cd) ---
    # Formula: Cd = m_dot / (A * sqrt(2 * rho * dP))
    # We need: Flow, Area (Injector), Density, Pressure Drop

    # Helper for Cd Calculation
    def calc_cd(m_dot, press_drop_bar, density, area_mm2):
        try:
            if area_mm2 <= 0 or density <= 0 or press_drop_bar <= 0: return None

            # Unit Conversion
            # P: Bar -> Pa (1e5)
            # A: mm^2 -> m^2 (1e-6)
            # m: g/s -> kg/s (1e-3)

            dp_pa = press_drop_bar * 1e5
            area_m2 = area_mm2 * 1e-6
            m_kg = m_dot * 1e-3

            # Bernoulli
            velocity_ideal = (2 * dp_pa * density) ** 0.5
            return m_kg / (area_m2 * velocity_ideal)
        except:
            return None

    # Calculate for Oxidizer
    if 'mass_flow_ox' in avg_stats and 'inlet_pressure_ox' in avg_stats:
        # P_chamber might be 0/ambient for cold flow, or measured
        p_out = avg_stats.get('chamber_pressure', 0.0)
        dp = avg_stats['inlet_pressure_ox'] - p_out

        cd_ox = calc_cd(
            avg_stats['mass_flow_ox'],
            dp,
            fluid.get('ox_density_kg_m3', 1000),  # Default to Water
            geom.get('ox_injector_area_mm2', 0)
        )
        if cd_ox: derived['Cd (Ox)'] = cd_ox

    # Calculate for Fuel
    if 'mass_flow_fuel' in avg_stats and 'inlet_pressure_fuel' in avg_stats:
        p_out = avg_stats.get('chamber_pressure', 0.0)
        dp = avg_stats['inlet_pressure_fuel'] - p_out

        cd_fuel = calc_cd(
            avg_stats['mass_flow_fuel'],
            dp,
            fluid.get('fuel_density_kg_m3', 0),
            geom.get('fuel_injector_area_mm2', 0)
        )
        if cd_fuel: derived['Cd (Fuel)'] = cd_fuel

    return derived
