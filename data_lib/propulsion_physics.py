import numpy as np
import pandas as pd

# --- CONSTANTS ---
g0 = 9.80665  # Standard Gravity (m/s^2)


# ==========================================
# 1. ACTUAL PERFORMANCE (Time-Series)
# ==========================================
def calculate_performance(df, config):
    """
    Enriches the DataFrame with propulsion metrics (Isp, C*, etc.)
    """
    df_calc = df.copy()
    cols = config.get('columns', {})
    geom = config.get('geometry', {})

    # Retrieve mapped column names (e.g., "IG-PT-01")
    col_pc = cols.get('chamber_pressure')
    col_thrust = cols.get('thrust')
    col_m_ox = cols.get('mass_flow_ox')
    col_m_fuel = cols.get('mass_flow_fuel')

    # 1. Total Mass Flow
    m_dot_total = None
    if col_m_ox and col_m_fuel and col_m_ox in df and col_m_fuel in df:
        df_calc['mass_flow_total'] = df[col_m_ox] + df[col_m_fuel]
        m_dot_total = df_calc['mass_flow_total']

        # O/F Ratio
        with np.errstate(divide='ignore', invalid='ignore'):
            df_calc['of_ratio'] = df[col_m_ox] / df[col_m_fuel]
            df_calc['of_ratio'] = df_calc['of_ratio'].replace([np.inf, -np.inf], np.nan)

    elif col_m_ox and col_m_ox in df:
        m_dot_total = df[col_m_ox]
        df_calc['mass_flow_total'] = m_dot_total

    # 2. C* (Characteristic Velocity)
    ath_mm2 = geom.get('throat_area_mm2')
    if col_pc and m_dot_total is not None and ath_mm2 and col_pc in df:
        ath_m2 = ath_mm2 * 1e-6
        pc_pa = df[col_pc] * 1e5
        m_dot_kg = m_dot_total * 1e-3

        with np.errstate(divide='ignore', invalid='ignore'):
            df_calc['c_star'] = (pc_pa * ath_m2) / m_dot_kg
            # Filter noise near zero flow
            df_calc.loc[m_dot_total < 0.1, 'c_star'] = np.nan

    # 3. Isp (Specific Impulse)
    if col_thrust and m_dot_total is not None and col_thrust in df:
        thrust_n = df[col_thrust]
        m_dot_kg = m_dot_total * 1e-3

        with np.errstate(divide='ignore', invalid='ignore'):
            df_calc['isp'] = thrust_n / (m_dot_kg * g0)
            df_calc.loc[m_dot_total < 0.1, 'isp'] = np.nan

    # 4. Cf (Thrust Coefficient)
    if col_thrust and col_pc and ath_mm2 and col_thrust in df and col_pc in df:
        ath_m2 = ath_mm2 * 1e-6
        thrust_n = df[col_thrust]
        pc_pa = df[col_pc] * 1e5

        with np.errstate(divide='ignore', invalid='ignore'):
            df_calc['cf'] = thrust_n / (pc_pa * ath_m2)
            df_calc.loc[df[col_pc] < 0.1, 'cf'] = np.nan

    return df_calc


# ==========================================
# 2. THEORETICAL PROFILES (Time-Series)
# ==========================================
def calculate_theoretical_profile(df, config):
    """
    Generates theoretical traces (Ideal Thrust, Ideal Flow) based on
    measured Chamber Pressure and Target C*/Cf.
    """
    cols = config.get('columns', {})
    col_pc = cols.get('chamber_pressure')

    if not col_pc or col_pc not in df:
        return None

    # Get Parameters
    geom = config.get('geometry', {})
    targets = config.get('settings', {})

    at_mm2 = geom.get('throat_area_mm2', 0)
    c_star_target = targets.get('target_c_star', 0)
    cf_target = targets.get('target_cf', 0)

    if at_mm2 <= 0: return None

    # Prepare Data
    pc_pa = df[col_pc] * 1e5
    at_m2 = at_mm2 * 1e-6

    theo_data = pd.DataFrame(index=df.index)

    # 1. Theoretical Mass Flow (Ideal Injector/Combustion)
    if c_star_target > 0:
        # m_dot = (Pc * At) / C*
        m_dot_kg = (pc_pa * at_m2) / c_star_target
        theo_data['mass_flow_ideal'] = m_dot_kg * 1000.0

    # 2. Theoretical Thrust (Ideal Nozzle)
    if cf_target > 0:
        # F = Pc * At * Cf
        theo_data['thrust_ideal'] = pc_pa * at_m2 * cf_target

    return theo_data


# ==========================================
# 3. UNCERTAINTIES (Steady State)
# ==========================================
def calculate_uncertainties(avg_stats, config):
    # (Copy the code from our previous discussion here)
    # I will summarize it briefly to keep this file block complete
    u_map = config.get('uncertainties', {})
    geom = config.get('geometry', {})
    cols = config.get('columns', {})

    def get_u_abs(col_alias, value):
        sensor_id = cols.get(col_alias)
        if not sensor_id or sensor_id not in u_map: return 0.0
        u = u_map[sensor_id]
        return abs(value * u['value']) if u['type'] == 'rel' else u['value']

    errors = {}

    # Retrieve Averages
    col_ox = cols.get('mass_flow_ox')
    col_fu = cols.get('mass_flow_fuel')
    col_th = cols.get('thrust')
    col_pc = cols.get('chamber_pressure')

    m_ox = avg_stats.get(col_ox, 0)
    m_fu = avg_stats.get(col_fu, 0)
    thrust = avg_stats.get(col_th, 0)
    pc = avg_stats.get(col_pc, 0)

    m_tot = avg_stats.get('mass_flow_total', 0)
    c_star = avg_stats.get('c_star', 0)
    isp = avg_stats.get('isp', 0)

    # Calculate Errors
    if col_pc: errors['chamber_pressure'] = get_u_abs('chamber_pressure', pc)
    if col_th: errors['thrust'] = get_u_abs('thrust', thrust)

    u_ox = get_u_abs('mass_flow_ox', m_ox)
    u_fu = get_u_abs('mass_flow_fuel', m_fu)

    if m_tot > 0:
        u_m_tot = np.sqrt(u_ox ** 2 + u_fu ** 2)
        errors['mass_flow_total'] = u_m_tot

        # Isp Error
        if isp > 0 and thrust > 0:
            rel_isp = np.sqrt((get_u_abs('thrust', thrust) / thrust) ** 2 + (u_m_tot / m_tot) ** 2)
            errors['isp'] = isp * rel_isp

        # C* Error
        at = geom.get('throat_area_mm2', 0)
        u_at = geom.get('throat_area_uncertainty_mm2', 0)
        if c_star > 0 and pc > 0 and at > 0:
            rel_c = np.sqrt((get_u_abs('chamber_pressure', pc) / pc) ** 2 + (u_at / at) ** 2 + (u_m_tot / m_tot) ** 2)
            errors['c_star'] = c_star * rel_c

    return errors


# ==========================================
# 4. DERIVED METRICS (Steady State)
# ==========================================
def calculate_derived_metrics(avg_stats, config):
    derived = {}
    geom = config.get('geometry', {})
    fluid = config.get('fluid', {})
    targets = config.get('settings', {})

    # Efficiencies
    if 'c_star' in avg_stats and targets.get('target_c_star'):
        derived['η C* (%)'] = (avg_stats['c_star'] / targets['target_c_star']) * 100.0
    if 'cf' in avg_stats and targets.get('target_cf'):
        derived['η Cf (%)'] = (avg_stats['cf'] / targets['target_cf']) * 100.0
    if 'isp' in avg_stats and targets.get('target_isp'):
        derived['η Isp (%)'] = (avg_stats['isp'] / targets['target_isp']) * 100.0

    # Cold Flow Cd
    def calc_cd(m_dot, press_drop_bar, density, area_mm2):
        if area_mm2 <= 0 or density <= 0 or press_drop_bar <= 0: return None
        return (m_dot * 1e-3) / ((area_mm2 * 1e-6) * (2 * press_drop_bar * 1e5 * density) ** 0.5)

    if 'mass_flow_ox' in avg_stats:
        p_out = avg_stats.get(config['columns'].get('chamber_pressure'), 0.0)
        p_in = avg_stats.get(config['columns'].get('inlet_pressure_ox'), 0.0)
        cd = calc_cd(avg_stats['mass_flow_ox'], p_in - p_out, fluid.get('ox_density_kg_m3'),
                     geom.get('ox_injector_area_mm2'))
        if cd: derived['Cd (Ox)'] = cd

    return derived