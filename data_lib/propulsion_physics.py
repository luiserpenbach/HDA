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

    # Retrieve mapped column names
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
# 2. THEORETICAL PROFILES
# ==========================================
def calculate_theoretical_profile(df, config):
    cols = config.get('columns', {})
    col_pc = cols.get('chamber_pressure')
    if not col_pc or col_pc not in df: return None

    geom = config.get('geometry', {})
    targets = config.get('settings', {})
    at_mm2 = geom.get('throat_area_mm2', 0)
    c_star_target = targets.get('target_c_star', 0)
    cf_target = targets.get('target_cf', 0)

    if at_mm2 <= 0: return None

    pc_pa = df[col_pc] * 1e5
    at_m2 = at_mm2 * 1e-6

    theo_data = pd.DataFrame(index=df.index)
    if c_star_target > 0:
        theo_data['mass_flow_ideal'] = (pc_pa * at_m2) / c_star_target * 1000.0
    if cf_target > 0:
        theo_data['thrust_ideal'] = pc_pa * at_m2 * cf_target

    return theo_data


# ==========================================
# 3. UNCERTAINTIES
# ==========================================
def calculate_uncertainties(avg_stats, config):
    u_map = config.get('uncertainties', {})
    geom = config.get('geometry', {})
    cols = config.get('columns', {})

    def get_u_abs(col_alias, value):
        sensor_id = cols.get(col_alias)
        if not sensor_id or sensor_id not in u_map: return 0.0
        u = u_map[sensor_id]
        return abs(value * u['value']) if u['type'] == 'rel' else u['value']

    errors = {}

    # Retrieve values safely
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

    # Calculate
    if col_pc: errors['chamber_pressure'] = get_u_abs('chamber_pressure', pc)
    if col_th: errors['thrust'] = get_u_abs('thrust', thrust)

    u_ox = get_u_abs('mass_flow_ox', m_ox)
    u_fu = get_u_abs('mass_flow_fuel', m_fu)

    if m_tot > 0:
        u_m_tot = np.sqrt(u_ox ** 2 + u_fu ** 2)
        errors['mass_flow_total'] = u_m_tot

        if isp > 0 and thrust > 0:
            rel_isp = np.sqrt((get_u_abs('thrust', thrust) / thrust) ** 2 + (u_m_tot / m_tot) ** 2)
            errors['isp'] = isp * rel_isp

        at = geom.get('throat_area_mm2', 0)
        u_at = geom.get('throat_area_uncertainty_mm2', 0)
        if c_star > 0 and pc > 0 and at > 0:
            rel_c = np.sqrt((get_u_abs('chamber_pressure', pc) / pc) ** 2 + (u_at / at) ** 2 + (u_m_tot / m_tot) ** 2)
            errors['c_star'] = c_star * rel_c

    return errors


# ==========================================
# 4. DERIVED METRICS (Robust for Cold Flow)
# ==========================================
def calculate_derived_metrics(avg_stats, config):
    derived = {}
    geom = config.get('geometry', {})
    fluid = config.get('fluid', {})
    targets = config.get('settings', {})
    cols = config.get('columns', {})

    # Hot Fire Efficiencies
    if 'c_star' in avg_stats and targets.get('target_c_star'):
        derived['η C* (%)'] = (avg_stats['c_star'] / targets['target_c_star']) * 100.0
    if 'cf' in avg_stats and targets.get('target_cf'):
        derived['η Cf (%)'] = (avg_stats['cf'] / targets['target_cf']) * 100.0
    if 'isp' in avg_stats and targets.get('target_isp'):
        derived['η Isp (%)'] = (avg_stats['isp'] / targets['target_isp']) * 100.0

    # Cold Flow Cd Helper
    def calc_cd(m_dot, press_drop_bar, density, area_mm2):
        if area_mm2 <= 0 or density <= 0 or press_drop_bar <= 0: return None
        # m_dot (g/s -> kg/s), Area (mm2 -> m2), P (bar -> Pa)
        return (m_dot * 1e-3) / ((area_mm2 * 1e-6) * (2 * press_drop_bar * 1e5 * density) ** 0.5)

    # Cd Ox (Robust check)
    col_ox_flow = cols.get('mass_flow_ox')
    if col_ox_flow and col_ox_flow in avg_stats:
        p_out = avg_stats.get(cols.get('chamber_pressure'), 0.0)
        p_in = avg_stats.get(cols.get('inlet_pressure_ox'),
                             p_out)  # If no inlet, assume P_out=P_in (user err) or check mapping

        # Cold Flow assumption: 'chamber_pressure' usually maps to the upstream sensor in simple configs,
        # or we assume P_out is ambient (0 gauge).
        # Let's use strict config mapping: user must map 'inlet_pressure_ox' for Cd.

        if 'inlet_pressure_ox' in cols:
            p_in = avg_stats.get(cols['inlet_pressure_ox'], 0.0)
            # Delta P = Pin - Pout (if Pout exists, else 0 gauge)
            dp = p_in - p_out if p_in > p_out else p_in

            cd = calc_cd(avg_stats[col_ox_flow], dp, fluid.get('ox_density_kg_m3'), geom.get('ox_injector_area_mm2'))
            if cd: derived['Cd (Ox)'] = cd

    # Cd Fuel
    col_fu_flow = cols.get('mass_flow_fuel')
    if col_fu_flow and col_fu_flow in avg_stats:
        # Similar logic for fuel
        pass

    return derived