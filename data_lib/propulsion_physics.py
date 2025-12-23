import numpy as np
import pandas as pd

# --- CONSTANTS ---
g0 = 9.80665


# ==========================================
# 1. HOT FIRE TIME-SERIES (Heavy Physics)
# ==========================================
def calculate_hot_fire_series(df, config):
    """
    Applies Combustion Physics (Isp, C*, O/F) to the time-series DataFrame.
    Only call this for Hot Fire tests.
    """
    df_calc = df.copy()
    cols = config.get('columns', {})
    geom = config.get('geometry', {})

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

    # 3. Isp & Cf
    if col_thrust and m_dot_total is not None and col_thrust in df:
        thrust_n = df[col_thrust]
        m_dot_kg = m_dot_total * 1e-3
        with np.errstate(divide='ignore', invalid='ignore'):
            df_calc['isp'] = thrust_n / (m_dot_kg * g0)

            if col_pc and ath_mm2 and col_pc in df:
                ath_m2 = ath_mm2 * 1e-6
                pc_pa = df[col_pc] * 1e5
                df_calc['cf'] = thrust_n / (pc_pa * ath_m2)

    return df_calc


# ==========================================
# 2. COLD FLOW METRICS (Single Point)
# ==========================================
def calculate_cold_flow_metrics(avg_stats, config):
    """
    Calculates Cd based on average values.
    Strictly follows: Fluid/Area from Config, Pressures/Flow from Columns.
    """
    derived = {}

    # 1. Extract Config Parameters
    fluid = config.get('fluid', {})
    geom = config.get('geometry', {})
    cols = config.get('columns', {})

    # Prioritize 'density_kg_m3' key, fallback to others
    density = fluid.get('density_kg_m3') or fluid.get('ox_density_kg_m3') or fluid.get('water_density_kg_m3')

    # Prioritize 'orifice_area_mm2', fallback to injector area
    area_mm2 = geom.get('orifice_area_mm2') or geom.get('injector_area_mm2')

    # 2. Extract Sensor Data
    # Look for explicitly named 'upstream'/'downstream' keys, fallback to standard mapping
    k_up = cols.get('upstream_pressure') or cols.get('inlet_pressure' )or cols.get('injector_pressure')
    k_down = cols.get('downstream_pressure')
    k_flow = cols.get('mass_flow') or cols.get('mf')

    p_up = avg_stats.get(k_up)
    m_dot = avg_stats.get(k_flow)

    # "if no downstream_pressure given assume 0 bar"
    p_down = avg_stats.get(k_down, 0.0) if k_down else 0.0

    # 3. Calculate Cd
    if p_up is not None and m_dot is not None and density and area_mm2:
        if p_up > p_down and area_mm2 > 0 and density > 0:
            dp_bar = p_up - p_down

            # Formula: m_dot = Cd * A * sqrt(2 * rho * dP)
            # Cd = m_dot / (A * sqrt(2 * rho * dP))
            # Units: m_dot(kg/s), A(m2), P(Pa), rho(kg/m3)

            m_kg_s = m_dot * 1e-3
            a_m2 = area_mm2 * 1e-6
            dp_pa = dp_bar * 1e5

            denom = a_m2 * np.sqrt(2 * density * dp_pa)
            if denom > 0:
                derived['Cd'] = m_kg_s / denom
                derived['dP (bar)'] = dp_bar

    return derived


# ==========================================
# 3. HOT FIRE METRICS (Single Point)
# ==========================================
def calculate_hot_fire_metrics(avg_stats, config):
    """
    Calculates Efficiencies (Eta Isp, etc.) for Hot Fire.
    """
    derived = {}
    targets = config.get('settings', {})

    if 'c_star' in avg_stats and targets.get('target_c_star'):
        derived['η C* (%)'] = (avg_stats['c_star'] / targets['target_c_star']) * 100.0

    if 'isp' in avg_stats and targets.get('target_isp'):
        derived['η Isp (%)'] = (avg_stats['isp'] / targets['target_isp']) * 100.0

    return derived


# ==========================================
# 4. THEORETICAL PROFILES
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
# 5. UNCERTAINTIES
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


