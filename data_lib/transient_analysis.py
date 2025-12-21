import pandas as pd
import numpy as np


def find_digital_edge(df, col_name, edge_type='rising', threshold=0.5):
    """
    Finds the timestamp of a digital signal edge (e.g. Valve Open).
    Assumes signal is roughly 0/1 or Low/High.
    """
    if col_name not in df:
        return None

    # Normalize signal to 0-1 range to handle 5V or 24V logic
    sig = df[col_name]
    sig_norm = (sig - sig.min()) / (sig.max() - sig.min())

    # Calculate difference
    diff = sig_norm.diff()

    # Find index where difference is large
    if edge_type == 'rising':
        events = df[diff > threshold]
    else:  # falling
        events = df[diff < -threshold]

    if not events.empty:
        return events.iloc[0]['timestamp']
    return None


def detect_transient_events(df, config, steady_bounds=None):
    """
    Analyzes the timeline to find key propulsion events.

    Returns a dictionary:
    {
        't_zero': 10050,         # Valve Open (ms)
        't_ignition': 10080,     # Pressure > 10% (ms)
        'ignition_delay_ms': 30, # Delta
        't_shutdown': 15000,     # Valve Close (ms)
        'shutdown_impulse': 50.5 # Ns
    }
    """
    events = {}
    col_map = config.get('columns', {})

    # 1. FIND T-0 (Fire Command / Valve Open)
    # We look for a 'fire_cmd' column in the config, or user can select it
    col_fire = col_map.get('fire_command')  # Needs to be added to config!

    t0 = None
    if col_fire:
        t0 = find_digital_edge(df, col_fire, 'rising')
        if t0 is not None:
            events['t_zero'] = t0

    # 2. FIND IGNITION (Pressure Rise)
    # Defined as: Chamber Pressure crosses 10% of Steady State Average
    # If steady state isn't defined, use max pressure as reference.
    col_pc = col_map.get('chamber_pressure')

    if col_pc and col_pc in df:
        # Determine target pressure (10% of steady)
        if steady_bounds:
            s_start, s_end = steady_bounds
            steady_mean = df[(df['timestamp'] >= s_start) & (df['timestamp'] <= s_end)][col_pc].mean()
        else:
            steady_mean = df[col_pc].max()  # Fallback

        p_threshold = 0.10 * steady_mean

        # Search for crossing
        # If T0 exists, search only AFTER T0
        search_mask = df['timestamp'] > (t0 if t0 else df['timestamp'].min())
        candidates = df[search_mask & (df[col_pc] > p_threshold)]

        if not candidates.empty:
            t_ign = candidates.iloc[0]['timestamp']
            events['t_ignition'] = t_ign

            # Calculate Delay
            if t0:
                events['ignition_delay_ms'] = t_ign - t0

    # 3. SHUTDOWN IMPULSE (Total Impulse after Cut-off)
    # Defined as Integral of Thrust from Valve Close until Thrust < 1%
    col_thrust = col_map.get('thrust')

    # Try to find cut-off signal
    t_cut = None
    if col_fire:
        t_cut = find_digital_edge(df, col_fire, 'falling')
        if t_cut:
            events['t_shutdown'] = t_cut

    # If we have a cut time and thrust data, calculate impulse
    if t_cut and col_thrust and col_thrust in df:
        # Slice data from Cut-off onwards
        tail_df = df[df['timestamp'] >= t_cut].copy()

        # Convert to seconds for integration
        # Integral F dt (where t is seconds)
        # We need to ensure we don't integrate offset drift forever
        # Stop when thrust drops below 0.5% of max (or steady)

        f_max = df[col_thrust].max()
        cutoff_thresh = 0.005 * f_max

        # Filter down to the "tail"
        active_tail = tail_df[tail_df[col_thrust] > cutoff_thresh]

        if not active_tail.empty:
            # Integrate
            # Trapz expects x in seconds for Result in N-s
            y = active_tail[col_thrust].values
            x = active_tail['timestamp'].values / 1000.0

            impulse = np.trapz(y, x)
            events['shutdown_impulse_ns'] = impulse
            events['t_tail_end'] = active_tail['timestamp'].max()

    return events