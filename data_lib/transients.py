
def analyze_rise_time(df, col_name, t_steady_start, t_steady_end):
    """
    Calculates the 10-90% rise time of a signal.
    For example determining valve opening speeds and ignition delay.
    """
    # 1. Determine Target "100%" Value (Steady State Average)
    mask = (df.index >= t_steady_start) & (df.index <= t_steady_end)
    steady_val = df.loc[mask, col_name].mean()

    # 2. Define Thresholds
    val_10 = 0.10 * steady_val
    val_90 = 0.90 * steady_val

    # 3. Find timestamps
    # Look at data BEFORE the steady state
    pre_steady = df[df.index < t_steady_start]

    try:
        # Get the LAST time it crossed these thresholds (to ignore pre-test noise)
        t_10 = pre_steady[pre_steady[col_name] >= val_10].index[-1]
        t_90 = pre_steady[pre_steady[col_name] >= val_90].index[-1]

        rise_time_ms = (t_90 - t_10).total_seconds() * 1000
        return rise_time_ms, t_10, t_90

    except IndexError:
        return None, None, None