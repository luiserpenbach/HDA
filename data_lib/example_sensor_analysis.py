import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from data_lib.sensor_data_tools import analyze_stability_fft, integrate_data, find_steady_window_dense, \
    smooth_signal_savgol, \
    resample_data, generate_pdf_report

# ---------------------------------------------------------
# EXAMPLE EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":

    # --- CONFIGURATION ---
    FILENAME = "../igniter_testing/local_data/IGN-CF-C1-003_raw.csv"
    OUTPUT_PDF = "Test_Report_003.pdf"

    TARGET_FLOW = '10009'  # Mass Flow Column
    TARGET_PRESS = '10003'  # Pressure Column
    FREQ = '10ms'  # Resampling Frequency (100Hz)
    FREQ_MS = 10

    # 1. Load & Preprocess
    print("Loading and cleaning data...")
    df_raw = pd.read_csv(FILENAME)
    df = resample_data(df_raw, freq=FREQ)
    df.index.name = 'timestamp'
    print(df.head())

    # 2. Analysis
    df['flow_smooth'] = smooth_signal_savgol(df, TARGET_FLOW, window=51, polyorder=3)
    bounds, cv_trace = find_steady_window_dense(df, 'flow_smooth',
                                                window_sec=1.0,
                                                threshold_pct=0.5,
                                                freq_ms=FREQ_MS)
    # Calculate Stats
    stats = {
        'total_mass': 0, 'steady_mass': 0, 'duration': 0,
        'avg_flow': 0, 'avg_press': 0, 'avg_cv': 0
    }
    # Active Mass
    df_active = df[df['flow_smooth'] > 0.5].reset_index()
    stats['total_mass'] = integrate_data(df_active, 'flow_smooth')

    if bounds:
        t_start, t_end = bounds
        steady_mask = (df.index >= t_start) & (df.index <= t_end)

        stats['steady_mass'] = integrate_data(df.reset_index(), 'flow_smooth', t_start=t_start, t_end=t_end)
        stats['avg_flow'] = df.loc[steady_mask, 'flow_smooth'].mean()
        stats['avg_press'] = df.loc[steady_mask, TARGET_PRESS].mean()
        stats['duration'] = (t_end - t_start).total_seconds()

        # Calculate avg CV in the stable region
        # Align cv_trace with df index if it's a series, or re-calculate
        # cv_trace computed on same index
        stats['avg_cv'] = cv_trace[steady_mask].mean()

        # C. Generate Report
        results = {
            'bounds': bounds,
            'cv_trace': cv_trace,
            'stats': stats,
            'col_flow': TARGET_FLOW,
            'col_press': TARGET_PRESS
        }
        generate_pdf_report(df, FILENAME, OUTPUT_PDF, results)