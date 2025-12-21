import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_lib.propulsion_physics import calculate_derived_metrics
from data_lib.sensor_data_tools import (
    analyze_stability_fft, integrate_data, smooth_signal_savgol,
    resample_data, find_steady_window, calculate_rise_time, generate_html_report, generate_pdf_report
)

# ---------------------------------------------------------
# EXAMPLE EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":

    # --- CONFIGURATION ---
    FILENAME = "../igniter_testing/local_data/IGN-CF-C1-003_raw.csv"
    OUTPUT_PDF = "Test_Report_002.pdf"

    CH_FLOW = '10009'  # Mass Flow Column
    CH_PRESS = '10003'  # Pressure Column
    FREQ = 100  # Resampling Frequency, Hz
    FREQ_MS = 10 # Resampling Frequency, Hz


    # 1. Load & Preprocess
    df_raw = pd.read_csv(FILENAME)
    print(f"Raw rows: {len(df_raw)}")
    # Resample to strict 10ms grid (100 Hz)
    df = resample_data(df_raw, time_col='timestamp', freq_ms=FREQ_MS)
    print(f"Resampled rows: {len(df)}")

    # PLOT 1: Resampling Check
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_raw['timestamp'], y=df_raw[CH_FLOW],
                             mode='markers', name='Raw', marker=dict(color='silver', size=4)))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df[CH_FLOW],
                             mode='lines', name=f'Resampled ({FREQ_MS} ms)', marker=dict(size=3)))
    fig.update_layout(template="plotly_dark+gridon", title="Resampling Check", xaxis_title="Time / ms", yaxis_title="Mass flow / g/s")
    fig.show()



    # 2. Smooth
    df['flow_smooth'] = smooth_signal_savgol(df, CH_FLOW, window=21)

    # 3. Find Steady State
    bounds, cv_trace = find_steady_window(df, CH_PRESS,
                                          time_col='timestamp',
                                          window_ms=500,
                                          threshold_pct=1.0)

    # Initialize basic stats
    stats = {
        'total_mass': 0.0, 'duration': 0.0,
        'avg_flow': 0.0, 'avg_press': 0.0, 'steady_mass': 0.0, 'avg_cv': 0.0,
        'rise_time': 0.0  # New field
    }

    if bounds:
        t_start, t_end = bounds
        print(f"Steady State Detected: {t_start:.0f} ms to {t_end:.0f} ms")

        # Calc stability stats
        steady_mask = (df['timestamp'] >= t_start) & (df['timestamp'] <= t_end)
        steady_df = df[steady_mask]

        steady_mean_flow = steady_df[CH_FLOW].mean()
        stats['avg_flow'] = steady_mean_flow
        stats['avg_press'] = steady_df[CH_PRESS].mean()
        stats['avg_cv'] = cv_trace[steady_mask].mean()
        stats['steady_mass'] = integrate_data(df, CH_FLOW, 'timestamp', t_start, t_end)
        stats['total_mass'] = integrate_data(df, CH_FLOW, 'timestamp')
        stats['duration'] = (df['timestamp'].max() - df['timestamp'].min()) / 1000.0
        avg_cd_calc = calc_cd("H2O", stats['avg_flow'], np.pi/4*1.0**2, stats['avg_press']+1, 10.0)
        print(f"CD VALUE: {avg_cd_calc}")

        plt.figure(figsize=(10, 6))
        plt.subplot(2,1,1)
        plt.plot(df['timestamp'], df[CH_FLOW], color='lightgray', label='Raw')
        plt.plot(df['timestamp'], df['flow_smooth'], color='blue', label='Smoothed')
        plt.axvspan(t_start, t_end, color='green', alpha=0.2, label='Steady Window')
        plt.ylabel("Flow (g/s)")
        plt.legend()

        plt.subplot(2,1,2)
        plt.plot(df['timestamp'], cv_trace, color='orange', label='CV %')
        plt.axhline(1.0, color='red', linestyle='--')
        plt.ylabel("CV %")
        plt.xlabel("Time (ms)")
        plt.show()

        print(f"Total Mass in Steady Window: {stats['steady_mass']:.4f} g")



        # 4. Rise Time
        t10, t90, rise_s = calculate_rise_time(df, 'flow_smooth', t_start, steady_mean_flow)

        if rise_s:
            print(f"Rise Time (10-90%): {rise_s:.3f} s")
            stats['rise_time'] = rise_s

            # --- PLOT RISE TIME ---
            plt.figure(figsize=(10, 5))
            plt.plot(df['timestamp'], df[CH_FLOW], color='lightgray', label='Raw')
            plt.plot(df['timestamp'], df['flow_smooth'], color='blue', label='Smooth')

            # Draw lines for 10% and 90%
            plt.axvline(t10, color='orange', linestyle='--', label='10%')
            plt.axvline(t90, color='red', linestyle='--', label='90%')

            # Highlight the Rise Interval
            plt.axvspan(t10, t90, color='yellow', alpha=0.3, label=f'Rise: {rise_s:.3f}s')

            # Highlight Steady State
            plt.axvspan(t_start, t_end, color='green', alpha=0.1, label='Steady')

            plt.title("Flow Rise Time Analysis")
            plt.ylabel("Mass Flow")
            plt.xlabel("Time / ms")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
        else:
            print("Could not calculate rise time (signal might start high).")


        # 5. FFT Analysis
        steady_df = df[(df['timestamp'] >= t_start) & (df['timestamp'] <= t_end)]
        freqs, psd, peak = analyze_stability_fft(steady_df, CH_PRESS, time_col='timestamp')

        # PLOT 3: FFT
        fig_fft = go.Figure()
        fig_fft.add_trace(go.Scatter(x=freqs, y=psd, mode='lines', name='PSD', line=dict(color='purple')))
        fig_fft.update_layout(title=f"Combustion Stability (Peak: {peak:.1f} Hz)",
                              xaxis_title="Frequency (Hz)", yaxis_title="PSD", yaxis_type="log")
        fig_fft.show()


    results = {
        'bounds': bounds,
        'cv_trace': cv_trace,
        'stats': stats,
        'col_flow': CH_FLOW,
        'col_press': CH_PRESS
    }

    base_name = os.path.splitext(os.path.basename(FILENAME))[0]
    OUTPUT_PDF = f"Report_{base_name}.pdf"
    OUTPUT_HTML = f"Report_{base_name}.html"

    generate_html_report(df, FILENAME, OUTPUT_HTML, results)
    #generate_pdf_report(df, f"Test {base_name}", OUTPUT_PDF, results)