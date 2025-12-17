import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams.update({
    "text.usetex": True,  # Enable LaTeX
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.labelsize": 20,
    "axes.titlesize": 16,
    "axes.grid": True,
    "grid.alpha": 0.2,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
})


def pre_process(df):
    '''
    - Save absolut start and end times
    - Convert time column to ms starting at zero
    '''

    # Save the start and end time of acquired data
    data_start_time = df['timestamp'][0]
    print("DAQ Start Timestamp: ", pd.to_datetime(df['timestamp'][0], unit='ms'))
    data_end_time = df['timestamp'].iloc[-1]
    print("DAQ End Timestamp: ", pd.to_datetime(df['timestamp'].iloc[-1], unit='ms'))

    # Shift start time to zero, convert to s.ms format
    df_processed = df.copy()
    df_processed['timestamp'] = ( df_processed['timestamp'] - df_processed['timestamp'][0] ) / 1000

    return df_processed, data_start_time, data_end_time

def get_valve_times(df):

    # Dynamically select timestamp and all valve columns (starting with 'V')
    valve_columns = ['timestamp'] + [col for col in df.columns if col.startswith('V')]
    df_valves = df[valve_columns]

    # Filtering rows where at least one valve is actuated (non-null)
    df_actuated = df_valves[df_valves[valve_columns[1:]].notnull().any(axis=1)]

    # Dropping columns where all values are NaN (valves that were never actuated)
    df_actuated = df_actuated.dropna(axis=1, how='all')

    return df_actuated

def plot_sensor_data(df, sensor_columns, test_id='sensor_plot', show_plot=True,
                     save_plot=False):
    """
    Generate plots for specified sensors and valves from test data with optional sample rate reduction

    Parameters:
    - df: pandas DataFrame containing the testbench data
    - sensor_columns: list of sensor column names to plot (e.g., ['P1', 'T2'])
    - reduce_sample_rate: integer factor to reduce sample rate (e.g., 10 keeps every 10th point, None for full data)
    - output_file_prefix: prefix for saved plot files (default: 'sensor_plot')
    - show_plot: whether to display the plot interactively (default: True)
    """


    # Validate sensor columns
    invalid_sensors = [col for col in sensor_columns if col not in df.columns]
    if invalid_sensors:
        raise ValueError(f"Sensor columns {invalid_sensors} not found in DataFrame")

    # Initialize sensor_columns if None 
    sensor_columns = sensor_columns or []

    # Find all valve columns (starting with 'V' followed by a number)
    valve_columns = [col for col in df.columns if col.startswith('V') and col[1:].isdigit()]

    # Validate sensor columns
    invalid_sensors = [col for col in sensor_columns if col not in df.columns]
    if invalid_sensors:
        raise ValueError(f"Sensor columns {invalid_sensors} not found in DataFrame")

    # Define color palettes (distinct for valves and sensors)
    valve_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    sensor_colors = ['#4c78a8', '#f58518', '#54a24b', '#e45756', '#b279a2',
                     '#9c755f', '#ff9da6', '#bab0ac', '#d4a71b', '#72b7b2']

    # Initialize the plot
    fig, ax = plt.subplots()

    # Plot sensor data as time series
    for idx, sensor_col in enumerate(sensor_columns):
        # Filter out NaN values for the sensor
        sensor_data = df[['timestamp', sensor_col]].dropna()
        if not sensor_data.empty:
            ax.plot(sensor_data['timestamp'], sensor_data[sensor_col],
                    color=sensor_colors[idx % len(sensor_colors)],
                    label=sensor_col, linewidth=2)

    # Plot valve actuation timestamps as vertical dotted lines
    for idx, valve_col in enumerate(valve_columns):
        # Filter rows where valve is non-zero and not NaN
        actuation_points = df[(df[valve_col] != 0) & (df[valve_col].notna())]
        # Get actuation timestamps
        timestamps = actuation_points['timestamp']

        # Plot vertical dotted lines for each actuation timestamp
        for ts in timestamps:
            ax.axvline(x=ts, color=valve_colors[idx % len(valve_colors)], linestyle=':',
                       label=valve_col if ts == timestamps.iloc[0] else None, alpha=0.7)

    # Customize the plot
    ax.set_xlabel('t / s')
    ax.set_title(f'{test_id} - Data', loc='left')
    ax.legend(title='Valves and Sensors')

    # Set y-axis label based on sensor data (generic if multiple sensors)
    ax.set_ylabel('Sensor Values' if sensor_columns else 'Valve Actuations')

    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    # Save the plot
    if save_plot:
        output_file = f"{test_id}_{'_'.join(sensor_columns)}.svg"
        plt.savefig(output_file)
        print(f"Plot saved as: {output_file}")

    # Show plot if requested
    if show_plot:
        plt.show()

    # Close the figure to free memory
    plt.close()

def trim_data_by_time(df, start_time, end_time, time_column='timestamp'):
    """
    Trim a DataFrame to include only rows between two timestamps

    Returns:
    - Trimmed DataFrame

    ADD:
    - if no end time or start time given than dont trim
    """

    # Trim the DataFrame
    trimmed_df = df[
        (df[time_column] >= start_time) &
        (df[time_column] <= end_time)
        ].copy()

    # Reset index for cleanliness
    trimmed_df = trimmed_df.reset_index(drop=True)

    print(f"Data was trimmed between {start_time} and {end_time}s: {trimmed_df.shape[0]} timestamps left.")
    return trimmed_df

def get_valve_act_timestamp(df, valve_channel, act_num, time_column='timestamp'):
    # Filter rows where the specified valve is actuated (non-null)
    df_actuated = df[df[valve_channel].notnull()][[time_column, valve_channel]]

    # Check if there are enough actuations
    if len(df_actuated) < act_num or act_num < 1:
        return None

    # Get the nth actuation timestamp (act_id is 1-based, so adjust to 0-based index)
    return df_actuated.iloc[act_num - 1][time_column]

#maybe unnecessary function (valve_act_timestamp + trim by time combined)
def trim_data_by_valveact(df, valve_channel, act_num, time_offset, trim_direction, time_column='timestamp'):
    # Trims test data to a specific valve actuation time with an offset
    # Trimming to left cuts out data left from valve act time!

    trim_time = get_valve_act_timestamp(df, valve_channel, act_num, time_column)

    if trim_direction=="left":
        # Trim the DataFrame
        trimmed_df = df[(df[time_column] >= trim_time+time_offset)].copy()
    elif trim_direction=="right":
        trimmed_df = df[(df[time_column] <= trim_time+time_offset)].copy()
    else:
        raise ValueError(f"Invalid trim direction: {trim_direction}")

    # Reset index for cleanliness
    trimmed_df = trimmed_df.reset_index(drop=True)

    print(f"Data was trimmed between {trimmed_df[time_column][0]} and {trimmed_df[time_column].iloc[-1]}: {trimmed_df.shape[0]} timestamps left.")
    return trimmed_df

def shift_t0_to(df, timestamp_target, time_column='timestamp'):
    """
    Move 0s timestamp to desired position
    """

    t0_time = df[time_column].iloc[timestamp_target]
    shifted_df = df[time_column] - t0_time
    return shifted_df

# Example usage
if __name__ == "__main__":
    # Load the data
    # Timestamps shall be in UNIX ms format!

    filename = "../injector_testing/local_data/INJ_B1_CFs_11122025_raw.csv"
    TEST_ID = "IGN-T001"
    df = pd.read_csv(filename)
    pp = pre_process(df)

    #print(get_valve_act_timestamp(df, "V4", 1, ))