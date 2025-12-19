import base64
import io
import dash
from dash import dcc, html, Input, Output, State, ctx, ALL, dash_table
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import plotly.io as pio  # For export

# Initialize the Dash app
app = dash.Dash(__name__, title="Hop Data Plotter")

# --- Enhanced CSS Styles for Visual Appeal ---
LAYOUT_STYLE = {
    "display": "flex",
    "flexDirection": "row",
    "height": "100vh",
    "overflow": "hidden",
    "fontFamily": "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
    "backgroundColor": "#121212",  # Dark background
    "color": "#e0e0e0"  # Light text
}

SIDEBAR_STYLE = {
    "width": "35rem",
    "minWidth": "300px",
    "maxWidth": "50vw",
    "padding": "2rem 1.5rem",
    "backgroundColor": "#1e1e1e",  # Darker sidebar
    "overflow": "auto",
    "resize": "horizontal",
    "borderRight": "3px solid #333",  # Subtle separator
    "display": "flex",
    "flexDirection": "column",
    "boxShadow": "2px 0 5px rgba(0,0,0,0.5)",  # Shadow for depth
}

CONTENT_STYLE = {
    "flex": "1",
    "padding": "2rem 1.5rem",
    "overflowY": "auto",
    "backgroundColor": "#121212",
}

BUTTON_STYLE = {
    'width': '100%',
    'height': '45px',
    'backgroundColor': '#007bff',
    'color': 'white',
    'border': 'none',
    'borderRadius': '5px',
    'fontWeight': 'bold',
    'cursor': 'pointer',
    'marginTop': '20px',
    'transition': 'background-color 0.3s'
}

UPLOAD_STYLE = {
    'width': '100%',
    'height': '60px',
    'lineHeight': '60px',
    'borderWidth': '1px',
    'borderStyle': 'dashed',
    'borderRadius': '5px',
    'borderColor': '#007bff',
    'textAlign': 'center',
    'margin-bottom': '20px',
    'backgroundColor': '#2a2a2a',
    'color': '#e0e0e0'
}

LABEL_STYLE = {
    'fontWeight': 'bold',
    'marginBottom': '8px',
    'color': '#bbbbbb'
}

INPUT_STYLE = {
    'width': '100%',
    'height': '35px',
    'backgroundColor': '#2a2a2a',
    'color': '#e0e0e0',
    'border': '1px solid #444',
    'borderRadius': '4px',
    'padding': '0 10px',
    'marginBottom': '15px'
}

CLEAR_BUTTON_STYLE = {
    'width': '100%',
    'height': '35px',
    'backgroundColor': '#dc3545',
    'color': 'white',
    'border': 'none',
    'borderRadius': '5px',
    'fontWeight': 'bold',
    'cursor': 'pointer',
    'marginTop': '10px'
}

# --- Layout Definition ---
sidebar = html.Div(
    [
        html.H2("Controls", style={'color': '#ffffff', 'marginBottom': '20px'}),
        html.Hr(style={'borderColor': '#444'}),

        # 1. File Uploader
        html.Label("1. Test Data File", style=LABEL_STYLE),
        dcc.Upload(
            id='upload-data',
            children=html.Div(['Drag and Drop or ', html.A('Select CSV', style={'color': '#007bff'})]),
            style=UPLOAD_STYLE,
            multiple=False
        ),

        # Plot Mode
        html.Label("Plot Mode", style=LABEL_STYLE),
        dcc.RadioItems(
            id='plot-mode',
            options=[
                {'label': 'Overlay', 'value': 'overlay'},
                {'label': 'Subplots', 'value': 'subplots'}
            ],
            value='overlay',
            labelStyle={'display': 'block', 'marginBottom': '10px', 'color': '#e0e0e0'},
            style={'margin-bottom': '20px'}
        ),

        # 2. Channel Selection
        html.Label("2. Select Channels", style=LABEL_STYLE),
        dcc.Dropdown(
            id='channel-dropdown',
            multi=True,
            placeholder="Upload file first...",
            style={'margin-bottom': '20px', 'backgroundColor': '#2a2a2a', 'color': '#e0e0e0', 'border': '1px solid #444'}
        ),

        # Axis Assignments (only for overlay)
        html.Div(id='axis-assignments-container', children=[
            html.Label("Axis Assignment", style=LABEL_STYLE),
            html.Div(id='axis-assignments', style={'margin-bottom': '20px'}),
        ]),

        # 3. Snipping Tools
        html.Label("3. Snipping Mode", style=LABEL_STYLE),
        dcc.RadioItems(
            id='snip-mode',
            options=[
                {'label': 'Full Data', 'value': 'full'},
                {'label': 'Manual Range', 'value': 'manual'},
                {'label': 'Valve Event', 'value': 'valve'}
            ],
            value='full',
            labelStyle={'display': 'block', 'marginBottom': '10px', 'color': '#e0e0e0'},
            style={'margin-bottom': '20px'}
        ),

        # Manual Inputs
        html.Div(id='manual-controls', children=[
            html.Label("Start Time (s)", style=LABEL_STYLE),
            dcc.Input(id='start-time', type='number', value=0, style=INPUT_STYLE),
            html.Label("End Time (s)", style=LABEL_STYLE),
            dcc.Input(id='end-time', type='number', value=100, style=INPUT_STYLE),
        ], style={'display': 'none'}),

        # Valve Event Inputs
        html.Div(id='valve-controls', children=[
            html.Label("Trigger Channel", style=LABEL_STYLE),
            dcc.Dropdown(id='trigger-channel', placeholder="Select trigger...", style={'margin-bottom': '15px', 'backgroundColor': '#2a2a2a', 'color': '#e0e0e0', 'border': '1px solid #444'}),
            html.Label("Threshold", style=LABEL_STYLE),
            dcc.Input(id='threshold', type='number', value=3.0, style=INPUT_STYLE),
            html.Label("Pre-Buffer (s)", style=LABEL_STYLE),
            dcc.Input(id='pre-buffer', type='number', value=5.0, style=INPUT_STYLE),
            html.Label("Post-Buffer (s)", style=LABEL_STYLE),
            dcc.Input(id='post-buffer', type='number', value=20.0, style=INPUT_STYLE),
        ], style={'display': 'none'}),

        html.Button('Update Plot', id='update-btn', n_clicks=0, style=BUTTON_STYLE),
        html.Button('Clear Annotations', id='clear-annot-btn', n_clicks=0, style=CLEAR_BUTTON_STYLE),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(
    [
        html.H1("Hopper General Data Plotter", style={'text-align': 'center', 'color': '#ffffff', 'fontSize': 40, 'marginBottom': '30px'}),
        dcc.Loading(
            id="loading-1",
            type="default",
            children=html.Div([
                dcc.Graph(id='main-graph', style={'height': '70vh'}, config={'responsive': True, 'displayModeBar': True}),
                html.Div(id='status-msg', style={'margin-top': '15px', 'color': '#888888', 'textAlign': 'center'}),
                html.Button('Export PNG', id='export-btn', style={**BUTTON_STYLE, 'width': '200px', 'margin': '20px auto', 'display': 'block'}),
                html.Details([
                    html.Summary('Data Statistics', style={'color': '#ffffff', 'cursor': 'pointer', 'marginTop': '20px'}),
                    dash_table.DataTable(
                        id='stats-table',
                        style_table={'overflowX': 'auto', 'backgroundColor': '#1e1e1e'},
                        style_cell={'backgroundColor': '#1e1e1e', 'color': '#e0e0e0', 'border': '1px solid #444'},
                        style_header={'backgroundColor': '#2a2a2a', 'fontWeight': 'bold', 'border': '1px solid #444'}
                    )
                ], style={'margin': '20px auto', 'width': '80%'})
            ])
        ),
        # Hidden download component for export
        dcc.Download(id="download-image"),
        # Store for annotations (list of x values)
        dcc.Store(id='annotations', data=[]),
        # Store for current figure (to update on clicks)
        dcc.Store(id='current-figure'),
        # Store for data
        dcc.Store(id='stored-data-meta')
    ],
    style=CONTENT_STYLE,
)

app.layout = html.Div(
    [sidebar, content],
    style=LAYOUT_STYLE
)


# --- Helper to parse upload ---
def parse_contents(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

        # Auto-detect timestamp column (more robust)
        time_cols = [c for c in df.columns if 'time' in c.lower() or 'timestamp' in c.lower()]
        time_col = time_cols[0] if time_cols else df.columns[0]  # Fallback to first col

        df = df.sort_values(time_col)
        df['relative_time'] = (df[time_col] - df[time_col].iloc[0]) / 1000.0 if 'timestamp' in time_col.lower() else df.index

        df = df.ffill()
        return df
    except Exception as e:
        print(e)
        return None


# --- Callbacks ---

# 1. Update Visibility of Snipping Controls
@app.callback(
    [Output('manual-controls', 'style'), Output('valve-controls', 'style')],
    Input('snip-mode', 'value')
)
def toggle_controls(mode):
    show = {'display': 'block'}
    hide = {'display': 'none'}
    if mode == 'manual':
        return show, hide
    elif mode == 'valve':
        return hide, show
    return hide, hide


# Toggle Axis Assignments Visibility (hide for subplots)
@app.callback(
    Output('axis-assignments-container', 'style'),
    Input('plot-mode', 'value')
)
def toggle_axis_assign(mode):
    if mode == 'subplots':
        return {'display': 'none'}
    return {'display': 'block'}


# 2. Update Dropdown Options when File Uploaded
@app.callback(
    [Output('channel-dropdown', 'options'),
     Output('channel-dropdown', 'value'),
     Output('trigger-channel', 'options')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_dropdowns(contents, filename):
    if contents is None:
        return [], [], []

    df = parse_contents(contents)
    if df is None:
        return [], [], []

    cols = [c for c in df.columns if c not in ['timestamp', 'relative_time', 'testCaseId']]

    # Default selection: first 3 non-time columns
    default_val = cols[:3] if len(cols) >= 3 else cols

    options = [{'label': c, 'value': c} for c in cols]
    return options, default_val, options


# 3. Dynamic Axis Assignment Dropdowns
@app.callback(
    Output('axis-assignments', 'children'),
    Input('channel-dropdown', 'value')
)
def render_axis_dropdowns(channels):
    if not channels:
        return []

    children = []
    for channel in channels:
        children.append(html.Label(f"Axis for {channel}", style=LABEL_STYLE))
        children.append(dcc.Dropdown(
            id={'type': 'axis-dropdown', 'index': channel},
            options=[
                {'label': 'Left', 'value': 'left'},
                {'label': 'Right', 'value': 'right'}
            ],
            value='left',  # Default to left
            style={'margin-bottom': '15px', 'backgroundColor': '#2a2a2a', 'color': '#e0e0e0', 'border': '1px solid #444'}
        ))
    return children


# 4. Main Plotting Callback (now with stats and plot mode)
@app.callback(
    [Output('main-graph', 'figure'), Output('status-msg', 'children'),
     Output('stats-table', 'data'), Output('stats-table', 'columns'),
     Output('current-figure', 'data')],
    [Input('update-btn', 'n_clicks')],
    [State('upload-data', 'contents'),
     State('plot-mode', 'value'),
     State('channel-dropdown', 'value'),
     State({'type': 'axis-dropdown', 'index': ALL}, 'value'),
     State('snip-mode', 'value'),
     State('start-time', 'value'),
     State('end-time', 'value'),
     State('trigger-channel', 'value'),
     State('threshold', 'value'),
     State('pre-buffer', 'value'),
     State('post-buffer', 'value'),
     State('annotations', 'data')]
)
def update_graph(n_clicks, contents, plot_mode, channels, axis_values, mode, t_start, t_end, trig_col, thresh, pre_b, post_b, annotations):
    if contents is None:
        return go.Figure(), "Waiting for file upload...", [], [], None

    df = parse_contents(contents)

    if not channels:
        return go.Figure(), "No channels selected.", [], [], None

    # Map axes to channels (only for overlay)
    axis_assign = dict(zip(channels, axis_values)) if plot_mode == 'overlay' else {}

    # --- Snipping Logic ---
    msg = f"Plotting {len(channels)} channels in {plot_mode} mode."

    snipped_df = df.copy()  # For stats, keep full resolution

    if mode == 'manual':
        mask = (snipped_df['relative_time'] >= (t_start or 0)) & (snipped_df['relative_time'] <= (t_end or snipped_df['relative_time'].max()))
        snipped_df = snipped_df.loc[mask]
        msg += f" (Manual Range: {t_start}s to {t_end}s)"

    elif mode == 'valve':
        if trig_col and trig_col in snipped_df.columns:
            events = snipped_df[snipped_df[trig_col] > (thresh or 0)]
            if not events.empty:
                event_time = events.iloc[0]['relative_time']
                start_win = event_time - (pre_b or 0)
                end_win = event_time + (post_b or 0)

                mask = (snipped_df['relative_time'] >= start_win) & (snipped_df['relative_time'] <= end_win)
                snipped_df = snipped_df.loc[mask].copy()

                # Re-zero time to event
                snipped_df['relative_time'] -= event_time
                msg += f" (Valve Event at {event_time:.2f}s)"
            else:
                msg += " (No event found, showing full data)"

    # --- Compute Stats on snipped_df (full res) ---
    stats_data = []
    for col in channels:
        series = snipped_df[col]
        stats_data.append({
            'Channel': col,
            'Min': series.min(),
            'Max': series.max(),
            'Mean': series.mean(),
            'Std': series.std()
        })
    stats_columns = [
        {'name': 'Channel', 'id': 'Channel'},
        {'name': 'Min', 'id': 'Min'},
        {'name': 'Max', 'id': 'Max'},
        {'name': 'Mean', 'id': 'Mean'},
        {'name': 'Std', 'id': 'Std'}
    ]

    # --- Performance Optimization for Plot (downsample after stats) ---
    df_plot = snipped_df.copy()
    if len(df_plot) > 15000:
        step = len(df_plot) // 15000 + 1
        df_plot = df_plot.iloc[::step]
        msg += f" [Downsampled {step}x for performance]"

    # --- Plotting ---
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']  # Nice color palette

    if plot_mode == 'subplots':
        fig = make_subplots(rows=len(channels), cols=1, shared_xaxes=True, vertical_spacing=0.05)
        for i, col in enumerate(channels):
            fig.add_trace(go.Scattergl(
                x=df_plot['relative_time'],
                y=df_plot[col],
                mode='lines',
                name=col,
                line={'color': colors[i % len(colors)]}
            ), row=i+1, col=1)
            fig.update_yaxes(title_text=col, row=i+1, col=1)
        fig.update_layout(height=200 * len(channels))  # Adjust height dynamically

    else:  # overlay
        fig = go.Figure()
        has_right = any(axis_assign.get(col, 'left') == 'right' for col in channels)
        for i, col in enumerate(channels):
            yaxis = 'y2' if axis_assign.get(col, 'left') == 'right' else 'y1'
            fig.add_trace(go.Scattergl(
                x=df_plot['relative_time'],
                y=df_plot[col],
                mode='lines',
                name=col,
                line={'color': colors[i % len(colors)]},
                yaxis=yaxis
            ))
        if has_right:
            fig.update_layout(yaxis2={'title': "Right Axis Value", 'overlaying': 'y', 'side': 'right'})

    # Common layout
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Time (s)",
        yaxis_title="Left Axis Value" if plot_mode == 'overlay' else None,
        hovermode="x unified",
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#121212'
    )

    # Add annotations as vertical lines
    shapes = []
    for x in annotations:
        shapes.append({
            'type': 'line',
            'x0': x, 'x1': x,
            'y0': 0, 'y1': 1,
            'yref': 'paper', 'xref': 'x',
            'line': {'color': 'red', 'width': 2, 'dash': 'dash'}
        })
    fig.update_layout(shapes=shapes)

    return fig, msg, stats_data, stats_columns, fig.to_dict()  # Store current figure dict


# Annotation Append on Click
@app.callback(
    Output('annotations', 'data'),
    Input('main-graph', 'clickData'),
    State('annotations', 'data')
)
def add_annotation(clickData, annotations):
    if clickData:
        x = clickData['points'][0]['x']
        if x not in annotations:
            annotations.append(x)
    return annotations


# Clear Annotations
@app.callback(
    Output('annotations', 'data', allow_duplicate=True),
    Input('clear-annot-btn', 'n_clicks'),
    prevent_initial_call=True
)
def clear_annotations(n_clicks):
    return []


# Update Figure with Annotations (reactive to annotations change)
@app.callback(
    Output('main-graph', 'figure', allow_duplicate=True),
    Input('annotations', 'data'),
    State('current-figure', 'data'),
    prevent_initial_call=True
)
def update_figure_annotations(annotations, current_fig):
    if current_fig:
        fig = go.Figure(current_fig)
        shapes = []
        for x in annotations:
            shapes.append({
                'type': 'line',
                'x0': x, 'x1': x,
                'y0': 0, 'y1': 1,
                'yref': 'paper', 'xref': 'x',
                'line': {'color': 'red', 'width': 2, 'dash': 'dash'}
            })
        fig.update_layout(shapes=shapes)
        return fig
    return dash.no_update


# 5. Export Callback
@app.callback(
    Output("download-image", "data"),
    Input("export-btn", "n_clicks"),
    State("main-graph", "figure"),
    prevent_initial_call=True,
)
def export_png(n_clicks, figure):
    if figure:
        img_bytes = pio.to_image(figure, format="png", width=1200, height=800)
        return dcc.send_bytes(img_bytes, "plot.png")
    return None


if __name__ == '__main__':
    app.run(debug=True)