import base64
import io
import dash
from dash import dcc, html, Input, Output, State, ctx
import plotly.graph_objects as go
import pandas as pd
import plotly.io as pio  # For export

# Initialize the Dash app
app = dash.Dash(__name__, title="Hop Data Plotter")

# --- Enhanced CSS Styles for Visual Appeal ---
# Dark theme for consistency with Plotly dark plots
# Improved typography, spacing, and button styles

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

        # 2. Channel Selection
        html.Label("2. Select Channels", style=LABEL_STYLE),
        dcc.Dropdown(
            id='channel-dropdown',
            multi=True,
            placeholder="Upload file first...",
            style={'margin-bottom': '20px', 'backgroundColor': '#2a2a2a', 'color': '#e0e0e0', 'border': '1px solid #444'}
        ),

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
                html.Button('Export PNG', id='export-btn', style={**BUTTON_STYLE, 'width': '200px', 'margin': '20px auto', 'display': 'block'})
            ])
        ),
        # Hidden download component for export
        dcc.Download(id="download-image"),
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


# 3. Main Plotting Callback
@app.callback(
    [Output('main-graph', 'figure'), Output('status-msg', 'children')],
    [Input('update-btn', 'n_clicks')],
    [State('upload-data', 'contents'),
     State('channel-dropdown', 'value'),
     State('snip-mode', 'value'),
     State('start-time', 'value'),
     State('end-time', 'value'),
     State('trigger-channel', 'value'),
     State('threshold', 'value'),
     State('pre-buffer', 'value'),
     State('post-buffer', 'value')]
)
def update_graph(n_clicks, contents, channels, mode, t_start, t_end, trig_col, thresh, pre_b, post_b):
    if contents is None:
        return go.Figure(), "Waiting for file upload..."

    df = parse_contents(contents)

    if not channels:
        return go.Figure(), "No channels selected."

    # --- Snipping Logic ---
    msg = f"Plotting {len(channels)} channels."

    if mode == 'manual':
        mask = (df['relative_time'] >= (t_start or 0)) & (df['relative_time'] <= (t_end or df['relative_time'].max()))
        df = df.loc[mask]
        msg += f" (Manual Range: {t_start}s to {t_end}s)"

    elif mode == 'valve':
        if trig_col and trig_col in df.columns:
            events = df[df[trig_col] > (thresh or 0)]
            if not events.empty:
                event_time = events.iloc[0]['relative_time']
                start_win = event_time - (pre_b or 0)
                end_win = event_time + (post_b or 0)

                mask = (df['relative_time'] >= start_win) & (df['relative_time'] <= end_win)
                df = df.loc[mask].copy()

                # Re-zero time to event
                df['relative_time'] -= event_time
                msg += f" (Valve Event at {event_time:.2f}s)"
            else:
                msg += " (No event found, showing full data)"

    # --- Performance Optimization ---
    if len(df) > 15000:
        step = len(df) // 15000 + 1
        df = df.iloc[::step]
        msg += f" [Downsampled {step}x for performance]"

    # --- Plotting with Color Cycle ---
    fig = go.Figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']  # Nice color palette
    for i, col in enumerate(channels):
        fig.add_trace(go.Scattergl(
            x=df['relative_time'],
            y=df[col],
            mode='lines',
            name=col,
            line={'color': colors[i % len(colors)]}
        ))

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Time (s)",
        yaxis_title="Value",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#121212'
    )

    return fig, msg


# 4. Export Callback
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