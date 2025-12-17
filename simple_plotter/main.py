import base64
import io
import dash
from dash import dcc, html, Input, Output, State, ctx
import plotly.graph_objects as go
import pandas as pd

# Initialize the Dash app
app = dash.Dash(__name__, title="Hop Data Plotter")

# --- CSS Styles for Side Panel Layout ---

# 1. The Main Container (Holds Sidebar + Content side-by-side)
# This uses Flexbox to manage the layout automatically.
LAYOUT_STYLE = {
    "display": "flex",
    "flexDirection": "row",
    "height": "100vh",       # Full viewport height
    "overflow": "hidden",    # Prevent double scrollbars
    "fontFamily": "Arial, sans-serif"
}

# 2. The Sidebar
SIDEBAR_STYLE = {
    "width": "35rem",        # Initial width
    "minWidth": "300px",     # Prevent it from becoming too narrow
    "maxWidth": "50vw",      # Prevent it from taking over the screen
    "padding": "2rem 1rem",
    "backgroundColor": "#e6e6e6",
    "overflow": "auto",     # Scroll vertically if controls get too long
    "resize": "horizontal",  # <--- ENABLES RESIZING
    "borderRight": "3px solid #ddd", # Visual separator
    "display": "flex",
    "flexDirection": "column",
    "fontSize": "20px",
}

# 3. The Main Content
CONTENT_STYLE = {
    "flex": "1",             # Take up all remaining space automatically
    "padding": "2rem 1rem",
    "overflowY": "auto",     # Scroll vertically independently of sidebar
}



# --- Layout Definition ---
sidebar = html.Div(
    [
        html.H2("Controls", className="display-4"),
        html.Hr(),

        # 1. File Uploader
        html.Label("1. Test Data File"),
        dcc.Upload(
            id='upload-data',
            children=html.Div(['Drag and Drop or ', html.A('Select CSV')]),
            style={
                'width': '100%', 'height': '60px', 'lineHeight': '60px',
                'borderWidth': '1px', 'borderStyle': 'dashed',
                'borderRadius': '5px', 'textAlign': 'center', 'margin-bottom': '20px'
            },
            multiple=False
        ),

        # 2. Channel Selection
        html.Label("2. Select Channels"),
        dcc.Dropdown(
            id='channel-dropdown',
            multi=True,
            placeholder="Upload file first...",
            style={'margin-bottom': '20px'}
        ),

        # 3. Snipping Tools
        html.Label("3. Snipping Mode"),
        dcc.RadioItems(
            id='snip-mode',
            options=[
                {'label': 'Full Data', 'value': 'full'},
                {'label': 'Manual Range', 'value': 'manual'},
                {'label': 'Valve Event', 'value': 'valve'}
            ],
            value='full',
            style={'margin-bottom': '15px'}
        ),

        # Manual Inputs
        html.Div(id='manual-controls', children=[
            html.Label("Start Time (s)"),
            dcc.Input(id='start-time', type='number', value=0, style={'width': '100%', 'margin-bottom': '5px'}),
            html.Label("End Time (s)"),
            dcc.Input(id='end-time', type='number', value=100, style={'width': '100%', 'margin-bottom': '15px'}),
        ], style={'display': 'none'}),

        # Valve Event Inputs
        html.Div(id='valve-controls', children=[
            html.Label("Trigger Channel"),
            dcc.Dropdown(id='trigger-channel', placeholder="Select trigger..."),
            html.Label("Threshold", style={'margin-top': '5px'}),
            dcc.Input(id='threshold', type='number', value=3.0, style={'width': '100%'}),
            html.Label("Pre-Buffer (s)", style={'margin-top': '5px'}),
            dcc.Input(id='pre-buffer', type='number', value=5.0, style={'width': '100%'}),
            html.Label("Post-Buffer (s)", style={'margin-top': '5px'}),
            dcc.Input(id='post-buffer', type='number', value=20.0, style={'width': '100%', 'margin-bottom': '15px'}),
        ], style={'display': 'none'}),

        html.Button('Update Plot', id='update-btn', n_clicks=0,
                    style={'width': '100%', 'height': '40px', 'background-color': '#007bff', 'color': 'white',
                           'border': 'none'}),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(
    [
        html.H1("Hopper General Data Plotter", style={'text-align': 'center', "fontSize": 40}),
        dcc.Loading(
            id="loading-1",
            type="default",
            children=html.Div([
                dcc.Graph(id='main-graph', style={'height': '80vh'}, config={'responsive': True}), # Ensures graph redraws on resize
                html.Div(id='status-msg', style={'margin-top': '10px', 'color': 'gray'})
            ])
        ),
        # Hidden div to store data (or share status)
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

        # Preprocessing
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
            df['relative_time'] = (df['timestamp'] - df['timestamp'].iloc[0]) / 1000.0
        else:
            # Fallback if no timestamp
            df['relative_time'] = df.index

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

    # Default selection: first 3
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
        mask = (df['relative_time'] >= t_start) & (df['relative_time'] <= t_end)
        df = df.loc[mask]
        msg += f" (Manual Range: {t_start}s to {t_end}s)"

    elif mode == 'valve':
        if trig_col and trig_col in df.columns:
            events = df[df[trig_col] > thresh]
            if not events.empty:
                event_time = events.iloc[0]['relative_time']
                start_win = event_time - pre_b
                end_win = event_time + post_b

                mask = (df['relative_time'] >= start_win) & (df['relative_time'] <= end_win)
                df = df.loc[mask].copy()

                # Re-zero time
                df['relative_time'] = df['relative_time'] - event_time
                msg += f" (Valve Event at {event_time:.2f}s)"
            else:
                msg += " (No event found, showing full data)"

    # --- Performance Optimization ---
    # If data > 15k points, downsample for browser speed
    if len(df) > 15000:
        step = len(df) // 15000
        df = df.iloc[::step]
        msg += f" [Downsampled {step}x]"

    # --- Plotting ---
    fig = go.Figure()
    for col in channels:
        fig.add_trace(go.Scattergl(
            x=df['relative_time'],
            y=df[col],
            mode='lines',
            name=col
        ))

    fig.update_layout(
        template="plotly_dark+gridon",
        xaxis_title="Time / s)",
        yaxis_title="Value",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig, msg


if __name__ == '__main__':
    app.run(debug=True)