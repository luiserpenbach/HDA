# Create new file: data_lib/report_generator.py

import pandas as pd
import plotly.io as pio
from datetime import datetime
import base64
from io import BytesIO


def fig_to_base64(fig):
    """Convert Plotly figure to base64 for embedding."""
    buffer = BytesIO()
    fig.write_image(buffer, format='png', width=1200, height=600)
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode()
    return f"data:image/png;base64,{img_base64}"


def generate_cold_flow_test_report(test_data, config, figures, steady_window_info):
    """
    Generate comprehensive HTML report for a single cold flow test.

    Args:
        test_data: dict with test metadata and results
        config: test configuration used
        figures: list of Plotly figures
        steady_window_info: dict with window detection details
    """

    # Extract key metrics
    test_id = test_data.get('test_id', 'Unknown')
    part = test_data.get('part', 'N/A')
    fluid = test_data.get('fluid', 'N/A')
    cd = test_data.get('avg_cd_CALC', 0)
    pressure = test_data.get('avg_p_up_bar', 0)
    flow = test_data.get('avg_mf_g_s', 0)
    temp = test_data.get('avg_T_up_K', 273.15) - 273.15

    # Convert figures to HTML
    fig_htmls = ""
    for i, fig in enumerate(figures):
        fig_htmls += f'''
        <div class="chart-container">
            <h3>Figure {i + 1}</h3>
            {pio.to_html(fig, full_html=False, include_plotlyjs='cdn' if i == 0 else False)}
        </div>
        '''

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Cold Flow Test Report - {test_id}</title>
        <style>
            body {{
                font-family: 'Segoe UI', Roboto, Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #333;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 12px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                overflow: hidden;
            }}
            .header {{
                background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
                color: white;
                padding: 40px;
                text-align: center;
            }}
            .header h1 {{
                margin: 0;
                font-size: 2.5em;
                font-weight: 300;
            }}
            .header .subtitle {{
                margin-top: 10px;
                font-size: 1.2em;
                opacity: 0.9;
            }}
            .content {{
                padding: 40px;
            }}
            .section {{
                margin-bottom: 40px;
            }}
            .section h2 {{
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
                margin-bottom: 20px;
                font-size: 1.8em;
            }}
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            .metric-card {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 25px;
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                transition: transform 0.2s;
            }}
            .metric-card:hover {{
                transform: translateY(-5px);
            }}
            .metric-label {{
                font-size: 0.9em;
                opacity: 0.9;
                margin-bottom: 5px;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            .metric-value {{
                font-size: 2.2em;
                font-weight: bold;
                margin: 5px 0;
            }}
            .metric-unit {{
                font-size: 0.9em;
                opacity: 0.8;
            }}
            .info-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            .info-table td {{
                padding: 12px;
                border-bottom: 1px solid #ecf0f1;
            }}
            .info-table td:first-child {{
                font-weight: 600;
                color: #7f8c8d;
                width: 200px;
            }}
            .chart-container {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                margin: 20px 0;
            }}
            .chart-container h3 {{
                margin-top: 0;
                color: #2c3e50;
            }}
            .pass-fail {{
                display: inline-block;
                padding: 8px 16px;
                border-radius: 20px;
                font-weight: bold;
                font-size: 0.9em;
            }}
            .pass {{
                background: #2ecc71;
                color: white;
            }}
            .fail {{
                background: #e74c3c;
                color: white;
            }}
            .footer {{
                background: #ecf0f1;
                padding: 20px;
                text-align: center;
                color: #7f8c8d;
                font-size: 0.9em;
            }}
            @media print {{
                body {{
                    background: white;
                }}
                .container {{
                    box-shadow: none;
                }}
                .metric-card {{
                    break-inside: avoid;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>❄️ Cold Flow Test Report</h1>
                <div class="subtitle">{test_id}</div>
            </div>

            <div class="content">
                <!-- Key Metrics -->
                <div class="section">
                    <h2>📊 Key Performance Metrics</h2>
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-label">Discharge Coefficient</div>
                            <div class="metric-value">{cd:.4f}</div>
                            <div class="metric-unit">Cd</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Upstream Pressure</div>
                            <div class="metric-value">{pressure:.2f}</div>
                            <div class="metric-unit">bar</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Mass Flow</div>
                            <div class="metric-value">{flow:.2f}</div>
                            <div class="metric-unit">g/s</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Temperature</div>
                            <div class="metric-value">{temp:.1f}</div>
                            <div class="metric-unit">°C</div>
                        </div>
                    </div>
                </div>

                <!-- Test Information -->
                <div class="section">
                    <h2>ℹ️ Test Information</h2>
                    <table class="info-table">
                        <tr>
                            <td>Test ID</td>
                            <td>{test_id}</td>
                        </tr>
                        <tr>
                            <td>Part Number</td>
                            <td>{part}</td>
                        </tr>
                        <tr>
                            <td>Fluid</td>
                            <td>{fluid}</td>
                        </tr>
                        <tr>
                            <td>Date</td>
                            <td>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td>
                        </tr>
                        <tr>
                            <td>Configuration</td>
                            <td>{config.get('config_name', 'Generic') if config else 'Generic'}</td>
                        </tr>
                        <tr>
                            <td>Steady Window Duration</td>
                            <td>{steady_window_info.get('duration_s', 0):.2f} s</td>
                        </tr>
                        <tr>
                            <td>Detection Method</td>
                            <td>{steady_window_info.get('method', 'CV-based')}</td>
                        </tr>
                    </table>
                </div>

                <!-- Configuration Details -->
                <div class="section">
                    <h2>⚙️ Test Configuration</h2>
                    <table class="info-table">
                        <tr>
                            <td>Orifice Area</td>
                            <td>{config.get('geometry', {}).get('orifice_area_mm2', 'N/A')} mm²</td>
                        </tr>
                        <tr>
                            <td>Fluid Density</td>
                            <td>{config.get('fluid', {}).get('density_kg_m3', 'N/A')} kg/m³</td>
                        </tr>
                        <tr>
                            <td>Resample Rate</td>
                            <td>{config.get('settings', {}).get('resample_freq_ms', 'N/A')} ms</td>
                        </tr>
                        <tr>
                            <td>CV Threshold</td>
                            <td>{config.get('settings', {}).get('cv_threshold', 'N/A')} %</td>
                        </tr>
                    </table>
                </div>

                <!-- Charts -->
                <div class="section">
                    <h2>📈 Data Visualization</h2>
                    {fig_htmls}
                </div>

                <!-- Comments -->
                <div class="section">
                    <h2>💬 Comments</h2>
                    <p>{test_data.get('comments', 'No comments provided.')}</p>
                </div>
            </div>

            <div class="footer">
                Generated by Hopper Data Studio • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>
        </div>
    </body>
    </html>
    """

    return html_content.encode('utf-8')


def generate_cold_flow_campaign_report(campaign_name, df, summary_stats, figures):
    """
    Generate campaign-level report for cold flow tests.
    """

    # Convert figures
    fig_htmls = ""
    for i, fig in enumerate(figures):
        fig_htmls += f'''
        <div class="chart-container">
            {pio.to_html(fig, full_html=False, include_plotlyjs='cdn' if i == 0 else False)}
        </div>
        '''

    # Build parts summary table
    parts_summary = ""
    for part in df['part'].unique():
        part_data = df[df['part'] == part]
        parts_summary += f"""
        <tr>
            <td>{part}</td>
            <td>{len(part_data)}</td>
            <td>{part_data['avg_cd_CALC'].mean():.4f}</td>
            <td>{part_data['avg_cd_CALC'].std():.4f}</td>
            <td>{part_data['avg_cd_CALC'].min():.4f}</td>
            <td>{part_data['avg_cd_CALC'].max():.4f}</td>
        </tr>
        """

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Campaign Report - {campaign_name}</title>
        <style>
            body {{
                font-family: 'Segoe UI', Roboto, Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                border-radius: 12px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            }}
            .header {{
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                color: white;
                padding: 50px;
                text-align: center;
            }}
            .header h1 {{
                margin: 0;
                font-size: 3em;
                font-weight: 300;
            }}
            .content {{
                padding: 40px;
            }}
            .section {{
                margin-bottom: 50px;
            }}
            .section h2 {{
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
                font-size: 2em;
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 30px 0;
            }}
            .stat-card {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 25px;
                border-radius: 8px;
                text-align: center;
            }}
            .stat-card .label {{
                font-size: 0.9em;
                opacity: 0.9;
                margin-bottom: 10px;
            }}
            .stat-card .value {{
                font-size: 2.5em;
                font-weight: bold;
            }}
            .data-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            .data-table th {{
                background: #34495e;
                color: white;
                padding: 15px;
                text-align: left;
            }}
            .data-table td {{
                padding: 12px;
                border-bottom: 1px solid #ecf0f1;
            }}
            .data-table tr:hover {{
                background: #f8f9fa;
            }}
            .chart-container {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                margin: 20px 0;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>❄️ Cold Flow Campaign Report</h1>
                <div style="font-size: 1.5em; margin-top: 10px;">{campaign_name}</div>
            </div>

            <div class="content">
                <!-- Summary Statistics -->
                <div class="section">
                    <h2>📊 Campaign Summary</h2>
                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="label">Total Tests</div>
                            <div class="value">{summary_stats['total_tests']}</div>
                        </div>
                        <div class="stat-card">
                            <div class="label">Unique Parts</div>
                            <div class="value">{summary_stats['unique_parts']}</div>
                        </div>
                        <div class="stat-card">
                            <div class="label">Avg Cd</div>
                            <div class="value">{summary_stats['cd_overall']['mean']:.4f}</div>
                        </div>
                        <div class="stat-card">
                            <div class="label">Cd Std Dev</div>
                            <div class="value">{summary_stats['cd_overall']['std']:.4f}</div>
                        </div>
                    </div>
                </div>

                <!-- Parts Summary -->
                <div class="section">
                    <h2>🔧 Parts Performance Summary</h2>
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>Part Number</th>
                                <th>Tests</th>
                                <th>Mean Cd</th>
                                <th>Std Dev</th>
                                <th>Min Cd</th>
                                <th>Max Cd</th>
                            </tr>
                        </thead>
                        <tbody>
                            {parts_summary}
                        </tbody>
                    </table>
                </div>

                <!-- Charts -->
                <div class="section">
                    <h2>📈 Campaign Analytics</h2>
                    {fig_htmls}
                </div>
            </div>
        </div>
    </body>
    </html>
    """

    return html_content.encode('utf-8')


def generate_hot_fire_test_report(test_data, test_df, config, figures):
    """
    Generate comprehensive HTML report for a single hot fire test.
    """

    test_id = test_data.get('test_id', 'Unknown')
    duration = test_data.get('duration_s', 0)
    pc = test_data.get('avg_pc_bar', 0)
    thrust = test_data.get('avg_thrust_n', 0)
    isp = test_data.get('avg_isp_s', 0)
    c_star = test_data.get('avg_c_star_m_s', 0)
    of_ratio = test_data.get('avg_of_ratio', 0)

    # Convert figures
    fig_htmls = ""
    for i, fig in enumerate(figures):
        fig_htmls += f'''
        <div class="chart-container">
            {pio.to_html(fig, full_html=False, include_plotlyjs='cdn' if i == 0 else False)}
        </div>
        '''

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Hot Fire Test Report - {test_id}</title>
        <style>
            body {{
                font-family: 'Segoe UI', Roboto, Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 12px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            }}
            .header {{
                background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
                color: white;
                padding: 40px;
                text-align: center;
            }}
            .header h1 {{
                margin: 0;
                font-size: 2.5em;
                font-weight: 300;
            }}
            .content {{
                padding: 40px;
            }}
            .section {{
                margin-bottom: 40px;
            }}
            .section h2 {{
                color: #e74c3c;
                border-bottom: 3px solid #e74c3c;
                padding-bottom: 10px;
                font-size: 1.8em;
            }}
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                gap: 20px;
            }}
            .metric-card {{
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                color: white;
                padding: 25px;
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            .metric-label {{
                font-size: 0.85em;
                opacity: 0.9;
                margin-bottom: 5px;
                text-transform: uppercase;
            }}
            .metric-value {{
                font-size: 2em;
                font-weight: bold;
                margin: 5px 0;
            }}
            .metric-unit {{
                font-size: 0.9em;
                opacity: 0.8;
            }}
            .info-table {{
                width: 100%;
                border-collapse: collapse;
            }}
            .info-table td {{
                padding: 12px;
                border-bottom: 1px solid #ecf0f1;
            }}
            .info-table td:first-child {{
                font-weight: 600;
                color: #7f8c8d;
                width: 200px;
            }}
            .chart-container {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                margin: 20px 0;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔥 Hot Fire Test Report</h1>
                <div style="font-size: 1.2em; margin-top: 10px;">{test_id}</div>
            </div>

            <div class="content">
                <!-- Key Metrics -->
                <div class="section">
                    <h2>🚀 Performance Metrics</h2>
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-label">Chamber Pressure</div>
                            <div class="metric-value">{pc:.1f}</div>
                            <div class="metric-unit">bar</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Thrust</div>
                            <div class="metric-value">{thrust:.1f}</div>
                            <div class="metric-unit">N</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Specific Impulse</div>
                            <div class="metric-value">{isp:.1f}</div>
                            <div class="metric-unit">s</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">C* Velocity</div>
                            <div class="metric-value">{c_star:.0f}</div>
                            <div class="metric-unit">m/s</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">O/F Ratio</div>
                            <div class="metric-value">{of_ratio:.2f}</div>
                            <div class="metric-unit">-</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Burn Duration</div>
                            <div class="metric-value">{duration:.2f}</div>
                            <div class="metric-unit">s</div>
                        </div>
                    </div>
                </div>

                <!-- Test Info -->
                <div class="section">
                    <h2>ℹ️ Test Information</h2>
                    <table class="info-table">
                        <tr>
                            <td>Test ID</td>
                            <td>{test_id}</td>
                        </tr>
                        <tr>
                            <td>Date</td>
                            <td>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td>
                        </tr>
                        <tr>
                            <td>Configuration</td>
                            <td>{config.get('config_name', 'Generic') if config else 'Generic'}</td>
                        </tr>
                        <tr>
                            <td>Propellants</td>
                            <td>{test_data.get('propellants', 'N/A')}</td>
                        </tr>
                    </table>
                </div>

                <!-- Charts -->
                <div class="section">
                    <h2>📈 Time-Series Data</h2>
                    {fig_htmls}
                </div>
            </div>
        </div>
    </body>
    </html>
    """

    return html_content.encode('utf-8')


def generate_hot_fire_campaign_report(campaign_name, df, summary_stats, figures):
    """
    Generate campaign-level report for hot fire tests.
    """

    fig_htmls = ""
    for i, fig in enumerate(figures):
        fig_htmls += f'''
        <div class="chart-container">
            {pio.to_html(fig, full_html=False, include_plotlyjs='cdn' if i == 0 else False)}
        </div>
        '''

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Hot Fire Campaign - {campaign_name}</title>
        <style>
            body {{
                font-family: 'Segoe UI', Roboto, Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                border-radius: 12px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            }}
            .header {{
                background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
                color: white;
                padding: 50px;
                text-align: center;
            }}
            .header h1 {{
                margin: 0;
                font-size: 3em;
                font-weight: 300;
            }}
            .content {{
                padding: 40px;
            }}
            .section {{
                margin-bottom: 50px;
            }}
            .section h2 {{
                color: #e74c3c;
                border-bottom: 3px solid #e74c3c;
                padding-bottom: 10px;
                font-size: 2em;
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
            }}
            .stat-card {{
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                color: white;
                padding: 25px;
                border-radius: 8px;
                text-align: center;
            }}
            .stat-card .label {{
                font-size: 0.9em;
                opacity: 0.9;
                margin-bottom: 10px;
            }}
            .stat-card .value {{
                font-size: 2.5em;
                font-weight: bold;
            }}
            .chart-container {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                margin: 20px 0;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔥 Hot Fire Campaign Report</h1>
                <div style="font-size: 1.5em; margin-top: 10px;">{campaign_name}</div>
            </div>

            <div class="content">
                <!-- Summary -->
                <div class="section">
                    <h2>📊 Campaign Summary</h2>
                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="label">Total Tests</div>
                            <div class="value">{summary_stats['total_tests']}</div>
                        </div>
                        <div class="stat-card">
                            <div class="label">Total Burn Time</div>
                            <div class="value">{summary_stats['total_firing_time']:.1f}</div>
                            <div style="font-size: 0.8em; opacity: 0.8;">seconds</div>
                        </div>
                        <div class="stat-card">
                            <div class="label">Avg Isp</div>
                            <div class="value">{summary_stats['performance']['avg_isp']:.1f}</div>
                            <div style="font-size: 0.8em; opacity: 0.8;">seconds</div>
                        </div>
                        <div class="stat-card">
                            <div class="label">Max Isp</div>
                            <div class="value">{summary_stats['performance']['max_isp']:.1f}</div>
                            <div style="font-size: 0.8em; opacity: 0.8;">seconds</div>
                        </div>
                    </div>
                </div>

                <!-- Charts -->
                <div class="section">
                    <h2>📈 Campaign Analytics</h2>
                    {fig_htmls}
                </div>
            </div>
        </div>
    </body>
    </html>
    """

    return html_content.encode('utf-8')