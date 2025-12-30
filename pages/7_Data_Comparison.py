"""
Data Comparison Page (P2)
=========================
Compare tests and perform regression analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json

from core.comparison import (
    compare_tests,
    compare_to_golden,
    create_golden_from_campaign,
    linear_regression,
    calculate_correlation_matrix,
    track_deviations,
    compare_campaigns,
    format_campaign_comparison,
    GoldenReference,
)
from core.campaign_manager_v2 import get_available_campaigns, get_campaign_data

st.set_page_config(page_title="Data Comparison", page_icon="", layout="wide")

st.title(" Data Comparison & Regression")
st.markdown("Compare tests, create golden references, and analyze correlations.")

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.header("Settings")
    
    mode = st.radio(
        "Analysis Mode",
        ["Test Comparison", "Golden Reference", "Regression", "Correlation", "Campaign Comparison"]
    )
    
    st.divider()
    
    campaigns = get_available_campaigns()
    if campaigns:
        campaign_names = [c['name'] for c in campaigns]
        selected_campaign = st.selectbox("Primary Campaign", campaign_names)
    else:
        selected_campaign = None
        st.warning("No campaigns found")

# =============================================================================
# MAIN CONTENT
# =============================================================================

if selected_campaign:
    df = get_campaign_data(selected_campaign)
    
    if df is None or len(df) == 0:
        st.warning("No data in selected campaign")
        st.stop()
    
    # Get numeric columns
    numeric_cols = [c for c in df.columns if df[c].dtype in ['float64', 'int64']]
    metric_cols = [c for c in numeric_cols if c.startswith('avg_')]
    
    # =============================================================================
    # TEST COMPARISON
    # =============================================================================
    
    if mode == "Test Comparison":
        st.header("Test-to-Test Comparison")
        
        if 'test_id' not in df.columns:
            st.error("Campaign must have test_id column")
            st.stop()
        
        test_ids = df['test_id'].tolist()
        
        col1, col2 = st.columns(2)
        
        with col1:
            test_a = st.selectbox("Test A", test_ids, key="test_a")
        with col2:
            test_b = st.selectbox("Test B", test_ids, index=min(1, len(test_ids)-1), key="test_b")
        
        # Parameter selection
        params_to_compare = st.multiselect(
            "Parameters to compare",
            metric_cols,
            default=metric_cols[:5] if len(metric_cols) > 5 else metric_cols
        )
        
        # Tolerance
        default_tol = st.slider("Default tolerance (%)", 1.0, 20.0, 5.0, 0.5)
        
        if st.button("Compare Tests", type="primary"):
            if test_a == test_b:
                st.warning("Select different tests")
            else:
                # Get test data
                row_a = df[df['test_id'] == test_a].iloc[0]
                row_b = df[df['test_id'] == test_b].iloc[0]
                
                data_a = {p: float(row_a[p]) for p in params_to_compare if pd.notna(row_a.get(p))}
                data_b = {p: float(row_b[p]) for p in params_to_compare if pd.notna(row_b.get(p))}
                
                result = compare_tests(data_a, data_b, test_a, test_b, default_tolerance=default_tol)
                
                # Display results
                st.subheader("Comparison Results")
                
                status_color = "green" if result.overall_pass else "red"
                st.markdown(f"### :{status_color}[{'[PASS] PASS' if result.overall_pass else '[FAIL] FAIL'}]")
                st.caption(f"{result.n_within_tolerance}/{result.n_parameters} parameters within tolerance")
                
                # Results table
                st.dataframe(result.to_dataframe(), use_container_width=True, hide_index=True)
                
                # Visual comparison
                st.subheader("Visual Comparison")
                
                fig = go.Figure()
                
                params = [c.parameter for c in result.comparisons]
                vals_a = [c.value_a for c in result.comparisons]
                vals_b = [c.value_b for c in result.comparisons]
                
                fig.add_trace(go.Bar(name=test_a, x=params, y=vals_a, marker_color='blue'))
                fig.add_trace(go.Bar(name=test_b, x=params, y=vals_b, marker_color='orange'))
                
                fig.update_layout(barmode='group', title="Parameter Comparison")
                st.plotly_chart(fig, use_container_width=True)
    
    # =============================================================================
    # GOLDEN REFERENCE
    # =============================================================================
    
    elif mode == "Golden Reference":
        st.header(" Golden Reference")
        
        tab1, tab2 = st.tabs(["Create Golden", "Compare to Golden"])
        
        with tab1:
            st.subheader("Create Golden Reference")
            
            golden_name = st.text_input("Reference Name", value=f"{selected_campaign}_golden")
            
            params_for_golden = st.multiselect(
                "Parameters to include",
                metric_cols,
                default=metric_cols[:5] if len(metric_cols) > 5 else metric_cols
            )
            
            col1, col2 = st.columns(2)
            with col1:
                method = st.selectbox("Central value method", ["mean", "median"])
            with col2:
                tol_mult = st.slider("Tolerance multiplier (× σ)", 1.0, 5.0, 3.0)
            
            if st.button(" Create Golden Reference"):
                if not params_for_golden:
                    st.warning("Select at least one parameter")
                else:
                    try:
                        golden = create_golden_from_campaign(
                            df, golden_name, params_for_golden,
                            tolerance_multiplier=tol_mult,
                            method=method,
                        )
                        
                        st.success(f"Created golden reference from {len(df)} tests")
                        
                        # Show golden values
                        golden_df = pd.DataFrame([
                            {
                                'Parameter': p,
                                'Value': f"{v:.4g}",
                                'Tolerance (%)': f"±{golden.tolerances.get(p, 5.0):.1f}",
                                'Uncertainty': f"±{golden.uncertainties.get(p, 0):.4g}" if golden.uncertainties else "-",
                            }
                            for p, v in golden.parameters.items()
                        ])
                        st.dataframe(golden_df, use_container_width=True, hide_index=True)
                        
                        # Download
                        st.download_button(
                            "Download Golden Reference",
                            json.dumps(golden.to_dict(), indent=2),
                            file_name=f"{golden_name}.json",
                            mime="application/json"
                        )
                        
                        # Store in session
                        st.session_state['golden_ref'] = golden
                        
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        with tab2:
            st.subheader("Compare Test to Golden")
            
            # Upload or use session golden
            golden_source = st.radio("Golden source", ["Upload JSON", "Use created golden"])
            
            golden = None
            if golden_source == "Upload JSON":
                uploaded_golden = st.file_uploader("Upload golden reference", type=['json'])
                if uploaded_golden:
                    data = json.load(uploaded_golden)
                    golden = GoldenReference.from_dict(data)
                    st.success(f"Loaded: {golden.name}")
            else:
                if 'golden_ref' in st.session_state:
                    golden = st.session_state['golden_ref']
                    st.info(f"Using: {golden.name}")
                else:
                    st.warning("Create a golden reference first")
            
            if golden and 'test_id' in df.columns:
                test_to_compare = st.selectbox("Select test", df['test_id'].tolist())
                
                if st.button("Compare to Golden"):
                    row = df[df['test_id'] == test_to_compare].iloc[0]
                    test_data = {p: float(row[p]) for p in golden.parameters.keys() if pd.notna(row.get(p))}
                    
                    result = compare_to_golden(test_data, test_to_compare, golden)
                    
                    status_color = "green" if result.overall_pass else "red"
                    st.markdown(f"### :{status_color}[{'[PASS] PASS' if result.overall_pass else '[FAIL] FAIL'}]")
                    
                    st.dataframe(result.to_dataframe(), use_container_width=True, hide_index=True)
    
    # =============================================================================
    # REGRESSION
    # =============================================================================
    
    elif mode == "Regression":
        st.header("Regression Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_param = st.selectbox("X Parameter (Independent)", numeric_cols)
        with col2:
            y_param = st.selectbox("Y Parameter (Dependent)", 
                                   [c for c in numeric_cols if c != x_param])
        
        if st.button("Run Regression"):
            x = df[x_param].values
            y = df[y_param].values
            
            try:
                result = linear_regression(x, y, x_param, y_param)
                
                # Results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("R²", f"{result.r_squared:.4f}")
                with col2:
                    st.metric("Slope", f"{result.slope:.4g}")
                with col3:
                    st.metric("Intercept", f"{result.intercept:.4g}")
                
                st.info(f"**Equation:** {result.prediction_equation}")
                
                # Plot
                fig = go.Figure()
                
                # Data points
                fig.add_trace(go.Scatter(
                    x=x, y=y,
                    mode='markers',
                    name='Data',
                    marker=dict(size=8, color='blue'),
                ))
                
                # Regression line
                x_line = np.linspace(np.nanmin(x), np.nanmax(x), 100)
                y_line = result.predict(x_line)
                
                fig.add_trace(go.Scatter(
                    x=x_line, y=y_line,
                    mode='lines',
                    name=f'Fit (R²={result.r_squared:.3f})',
                    line=dict(color='red', width=2),
                ))
                
                fig.update_layout(
                    title=f"Regression: {y_param} vs {x_param}",
                    xaxis_title=x_param,
                    yaxis_title=y_param,
                    height=500,
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.text(result.summary())
                
            except Exception as e:
                st.error(f"Regression error: {e}")
    
    # =============================================================================
    # CORRELATION
    # =============================================================================
    
    elif mode == "Correlation":
        st.header(" Correlation Analysis")
        
        params_for_corr = st.multiselect(
            "Parameters to analyze",
            metric_cols,
            default=metric_cols[:8] if len(metric_cols) > 8 else metric_cols
        )
        
        threshold = st.slider("Strong correlation threshold", 0.5, 0.95, 0.7, 0.05)
        
        if st.button(" Calculate Correlations") and params_for_corr:
            try:
                corr_matrix = calculate_correlation_matrix(df, params_for_corr)
                
                # Heatmap
                fig = px.imshow(
                    corr_matrix.matrix,
                    x=corr_matrix.parameters,
                    y=corr_matrix.parameters,
                    color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1,
                    title="Correlation Matrix"
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                # Strong correlations
                strong = corr_matrix.get_strong_correlations(threshold)
                
                if strong:
                    st.subheader(f"Strong Correlations (|r| ≥ {threshold})")
                    
                    strong_df = pd.DataFrame([
                        {'Parameter 1': p1, 'Parameter 2': p2, 'Correlation': f"{r:.3f}"}
                        for p1, p2, r in strong
                    ])
                    st.dataframe(strong_df, use_container_width=True, hide_index=True)
                else:
                    st.info(f"No correlations above |{threshold}| found")
                
                # Download matrix
                st.download_button(
                    "Download Correlation Matrix",
                    corr_matrix.to_dataframe().to_csv(),
                    file_name="correlation_matrix.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Error: {e}")
    
    # =============================================================================
    # CAMPAIGN COMPARISON
    # =============================================================================
    
    elif mode == "Campaign Comparison":
        st.header("Campaign Comparison")
        
        if len(campaigns) < 2:
            st.warning("Need at least 2 campaigns to compare")
        else:
            other_campaigns = [c['name'] for c in campaigns if c['name'] != selected_campaign]
            campaign_b = st.selectbox("Compare to campaign", other_campaigns)
            
            params_to_compare = st.multiselect(
                "Parameters to compare",
                metric_cols,
                default=metric_cols[:5] if len(metric_cols) > 5 else metric_cols
            )
            
            if st.button("Compare Campaigns") and params_to_compare:
                df_b = get_campaign_data(campaign_b)
                
                if df_b is None or len(df_b) == 0:
                    st.error(f"No data in {campaign_b}")
                else:
                    result = compare_campaigns(
                        df, df_b,
                        selected_campaign, campaign_b,
                        params_to_compare
                    )
                    
                    # Summary
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(f"{selected_campaign}", f"n={result['n_tests_a']}")
                    with col2:
                        st.metric(f"{campaign_b}", f"n={result['n_tests_b']}")
                    
                    # Results table
                    rows = []
                    for param, data in result['parameters'].items():
                        rows.append({
                            'Parameter': param,
                            f'Mean ({selected_campaign})': f"{data['mean_a']:.4g}",
                            f'Mean ({campaign_b})': f"{data['mean_b']:.4g}",
                            'Δ%': f"{data['mean_diff_pct']:+.2f}%",
                            'Status': '' if data['means_equivalent'] else '',
                        })
                    
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                    
                    # Visual comparison
                    st.subheader("Distribution Comparison")
                    
                    param_to_plot = st.selectbox("Parameter to visualize", params_to_compare)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(x=df[param_to_plot].dropna(), name=selected_campaign, opacity=0.7))
                    fig.add_trace(go.Histogram(x=df_b[param_to_plot].dropna(), name=campaign_b, opacity=0.7))
                    fig.update_layout(barmode='overlay', title=f"{param_to_plot} Distribution")
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.text(format_campaign_comparison(result))

else:
    st.info("Select a campaign in the sidebar")
