# app.py - Main navigation hub

import streamlit as st

st.set_page_config(
    page_title="Hopper Data Studio",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .feature-card {
        padding: 1.5rem;
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        margin: 1rem 0;
        transition: all 0.3s;
    }
    .feature-card:hover {
        border-color: #667eea;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
    }
    </style>
""", unsafe_allow_html=True)

# Main Header
st.markdown("""
    <div class="main-header">
        <h1>Hopper Data Studio</h1>
        <p> Rocket Engine Test Data Viewer and  Analysis Platform</p>
    </div>
""", unsafe_allow_html=True)

# Introduction
st.markdown("""
## Welcome to Hopper Data Studio

A comprehensive platform for managing and analyzing rocket engine test data, from data ingestion to campaign-level analytics.

### Quick Start Guide

1. **Data Ingest** - Import test data with structured metadata
2. **Inspector** - Explore raw sensor data
3. **Cold Flow** - Analyze injector and valve characterization tests
4. **Hot Fire** - Process combustion tests and performance metrics
5. **Campaigns** - Track trends and generate reports across test series

---

### Features at a Glance
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    #### Analysis Capabilities
    - ✅ ML-based steady-state detection
    - ✅ Automatic Cd calculation
    - ✅ Combustion performance metrics (Isp, C*, Cf)
    - ✅ Statistical Process Control (SPC)
    - ✅ Operating envelope analysis
    """)

with col2:
    st.markdown("""
    #### Data Management
    - ✅ Multi-campaign database support
    - ✅ Automated HTML report generation
    - ✅ Flexible test configuration system
    - ✅ Time-series visualization
    - ✅ Batch processing workflows
    """)

st.markdown("---")

# Sidebar navigation instructions
with st.sidebar:
    st.markdown("""
    ### 📖 Navigation

    Use the page selector above to access different modules:

    - **Data Ingest**: Import new test data
    - **Inspector**: Browse raw data
    - **Cold Flow**: Single test analysis
    - **CF Campaign**: Campaign analytics
    - **Hot Fire**: Combustion analysis
    - **HF Campaign**: Performance tracking
    - **Config Manager**: Test configurations
    """)

    st.markdown("---")

    st.info("💡 **Tip**: Start by configuring your test in Config Manager, then use Data Ingest to import test files.")

# Version info
st.sidebar.markdown("---")
st.sidebar.caption("Hopper Data Studio v2.0")
