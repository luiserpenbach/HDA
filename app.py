"""
Hopper Data Studio v2.0
=======================
Propulsion Test Data Analysis Platform with Engineering Integrity

Main entry point for the Streamlit application.
"""

import streamlit as st
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Page configuration
st.set_page_config(
    page_title="Hopper Data Studio",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .feature-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    .integrity-badge {
        display: inline-block;
        background: #d4edda;
        color: #155724;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">Hopper Data Studio</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Propulsion Test Data Analysis Platform</p>', unsafe_allow_html=True)

# Version and integrity badge
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    st.markdown('<span class="integrity-badge">✓ Engineering Integrity v2.0</span>', unsafe_allow_html=True)
with col2:
    st.caption("Core Version: 2.0.0")

st.divider()

# Main content
st.header("Welcome")

st.markdown("""
Hopper Data Studio provides comprehensive analysis tools for rocket propulsion testing with 
**built-in engineering integrity** - ensuring test data is traceable, your uncertainties 
are properly quantified, and your quality control is rigorous.
""")

# Feature cards
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="feature-card">
        <h4>📊 Single Test Analysis</h4>
        <p>Analyze individual cold flow or hot fire tests with automatic steady-state detection, 
        full uncertainty propagation, and pre-analysis QC checks.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <h4>📈 Campaign Management</h4>
        <p>Track test campaigns with full data traceability. Every result is linked to its 
        source data via cryptographic hashes.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <h4>🔄 Batch Processing</h4>
        <p>Process multiple test files efficiently with consistent configuration, 
        parallel execution, and aggregate reporting.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h4>📉 Statistical Process Control</h4>
        <p>Monitor process stability with control charts, Western Electric rules, 
        trend detection, and capability indices (Cpk).</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <h4>📋 Report Generation</h4>
        <p>Generate professional HTML reports with full traceability records, 
        uncertainty tables, and QC summaries.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <h4>💾 Data Export</h4>
        <p>Export campaign data with full metadata to CSV, Excel, JSON, or 
        create qualification-ready documentation packages.</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# Quick stats if campaigns exist
st.header("Quick Start")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("**Step 1: Configure**\n\nSet up your test configuration with sensor mappings, uncertainties, and geometry.")

with col2:
    st.info("**Step 2: Analyze**\n\nUpload test data, detect steady-state, run QC checks, and calculate results.")

with col3:
    st.info("**Step 3: Track**\n\nSave results to campaigns with full traceability for SPC and reporting.")

# Navigation hint
st.markdown("---")
st.markdown("👈 **Use the sidebar to navigate between analysis pages**")

# Footer
st.markdown("---")
st.caption("Hopper Data Studio v2.0 | Engineering Integrity System | © 2024")
