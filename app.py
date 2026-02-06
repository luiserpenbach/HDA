"""
Hopper Data Studio v2.5
=======================
Propulsion Test Data Analysis Platform with Engineering Integrity

Main entry point for the Streamlit application.
"""

import streamlit as st
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import custom styles
from pages._shared_styles import apply_custom_styles, render_page_header, render_metric_card, render_feature_card

# Page configuration
st.set_page_config(
    page_title="Hopper Data Studio",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom styles
apply_custom_styles()

# ============================================
# HEADER SECTION
# ============================================

render_page_header(
    title="Hopper Data Studio",
    description="Professional propulsion test data analysis platform with built-in engineering integrity",
    badge_text="v2.5",
    badge_type="default"
)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================
# KEY METRICS SECTION
# ============================================

col1, col2, col3 = st.columns(3)

with col1:
    render_metric_card(
        title="Engineering Integrity",
        value="P0 Core",
        subtitle="Traceability • Uncertainty • QC"
    )

with col2:
    render_metric_card(
        title="Test Types",
        value="Plugin Architecture",
        subtitle="Cold Flow • Hot Fire • Extensible"
    )

with col3:
    render_metric_card(
        title="Campaign Tracking",
        value="SPC + CUSUM + EWMA",
        subtitle="Full traceability & process control"
    )

st.markdown("<br><br>", unsafe_allow_html=True)

# ============================================
# CORE PHILOSOPHY SECTION
# ============================================

st.markdown("""
<div class="card-elevated animate-fade-in">
    <h3 style="margin-top: 0;">🎯 Core Philosophy: Engineering Integrity First</h3>
    <p style="margin-bottom: 1rem;">
        Hopper Data Studio is built on three non-negotiable principles that ensure every result is
        trustworthy, traceable, and scientifically rigorous:
    </p>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin-top: 1.5rem;">
        <div style="padding: 1rem; background: #fafafa; border-radius: 0.5rem;">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🔐</div>
            <h4 style="margin: 0 0 0.5rem 0;">Traceability</h4>
            <p style="margin: 0; font-size: 0.875rem; color: #71717a;">
                Every result is cryptographically linked to its source data via SHA-256 hashing
            </p>
        </div>
        <div style="padding: 1rem; background: #fafafa; border-radius: 0.5rem;">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">📊</div>
            <h4 style="margin: 0 0 0.5rem 0;">Uncertainty</h4>
            <p style="margin: 0; font-size: 0.875rem; color: #71717a;">
                Every metric includes error bars—no naked numbers allowed
            </p>
        </div>
        <div style="padding: 1rem; background: #fafafa; border-radius: 0.5rem;">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">✅</div>
            <h4 style="margin: 0 0 0.5rem 0;">Quality Control</h4>
            <p style="margin: 0; font-size: 0.875rem; color: #71717a;">
                Bad data is blocked, not silently processed—QC checks are non-negotiable
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================
# FEATURES GRID
# ============================================

st.markdown('<h2 style="margin-bottom: 1.5rem;">Features</h2>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    render_feature_card(
        title="Single Test Analysis",
        description="Analyze cold flow or hot fire tests with steady-state detection, uncertainty propagation, and QC checks.",
        icon="🔬"
    )

    render_feature_card(
        title="Transient Analysis",
        description="Multi-phase test segmentation with startup/shutdown transient characterization — rise time, overshoot, settling.",
        icon="⚡"
    )

    render_feature_card(
        title="Campaign Management",
        description="Track test campaigns with full traceability. Results linked to source data via SHA-256 hashes.",
        icon="🗂️"
    )

with col2:
    render_feature_card(
        title="SPC Control Charts",
        description="I-MR, CUSUM, and EWMA charts with Western Electric rules, trend detection, and capability indices (Cpk).",
        icon="📈"
    )

    render_feature_card(
        title="Frequency Analysis",
        description="PSD estimation, spectrogram visualization, harmonic detection, cross-spectrum coherence, and resonance Q-factor.",
        icon="🔊"
    )

    render_feature_card(
        title="Report Generation",
        description="Professional HTML reports with traceability records, uncertainty tables, and qualification packages.",
        icon="📄"
    )

with col3:
    render_feature_card(
        title="Batch Processing",
        description="Process multiple test files with consistent configuration, parallel execution, and aggregate reporting.",
        icon="📦"
    )

    render_feature_card(
        title="Operating Envelope",
        description="Visualize O/F ratio vs chamber pressure operating space with safety margins and success/failure mapping.",
        icon="🎯"
    )

    render_feature_card(
        title="Anomaly Detection",
        description="Multi-type anomaly detection with spike, drift, flatline, and correlation analysis across sensor channels.",
        icon="🔍"
    )

st.markdown("<br>", unsafe_allow_html=True)

# ============================================
# QUICK START GUIDE
# ============================================

st.markdown('<h2 style="margin-bottom: 1.5rem;">Quick Start</h2>', unsafe_allow_html=True)

st.markdown("""
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem;">
    <div class="card">
        <div style="background: #18181b; color: white; width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; margin-bottom: 1rem;">1</div>
        <h4 style="margin: 0 0 0.5rem 0;">Configure Your Test</h4>
        <p style="margin: 0; font-size: 0.875rem; color: #71717a;">
            Set up your test configuration with sensor mappings, calibration uncertainties, and test article geometry.
        </p>
    </div>
    <div class="card">
        <div style="background: #18181b; color: white; width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; margin-bottom: 1rem;">2</div>
        <h4 style="margin: 0 0 0.5rem 0;">Analyze Your Data</h4>
        <p style="margin: 0; font-size: 0.875rem; color: #71717a;">
            Upload test data, detect steady-state automatically or manually, run QC checks, and calculate results with uncertainties.
        </p>
    </div>
    <div class="card">
        <div style="background: #18181b; color: white; width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; margin-bottom: 1rem;">3</div>
        <h4 style="margin: 0 0 0.5rem 0;">Track & Report</h4>
        <p style="margin: 0; font-size: 0.875rem; color: #71717a;">
            Save results to campaigns with full traceability, perform SPC analysis, and generate professional reports.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================
# PRIORITY SYSTEM INFO
# ============================================

with st.expander("📋 System Architecture & Priorities", expanded=False):
    st.markdown("""
    <div style="font-size: 0.875rem;">
        <p style="margin-bottom: 1rem;">The codebase uses a P0/P1/P2 priority system:</p>

        <div class="card-muted" style="margin-bottom: 0.75rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                <span class="badge badge-error">P0</span>
                <strong>Core Foundation (Non-Negotiable)</strong>
            </div>
            <p style="margin: 0; font-size: 0.875rem; color: #71717a;">
                Traceability, Uncertainty Propagation, QC Checks, Config Validation, Campaign Manager
            </p>
        </div>

        <div class="card-muted" style="margin-bottom: 0.75rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                <span class="badge badge-warning">P1</span>
                <strong>High Priority Analysis</strong>
            </div>
            <p style="margin: 0; font-size: 0.875rem; color: #71717a;">
                SPC, Reporting, Batch Analysis, Export
            </p>
        </div>

        <div class="card-muted">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                <span class="badge badge-info">P2</span>
                <strong>Advanced Features</strong>
            </div>
            <p style="margin: 0; font-size: 0.875rem; color: #71717a;">
                Anomaly Detection, Comparison Tools, Transient Analysis, Frequency Analysis, Operating Envelope
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================
# NAVIGATION HINT
# ============================================

st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #f4f4f5 0%, #fafafa 100%); border-radius: 0.5rem;">
    <h3 style="margin: 0 0 0.5rem 0;">Ready to Get Started?</h3>
    <p style="margin: 0; font-size: 0.875rem; color: #71717a;">
        Use the sidebar to navigate between analysis pages →
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================
# FOOTER
# ============================================

st.divider()

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.caption("**Hopper Data Studio** — Engineering Integrity System")

with col2:
    st.caption("Version 2.5.0")

with col3:
    st.caption("© 2024-2026")
