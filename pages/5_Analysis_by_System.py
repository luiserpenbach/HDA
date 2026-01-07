"""
Analysis by System Page
=======================
System-level analysis and cross-campaign comparison.
"""

import streamlit as st
from pages._shared_sidebar import render_global_context

st.set_page_config(page_title="Analysis by System", page_icon="AS", layout="wide")

# =============================================================================
# SIDEBAR - Global Context
# =============================================================================

with st.sidebar:
    context = render_global_context()
    st.divider()
    st.caption("Select a Test Root and Program to view system-level analysis.")

# =============================================================================
# MAIN CONTENT
# =============================================================================

st.title("Analysis by System")
st.markdown("System-level analysis and cross-campaign comparison.")

st.divider()

st.info("This page is under development.")

st.markdown("""
### Planned Features

- **System Overview**: View all campaigns and tests for a selected system
- **Cross-Campaign Comparison**: Compare performance across different campaigns
- **System Trends**: Track system performance over time
- **Hardware Configuration**: View and manage system hardware configurations
- **System Reports**: Generate system-level summary reports

### Coming Soon

Select a system from your test data hierarchy to view:
- All test programs using this system
- Campaign summaries grouped by system
- Historical performance data
- System configuration history
""")

# Placeholder for future implementation
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    #### System Selection

    Future: Select from available systems in your test data folder.
    """)

with col2:
    st.markdown("""
    #### Analysis Options

    Future: Configure system-level analysis parameters.
    """)
