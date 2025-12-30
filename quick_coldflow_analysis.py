import pandas as pd

from core.integrated_analysis import analyze_cold_flow_test
from core.campaign_manager_v2 import save_to_campaign
from data_lib.config_loader import load_config

# Load your data and config
df = pd.read_csv("test_data.csv")
config = load_config("LCSC_B1_SingSide_ColdFlow")

# Run complete analysis with all P0 components
result = analyze_cold_flow_test(
    df=df,
    config=config,
    steady_window=(1500, 5000),  # ms
    test_id="INJ-CF-001",
    file_path="test_data.csv",
    detection_method="CV-based",
    metadata={
        'part': 'INJ-V1',
        'serial_num': 'SN-001',
        'operator': 'jsmith',
        'fluid': 'N2',
    }
)

# Check results
print(f"QC Passed: {result.passed_qc}")
print(f"Cd: {result.measurements['Cd']}")  # 0.654 ± 0.018 (2.8%)

# Save to campaign database
record = result.to_database_record('cold_flow')
save_to_campaign("INJ_Acceptance_Q1", record)