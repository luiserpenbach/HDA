import os

import pandas as pd

results = []
for filename in os.listdir('./local_data'):
    if filename.endswith('.parquet'):
        # ... Run analysis functions ...
        avg_p=1
        rise_ms = 1
        mass = 1

        results.append({
            'test_id': filename,
            'avg_p_up_bar': avg_p,
            't_rise_ms': rise_ms,
            'm_total_kg': mass
        })

# Save a master summary of ALL tests
summary_df = pd.DataFrame(results)
summary_df.to_excel("IGN-CF-C1_Test_Log.xlsx")