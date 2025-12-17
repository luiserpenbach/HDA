import pandas as pd

file = "../igniter_testing/local_data/IGN-CF-C1-003_raw.csv"
output_file = "test_003.parquet"

df = pd.read_csv(file)
print("Converting file to Parquet ...")
df.to_parquet(output_file)