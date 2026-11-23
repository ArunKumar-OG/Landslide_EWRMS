import pandas as pd

df = pd.read_csv(
    "data/processed/DEM/district_dem_coverage.csv"
)

print("Columns:")
print(df.columns.tolist())

print("\nFirst 5 records:")
print(df.head().to_string())