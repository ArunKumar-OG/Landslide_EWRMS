import pandas as pd

df = pd.read_csv(
    "data/processed/DEM/district_dem_validation_report.csv"
)

print("Resolution X:")
print(df["pixel_x"].describe())

print("\nResolution Y:")
print(df["pixel_y"].describe())

print("\nElevation statistics:")
print(
    df[
        [
            "min_elevation_m",
            "max_elevation_m",
            "mean_elevation_m"
        ]
    ].describe()
)

print("\nUnique resolutions:")
print(df[["pixel_x", "pixel_y"]].drop_duplicates())