from pathlib import Path

import pandas as pd
import geopandas as gpd

ROOT = Path(__file__).resolve().parent.parent

CSV = ROOT / "data/raw/landslides/glc_global.csv"

BOUNDARY = (
    ROOT / "data/processed/boundary/ihr_districts.gpkg"
)

print("=" * 65)
print("LANDSLIDE INVENTORY INITIAL INSPECTION")
print("=" * 65)

df = pd.read_csv(CSV, low_memory=False)

df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

print("\nDATASET INFORMATION")
print("Total global records:", len(df))
print("Total columns:", len(df.columns))

print("\nCOLUMN NAMES")
for col in df.columns:
    print(col)

print("\nCOLUMN DATA TYPES")
print(df.dtypes)

print("\nMISSING VALUE SUMMARY")
print(
    pd.DataFrame({
        "missing": df.isna().sum(),
        "missing_pct": df.isna().mean() * 100
    }).to_string()
)

print("\nExact duplicate rows:", df.duplicated().sum())

# Inspect available event attributes

fields = [
    "event_date",
    "event_time",
    "event_id",
    "country_name",
    "landslide_category",
    "landslide_trigger",
    "landslide_size",
    "location_accuracy",
    "latitude",
    "longitude"
]

print("\nSAMPLE EVENT RECORDS")
print(
    df[[c for c in fields if c in df.columns]]
    .head(10)
    .to_string(index=False)
)

if not {"latitude", "longitude"}.issubset(df.columns):
    raise RuntimeError(
        "Coordinate column names require inspection."
    )

# Validate coordinates

lat = pd.to_numeric(df["latitude"], errors="coerce")
lon = pd.to_numeric(df["longitude"], errors="coerce")

valid = (
    lat.between(-90, 90)
    & lon.between(-180, 180)
)

print("\nValid coordinate records:", valid.sum())
print("Invalid coordinate records:", (~valid).sum())

points = gpd.GeoDataFrame(
    df.loc[valid].copy(),
    geometry=gpd.points_from_xy(
        lon[valid], lat[valid]
    ),
    crs="EPSG:4326"
)

points["_row_id"] = points.index

# Load our 137 districts

districts = gpd.read_file(BOUNDARY)

if len(districts) != 137:
    raise RuntimeError("Unexpected district count.")

districts = districts[
    ["state", "district", "geometry"]
].to_crs("EPSG:4326")

districts = districts.rename(columns={
    "state": "ihr_state",
    "district": "ihr_district"
})

districts["geometry"] = districts.geometry.make_valid()

# Find records inside our study region

regional = gpd.sjoin(
    points,
    districts,
    how="inner",
    predicate="intersects"
)

unique_events = regional.drop_duplicates("_row_id")

print("\nREGIONAL LANDSLIDE SUMMARY")
print("Matched records:", len(unique_events))

print(
    "Records matching multiple districts:",
    regional["_row_id"].duplicated(keep=False)
    .groupby(regional["_row_id"]).any().sum()
)

print(
    "Districts with recorded landslides:",
    regional[["ihr_state", "ihr_district"]]
    .drop_duplicates().shape[0]
)

print("\nRECORDS BY STATE")
print(
    regional.groupby("ihr_state")["_row_id"]
    .nunique()
    .sort_values(ascending=False)
)

print("\nTOP 20 DISTRICTS")
print(
    regional.groupby(
        ["ihr_state", "ihr_district"]
    )["_row_id"]
    .nunique()
    .sort_values(ascending=False)
    .head(20)
)

for col in [
    "location_accuracy",
    "landslide_trigger",
    "landslide_category",
    "landslide_size"
]:
    if col in unique_events.columns:
        print(f"\nREGIONAL {col.upper()}")
        print(unique_events[col].value_counts(dropna=False))

if "event_date" in unique_events.columns:
    print("\nREGIONAL DATE EXAMPLES")
    print(unique_events["event_date"].head(20).to_string())

    print(
        "Missing event dates:",
        unique_events["event_date"].isna().sum()
    )

if "event_id" in unique_events.columns:
    print(
        "\nRepeated non-null event IDs:",
        unique_events["event_id"].dropna().duplicated().sum()
    )

print("\nInspection complete.")
print("Original dataset unchanged.")