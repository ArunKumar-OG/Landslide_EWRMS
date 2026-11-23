
import geopandas as gpd
from pathlib import Path

files = list(Path("data/raw").rglob("district_nwic.GeoJSON"))

if not files:
    raise FileNotFoundError("district_nwic.GeoJSON not found")

source = files[0]

print(f"Reading: {source}")

gdf = gpd.read_file(source)

print("\n--- BASIC INFORMATION ---")
print("Features:", len(gdf))
print("CRS:", gdf.crs)
print("Geometry types:")
print(gdf.geom_type.value_counts())

print("\n--- COLUMNS ---")
print(gdf.columns.tolist())

print("\n--- FIRST FIVE RECORDS ---")
print(gdf.drop(columns="geometry").head().to_string())

print("\n--- UNIQUE VALUES BY COLUMN ---")
for col in gdf.columns:
    if col != "geometry" and gdf[col].nunique() < 100:
        print(f"\n{col}:")
        print(sorted(gdf[col].dropna().astype(str).unique()))

print("\n--- GEOMETRY VALIDITY ---")
print("Invalid geometries:", (~gdf.geometry.is_valid).sum())
print("Empty geometries:", gdf.geometry.is_empty.sum())
