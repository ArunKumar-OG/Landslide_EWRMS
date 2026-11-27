
import geopandas as gpd
from pathlib import Path

file_path = Path(
    "data/processed/geology/gliм_provisional_region.gpkg"
)

print("Checking saved geology dataset...")

gdf = gpd.read_file(file_path)

print("\nShape:", gdf.shape)
print("\nCRS:", gdf.crs)

print("\nColumns:")
print(gdf.columns.tolist())

print("\nLithology classes:")
print(gdf["Litho"].nunique())

print("\nMissing values:")
print(gdf.isnull().sum())

print("\nBounds:")
print(gdf.total_bounds)

print("\nFirst 5 records:")
print(gdf.head())
