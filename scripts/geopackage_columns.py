import geopandas as gpd

gdf = gpd.read_file(
    "data/processed/boundary/ihr_districts.gpkg"
)

print("Columns:")
print(gdf.columns.tolist())

print("\nFirst 5 records:")
print(gdf.drop(columns="geometry").head())