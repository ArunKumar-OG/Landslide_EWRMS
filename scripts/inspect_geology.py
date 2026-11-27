import geopandas as gpd
from pathlib import Path

gdb_path = Path(
    "data/raw/geology/LiMW_GIS 2015.gdb"
).resolve()

layer_name = "GLiM_export"

try:
    gdf = gpd.read_file(
        gdb_path,
        layer=layer_name,
        engine="pyogrio"
    )

    print("\n========== GEOLOGY DATASET ==========")

    print("\nShape:", gdf.shape)

    print("\nCRS:")
    print(gdf.crs)

    print("\nGeometry types:")
    print(gdf.geometry.geom_type.value_counts())

    print("\nColumns:")
    print(gdf.columns.tolist())

    print("\nFirst 5 records:")
    print(gdf.head())

    print("\nMissing values:")
    print(gdf.isnull().sum())

    print("\nGeographic bounds:")
    print(gdf.total_bounds)

except Exception as e:
    print("Error:", repr(e))
