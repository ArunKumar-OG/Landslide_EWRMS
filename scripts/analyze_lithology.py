import geopandas as gpd
from pathlib import Path

gdb_path = Path(
    "data/raw/geology/LiMW_GIS 2015.gdb"
).resolve()

gdf = gpd.read_file(
    gdb_path,
    layer="GLiM_export",
    engine="fiona"
)

# Extract primary lithology code
gdf["Litho_primary"] = gdf["Litho"].str[:2]

print("\nPrimary lithology distribution:")
print(gdf["Litho_primary"].value_counts())

print("\nComplete Litho code distribution:")
print(gdf["Litho"].value_counts().head(20))

print("\nAdditional attribute xx:")
print(gdf["xx"].value_counts().head(20))