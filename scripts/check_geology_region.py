
import geopandas as gpd
from pathlib import Path
from pyproj import Transformer
from shapely.geometry import box

# Geology database path
gdb_path = Path(
    "data/raw/geology/LiMW_GIS 2015.gdb"
).resolve()

print("Loading geology dataset...")

gdf = gpd.read_file(
    gdb_path,
    layer="GLiM_export",
    engine="fiona"
)

print("Total global features:", len(gdf))

# Temporary approximate study boundary
study_area = box(72, 21, 98, 38)

# Transform the boundary into the geology CRS
transformer = Transformer.from_crs(
    "EPSG:4326",
    gdf.crs,
    always_xy=True
)

minx, miny, maxx, maxy = transformer.transform_bounds(
    72, 21, 98, 38,
    densify_pts=101
)

# First filter: reduce the number of polygons
print("\nFiltering candidate polygons...")

candidates = gdf.cx[
    minx:maxx,
    miny:maxy
].copy()

print("Candidate features:", len(candidates))

# Reproject only selected polygons
print("\nReprojecting candidates...")

candidates = candidates.to_crs("EPSG:4326")

# Second filter: exact intersection with the rectangle
print("\nClipping to provisional study boundary...")

region = gpd.clip(candidates, study_area)

print("Clipped features:", len(region))
print("Regional bounds:", region.total_bounds)

print("\nPrimary lithology distribution:")
print(region["Litho"].str[:2].value_counts())

# Save the provisional regional dataset
output_dir = Path("data/processed/geology")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "gliм_provisional_region.gpkg"

region.to_file(
    output_file,
    driver="GPKG"
)

print("\nSaved to:", output_file)
