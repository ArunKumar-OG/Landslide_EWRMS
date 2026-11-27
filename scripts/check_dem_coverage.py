
from pathlib import Path
import csv

import geopandas as gpd
import rasterio
from rasterio.warp import transform_bounds
from shapely.geometry import box


# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DEM_DIR = ROOT / "data" / "raw" / "DEM"

DISTRICT_FILE = (
    ROOT / "data" / "processed" / "boundary" / "ihr_districts.gpkg"
)

OUTPUT_DIR = ROOT / "data" / "processed" / "DEM"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# LOAD DISTRICTS
# --------------------------------------------------

print("=" * 60)
print("DEM TILE COVERAGE VALIDATION")
print("=" * 60)

print("\nLoading district boundaries...")

districts = gpd.read_file(
    DISTRICT_FILE,
    layer="ihr_districts"
)

if districts.empty:
    raise ValueError("District dataset is empty.")

if districts.crs is None:
    raise ValueError("District dataset has no CRS.")

if not {"state_name", "district"}.issubset(districts.columns):
    raise ValueError(
        "Expected columns 'state_name' and 'district' "
        "were not found."
    )

# Convert district geometries to geographic coordinates
districts = districts.to_crs("EPSG:4326")

districts["district_key"] = (
    districts["state_name"].astype(str)
    + " | "
    + districts["district"].astype(str)
)

print(f"Districts loaded: {len(districts)}")


# --------------------------------------------------
# FIND DEM TILES
# --------------------------------------------------

dem_files = sorted(DEM_DIR.glob("*.tif"))

if not dem_files:
    raise FileNotFoundError(
        f"No .tif files found in {DEM_DIR}"
    )

print(f"DEM tiles found: {len(dem_files)}")


# --------------------------------------------------
# COVERAGE TRACKING
# --------------------------------------------------

district_tiles = {
    key: []
    for key in districts["district_key"]
}

tile_records = []

invalid_tiles = []
unmatched_tiles = []


# --------------------------------------------------
# CHECK EACH TILE
# --------------------------------------------------

for index, dem_path in enumerate(dem_files, start=1):

    try:
        with rasterio.open(dem_path) as src:

            if src.crs is None:
                raise ValueError("Missing raster CRS")

            # Convert raster bounds into EPSG:4326
            bounds = transform_bounds(
                src.crs,
                "EPSG:4326",
                *src.bounds,
                densify_pts=21
            )

            tile_geometry = box(*bounds)

            intersecting_districts = []

            for _, district in districts.iterrows():

                if tile_geometry.intersects(district.geometry):

                    key = district["district_key"]

                    district_tiles[key].append(
                        dem_path.name
                    )

                    intersecting_districts.append(key)

            tile_records.append({
                "tile": dem_path.name,
                "crs": str(src.crs),
                "width": src.width,
                "height": src.height,
                "resolution_x": src.res[0],
                "resolution_y": src.res[1],
                "intersecting_district_count": len(
                    intersecting_districts
                ),
                "intersecting_districts": "; ".join(
                    intersecting_districts
                )
            })

            if not intersecting_districts:
                unmatched_tiles.append(dem_path.name)

    except Exception as error:

        invalid_tiles.append({
            "tile": dem_path.name,
            "error": str(error)
        })

    if index % 25 == 0 or index == len(dem_files):
        print(f"Processed {index}/{len(dem_files)} tiles")


# --------------------------------------------------
# DISTRICT COVERAGE REPORT
# --------------------------------------------------

district_records = []

for key, tiles in district_tiles.items():

    state_name, district_name = key.split(" | ", 1)

    unique_tiles = sorted(set(tiles))

    district_records.append({
        "state": state_name,
        "district": district_name,
        "intersecting_tile_count": len(unique_tiles),
        "coverage_status": (
            "COVERED" if unique_tiles else "NO TILE"
        ),
        "tiles": "; ".join(unique_tiles)
    })


# --------------------------------------------------
# SAVE REPORTS
# --------------------------------------------------

tile_report = OUTPUT_DIR / "dem_tile_coverage.csv"
district_report = OUTPUT_DIR / "district_dem_coverage.csv"

with open(tile_report, "w", newline="", encoding="utf-8") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=tile_records[0].keys()
        if tile_records else []
    )

    if tile_records:
        writer.writeheader()
        writer.writerows(tile_records)


with open(district_report, "w", newline="", encoding="utf-8") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=district_records[0].keys()
    )

    writer.writeheader()
    writer.writerows(district_records)


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

covered = sum(
    1 for record in district_records
    if record["coverage_status"] == "COVERED"
)

uncovered = len(district_records) - covered

tiles_used = sum(
    1 for record in tile_records
    if record["intersecting_district_count"] > 0
)

print("\n" + "=" * 60)
print("COVERAGE SUMMARY")
print("=" * 60)

print(f"Total DEM tiles: {len(dem_files)}")
print(f"Successfully inspected tiles: {len(tile_records)}")
print(f"Tiles intersecting districts: {tiles_used}")
print(f"Tiles not intersecting districts: {len(unmatched_tiles)}")
print(f"Invalid/unreadable tiles: {len(invalid_tiles)}")

print(f"\nTotal districts: {len(district_records)}")
print(f"Districts with DEM tile coverage: {covered}")
print(f"Districts without DEM tile coverage: {uncovered}")

if uncovered:
    print("\nDISTRICTS WITHOUT TILE COVERAGE:")

    for record in district_records:
        if record["coverage_status"] == "NO TILE":
            print(
                f"- {record['district']}, {record['state']}"
            )

if invalid_tiles:
    print("\nINVALID TILES:")

    for item in invalid_tiles:
        print(f"- {item['tile']}: {item['error']}")

print("\nReports saved:")
print(tile_report)
print(district_report)

print("\nNOTE: This checks spatial intersection,")
print("not valid elevation pixels or complete raster coverage.")
print("=" * 60)
