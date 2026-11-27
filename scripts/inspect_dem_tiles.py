
import rasterio
from rasterio.warp import transform_bounds
from pathlib import Path
import csv

BASE_DIR = Path(__file__).resolve().parents[1]
DEM_DIR = BASE_DIR / "data" / "raw" / "dem"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

# Temporary broad screening boundary, not the final IHR boundary
REGION = {
    "min_lon": 72,
    "min_lat": 21,
    "max_lon": 98,
    "max_lat": 38
}

records = []

files = sorted(DEM_DIR.glob("*.tif"))

print(f"Total DEM tiles found: {len(files)}")

for file in files:
    try:
        with rasterio.open(file) as src:

            if src.crs is None:
                print("Missing CRS:", file.name)
                continue

            bounds = transform_bounds(
                src.crs,
                "EPSG:4326",
                *src.bounds,
                densify_pts=21
            )

            min_lon, min_lat, max_lon, max_lat = bounds

            intersects = (
                max_lon > REGION["min_lon"]
                and min_lon < REGION["max_lon"]
                and max_lat > REGION["min_lat"]
                and min_lat < REGION["max_lat"]
            )

            records.append({
                "filename": file.name,
                "width": src.width,
                "height": src.height,
                "crs": str(src.crs),
                "resolution_x": src.res[0],
                "resolution_y": src.res[1],
                "nodata": src.nodata,
                "min_lon": min_lon,
                "min_lat": min_lat,
                "max_lon": max_lon,
                "max_lat": max_lat,
                "intersects_provisional_region": intersects
            })

            status = "CANDIDATE" if intersects else "OUTSIDE"

            print(f"{status}: {file.name}")

    except Exception as e:
        print(f"ERROR: {file.name}: {e}")

# Save inventory
csv_path = OUTPUT_DIR / "dem_tile_inventory.csv"

if records:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=records[0].keys()
        )
        writer.writeheader()
        writer.writerows(records)

candidate_count = sum(
    r["intersects_provisional_region"] for r in records
)

print("\n========== DEM INVENTORY SUMMARY ==========")
print("Total files:", len(files))
print("Successfully inspected:", len(records))
print("Candidate tiles:", candidate_count)
print("Outside provisional region:", len(records) - candidate_count)
print("Inventory saved:", csv_path)
