from pathlib import Path
from math import floor, ceil

import geopandas as gpd
import pandas as pd

from shapely.geometry import box
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parent.parent

RAW = ROOT / "data/raw/landcover"
OUT = ROOT / "data/processed/landcover"

BOUNDARY = ROOT / "data/processed/boundary/ihr_districts.gpkg"
INVENTORY = OUT / "landcover_tile_inventory.csv"

print("=" * 65)
print("PITHORAGARH LAND COVER GAP DIAGNOSTIC")
print("=" * 65)

# 1. Load district boundary

districts = gpd.read_file(BOUNDARY).to_crs("EPSG:4326")

selected = districts[
    (districts["state"] == "UK")
    & (districts["district"].str.lower() == "pithoragarh")
]

if len(selected) != 1:
    raise RuntimeError("District boundary match failed.")

geom = selected.geometry.iloc[0]

if not geom.is_valid:
    geom = geom.buffer(0)

# 2. Load validated tile footprints

inventory = pd.read_csv(INVENTORY)

valid = inventory[
    inventory["status"] == "PASS"
]

footprints = []
valid_names = set()

for _, row in valid.iterrows():

    path = RAW / row["file"]

    if not path.is_file():
        raise FileNotFoundError(path)

    footprints.append(
        box(
            row["west"],
            row["south"],
            row["east"],
            row["north"]
        )
    )

    valid_names.add(path.name)

if not footprints:
    raise RuntimeError("No validated tile footprints.")

covered = unary_union(footprints)

# 3. Identify uncovered geometry

gap = geom.difference(covered)

def area_km2(geometry):

    return (
        gpd.GeoSeries(
            [geometry],
            crs="EPSG:4326"
        ).to_crs("EPSG:6933").area.iloc[0]
        / 1e6
    )

district_area = area_km2(geom)
gap_area = area_km2(gap)

print(f"\nDistrict area: {district_area:.6f} km2")
print(f"Uncovered area: {gap_area:.6f} km2")

print(
    f"Uncovered percentage: "
    f"{gap_area / district_area * 100:.6f}%"
)

if gap.is_empty or gap_area <= 0:

    print("\nNo positive-area gap detected.")
    raise SystemExit(0)

print("\nUncovered geographical bounds:")
print(gap.bounds)

# 4. Identify expected WorldCover tiles

xmin, ymin, xmax, ymax = gap.bounds

existing_names = {
    p.name for p in RAW.rglob("*")
    if p.is_file()
}

records = []

print("\nREQUIRED TILES FOR UNCOVERED AREA")
print("-" * 65)

for lat in range(
    floor(ymin / 3) * 3,
    ceil(ymax / 3) * 3,
    3
):

    for lon in range(
        floor(xmin / 3) * 3,
        ceil(xmax / 3) * 3,
        3
    ):

        tile_geom = box(
            lon, lat,
            lon + 3, lat + 3
        )

        intersection = gap.intersection(tile_geom)

        if intersection.is_empty:
            continue

        missing_area = area_km2(intersection)

        if missing_area <= 0:
            continue

        lat_code = (
            f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}"
        )

        lon_code = (
            f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}"
        )

        tile_id = lat_code + lon_code

        filename = (
            f"ESA_WorldCover_10m_2021_v200_"
            f"{tile_id}_Map.tif"
        )

        if filename in valid_names:

            status = "FOOTPRINT_REVIEW"

        elif filename in existing_names:

            status = "EXISTS_BUT_NOT_VALIDATED"

        else:

            status = "DOWNLOAD_REQUIRED"

        records.append({
            "tile_id": tile_id,
            "filename": filename,
            "uncovered_area_km2": missing_area,
            "status": status
        })

        print(f"\nTile: {tile_id}")
        print(f"Filename: {filename}")
        print(f"Uncovered area: {missing_area:.6f} km2")
        print(f"Status: {status}")

# 5. Save diagnostic report

report = pd.DataFrame(records)

report.to_csv(
    OUT / "landcover_missing_tiles.csv",
    index=False
)

print("\n" + "=" * 65)
print("DIAGNOSTIC SUMMARY")
print("=" * 65)

if not report.empty:
    print(report["status"].value_counts())
else:
    print("No missing tile identified.")

print("\nReport saved:")
print(OUT / "landcover_missing_tiles.csv")

print("\nOriginal data unchanged.")