from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio

from shapely.geometry import box
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parent.parent

RAW = ROOT / "data/raw/landcover"
BOUNDARY = ROOT / "data/processed/boundary/ihr_districts.gpkg"
OUT = ROOT / "data/processed/landcover"

OUT.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("ESA WORLDCOVER TILE INVENTORY")
print("=" * 65)

districts = gpd.read_file(BOUNDARY).to_crs("EPSG:4326")

if len(districts) != 137:
    raise RuntimeError("Unexpected district count.")

tiles = sorted(
    p for p in RAW.rglob("*")
    if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}
)

if not tiles:
    raise RuntimeError("No land cover GeoTIFFs found.")

print("Districts:", len(districts))
print("GeoTIFFs found:", len(tiles))

records = []
valid_footprints = []
valid_paths = []

# --------------------------------------------------
# 1. Inspect tile metadata
# --------------------------------------------------

for i, path in enumerate(tiles, 1):

    try:
        with rasterio.open(path) as src:

            b = src.bounds

            correct_product = (
                path.name.startswith(
                    "ESA_WorldCover_10m_2021_v200_"
                )
                and path.name.endswith("_Map.tif")
            )

            metadata_ok = (
                correct_product
                and src.crs is not None
                and src.crs.to_epsg() == 4326
                and src.count == 1
                and src.dtypes[0] == "uint8"
                and src.nodata == 0
                and src.width == 36000
                and src.height == 36000
                and np.isclose(
                    src.res[0], 1 / 12000,
                    rtol=0, atol=1e-10
                )
                and np.isclose(
                    src.res[1], 1 / 12000,
                    rtol=0, atol=1e-10
                )
                and src.transform.a > 0
                and src.transform.e < 0
                and src.transform.b == 0
                and src.transform.d == 0
            )

            footprint = box(
                b.left, b.bottom, b.right, b.top
            )

            relevant = False

            if src.crs is not None and src.crs.to_epsg() == 4326:
                relevant = bool(
                    districts.geometry.intersects(
                        footprint
                    ).any()
                )

            records.append({
                "file": str(path.relative_to(RAW)),
                "size_mb": path.stat().st_size / 1024**2,
                "crs": str(src.crs),
                "width": src.width,
                "height": src.height,
                "bands": src.count,
                "dtype": src.dtypes[0],
                "nodata": src.nodata,
                "pixel_x": src.res[0],
                "pixel_y": src.res[1],
                "west": b.left,
                "south": b.bottom,
                "east": b.right,
                "north": b.top,
                "relevant": relevant,
                "status": "PASS" if metadata_ok else "REVIEW"
            })

            if metadata_ok and relevant:
                valid_footprints.append(footprint)
                valid_paths.append(path.name)

    except Exception as e:

        records.append({
            "file": str(path.relative_to(RAW)),
            "status": "ERROR",
            "error": str(e)
        })

    print(f"Inspected {i}/{len(tiles)}")

inventory = pd.DataFrame(records)

inventory.to_csv(
    OUT / "landcover_tile_inventory.csv",
    index=False
)

print("\nTILE METADATA SUMMARY")
print(inventory["status"].value_counts())

print("Relevant valid tiles:", len(valid_paths))

if not valid_footprints:
    raise RuntimeError("No valid relevant tiles.")

if len(set(valid_paths)) != len(valid_paths):
    print("WARNING: Duplicate relevant tile filenames.")

# --------------------------------------------------
# 2. District footprint coverage audit
# --------------------------------------------------

combined = unary_union(valid_footprints)

districts_ea = districts.to_crs("EPSG:6933")

combined_ea = gpd.GeoSeries(
    [combined],
    crs="EPSG:4326"
).to_crs("EPSG:6933").iloc[0]

coverage_records = []

print("\nDISTRICT FOOTPRINT COVERAGE")

for i in range(len(districts)):

    row = districts.iloc[i]

    geom = districts_ea.geometry.iloc[i]

    if not geom.is_valid:
        geom = geom.make_valid()

    covered = geom.intersection(combined_ea).area

    pct = min(100.0, covered / geom.area * 100)

    status = "PASS" if pct >= 99.999 else "REVIEW"

    required = [
        name
        for name, footprint in zip(
            valid_paths, valid_footprints
        )
        if row.geometry.intersects(footprint)
    ]

    coverage_records.append({
        "state": row["state"],
        "district": row["district"],
        "required_tiles": len(required),
        "tile_names": ";".join(required),
        "footprint_coverage_pct": pct,
        "status": status
    })

    print(
        f"[{i+1}/{len(districts)}] "
        f"{row['state']}/{row['district']} | "
        f"{pct:.6f}% | {status}"
    )

coverage = pd.DataFrame(coverage_records)

coverage.to_csv(
    OUT / "landcover_district_tile_coverage.csv",
    index=False
)

print("\n" + "=" * 65)
print("FINAL LAND COVER INVENTORY SUMMARY")
print("=" * 65)

print("Total tiles:", len(tiles))
print("Relevant valid tiles:", len(valid_paths))

print("\nMetadata status:")
print(inventory["status"].value_counts())

print("\nDistrict footprint coverage:")
print(coverage["status"].value_counts())

print("\nCoverage statistics:")
print(
    coverage["footprint_coverage_pct"].describe()
)

print("\nLowest coverage districts:")
print(
    coverage.nsmallest(
        5, "footprint_coverage_pct"
    ).to_string(index=False)
)

print("\nReports saved to:", OUT)
print("Original files unchanged.")
print("=" * 65)