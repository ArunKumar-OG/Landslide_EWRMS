
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DEM_DIR = ROOT / "data" / "raw" / "DEM"
REPORT_DIR = ROOT / "data" / "processed" / "DEM"

TILE_REPORT = REPORT_DIR / "dem_tile_coverage.csv"

OUTPUT_FILE = REPORT_DIR / "dem_pixel_validation.csv"


# --------------------------------------------------
# LOAD RELEVANT TILES
# --------------------------------------------------

tile_df = pd.read_csv(TILE_REPORT)

relevant_tiles = tile_df.loc[
    tile_df["intersecting_district_count"] > 0,
    "tile"
].tolist()

print("=" * 60)
print("DEM PIXEL VALIDATION")
print("=" * 60)

print(f"Relevant tiles: {len(relevant_tiles)}")


# --------------------------------------------------
# VALIDATE EACH TILE
# --------------------------------------------------

results = []

for index, filename in enumerate(relevant_tiles, start=1):

    path = DEM_DIR / filename

    try:
        with rasterio.open(path) as src:

            elevation = src.read(1, masked=True)

            values = elevation.data

            valid_mask = (
                ~np.ma.getmaskarray(elevation)
                & np.isfinite(values)
            )

            valid_count = int(np.count_nonzero(valid_mask))
            total_count = int(values.size)

            invalid_count = total_count - valid_count

            valid_percentage = (
                valid_count / total_count * 100
                if total_count else 0
            )

            if valid_count:
                minimum = float(values[valid_mask].min())
                maximum = float(values[valid_mask].max())
                mean = float(values[valid_mask].mean())
            else:
                minimum = None
                maximum = None
                mean = None

            status = (
                "OK" if valid_percentage >= 99
                else "REVIEW"
            )

            results.append({
                "tile": filename,
                "valid_pixels": valid_count,
                "invalid_pixels": invalid_count,
                "total_pixels": total_count,
                "valid_percentage": round(valid_percentage, 4),
                "minimum_elevation": minimum,
                "maximum_elevation": maximum,
                "mean_elevation": mean,
                "status": status
            })

    except Exception as error:

        results.append({
            "tile": filename,
            "valid_pixels": 0,
            "invalid_pixels": 0,
            "total_pixels": 0,
            "valid_percentage": 0,
            "minimum_elevation": None,
            "maximum_elevation": None,
            "mean_elevation": None,
            "status": f"ERROR: {error}"
        })

    if index % 10 == 0 or index == len(relevant_tiles):
        print(f"Validated {index}/{len(relevant_tiles)} tiles")


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

report = pd.DataFrame(results)

report.to_csv(OUTPUT_FILE, index=False)


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print("\n" + "=" * 60)
print("PIXEL VALIDATION SUMMARY")
print("=" * 60)

print(f"Tiles checked: {len(report)}")

print(
    "Tiles OK:",
    (report["status"] == "OK").sum()
)

print(
    "Tiles requiring review:",
    (report["status"] == "REVIEW").sum()
)

print(
    "Tiles with errors:",
    report["status"].astype(str).str.startswith("ERROR").sum()
)

print(
    "Overall valid pixel percentage:",
    round(
        report["valid_pixels"].sum()
        / max(report["total_pixels"].sum(), 1) * 100,
        4
    ),
    "%"
)

print("\nElevation range across valid tiles:")

valid_tiles = report.dropna(subset=["minimum_elevation"])

if not valid_tiles.empty:
    print("Minimum:", valid_tiles["minimum_elevation"].min(), "m")
    print("Maximum:", valid_tiles["maximum_elevation"].max(), "m")

print("\nReport saved to:")
print(OUTPUT_FILE)

print("=" * 60)
