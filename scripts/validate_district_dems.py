
import geopandas as gpd
import pandas as pd
import rasterio
import numpy as np

from pathlib import Path
from rasterio.features import geometry_mask
from shapely.geometry import mapping


# ==================================================
# PATHS
# ==================================================

DISTRICT_PATH = Path(
    "data/processed/boundary/ihr_districts.gpkg"
)

EXTRACTION_REPORT = Path(
    "data/processed/DEM/district_dem_extraction_report.csv"
)

DEM_DIR = Path(
    "data/processed/DEM/district_dem"
)

AUDIT_REPORT = Path(
    "data/processed/DEM/district_dem_validation_report.csv"
)

EXPECTED_DISTRICTS = 137

TARGET_CRS = "EPSG:4326"


# ==================================================
# HELPERS
# ==================================================

def normalize(value):
    return str(value).strip().casefold()


# ==================================================
# LOAD DATA
# ==================================================

print("=" * 65)
print("DISTRICT DEM QUALITY VALIDATION")
print("=" * 65)

districts = gpd.read_file(DISTRICT_PATH)
extraction = pd.read_csv(EXTRACTION_REPORT)

if len(districts) != EXPECTED_DISTRICTS:
    raise ValueError("Unexpected district count.")

if len(extraction) != EXPECTED_DISTRICTS:
    raise ValueError("Unexpected extraction report count.")

if not districts.crs:
    raise ValueError("District CRS missing.")

districts = districts.to_crs(TARGET_CRS)

district_lookup = {}

for _, row in districts.iterrows():

    key = (
        normalize(row["state_name"]),
        normalize(row["district"])
    )

    district_lookup[key] = row.geometry


# ==================================================
# FILE INVENTORY
# ==================================================

raster_files = list(DEM_DIR.rglob("*.tif"))

print(f"\nExpected rasters: {EXPECTED_DISTRICTS}")
print(f"Found rasters: {len(raster_files)}")

results = []
resolutions = []

success_count = 0
failure_count = 0

# Check for duplicate output paths.
if extraction["output"].dropna().duplicated().any():
    raise ValueError("Duplicate output paths in extraction report.")


# ==================================================
# VALIDATE EACH DISTRICT
# ==================================================

for _, row in extraction.iterrows():

    state = str(row["state"])
    district = str(row["district"])

    key = (
        normalize(state),
        normalize(district)
    )

    issues = []

    try:

        if row["status"] != "SUCCESS":
            raise ValueError("Extraction report shows failure.")

        if key not in district_lookup:
            raise ValueError("District geometry not found.")

        raster_path = Path(str(row["output"]))

        if not raster_path.is_file():
            raise FileNotFoundError(
                f"Raster missing: {raster_path}"
            )

        geometry = district_lookup[key]

        with rasterio.open(raster_path) as src:

            # Basic metadata

            if src.crs is None or src.crs.to_string() != TARGET_CRS:
                issues.append("Incorrect CRS")

            if src.count != 1:
                issues.append("Unexpected band count")

            if src.dtypes[0] != "float32":
                issues.append("Unexpected data type")

            if src.nodata is None:
                issues.append("NoData not defined")

            if src.width <= 0 or src.height <= 0:
                issues.append("Invalid raster dimensions")

            resolution = (
                abs(src.transform.a),
                abs(src.transform.e)
            )

            resolutions.append(resolution)

            elevation = src.read(1)

            # Valid pixel detection

            valid = np.isfinite(elevation)

            if src.nodata is not None:
                valid &= elevation != src.nodata

            valid_count = int(valid.sum())
            total_pixels = elevation.size

            if valid_count == 0:
                issues.append("No valid elevation pixels")

            # Clipping validation

            inside_mask = geometry_mask(
                [mapping(geometry)],
                out_shape=elevation.shape,
                transform=src.transform,
                invert=True
            )

            outside_valid = valid & ~inside_mask

            outside_count = int(outside_valid.sum())

            if outside_count > 0:
                issues.append(
                    f"{outside_count} valid pixels outside boundary"
                )

            # Elevation statistics

            if valid_count > 0:

                values = elevation[valid]

                min_elevation = float(values.min())
                max_elevation = float(values.max())
                mean_elevation = float(values.mean())

                if min_elevation < -500 or max_elevation > 9000:
                    issues.append(
                        "Elevation outside expected screening range"
                    )

            else:

                min_elevation = None
                max_elevation = None
                mean_elevation = None

            # Pixel statistics

            nodata_count = total_pixels - valid_count

            valid_percentage = (
                valid_count / total_pixels * 100
                if total_pixels else 0
            )

            # Result

            if issues:
                status = "FAILED"
                failure_count += 1
            else:
                status = "PASSED"
                success_count += 1

            print(
                f"[{status}] {state} - {district} "
                f"| {valid_count:,} valid pixels"
            )

            results.append({
                "state": state,
                "district": district,
                "status": status,
                "crs": str(src.crs),
                "dtype": src.dtypes[0],
                "width": src.width,
                "height": src.height,
                "pixel_x": resolution[0],
                "pixel_y": resolution[1],
                "valid_pixels": valid_count,
                "nodata_pixels": nodata_count,
                "valid_percentage": valid_percentage,
                "outside_boundary_pixels": outside_count,
                "min_elevation_m": min_elevation,
                "max_elevation_m": max_elevation,
                "mean_elevation_m": mean_elevation,
                "issues": "; ".join(issues),
                "output": str(raster_path)
            })

    except Exception as error:

        failure_count += 1

        print(
            f"[FAILED] {state} - {district}: {error}"
        )

        results.append({
            "state": state,
            "district": district,
            "status": "FAILED",
            "issues": str(error),
            "output": str(row.get("output", ""))
        })


# ==================================================
# CHECK FILE COUNT AND RESOLUTION
# ==================================================

if len(raster_files) != EXPECTED_DISTRICTS:
    print("\nWARNING: Raster file count mismatch!")

if resolutions:

    unique_resolutions = set(
        (round(x, 10), round(y, 10))
        for x, y in resolutions
    )

    print(
        "\nUnique raster resolutions:",
        len(unique_resolutions)
    )

    if len(unique_resolutions) > 1:
        print("WARNING: Resolution differences detected.")


# ==================================================
# SAVE REPORT
# ==================================================

report = pd.DataFrame(results)

report.to_csv(
    AUDIT_REPORT,
    index=False
)


# ==================================================
# FINAL SUMMARY
# ==================================================

print("\n" + "=" * 65)
print("DEM VALIDATION SUMMARY")
print("=" * 65)

print(f"Expected districts : {EXPECTED_DISTRICTS}")
print(f"Raster files found : {len(raster_files)}")
print(f"Validation passed : {success_count}")
print(f"Validation failed : {failure_count}")

print(f"\nAudit report: {AUDIT_REPORT}")

print("=" * 65)

if (
    success_count == EXPECTED_DISTRICTS
    and failure_count == 0
    and len(raster_files) == EXPECTED_DISTRICTS
):
    print("ALL DISTRICT DEMs PASSED VALIDATION.")
else:
    print("VALIDATION ISSUES FOUND. REVIEW THE REPORT.")
