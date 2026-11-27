
import geopandas as gpd
import pandas as pd
import rasterio
import numpy as np

from pathlib import Path
from rasterio.warp import (
    calculate_default_transform,
    reproject,
    Resampling
)
from rasterio.features import geometry_mask
from rasterio.crs import CRS
from shapely.geometry import mapping


# ==================================================
# CONFIGURATION
# ==================================================

DISTRICT_PATH = Path(
    "data/processed/boundary/ihr_districts.gpkg"
)

EXTRACTION_REPORT = Path(
    "data/processed/DEM/district_dem_extraction_report.csv"
)

INPUT_DIR = Path(
    "data/processed/DEM/district_dem"
)

OUTPUT_DIR = Path(
    "data/processed/DEM/district_dem_utm"
)

REPORT_PATH = Path(
    "data/processed/DEM/district_dem_utm_report.csv"
)

TARGET_RESOLUTION = 30.0
NODATA = -9999.0

EXPECTED_DISTRICTS = 137


# ==================================================
# HELPERS
# ==================================================

def normalize(value):
    return str(value).strip().casefold()


def safe_name(value):
    import re

    value = re.sub(
        r'[<>:"/\\|?*]',
        "_",
        str(value)
    )

    return re.sub(r"\s+", "_", value.strip())


def get_utm_crs(geometry):

    centroid = geometry.centroid

    longitude = centroid.x
    latitude = centroid.y

    zone = int((longitude + 180) // 6) + 1
    zone = max(1, min(zone, 60))

    if latitude >= 0:
        epsg = 32600 + zone
    else:
        epsg = 32700 + zone

    return CRS.from_epsg(epsg), zone


# ==================================================
# LOAD DATA
# ==================================================

print("=" * 65)
print("DISTRICT DEM REPROJECTION")
print("=" * 65)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

districts = gpd.read_file(DISTRICT_PATH)

extraction = pd.read_csv(
    EXTRACTION_REPORT
)

if len(districts) != EXPECTED_DISTRICTS:
    raise ValueError("Unexpected district count.")

if len(extraction) != EXPECTED_DISTRICTS:
    raise ValueError("Unexpected extraction report count.")

if not districts.crs:
    raise ValueError("District CRS missing.")

if not districts.geometry.is_valid.all():
    raise ValueError("Invalid district geometries.")

districts = districts.to_crs("EPSG:4326")

district_lookup = {}

for _, row in districts.iterrows():

    key = (
        normalize(row["state_name"]),
        normalize(row["district"])
    )

    if key in district_lookup:
        raise ValueError(f"Duplicate district: {key}")

    district_lookup[key] = row.geometry


# ==================================================
# PROCESS DISTRICTS
# ==================================================

results = []

success_count = 0
failure_count = 0

print("\nStarting reprojection...\n")

for _, row in extraction.iterrows():

    state = str(row["state"])
    district = str(row["district"])

    state_code = str(
        districts.loc[
            districts["district"].apply(normalize)
            == normalize(district),
            "state"
        ].iloc[0]
    ) if False else None

    key = (
        normalize(state),
        normalize(district)
    )

    output_path = None

    try:

        if row["status"] != "SUCCESS":
            raise ValueError("Source DEM extraction failed.")

        if key not in district_lookup:
            raise ValueError(f"District not found: {key}")

        geometry = district_lookup[key]

        source_path = Path(str(row["output"]))

        if not source_path.is_file():
            raise FileNotFoundError(source_path)

        # Determine appropriate UTM zone.

        target_crs, utm_zone = get_utm_crs(geometry)

        # State code from original GeoPackage.

        matching_district = districts[
            (districts["state_name"].apply(normalize) == normalize(state))
            & (districts["district"].apply(normalize) == normalize(district))
        ]

        if len(matching_district) != 1:
            raise ValueError("Unable to uniquely identify district.")

        state_code = str(
            matching_district.iloc[0]["state"]
        )

        output_path = (
            OUTPUT_DIR
            / safe_name(state_code)
            / f"{safe_name(district)}.tif"
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with rasterio.open(source_path) as src:

            if src.crs is None:
                raise ValueError("Source CRS missing.")

            if src.crs != CRS.from_epsg(4326):
                raise ValueError(
                    f"Unexpected source CRS: {src.crs}"
                )

            if src.count != 1:
                raise ValueError("Expected one DEM band.")

            # Calculate projected raster dimensions.

            transform, width, height = (
                calculate_default_transform(
                    src.crs,
                    target_crs,
                    src.width,
                    src.height,
                    *src.bounds,
                    resolution=TARGET_RESOLUTION
                )
            )

            if width <= 0 or height <= 0:
                raise ValueError("Invalid output dimensions.")

            # Read source elevation.

            source_data = src.read(1)

            source_nodata = src.nodata

            if source_nodata is None:
                source_nodata = NODATA

            # Prepare destination array.

            destination = np.full(
                (height, width),
                NODATA,
                dtype=np.float32
            )

            # Reproject elevation.

            reproject(
                source=source_data,
                destination=destination,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=transform,
                dst_crs=target_crs,
                dst_nodata=NODATA,
                resampling=Resampling.bilinear,
                init_dest_nodata=True
            )

            # Transform district geometry to UTM.

            geometry_utm = gpd.GeoSeries(
                [geometry],
                crs="EPSG:4326"
            ).to_crs(target_crs).iloc[0]

            # Apply exact district boundary mask.

            inside_mask = geometry_mask(
                [mapping(geometry_utm)],
                out_shape=destination.shape,
                transform=transform,
                invert=True
            )

            destination[~inside_mask] = NODATA

            # Validate pixels.

            valid = (
                np.isfinite(destination)
                & (destination != NODATA)
                & inside_mask
            )

            valid_pixels = int(valid.sum())

            if valid_pixels == 0:
                raise ValueError(
                    "No valid pixels after reprojection."
                )

            values = destination[valid]

            # Output metadata.

            profile = {
                "driver": "GTiff",
                "height": height,
                "width": width,
                "count": 1,
                "dtype": "float32",
                "crs": target_crs,
                "transform": transform,
                "nodata": NODATA,
                "compress": "deflate",
                "predictor": 3,
                "tiled": True,
                "BIGTIFF": "IF_SAFER"
            }

            with rasterio.open(
                output_path,
                "w",
                **profile
            ) as dst:

                dst.write(destination, 1)

        success_count += 1

        print(
            f"[SUCCESS] {state} - {district} "
            f"| UTM {utm_zone}N "
            f"| {valid_pixels:,} pixels"
        )

        results.append({
            "state": state,
            "district": district,
            "status": "SUCCESS",
            "utm_zone": utm_zone,
            "target_crs": target_crs.to_string(),
            "width": width,
            "height": height,
            "resolution_x": TARGET_RESOLUTION,
            "resolution_y": TARGET_RESOLUTION,
            "valid_pixels": valid_pixels,
            "min_elevation": float(values.min()),
            "max_elevation": float(values.max()),
            "mean_elevation": float(values.mean()),
            "output": str(output_path),
            "error": ""
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
            "output": str(output_path) if output_path else "",
            "error": str(error)
        })


# ==================================================
# SAVE REPORT
# ==================================================

report = pd.DataFrame(results)

report.to_csv(
    REPORT_PATH,
    index=False
)


# ==================================================
# FINAL SUMMARY
# ==================================================

print("\n" + "=" * 65)
print("REPROJECTION SUMMARY")
print("=" * 65)

print(f"Total districts : {EXPECTED_DISTRICTS}")
print(f"Successful      : {success_count}")
print(f"Failed          : {failure_count}")

print(f"\nOutput directory: {OUTPUT_DIR}")
print(f"Report: {REPORT_PATH}")

print("=" * 65)

if success_count == EXPECTED_DISTRICTS and failure_count == 0:
    print("ALL DISTRICTS REPROJECTED SUCCESSFULLY.")
else:
    print("REPROJECTION ISSUES FOUND.")
