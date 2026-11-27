
import geopandas as gpd
import pandas as pd
import rasterio
import numpy as np
import re

from pathlib import Path
from contextlib import ExitStack
from rasterio.merge import merge
from rasterio.features import geometry_mask
from rasterio.crs import CRS
from shapely.geometry import mapping


# ==================================================
# CONFIGURATION
# ==================================================

DEM_DIR = Path("data/raw/dem")

DISTRICT_PATH = Path(
    "data/processed/boundary/ihr_districts.gpkg"
)

COVERAGE_PATH = Path(
    "data/processed/DEM/district_dem_coverage.csv"
)

OUTPUT_DIR = Path(
    "data/processed/DEM/district_dem"
)

REPORT_PATH = Path(
    "data/processed/DEM/district_dem_extraction_report.csv"
)

EXPECTED_DISTRICTS = 137

TARGET_CRS = CRS.from_epsg(4326)

# Used only if source DEM has no defined NoData value.
DEFAULT_NODATA = -9999.0


# ==================================================
# HELPER FUNCTIONS
# ==================================================

def normalize(value):
    return str(value).strip().casefold()


def safe_name(value):
    value = re.sub(
        r'[<>:"/\\|?*]',
        "_",
        str(value)
    )

    return re.sub(r"\s+", "_", value.strip())


# ==================================================
# LOAD DATA
# ==================================================

print("=" * 65)
print("DISTRICT-WISE DEM EXTRACTION")
print("=" * 65)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

districts = gpd.read_file(DISTRICT_PATH)
coverage = pd.read_csv(COVERAGE_PATH)

print(f"\nDistrict records: {len(districts)}")
print(f"Coverage records: {len(coverage)}")

# Basic validation

if len(districts) != EXPECTED_DISTRICTS:
    raise ValueError("Unexpected district count.")

if len(coverage) != EXPECTED_DISTRICTS:
    raise ValueError("Coverage report count mismatch.")

if districts.crs is None:
    raise ValueError("District CRS is missing.")

if not districts.geometry.is_valid.all():
    raise ValueError("Invalid district geometries found.")

if districts.geometry.is_empty.any():
    raise ValueError("Empty district geometries found.")

required_columns = {
    "state", "district"
}

if not required_columns.issubset(districts.columns):
    raise ValueError("Missing district columns.")

if not {
    "state",
    "district",
    "coverage_status",
    "tiles"
}.issubset(coverage.columns):
    raise ValueError("Missing coverage CSV columns.")


# ==================================================
# MATCH DISTRICTS
# ==================================================

# districts["_state_key"] = districts["state"].apply(normalize)
# districts["_district_key"] = districts["district"].apply(normalize)

districts["_state_key"] = (
    districts["state_name"].apply(normalize)
)

districts["_district_key"] = (
    districts["district"].apply(normalize)
)

coverage["_state_key"] = coverage["state"].apply(normalize)
coverage["_district_key"] = coverage["district"].apply(normalize)

district_keys = list(zip(
    districts["_state_key"],
    districts["_district_key"]
))

coverage_keys = list(zip(
    coverage["_state_key"],
    coverage["_district_key"]
))

if len(set(district_keys)) != EXPECTED_DISTRICTS:
    raise ValueError("Duplicate district keys in GeoPackage.")

if len(set(coverage_keys)) != EXPECTED_DISTRICTS:
    raise ValueError("Duplicate district keys in coverage CSV.")

coverage_lookup = {
    key: row
    for key, (_, row) in zip(coverage_keys, coverage.iterrows())
}

missing = set(district_keys) - set(coverage_keys)

if missing:
    raise ValueError(
        f"Districts missing from coverage CSV: {missing}"
    )

print("District matching: PASSED")


# ==================================================
# PREPARE EXTRACTION
# ==================================================

results = []

success_count = 0
failure_count = 0

print("\nStarting extraction...\n")


# ==================================================
# PROCESS ONE DISTRICT AT A TIME
# ==================================================

# for index, district in districts.iterrows():

#     state = str(district["state"])
#     district_name = str(district["district"])

#     key = (
#         normalize(state),
#         normalize(district_name)
#     )

for index, district in districts.iterrows():

    state = str(district["state_name"])
    state_code = str(district["state"])
    district_name = str(district["district"])

    key = (
        normalize(state),
        normalize(district_name)
    )

    coverage_row = coverage_lookup[key]

    output_path = (
    OUTPUT_DIR
    / safe_name(state_code)
    / f"{safe_name(district_name)}.tif"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )
    tile_names = []
    try:

        if coverage_row["coverage_status"] != "COVERED":
            raise ValueError("District is not marked COVERED.")

        tile_names = [
            name.strip()
            for name in str(coverage_row["tiles"]).split(";")
            if name.strip()
        ]

        if not tile_names:
            raise ValueError("No DEM tiles listed.")

        tile_paths = [
            DEM_DIR / name
            for name in tile_names
        ]

        missing_tiles = [
            str(path)
            for path in tile_paths
            if not path.is_file()
        ]

        if missing_tiles:
            raise FileNotFoundError(
                f"Missing DEM tiles: {missing_tiles}"
            )

        # Open only the tiles required by this district.

        with ExitStack() as stack:

            sources = [
                stack.enter_context(rasterio.open(path))
                for path in tile_paths
            ]

            source_crs = sources[0].crs

            if source_crs != TARGET_CRS:
                raise ValueError(
                    f"Expected EPSG:4326, found {source_crs}"
                )

            for src in sources:

                if src.crs != source_crs:
                    raise ValueError("Inconsistent tile CRS.")

                if src.count < 1:
                    raise ValueError("DEM has no raster bands.")

            # Transform district geometry into DEM CRS.

            geometry = gpd.GeoSeries(
                [district.geometry],
                crs=districts.crs
            ).to_crs(source_crs).iloc[0]

            # Determine NoData.

            nodata_values = {
                src.nodata for src in sources
            }

            if len(nodata_values) > 1:
                raise ValueError(
                    "Inconsistent source NoData values."
                )

            source_nodata = sources[0].nodata

            dst_nodata = (
                float(source_nodata)
                if source_nodata is not None
                else DEFAULT_NODATA
            )

            # Merge only the tiles intersecting this district,
            # restricted to the district's bounding box.

            mosaic, transform = merge(
                sources,
                bounds=geometry.bounds,
                nodata=dst_nodata,
                dtype="float32",
                indexes=1
            )

            elevation = mosaic[0]

            # Create a mask for the exact district boundary.

            inside_mask = geometry_mask(
                [mapping(geometry)],
                out_shape=elevation.shape,
                transform=transform,
                invert=True
            )

            # Set pixels outside district to NoData.

            elevation[~inside_mask] = dst_nodata

            valid_mask = (
                inside_mask
                & np.isfinite(elevation)
                & (elevation != dst_nodata)
            )

            valid_pixels = int(valid_mask.sum())

            if valid_pixels == 0:
                raise ValueError(
                    "No valid elevation pixels inside district."
                )

            # Output GeoTIFF profile.

            profile = sources[0].profile.copy()

            profile.update({
                "driver": "GTiff",
                "height": elevation.shape[0],
                "width": elevation.shape[1],
                "count": 1,
                "dtype": "float32",
                "crs": source_crs,
                "transform": transform,
                "nodata": dst_nodata,
                "compress": "deflate",
                "predictor": 3,
                "tiled": True,
                "BIGTIFF": "IF_SAFER"
            })

            # Write district raster.

            with rasterio.open(
                output_path,
                "w",
                **profile
            ) as dst:

                dst.write(elevation.astype("float32"), 1)

        success_count += 1

        print(
            f"[SUCCESS] {state} - {district_name} "
            f"| {valid_pixels:,} valid pixels"
        )

        results.append({
            "state": state,
            "district": district_name,
            "status": "SUCCESS",
            "tile_count": len(tile_names),
            "valid_pixels": valid_pixels,
            "output": str(output_path),
            "error": ""
        })

    except Exception as error:

        failure_count += 1

        print(
            f"[FAILED] {state} - {district_name}: {error}"
        )

        results.append({
            "state": state,
            "district": district_name,
            "status": "FAILED",
            "tile_count": len(tile_names) if "tile_names" in locals() else 0,
            "valid_pixels": 0,
            "output": "",
            "error": str(error)
        })


# ==================================================
# SAVE EXTRACTION REPORT
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
print("DEM EXTRACTION SUMMARY")
print("=" * 65)

print(f"Total districts : {len(districts)}")
print(f"Successful      : {success_count}")
print(f"Failed          : {failure_count}")

print(f"\nOutput directory: {OUTPUT_DIR}")
print(f"Extraction report: {REPORT_PATH}")

print("=" * 65)

if failure_count == 0:
    print("ALL DISTRICTS PROCESSED SUCCESSFULLY.")
else:
    print("SOME DISTRICTS FAILED. CHECK THE REPORT.")
