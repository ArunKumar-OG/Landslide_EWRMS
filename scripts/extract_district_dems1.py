
import geopandas as gpd
import pandas as pd
import rasterio
import numpy as np

from pathlib import Path
from contextlib import ExitStack
from rasterio.merge import merge
from rasterio.features import geometry_mask
from shapely.geometry import mapping
import re

# --------------------------------------------------
# PATHS
# --------------------------------------------------

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

EXPECTED_DISTRICTS = 137


def normalize(value):
    return str(value).strip().casefold()


def safe_name(value):
    value = re.sub(r'[<>:"/\\|?*]', "_", str(value))
    return re.sub(r"\s+", "_", value.strip())


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("=" * 65)
print("DISTRICT-WISE DEM EXTRACTION")
print("=" * 65)

districts = gpd.read_file(DISTRICT_PATH)
coverage = pd.read_csv(COVERAGE_PATH)

if len(districts) != EXPECTED_DISTRICTS:
    raise ValueError("Unexpected district count.")

if len(coverage) != EXPECTED_DISTRICTS:
    raise ValueError("Coverage report district count mismatch.")

if not districts.geometry.is_valid.all():
    raise ValueError("Invalid district geometries found.")

if districts.crs is None:
    raise ValueError("District CRS is missing.")

if districts.crs.to_epsg() != 4326:
    raise ValueError(
        "Expected EPSG:4326 district geometries. "
        "Reproject explicitly before extraction."
    )

# Create normalized lookup keys
districts["_key"] = list(zip(
    districts["state"].map(normalize),
    districts["district"].map(normalize)
))

coverage["_key"] = list(zip(
    coverage["state"].map(normalize),
    coverage["district"].map(normalize)
))

if districts["_key"].duplicated().any():
    raise ValueError("Duplicate districts in boundary file.")

if coverage["_key"].duplicated().any():
    raise ValueError("Duplicate districts in coverage report.")

geometry_lookup = {
    row["_key"]: row.geometry
    for _, row in districts.iterrows()
}

if set(geometry_lookup) != set(coverage["_key"]):
    raise ValueError(
        "District names do not match between boundary and coverage."
    )

# --------------------------------------------------
# BUILD TILE LOOKUP
# --------------------------------------------------

print("\nBuilding DEM tile lookup...")

tile_lookup = {
    path.name: path
    for path in DEM_DIR.glob("*.tif")
}

print("Available DEM tiles:", len(tile_lookup))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# PROCESS EACH DISTRICT
# --------------------------------------------------

success = 0
failed = []

for index, row in coverage.iterrows():

    state = str(row["state"])
    district = str(row["district"])

    print(
        f"\n[{index + 1}/{len(coverage)}] "
        f"{state} - {district}"
    )

    try:

        if row["coverage_status"] != "COVERED":
            raise ValueError("District is not marked COVERED.")

        tile_names = [
            name.strip()
            for name in str(row["tiles"]).split(";")
            if name.strip()
        ]

        if not tile_names:
            raise ValueError("No DEM tiles listed.")

        missing = [
            name for name in tile_names
            if name not in tile_lookup
        ]

        if missing:
            raise FileNotFoundError(
                f"Missing DEM tiles: {missing}"
            )

        geometry = geometry_lookup[row["_key"]]

        # ------------------------------------------
        # MERGE ONLY THE DISTRICT'S REQUIRED TILES
        # ------------------------------------------

        with ExitStack() as stack:

            sources = [
                stack.enter_context(
                    rasterio.open(tile_lookup[name])
                )
                for name in tile_names
            ]

            for src in sources:

                if src.crs != rasterio.crs.CRS.from_epsg(4326):
                    raise ValueError(
                        f"Unexpected CRS in {src.name}"
                    )

            nodata_values = {
                src.nodata for src in sources
            }

            if len(nodata_values) != 1:
                raise ValueError(
                    "Source tiles have inconsistent NoData values."
                )

            nodata = next(iter(nodata_values))

            if nodata is None:
                nodata = -9999.0

            mosaic, transform = merge(
                sources,
                bounds=geometry.bounds,
                nodata=nodata,
                dtype="float32"
            )

        # ------------------------------------------
        # MASK OUTSIDE THE DISTRICT
        # ------------------------------------------

        inside = geometry_mask(
            [mapping(geometry)],
            out_shape=mosaic.shape[1:],
            transform=transform,
            invert=True,
            all_touched=False
        )

        dem = mosaic[0].astype("float32")

        dem[~inside] = nodata

        # ------------------------------------------
        # CHECK VALID PIXELS
        # ------------------------------------------

        valid = (
            np.isfinite(dem)
            & (dem != nodata)
        )

        if not valid.any():
            raise ValueError("No valid elevation pixels.")

        elevation_min = float(dem[valid].min())
        elevation_max = float(dem[valid].max())
        valid_pixels = int(valid.sum())

        # ------------------------------------------
        # SAVE OUTPUT
        # ------------------------------------------

        state_folder = OUTPUT_DIR / safe_name(state)
        state_folder.mkdir(parents=True, exist_ok=True)

        output_path = (
            state_folder /
            f"{safe_name(district)}_dem.tif"
        )

        profile = {
            "driver": "GTiff",
            "height": dem.shape[0],
            "width": dem.shape[1],
            "count": 1,
            "dtype": "float32",
            "crs": "EPSG:4326",
            "transform": transform,
            "nodata": nodata,
            "compress": "deflate",
            "predictor": 3,
            "tiled": True,
            "BIGTIFF": "IF_SAFER"
        }

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(dem, 1)
            dst.set_band_description(1, "Elevation")

        success += 1

        print("  Tiles:", len(tile_names))
        print("  Valid pixels:", valid_pixels)
        print("  Elevation range:", elevation_min, "to", elevation_max)
        print("  Saved:", output_path)

    except Exception as error:

        failed.append({
            "state": state,
            "district": district,
            "error": str(error)
        })

        print("  ERROR:", error)

# --------------------------------------------------
# FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 65)
print("EXTRACTION SUMMARY")
print("=" * 65)

print("Total districts:", len(coverage))
print("Successfully processed:", success)
print("Failed:", len(failed))

if failed:

    print("\nFAILED DISTRICTS")

    for item in failed:
        print(
            item["state"],
            "|",
            item["district"],
            "|",
            item["error"]
        )

    pd.DataFrame(failed).to_csv(
        OUTPUT_DIR / "extraction_errors.csv",
        index=False
    )

print("\nOutput directory:", OUTPUT_DIR)

if success == EXPECTED_DISTRICTS:
    print("\nALL 137 DISTRICTS PROCESSED SUCCESSFULLY.")
else:
    print("\nWARNING: Extraction incomplete. Investigate failures.")
