from pathlib import Path
import re
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import mapping


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DEM_ROOT = PROJECT_DIR / "data" / "processed" / "DEM" / "district_dem_utm"
DISTRICT_FILE = PROJECT_DIR / "data" / "processed" / "boundary" / "ihr_districts.gpkg"

OUTPUT_DIR = PROJECT_DIR / "data" / "processed" / "DEM"
OUTPUT_CSV = OUTPUT_DIR / "district_dem_boundary_nodata_audit.csv"


# ============================================================
# HELPERS
# ============================================================

def normalize(value):
    """Normalize names so small formatting differences don't matter."""
    if pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def build_district_lookup(gdf):
    """
    Create a lookup using the state code and district name.
    The raster folders are expected to use state codes,
    for example: UK/Haridwar.tif
    """
    required = ["district", "state"]
    for column in required:
        if column not in gdf.columns:
            raise ValueError(
                f"Required column '{column}' not found in district file. "
                f"Available columns: {list(gdf.columns)}"
            )

    lookup = {}

    for _, row in gdf.iterrows():
        key = (
            normalize(row["state"]),
            normalize(row["district"])
        )

        if key in lookup:
            raise ValueError(f"Duplicate district boundary found: {key}")

        if row.geometry is None or row.geometry.is_empty:
            print(f"WARNING: Empty geometry for {key}")
            continue

        lookup[key] = row.geometry

    return lookup


# ============================================================
# MAIN AUDIT
# ============================================================

def audit_raster(raster_path, district_geometry, state_code, district_name):
    """Count district pixels and valid/missing DEM pixels block by block."""

    with rasterio.open(raster_path) as src:

        if src.crs is None:
            raise ValueError("Raster has no CRS.")

        # Convert the district boundary into this raster's CRS.
        district_projected = gpd.GeoSeries(
            [district_geometry],
            crs=DISTRICT_CRS
        ).to_crs(src.crs).iloc[0]

        if not district_projected.is_valid:
            district_projected = district_projected.buffer(0)

        if district_projected.is_empty:
            raise ValueError("District geometry is empty after reprojection.")

        total_pixels = src.width * src.height

        inside_pixels = 0
        valid_inside_pixels = 0

        # Process small blocks instead of loading the entire raster.
        for _, window in src.block_windows(1):

            data = src.read(1, window=window, masked=True)
            data_values = data.data
            data_mask = np.ma.getmaskarray(data)

            inside_mask = geometry_mask(
                [mapping(district_projected)],
                out_shape=(int(window.height), int(window.width)),
                transform=src.window_transform(window),
                invert=True,
                all_touched=False
            )

            valid_data = (~data_mask) & np.isfinite(data_values)

            # Explicitly exclude the raster's declared NoData value too.
            if src.nodata is not None:
                valid_data &= data_values != src.nodata

            inside_pixels += int(np.count_nonzero(inside_mask))
            valid_inside_pixels += int(
                np.count_nonzero(inside_mask & valid_data)
            )

        outside_pixels = total_pixels - inside_pixels
        missing_inside_pixels = inside_pixels - valid_inside_pixels

        coverage_percentage = (
            valid_inside_pixels / inside_pixels * 100
            if inside_pixels > 0 else 0
        )

        missing_percentage = (
            missing_inside_pixels / inside_pixels * 100
            if inside_pixels > 0 else 0
        )

        return {
            "state_code": state_code,
            "district": district_name,
            "file": str(raster_path.relative_to(PROJECT_DIR)),
            "crs": src.crs.to_string(),
            "width": src.width,
            "height": src.height,
            "total_pixels": total_pixels,
            "pixels_inside_district": inside_pixels,
            "pixels_outside_district": outside_pixels,
            "valid_dem_pixels_inside": valid_inside_pixels,
            "missing_dem_pixels_inside": missing_inside_pixels,
            "dem_coverage_percentage": round(coverage_percentage, 2),
            "missing_percentage_inside_district": round(missing_percentage, 2),
            "status": "PASS" if coverage_percentage >= 95 else "REVIEW"
        }


def main():

    print("=" * 65)
    print("DISTRICT DEM BOUNDARY AND NODATA AUDIT")
    print("=" * 65)

    if not DEM_ROOT.exists():
        raise FileNotFoundError(f"DEM directory not found: {DEM_ROOT}")

    if not DISTRICT_FILE.exists():
        raise FileNotFoundError(f"District boundary file not found: {DISTRICT_FILE}")

    print("\nLoading district boundaries...")
    districts = gpd.read_file(DISTRICT_FILE)

    if districts.crs is None:
        raise ValueError("District boundary file has no CRS.")

    global DISTRICT_CRS
    DISTRICT_CRS = districts.crs

    district_lookup = build_district_lookup(districts)

    raster_files = sorted(DEM_ROOT.rglob("*.tif"))

    print(f"District boundaries loaded: {len(district_lookup)}")
    print(f"District DEM files found: {len(raster_files)}")

    results = []

    for index, raster_path in enumerate(raster_files, start=1):

        state_code = raster_path.parent.name
        district_name = raster_path.stem

        key = (normalize(state_code), normalize(district_name))

        print(
            f"\n[{index}/{len(raster_files)}] "
            f"{state_code} - {district_name}"
        )

        if key not in district_lookup:
            print("  WARNING: Matching district boundary not found.")
            results.append({
                "state_code": state_code,
                "district": district_name,
                "file": str(raster_path.relative_to(PROJECT_DIR)),
                "status": "BOUNDARY_NOT_FOUND"
            })
            continue

        try:
            result = audit_raster(
                raster_path,
                district_lookup[key],
                state_code,
                district_name
            )

            results.append(result)

            print(
                f"  District pixels: {result['pixels_inside_district']:,}"
            )
            print(
                f"  Valid DEM pixels: {result['valid_dem_pixels_inside']:,}"
            )
            print(
                f"  Actual coverage: "
                f"{result['dem_coverage_percentage']}%"
            )
            print(f"  Status: {result['status']}")

        except Exception as error:
            print(f"  ERROR: {error}")

            results.append({
                "state_code": state_code,
                "district": district_name,
                "file": str(raster_path.relative_to(PROJECT_DIR)),
                "status": "ERROR",
                "error": str(error)
            })

    # Save results.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report = pd.DataFrame(results)
    report.to_csv(OUTPUT_CSV, index=False)

    # Summary.
    print("\n" + "=" * 65)
    print("AUDIT SUMMARY")
    print("=" * 65)

    if "status" in report.columns:
        print(report["status"].value_counts(dropna=False).to_string())

    if "dem_coverage_percentage" in report.columns:
        coverage = pd.to_numeric(
            report["dem_coverage_percentage"], errors="coerce"
        ).dropna()

        if not coverage.empty:
            print(f"\nAverage DEM coverage: {coverage.mean():.2f}%")
            print(f"Minimum DEM coverage: {coverage.min():.2f}%")
            print(f"Maximum DEM coverage: {coverage.max():.2f}%")

    print(f"\nReport saved to:\n{OUTPUT_CSV}")
    print("=" * 65)


if __name__ == "__main__":
    main()