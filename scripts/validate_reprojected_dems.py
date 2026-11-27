
"""
Validate district-level reprojected DEM rasters.

Input:
    data/processed/DEM/district_dem_utm/

Output:
    data/processed/DEM/district_dem_utm_validation.csv
"""

from pathlib import Path
import math

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = (
    BASE_DIR / "data" / "processed" / "DEM" / "district_dem_utm"
)

OUTPUT_CSV = (
    BASE_DIR / "data" / "processed" / "DEM"
    / "district_dem_utm_validation.csv"
)


def validate_dem(filepath):

    result = {
        "file": str(filepath.relative_to(INPUT_DIR)),
        "status": "FAILED",
        "crs": None,
        "epsg": None,
        "utm_zone": None,
        "units": None,
        "width": None,
        "height": None,
        "pixel_size_x": None,
        "pixel_size_y": None,
        "min_elevation": None,
        "max_elevation": None,
        "mean_elevation": None,
        "valid_pixels": 0,
        "nodata_pixels": None,
        "valid_percentage": None,
        "bounds": None,
        "zone_check": None,
        "message": ""
    }

    try:
        with rasterio.open(filepath) as src:

            # 1. Basic raster information
            crs = src.crs

            result["crs"] = str(crs) if crs else None
            result["epsg"] = crs.to_epsg() if crs else None
            result["width"] = src.width
            result["height"] = src.height
            result["units"] = (
                crs.linear_units if crs and crs.is_projected else None
            )

            result["pixel_size_x"] = abs(src.transform.a)
            result["pixel_size_y"] = abs(src.transform.e)

            result["bounds"] = str(src.bounds)

            total_pixels = src.width * src.height

            if not crs or not crs.is_projected:
                raise ValueError("Raster does not have a projected CRS.")

            if result["units"] not in ("metre", "meter"):
                raise ValueError(
                    f"Unexpected projected units: {result['units']}"
                )

            if src.count < 1 or total_pixels == 0:
                raise ValueError("Raster has no valid dimensions or bands.")

            # 2. Determine UTM zone from EPSG
            epsg = result["epsg"]

            if epsg and 32601 <= epsg <= 32660:
                zone = epsg - 32600
                result["utm_zone"] = f"{zone}N"

            elif epsg and 32701 <= epsg <= 32760:
                zone = epsg - 32700
                result["utm_zone"] = f"{zone}S"

            else:
                zone = None

            # 3. Estimate expected UTM zone from raster center
            center_x = (src.bounds.left + src.bounds.right) / 2
            center_y = (src.bounds.bottom + src.bounds.top) / 2

            longitude, latitude = transform(
                crs,
                "EPSG:4326",
                [center_x],
                [center_y]
            )

            lon = longitude[0]
            lat = latitude[0]

            expected_zone = int(math.floor((lon + 180) / 6) + 1)
            expected_zone = max(1, min(60, expected_zone))

            if zone is not None:
                hemisphere = "N" if lat >= 0 else "S"
                expected_utm = f"{expected_zone}{hemisphere}"
                actual_utm = result["utm_zone"]

                result["zone_check"] = (
                    "PASS" if actual_utm == expected_utm else
                    f"CHECK: expected {expected_utm}"
                )

            # 4. Scan raster in blocks (memory efficient)
            min_elev = float("inf")
            max_elev = float("-inf")
            total_sum = 0.0
            valid_count = 0

            for _, window in src.block_windows(1):

                data = src.read(
                    1,
                    window=window,
                    masked=True
                )

                values = np.asarray(data.compressed())

                if values.size == 0:
                    continue

                values = values[np.isfinite(values)]

                if values.size == 0:
                    continue

                block_min = float(values.min())
                block_max = float(values.max())

                min_elev = min(min_elev, block_min)
                max_elev = max(max_elev, block_max)

                total_sum += float(
                    np.sum(values, dtype=np.float64)
                )

                valid_count += values.size

            # 5. Calculate statistics
            result["valid_pixels"] = valid_count
            result["nodata_pixels"] = total_pixels - valid_count

            result["valid_percentage"] = round(
                valid_count / total_pixels * 100, 4
            )

            if valid_count > 0:

                result["min_elevation"] = min_elev
                result["max_elevation"] = max_elev
                result["mean_elevation"] = round(
                    total_sum / valid_count, 3
                )

            else:
                raise ValueError("Raster contains no valid elevation pixels.")

            # 6. Final checks
            if result["pixel_size_x"] <= 0 or result["pixel_size_y"] <= 0:
                raise ValueError("Invalid pixel resolution.")

            if min_elev > max_elev:
                raise ValueError("Invalid elevation range.")

            result["status"] = "PASS"

            if result["zone_check"] and result["zone_check"] != "PASS":
                result["status"] = "REVIEW"

            result["message"] = "Validation completed."

    except Exception as exc:
        result["message"] = str(exc)

    return result


def main():

    print("=" * 65)
    print("DISTRICT DEM VALIDATION")
    print("=" * 65)

    if not INPUT_DIR.exists():
        print(f"\nERROR: Input directory not found: {INPUT_DIR}")
        return

    files = sorted(
    list(INPUT_DIR.rglob("*.tif"))
    + list(INPUT_DIR.rglob("*.tiff"))
    )
    print(f"\nInput directory: {INPUT_DIR}")
    print(f"DEM files found: {len(files)}")

    if not files:
        print("No DEM files found.")
        return

    results = []

    for index, filepath in enumerate(files, 1):

        result = validate_dem(filepath)
        results.append(result)

        print(
            f"[{index}/{len(files)}] "
            f"{result['status']} | {filepath.name}"
        )

    # Save detailed report
    df = pd.DataFrame(results)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    # Summary
    print("\n" + "=" * 65)
    print("VALIDATION SUMMARY")
    print("=" * 65)

    print(f"Total DEM files : {len(results)}")
    print(f"Passed          : {(df['status'] == 'PASS').sum()}")
    print(f"Review required : {(df['status'] == 'REVIEW').sum()}")
    print(f"Failed          : {(df['status'] == 'FAILED').sum()}")

    print(f"\nDetailed report: {OUTPUT_CSV}")

    print("\nUTM zone distribution:")
    print(df["utm_zone"].value_counts(dropna=False).to_string())

    print("\nPixel resolution distribution:")
    print(
        df.groupby(
            ["pixel_size_x", "pixel_size_y"],
            dropna=False
        ).size().to_string()
    )

    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()
