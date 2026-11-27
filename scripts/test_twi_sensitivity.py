from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio

from pyproj import Geod
from rasterio.features import geometry_mask

ROOT = Path(__file__).resolve().parent.parent

NATIVE = (
    ROOT / "data/processed/hydrology/pilot/UK/Haridwar/native"
)

BOUNDARY = ROOT / "data/processed/boundary/ihr_districts.gpkg"

FLOORS = [0.05, 0.1, 0.5]

print("=" * 65)
print("HARIDWAR TWI SENSITIVITY ANALYSIS")
print("=" * 65)

with rasterio.open(NATIVE / "upa.tif") as src:

    upa = src.read(1, masked=True).astype(
        np.float64
    ).filled(np.nan)

    transform = src.transform
    shape = src.shape
    crs = src.crs

    # Latitude of each native raster row.
    lat = (
        transform.f
        + (np.arange(src.height) + 0.5) * transform.e
    )

    rx, ry = src.res


with rasterio.open(NATIVE / "hydro_slope.tif") as src:

    if (
        src.shape != shape
        or src.transform != transform
        or src.crs != crs
    ):
        raise RuntimeError("Slope grid mismatch.")

    slope = src.read(1, masked=True).astype(
        np.float64
    ).filled(np.nan)


districts = gpd.read_file(BOUNDARY).to_crs(crs)

selected = districts[
    (districts["state"].str.upper() == "UK")
    & (districts["district"].str.lower() == "haridwar")
]

if len(selected) != 1:
    raise RuntimeError("District boundary match failed.")

geometry = selected.geometry.iloc[0]

if not geometry.is_valid:
    geometry = geometry.buffer(0)

inside = geometry_mask(
    [geometry],
    out_shape=shape,
    transform=transform,
    invert=True
)

# Native geographic cell dimensions in metres.
geod = Geod(ellps="WGS84")

zeros = np.zeros_like(lat)

_, _, dx = geod.inv(
    zeros, lat, zeros + rx, lat
)

_, _, dy = geod.inv(
    zeros, lat - ry / 2,
    zeros, lat + ry / 2
)

width = np.sqrt(dx * dy)[:, None]

sca = upa * 1e6 / width

valid = (
    inside
    & np.isfinite(sca)
    & np.isfinite(slope)
    & (sca > 0)
    & (slope >= 0)
    & (slope < 90)
)

if not valid.any():
    raise RuntimeError("No valid TWI pixels.")


def calculate_twi(floor):

    beta = np.maximum(slope[valid], floor)

    return np.log(
        sca[valid] / np.tan(np.radians(beta))
    )


baseline = calculate_twi(0.1)

print(f"\nDistrict pixels: {inside.sum():,}")
print(f"Comparable pixels: {valid.sum():,}")

for floor in FLOORS:

    twi = calculate_twi(floor)

    affected = np.count_nonzero(
        valid & (slope < floor)
    )

    difference = np.abs(twi - baseline)

    print("\n" + "-" * 55)
    print(f"MINIMUM SLOPE: {floor} degrees")
    print("-" * 55)

    print(f"Affected pixels: {affected:,}")

    print(
        f"Affected percentage: "
        f"{affected / valid.sum() * 100:.4f}%"
    )

    print(f"Minimum TWI: {twi.min():.6f}")
    print(f"Maximum TWI: {twi.max():.6f}")
    print(f"Mean TWI: {twi.mean():.6f}")
    print(f"Median TWI: {np.median(twi):.6f}")
    print(f"P95 TWI: {np.percentile(twi, 95):.6f}")
    print(f"P99 TWI: {np.percentile(twi, 99):.6f}")

    print(
        f"MAE against 0.1 degree baseline: "
        f"{difference.mean():.6f}"
    )

    print(
        f"Maximum difference: "
        f"{difference.max():.6f}"
    )

print("\n" + "=" * 65)
print("Sensitivity analysis complete.")
print("No source rasters modified.")
print("=" * 65)