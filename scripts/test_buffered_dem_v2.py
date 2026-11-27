from pathlib import Path
import math

import numpy as np
import rasterio

from rasterio.merge import merge
from rasterio.warp import reproject, transform_bounds
from rasterio.enums import Resampling
from rasterio.transform import Affine, array_bounds


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT / "data/raw/dem"
DEM_DIR = ROOT / "data/processed/DEM"

REFERENCE = DEM_DIR / "district_dem_utm/UK/Haridwar.tif"

PREVIOUS = DEM_DIR / "district_dem_buffered/UK/Haridwar.tif"

OUTPUT = DEM_DIR / "district_dem_buffered_v2/UK/Haridwar.tif"

BUFFER = 10
NODATA = -9999.0
SOURCE_RES = 1 / 3600


# ============================================================
# SOURCE GRID ALIGNMENT
# ============================================================

def align_bounds(bounds, origin_x, origin_y, rx, ry):

    left, bottom, right, top = bounds

    aligned_left = origin_x + math.floor(
        (left - origin_x) / rx
    ) * rx

    aligned_right = origin_x + math.ceil(
        (right - origin_x) / rx
    ) * rx

    aligned_bottom = origin_y + math.floor(
        (bottom - origin_y) / ry
    ) * ry

    aligned_top = origin_y + math.ceil(
        (top - origin_y) / ry
    ) * ry

    return (
        aligned_left,
        aligned_bottom,
        aligned_right,
        aligned_top
    )


def is_integer(value):

    return abs(value - round(value)) < 1e-6


print("=" * 65)
print("SOURCE-ALIGNED BUFFERED DEM - HARIDWAR V2")
print("=" * 65)


# ============================================================
# TARGET UTM GRID
# ============================================================

with rasterio.open(REFERENCE) as ref:

    target_crs = ref.crs

    width = ref.width + 2 * BUFFER
    height = ref.height + 2 * BUFFER

    target_transform = ref.transform * Affine.translation(
        -BUFFER, -BUFFER
    )

    target_bounds = array_bounds(
        height, width, target_transform
    )

    print(f"\nTarget CRS: {target_crs}")
    print(f"Target resolution: {ref.res}")
    print(f"Buffered dimensions: {width} x {height}")


# ============================================================
# FILTER RAW DEM TILES
# ============================================================

raw_tiles = sorted(RAW_DIR.rglob("*.tif"))

selected_tiles = []

high_resolution_count = 0
excluded_count = 0

print("\nInspecting source tiles...")

for tile in raw_tiles:

    with rasterio.open(tile) as src:

        # Only accept native 1 arc-second DEMs.
        is_high_resolution = (
            src.crs is not None
            and src.crs.to_epsg() == 4326
            and src.width == 3600
            and src.height == 3600
            and np.allclose(
                src.res,
                (SOURCE_RES, SOURCE_RES),
                rtol=0,
                atol=1e-10
            )
        )

        if not is_high_resolution:
            excluded_count += 1
            continue

        high_resolution_count += 1

        left, bottom, right, top = transform_bounds(
            target_crs,
            src.crs,
            *target_bounds,
            densify_pts=21
        )

        pad = 2 * SOURCE_RES

        intersects = (
            left - pad < src.bounds.right
            and right + pad > src.bounds.left
            and bottom - pad < src.bounds.top
            and top + pad > src.bounds.bottom
        )

        if intersects:
            selected_tiles.append(tile)


print(f"\nTotal raw tiles: {len(raw_tiles)}")
print(f"Accepted 30 m tiles: {high_resolution_count}")
print(f"Excluded tiles: {excluded_count}")
print(f"Selected Haridwar tiles: {len(selected_tiles)}")

for tile in selected_tiles:
    print(f"  {tile.name}")

if not selected_tiles:
    raise RuntimeError("No suitable 30 m tiles found.")


# ============================================================
# VERIFY SOURCE TILE ALIGNMENT
# ============================================================

with rasterio.open(selected_tiles[0]) as src:

    source_crs = src.crs

    rx, ry = src.res

    origin_x = src.transform.c
    origin_y = src.transform.f


print("\nChecking native source grid alignment...")

for tile in selected_tiles:

    with rasterio.open(tile) as src:

        x_offset = (src.transform.c - origin_x) / rx
        y_offset = (src.transform.f - origin_y) / ry

        if not (
            src.crs == source_crs
            and np.allclose(
                src.res, (rx, ry),
                rtol=0, atol=1e-10
            )
            and src.transform.b == 0
            and src.transform.d == 0
            and src.transform.a > 0
            and src.transform.e < 0
            and is_integer(x_offset)
            and is_integer(y_offset)
        ):
            raise RuntimeError(
                f"Source grid misalignment: {tile.name}"
            )

        print(f"  PASS: {tile.name}")


# ============================================================
# SOURCE-ALIGNED MOSAIC BOUNDS
# ============================================================

left, bottom, right, top = transform_bounds(
    target_crs,
    source_crs,
    *target_bounds,
    densify_pts=21
)

requested_bounds = (
    left - 2 * rx,
    bottom - 2 * ry,
    right + 2 * rx,
    top + 2 * ry
)

mosaic_bounds = align_bounds(
    requested_bounds,
    origin_x,
    origin_y,
    rx,
    ry
)

print("\nAligned mosaic bounds:")

for value in mosaic_bounds:
    print(f"  {value:.12f}")


# ============================================================
# BUILD SOURCE MOSAIC
# ============================================================

print("\nBuilding source-aligned mosaic...")

mosaic, mosaic_transform = merge(
    selected_tiles,
    bounds=mosaic_bounds,
    res=(rx, ry),
    nodata=NODATA,
    dtype="float32",
    method="first",
    resampling=Resampling.nearest,
    target_aligned_pixels=False
)

mosaic_x_offset = (
    mosaic_transform.c - origin_x
) / rx

mosaic_y_offset = (
    mosaic_transform.f - origin_y
) / ry

if not (
    is_integer(mosaic_x_offset)
    and is_integer(mosaic_y_offset)
):
    raise RuntimeError("Mosaic grid alignment failed.")

print(f"Mosaic dimensions: {mosaic.shape}")
print(f"Mosaic X offset: {mosaic_x_offset:.6f} pixels")
print(f"Mosaic Y offset: {mosaic_y_offset:.6f} pixels")
print("Native grid alignment: PASS")

mosaic[~np.isfinite(mosaic)] = NODATA


# ============================================================
# REPROJECT TO UTM
# ============================================================

print("\nReprojecting DEM to buffered UTM grid...")

destination = np.full(
    (height, width),
    NODATA,
    dtype=np.float32
)

reproject(
    source=mosaic[0],
    destination=destination,
    src_transform=mosaic_transform,
    src_crs=source_crs,
    src_nodata=NODATA,
    dst_transform=target_transform,
    dst_crs=target_crs,
    dst_nodata=NODATA,
    resampling=Resampling.bilinear,
    num_threads=2
)

del mosaic

destination[~np.isfinite(destination)] = NODATA


# ============================================================
# SAVE V2 DEM
# ============================================================

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with rasterio.open(
    OUTPUT,
    "w",
    driver="GTiff",
    width=width,
    height=height,
    count=1,
    dtype="float32",
    crs=target_crs,
    transform=target_transform,
    nodata=NODATA,
    compress="deflate",
    predictor=3,
    tiled=True,
    blockxsize=256,
    blockysize=256,
    BIGTIFF="IF_SAFER"
) as dst:

    dst.write(destination, 1)

    dst.update_tags(
        source="Copernicus 1 arc-second DEM",
        mosaic_alignment="Native source grid",
        resampling="Bilinear",
        buffer_pixels=str(BUFFER),
        buffer_metres="300"
    )


# ============================================================
# VALIDATION
# ============================================================

valid = (
    np.isfinite(destination)
    & (destination != NODATA)
)

valid_count = np.count_nonzero(valid)

print("\n" + "=" * 65)
print("BUFFERED DEM V2 SUMMARY")
print("=" * 65)

print(f"Total pixels: {destination.size:,}")
print(f"Valid pixels: {valid_count:,}")
print(f"NoData pixels: {destination.size - valid_count:,}")

print(
    f"Valid coverage: "
    f"{valid_count / destination.size * 100:.6f}%"
)

if valid_count:

    print(f"Minimum elevation: {destination[valid].min():.4f} m")
    print(f"Maximum elevation: {destination[valid].max():.4f} m")


# ============================================================
# COMPARE V1 AND V2
# ============================================================

if PREVIOUS.exists():

    print("\n" + "=" * 65)
    print("V1 VS V2 ELEVATION COMPARISON")
    print("=" * 65)

    with rasterio.open(PREVIOUS) as old:

        if (
            old.crs != target_crs
            or old.width != width
            or old.height != height
            or not np.allclose(
                tuple(old.transform),
                tuple(target_transform),
                rtol=0,
                atol=1e-7
            )
        ):
            raise RuntimeError("V1 and V2 grid mismatch.")

        previous = old.read(1, masked=True)

        previous_valid = (
            ~np.ma.getmaskarray(previous)
            & np.isfinite(previous.data)
        )

        common = valid & previous_valid

        signed = (
            destination[common].astype(np.float64)
            - previous.data[common].astype(np.float64)
        )

        difference = np.abs(signed)

        print(f"Compared pixels: {difference.size:,}")

        if difference.size:

            print(f"Mean absolute difference: {difference.mean():.6f} m")
            print(f"Median difference: {np.median(difference):.6f} m")
            print(f"Mean signed difference: {signed.mean():.6f} m")
            print(f"Maximum difference: {difference.max():.6f} m")

            print("\nDifference percentiles:")

            for p in [50, 90, 95, 99, 99.9]:

                print(
                    f"  P{p}: "
                    f"{np.percentile(difference, p):.6f} m"
                )

        print(
            f"\nRecovered pixels: "
            f"{np.count_nonzero(valid & ~previous_valid):,}"
        )

        print(
            f"Lost pixels: "
            f"{np.count_nonzero(~valid & previous_valid):,}"
        )


print("\n" + "=" * 65)
print(f"Saved to:\n{OUTPUT}")
print("Source-aligned DEM generation complete.")
print("=" * 65)