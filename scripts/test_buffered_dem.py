from pathlib import Path

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

RAW_DEM_DIR = ROOT / "data/raw/dem"

REFERENCE_DEM = (
    ROOT / "data/processed/DEM/district_dem_utm/UK/Haridwar.tif"
)

OUTPUT_DIR = ROOT / "data/processed/DEM/district_dem_buffered/UK"
OUTPUT_FILE = OUTPUT_DIR / "Haridwar.tif"

BUFFER_PIXELS = 10
NODATA = -9999.0


# ============================================================
# REFERENCE GRID
# ============================================================

print("=" * 65)
print("BUFFERED DEM - HARIDWAR PILOT TEST")
print("=" * 65)

with rasterio.open(REFERENCE_DEM) as ref:

    target_crs = ref.crs

    original_width = ref.width
    original_height = ref.height

    target_width = ref.width + 2 * BUFFER_PIXELS
    target_height = ref.height + 2 * BUFFER_PIXELS

    target_transform = ref.transform * Affine.translation(
        -BUFFER_PIXELS,
        -BUFFER_PIXELS
    )

    target_bounds = array_bounds(
        target_height,
        target_width,
        target_transform
    )

    print(f"\nTarget CRS: {target_crs}")
    print(f"Resolution: {ref.res}")
    print(f"Original size: {ref.width} x {ref.height}")
    print(f"Buffered size: {target_width} x {target_height}")
    print(f"Buffer distance: {BUFFER_PIXELS * ref.res[0]} m")


# ============================================================
# FIND INTERSECTING RAW DEM TILES
# ============================================================

raw_tiles = sorted(RAW_DEM_DIR.rglob("*.tif"))

print(f"\nRaw tiles available: {len(raw_tiles)}")

selected_tiles = []

for tile in raw_tiles:

    with rasterio.open(tile) as src:

        bounds = transform_bounds(
            target_crs,
            src.crs,
            *target_bounds,
            densify_pts=21
        )

        left, bottom, right, top = bounds

        # Small additional margin for interpolation.
        pad_x = 2 * abs(src.res[0])
        pad_y = 2 * abs(src.res[1])

        intersects = (
            left - pad_x < src.bounds.right
            and right + pad_x > src.bounds.left
            and bottom - pad_y < src.bounds.top
            and top + pad_y > src.bounds.bottom
        )

        if intersects:
            selected_tiles.append(tile)


print(f"Required raw tiles: {len(selected_tiles)}")

for tile in selected_tiles:
    print(f"  {tile.name}")

if not selected_tiles:
    raise RuntimeError("No intersecting DEM tiles found.")


# ============================================================
# MOSAIC ORIGINAL TILES
# ============================================================

print("\nBuilding source DEM mosaic...")

with rasterio.open(selected_tiles[0]) as src:

    source_crs = src.crs
    source_resolution = src.res

    left, bottom, right, top = transform_bounds(
        target_crs,
        source_crs,
        *target_bounds,
        densify_pts=21
    )

    pad_x = 2 * abs(source_resolution[0])
    pad_y = 2 * abs(source_resolution[1])

    mosaic_bounds = (
        left - pad_x,
        bottom - pad_y,
        right + pad_x,
        top + pad_y
    )

mosaic, mosaic_transform = merge(
    selected_tiles,
    bounds=mosaic_bounds,
    nodata=NODATA,
    dtype="float32",
    method="first"
)

print(f"Mosaic dimensions: {mosaic.shape}")


# ============================================================
# REPROJECT TO BUFFERED UTM GRID
# ============================================================

print("\nReprojecting buffered DEM...")

destination = np.full(
    (target_height, target_width),
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

destination[~np.isfinite(destination)] = NODATA

del mosaic


# ============================================================
# SAVE BUFFERED DEM
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with rasterio.open(
    OUTPUT_FILE,
    "w",
    driver="GTiff",
    height=target_height,
    width=target_width,
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


# ============================================================
# VALIDATION
# ============================================================

valid = np.isfinite(destination) & (destination != NODATA)

valid_count = np.count_nonzero(valid)
total_count = destination.size

print("\n" + "=" * 65)
print("BUFFERED DEM SUMMARY")
print("=" * 65)

print(f"CRS: {target_crs}")
print(f"Width: {target_width}")
print(f"Height: {target_height}")
print(f"Total pixels: {total_count:,}")
print(f"Valid pixels: {valid_count:,}")
print(f"NoData pixels: {total_count - valid_count:,}")
print(f"Valid coverage: {valid_count / total_count * 100:.4f}%")

if valid_count > 0:
    print(f"Minimum elevation: {destination[valid].min():.2f} m")
    print(f"Maximum elevation: {destination[valid].max():.2f} m")

print(f"\nSaved to:\n{OUTPUT_FILE}")
print("=" * 65)