from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio

from rasterio.features import geometry_mask
from rasterio.windows import Window
from rasterio.transform import Affine
from shapely.geometry import mapping


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DEM_DIR = ROOT / "data/processed/DEM"

REFERENCE_DEM = DEM_DIR / "district_dem_utm/UK/Haridwar.tif"
BUFFERED_DEM = DEM_DIR / "district_dem_buffered/UK/Haridwar.tif"

BOUNDARY_FILE = (
    ROOT / "data/processed/boundary/ihr_districts.gpkg"
)

OUTPUT_DIR = ROOT / "data/processed/terrain/UK/Haridwar"

SLOPE_FILE = OUTPUT_DIR / "slope.tif"
ASPECT_FILE = OUTPUT_DIR / "aspect.tif"

BUFFER_PIXELS = 10
NODATA = -9999.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HORN'S ALGORITHM
# ============================================================

def calculate_terrain(z, valid, dx, dy):

    # 3x3 neighborhood
    #
    # a b c
    # d e f
    # g h i

    a = z[:-2, :-2]
    b = z[:-2, 1:-1]
    c = z[:-2, 2:]

    d = z[1:-1, :-2]
    f = z[1:-1, 2:]

    g = z[2:, :-2]
    h = z[2:, 1:-1]
    i = z[2:, 2:]

    # Elevation gradient towards east.
    dz_dx = (
        (c - a) + 2 * (f - d) + (i - g)
    ) / (8 * dx)

    # Elevation gradient towards south.
    dz_dy = (
        (g - a) + 2 * (h - b) + (i - c)
    ) / (8 * dy)

    # Gradient magnitude.
    gradient = np.hypot(dz_dx, dz_dy)

    # Slope in degrees.
    slope = np.degrees(np.arctan(gradient))

    # Downhill aspect, clockwise from grid north.
    aspect = (
        np.degrees(np.arctan2(-dz_dx, dz_dy)) + 360
    ) % 360

    # Require all nine neighborhood pixels to be valid.
    valid_neighborhood = (
        valid[:-2, :-2]
        & valid[:-2, 1:-1]
        & valid[:-2, 2:]
        & valid[1:-1, :-2]
        & valid[1:-1, 1:-1]
        & valid[1:-1, 2:]
        & valid[2:, :-2]
        & valid[2:, 1:-1]
        & valid[2:, 2:]
    )

    # Aspect is undefined on perfectly flat terrain.
    flat = gradient == 0

    slope[~valid_neighborhood] = NODATA

    aspect[
        ~valid_neighborhood | flat
    ] = NODATA

    return slope, aspect, flat


# ============================================================
# STATISTICS
# ============================================================

class RasterStats:

    def __init__(self):

        self.count = 0
        self.total = 0.0
        self.minimum = np.inf
        self.maximum = -np.inf

    def update(self, array):

        values = array[
            np.isfinite(array) & (array != NODATA)
        ]

        if values.size == 0:
            return

        self.count += values.size
        self.total += values.sum(dtype=np.float64)

        self.minimum = min(self.minimum, values.min())
        self.maximum = max(self.maximum, values.max())

    def display(self, name):

        print(f"\n{name}")
        print("-" * 40)
        print(f"Valid pixels: {self.count:,}")

        if self.count > 0:

            print(f"Minimum: {self.minimum:.4f}")
            print(f"Maximum: {self.maximum:.4f}")
            print(f"Mean: {self.total / self.count:.4f}")


# ============================================================
# LOAD DISTRICT BOUNDARY
# ============================================================

print("=" * 65)
print("HARIDWAR SLOPE AND ASPECT PILOT TEST")
print("=" * 65)

print("\nLoading district boundary...")

districts = gpd.read_file(BOUNDARY_FILE).to_crs("EPSG:4326")

selected = districts[
    (districts["district"].str.lower() == "haridwar")
    & (districts["state"].str.upper() == "UK")
]

if len(selected) != 1:

    raise RuntimeError(
        f"Expected one Haridwar boundary, found {len(selected)}"
    )


# ============================================================
# OPEN DEMS
# ============================================================

with rasterio.open(REFERENCE_DEM) as ref, \
     rasterio.open(BUFFERED_DEM) as src:

    print("\nChecking DEM alignment...")

    expected_transform = ref.transform * Affine.translation(
        -BUFFER_PIXELS,
        -BUFFER_PIXELS
    )

    if ref.crs != src.crs:

        raise RuntimeError("DEM CRS mismatch.")

    if not ref.crs.is_projected:

        raise RuntimeError("DEM must use a projected CRS.")

    if (
        src.width != ref.width + 2 * BUFFER_PIXELS
        or src.height != ref.height + 2 * BUFFER_PIXELS
    ):

        raise RuntimeError("Buffered raster dimensions mismatch.")

    if not np.allclose(
        tuple(src.transform),
        tuple(expected_transform),
        rtol=0,
        atol=1e-7
    ):

        raise RuntimeError("Buffered raster grid alignment mismatch.")

    if (
        ref.transform.b != 0
        or ref.transform.d != 0
        or ref.transform.a <= 0
        or ref.transform.e >= 0
    ):

        raise RuntimeError("Expected a north-up raster.")

    if not np.allclose(ref.res, (30, 30)):

        raise RuntimeError("Expected 30 m resolution.")

    print("CRS match: PASS")
    print("Dimensions: PASS")
    print("Grid alignment: PASS")

    dx, dy = ref.res

    geometry = selected.to_crs(ref.crs).geometry.iloc[0]

    if not geometry.is_valid:
        geometry = geometry.buffer(0)

    if geometry.is_empty:
        raise RuntimeError("District geometry is empty.")

    geometry = mapping(geometry)

    # Outputs use the original district grid.
    profile = ref.profile.copy()

    profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        nodata=NODATA,
        compress="deflate",
        predictor=3,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        BIGTIFF="IF_SAFER"
    )

    slope_stats = RasterStats()
    aspect_stats = RasterStats()

    district_pixels = 0
    flat_pixels = 0

    comparison_pixels = 0
    difference_sum = 0.0
    maximum_difference = 0.0

    print("\nCalculating terrain features...")

    with rasterio.open(SLOPE_FILE, "w", **profile) as slope_dst, \
         rasterio.open(ASPECT_FILE, "w", **profile) as aspect_dst:

        slope_dst.set_band_description(1, "Slope in degrees")
        aspect_dst.set_band_description(1, "Aspect in degrees")

        slope_dst.update_tags(algorithm="Horn")
        aspect_dst.update_tags(
            algorithm="Horn",
            direction="Clockwise from grid north",
            flat_value=str(NODATA)
        )

        for index, (_, window) in enumerate(
            ref.block_windows(1), start=1
        ):

            # Read an extra pixel on every side.
            # This provides a complete 3x3 neighborhood.

            source_window = Window(
                col_off=window.col_off + BUFFER_PIXELS - 1,
                row_off=window.row_off + BUFFER_PIXELS - 1,
                width=window.width + 2,
                height=window.height + 2
            )

            data = src.read(
                1,
                window=source_window,
                masked=True
            )

            z = data.data.astype(np.float64)

            valid = (
                ~np.ma.getmaskarray(data)
                & np.isfinite(z)
            )

            if src.nodata is not None:
                valid &= z != src.nodata

            z[~valid] = 0

            slope, aspect, flat = calculate_terrain(
                z, valid, dx, dy
            )

            # Identify pixels inside the actual district.
            inside = geometry_mask(
                [geometry],
                out_shape=(
                    int(window.height),
                    int(window.width)
                ),
                transform=ref.window_transform(window),
                invert=True,
                all_touched=False
            )

            district_pixels += np.count_nonzero(inside)

            # Clip only after calculating terrain features.
            slope[~inside] = NODATA
            aspect[~inside] = NODATA

            flat_pixels += np.count_nonzero(
                inside & flat & (slope != NODATA)
            )

            # Ensure floating-point rounding doesn't produce
            # an aspect value of exactly 360 degrees.
            slope = slope.astype(np.float32)
            aspect = aspect.astype(np.float32)

            aspect[aspect >= 360] = 0

            slope_stats.update(slope)
            aspect_stats.update(aspect)

            slope_dst.write(slope, 1, window=window)
            aspect_dst.write(aspect, 1, window=window)

            # Compare the original DEM with the new DEM
            # at locations where both contain valid data.
            reference = ref.read(
                1, window=window, masked=True
            )

            buffered_center = z[1:-1, 1:-1]

            common_valid = (
                inside
                & ~np.ma.getmaskarray(reference)
                & np.isfinite(reference.data)
                & valid[1:-1, 1:-1]
            )

            differences = np.abs(
                reference.data[common_valid].astype(np.float64)
                - buffered_center[common_valid]
            )

            if differences.size > 0:

                comparison_pixels += differences.size
                difference_sum += differences.sum()

                maximum_difference = max(
                    maximum_difference,
                    differences.max()
                )

            if index % 50 == 0:

                print(f"Processed {index} raster blocks")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("TERRAIN FEATURE SUMMARY")
print("=" * 65)

print(f"\nDistrict pixels: {district_pixels:,}")

slope_stats.display("SLOPE (degrees)")
aspect_stats.display("ASPECT (degrees)")

print(f"\nFlat terrain pixels: {flat_pixels:,}")

if district_pixels > 0:

    slope_coverage = (
        slope_stats.count / district_pixels * 100
    )

    aspect_coverage = (
        aspect_stats.count / district_pixels * 100
    )

    print(f"Slope coverage: {slope_coverage:.4f}%")
    print(f"Aspect coverage: {aspect_coverage:.4f}%")

    print(
        "Undefined aspect due to flat terrain: "
        f"{flat_pixels:,}"
    )

print("\nELEVATION CONSISTENCY CHECK")
print("-" * 40)

print(f"Compared pixels: {comparison_pixels:,}")

if comparison_pixels > 0:

    print(
        f"Mean absolute elevation difference: "
        f"{difference_sum / comparison_pixels:.6f} m"
    )

    print(
        f"Maximum absolute elevation difference: "
        f"{maximum_difference:.6f} m"
    )

print("\nOUTPUT FILES")
print("-" * 40)

print(SLOPE_FILE)
print(ASPECT_FILE)

print("\nTerrain extraction complete.")
print("=" * 65)