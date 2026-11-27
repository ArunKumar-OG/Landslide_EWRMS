from pathlib import Path
from contextlib import ExitStack

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

REFERENCE = DEM_DIR / "district_dem_utm/UK/Haridwar.tif"
BUFFERED = DEM_DIR / "district_dem_buffered_v2/UK/Haridwar.tif"

BOUNDARY = ROOT / "data/processed/boundary/ihr_districts.gpkg"

OUTPUT = ROOT / "data/processed/terrain_v2/UK/Haridwar"

BUFFER = 10
NODATA = -9999.0

# Avoid unstable curvature calculations on nearly flat surfaces.
MIN_CURVATURE_GRADIENT = 0.0001

FEATURES = {
    "elevation": ("metres", "Source-aligned Copernicus DEM"),
    "slope": ("degrees", "Horn"),
    "aspect": ("degrees", "Horn, clockwise from grid north"),
    "plan_curvature": ("1/metre", "Projected contour curvature"),
    "profile_curvature": ("1/metre", "Profile curvature"),
    "tri": ("metres", "Riley root sum of squares"),
    "tpi": ("metres", "Centre minus eight-neighbour mean")
}

OUTPUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# TERRAIN CALCULATIONS
# ============================================================

def calculate_features(z, valid, dx, dy):

    # 3x3 neighbourhood:
    #
    # a b c
    # d e f
    # g h i

    a = z[:-2, :-2]
    b = z[:-2, 1:-1]
    c = z[:-2, 2:]

    d = z[1:-1, :-2]
    e = z[1:-1, 1:-1]
    f = z[1:-1, 2:]

    g = z[2:, :-2]
    h = z[2:, 1:-1]
    i = z[2:, 2:]

    neighbours = [a, b, c, d, f, g, h, i]

    valid_neighbourhood = np.ones(e.shape, dtype=bool)

    for row in range(3):
        for col in range(3):

            valid_neighbourhood &= valid[
                row:row + e.shape[0],
                col:col + e.shape[1]
            ]

    # --------------------------------------------------------
    # FIRST DERIVATIVES - HORN
    # --------------------------------------------------------

    p = (
        (c - a) + 2 * (f - d) + (i - g)
    ) / (8 * dx)

    # Positive y points south (raster row direction).
    q = (
        (g - a) + 2 * (h - b) + (i - c)
    ) / (8 * dy)

    gradient_squared = p**2 + q**2
    gradient = np.sqrt(gradient_squared)

    # --------------------------------------------------------
    # ELEVATION, SLOPE AND ASPECT
    # --------------------------------------------------------

    elevation = e.copy()

    slope = np.degrees(np.arctan(gradient))

    aspect = (
        np.degrees(np.arctan2(-p, q)) + 360
    ) % 360

    flat = gradient == 0

    # --------------------------------------------------------
    # SECOND DERIVATIVES
    # --------------------------------------------------------

    r = (d - 2 * e + f) / dx**2

    t = (b - 2 * e + h) / dy**2

    s = (a - c - g + i) / (4 * dx * dy)

    # --------------------------------------------------------
    # PLAN AND PROFILE CURVATURE
    # Positive = convex; negative = concave.
    # --------------------------------------------------------

    curvature_valid = (
        valid_neighbourhood
        & (gradient >= MIN_CURVATURE_GRADIENT)
    )

    plan = np.full(e.shape, NODATA, dtype=np.float64)
    profile = np.full(e.shape, NODATA, dtype=np.float64)

    np.divide(
        -(q**2 * r - 2 * p * q * s + p**2 * t),
        gradient_squared**1.5,
        out=plan,
        where=curvature_valid
    )

    np.divide(
        -(p**2 * r + 2 * p * q * s + q**2 * t),
        gradient_squared * (1 + gradient_squared)**1.5,
        out=profile,
        where=curvature_valid
    )

    # --------------------------------------------------------
    # TRI - RILEY ALGORITHM
    # --------------------------------------------------------

    tri_squared = np.zeros(e.shape, dtype=np.float64)

    for neighbour in neighbours:
        tri_squared += (neighbour - e)**2

    tri = np.sqrt(tri_squared)

    # --------------------------------------------------------
    # TPI - CENTRE MINUS NEIGHBOUR MEAN
    # --------------------------------------------------------

    neighbour_sum = np.zeros(e.shape, dtype=np.float64)

    for neighbour in neighbours:
        neighbour_sum += neighbour

    tpi = e - neighbour_sum / 8

    # --------------------------------------------------------
    # APPLY VALIDITY MASKS
    # --------------------------------------------------------

    elevation[~valid[1:-1, 1:-1]] = NODATA

    for array in [slope, aspect, tri, tpi]:
        array[~valid_neighbourhood] = NODATA

    aspect[flat] = NODATA

    return {
        "elevation": elevation,
        "slope": slope,
        "aspect": aspect,
        "plan_curvature": plan,
        "profile_curvature": profile,
        "tri": tri,
        "tpi": tpi
    }


# ============================================================
# SYNTHETIC ALGORITHM TESTS
# ============================================================

def run_algorithm_tests():

    valid = np.ones((5, 5), dtype=bool)

    # Test 1: Perfectly flat terrain.
    flat_dem = np.full((5, 5), 100.0)

    result = calculate_features(flat_dem, valid, 30, 30)

    assert np.allclose(result["slope"], 0)
    assert np.allclose(result["tri"], 0)
    assert np.allclose(result["tpi"], 0)
    assert np.all(result["aspect"] == NODATA)
    assert np.all(result["plan_curvature"] == NODATA)
    assert np.all(result["profile_curvature"] == NODATA)

    print("Flat terrain test: PASS")

    # Test 2: Terrain rises 30 m towards the east
    # for every 30 m of horizontal distance.
    # Expected slope: 45 degrees.
    # Expected downhill direction: West (270 degrees).

    plane = np.tile(np.arange(5) * 30.0, (5, 1))

    result = calculate_features(plane, valid, 30, 30)

    assert np.allclose(result["slope"], 45)
    assert np.allclose(result["aspect"], 270)
    assert np.allclose(result["tpi"], 0)
    assert np.allclose(result["plan_curvature"], 0)
    assert np.allclose(result["profile_curvature"], 0)
    assert np.allclose(result["tri"], np.sqrt(6 * 30**2))

    print("Inclined plane test: PASS")

    # Test 3: Missing centre elevation.
    invalid = valid.copy()
    invalid[2, 2] = False

    result = calculate_features(plane, invalid, 30, 30)

    assert result["elevation"][1, 1] == NODATA
    assert np.all(result["slope"] == NODATA)
    assert np.all(result["tri"] == NODATA)

    print("NoData propagation test: PASS")


# ============================================================
# MAIN PROCESS
# ============================================================

def main():

    print("=" * 65)
    print("HARIDWAR TERRAIN FEATURE EXTRACTION V2")
    print("=" * 65)

    print("\nRunning algorithm tests...")
    run_algorithm_tests()

    districts = gpd.read_file(BOUNDARY).to_crs("EPSG:4326")

    selected = districts[
        (districts["state"].str.upper() == "UK")
        & (districts["district"].str.lower() == "haridwar")
    ]

    if len(selected) != 1:
        raise RuntimeError("District boundary matching failed.")

    stats = {
        name: {
            "count": 0,
            "min": np.inf,
            "max": -np.inf,
            "sum": 0.0
        }
        for name in FEATURES
    }

    district_pixels = 0

    with rasterio.open(REFERENCE) as ref, \
         rasterio.open(BUFFERED) as src:

        expected_transform = (
            ref.transform
            * Affine.translation(-BUFFER, -BUFFER)
        )

        if not (
            ref.crs == src.crs
            and ref.crs.is_projected
            and ref.width + 2 * BUFFER == src.width
            and ref.height + 2 * BUFFER == src.height
            and np.allclose(
                tuple(src.transform),
                tuple(expected_transform),
                rtol=0,
                atol=1e-7
            )
        ):
            raise RuntimeError("DEM grid alignment failed.")

        if not np.allclose(ref.res, (30, 30)):
            raise RuntimeError("Unexpected DEM resolution.")

        if (
            ref.transform.b != 0
            or ref.transform.d != 0
            or ref.transform.a <= 0
            or ref.transform.e >= 0
        ):
            raise RuntimeError("Expected a north-up DEM.")

        print("\nDEM grid alignment: PASS")

        dx, dy = ref.res

        geometry = selected.to_crs(ref.crs).geometry.iloc[0]

        if not geometry.is_valid:
            geometry = geometry.buffer(0)

        if geometry.is_empty:
            raise RuntimeError("District geometry is empty.")

        geometry = mapping(geometry)

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

        print("\nGenerating terrain features...")

        with ExitStack() as stack:

            writers = {}

            for name, (unit, algorithm) in FEATURES.items():

                dst = stack.enter_context(
                    rasterio.open(
                        OUTPUT / f"{name}.tif",
                        "w",
                        **profile
                    )
                )

                dst.set_band_description(1, name)

                dst.update_tags(
                    source="Buffered DEM V2",
                    units=unit,
                    algorithm=algorithm,
                    neighbourhood="3x3",
                    resolution_metres="30"
                )

                if "curvature" in name:

                    dst.update_tags(
                        sign_convention="Positive convex",
                        min_gradient=str(
                            MIN_CURVATURE_GRADIENT
                        )
                    )

                writers[name] = dst

            for index, (_, window) in enumerate(
                ref.block_windows(1), start=1
            ):

                source_window = Window(
                    window.col_off + BUFFER - 1,
                    window.row_off + BUFFER - 1,
                    window.width + 2,
                    window.height + 2
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

                results = calculate_features(
                    z, valid, dx, dy
                )

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

                district_pixels += int(
                    np.count_nonzero(inside)
                )

                for name, array in results.items():

                    array[~inside] = NODATA
                    array[~np.isfinite(array)] = NODATA

                    array = array.astype(np.float32)

                    if name == "aspect":
                        array[array >= 360] = 0

                    values = array[array != NODATA]

                    if values.size:

                        stat = stats[name]

                        stat["count"] += values.size
                        stat["min"] = min(
                            stat["min"], values.min()
                        )
                        stat["max"] = max(
                            stat["max"], values.max()
                        )
                        stat["sum"] += values.sum(
                            dtype=np.float64
                        )

                    writers[name].write(
                        array, 1, window=window
                    )

                if index % 25 == 0:
                    print(f"Processed {index} raster blocks")


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 65)
    print("TERRAIN FEATURE SUMMARY V2")
    print("=" * 65)

    print(f"\nDistrict pixels: {district_pixels:,}")

    for name, stat in stats.items():

        print(f"\n{name.upper()}")
        print("-" * 45)

        count = stat["count"]

        print(f"Valid pixels: {count:,}")
        print(f"Missing pixels: {district_pixels - count:,}")

        if district_pixels > 0:

            print(
                f"Coverage: "
                f"{count / district_pixels * 100:.4f}%"
            )

        if count > 0:

            print(f"Minimum: {stat['min']:.8f}")
            print(f"Maximum: {stat['max']:.8f}")

            if name != "aspect":
                print(
                    f"Mean: {stat['sum'] / count:.8f}"
                )

    print("\nGenerated files:")

    for name in FEATURES:
        print(OUTPUT / f"{name}.tif")

    print("\nTerrain feature extraction complete.")
    print("=" * 65)


if __name__ == "__main__":
    main()