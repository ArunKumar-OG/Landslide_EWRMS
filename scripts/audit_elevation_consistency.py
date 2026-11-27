from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio

from rasterio.windows import Window
from rasterio.features import geometry_mask
from rasterio.transform import Affine
from scipy.ndimage import distance_transform_edt
from shapely.geometry import mapping


ROOT = Path(__file__).resolve().parent.parent

DEM_DIR = ROOT / "data/processed/DEM"

ORIGINAL = DEM_DIR / "district_dem_utm/UK/Haridwar.tif"
BUFFERED = DEM_DIR / "district_dem_buffered/UK/Haridwar.tif"

BOUNDARY = ROOT / "data/processed/boundary/ihr_districts.gpkg"

OUTPUT = DEM_DIR / "haridwar_elevation_difference.tif"

BUFFER = 10
NODATA = -9999.0


print("=" * 65)
print("HARIDWAR ELEVATION CONSISTENCY AUDIT")
print("=" * 65)


with rasterio.open(ORIGINAL) as ref, \
     rasterio.open(BUFFERED) as src:

    if (
        ref.crs != src.crs
        or src.width != ref.width + 2 * BUFFER
        or src.height != ref.height + 2 * BUFFER
        or not np.allclose(
            tuple(src.transform),
            tuple(ref.transform * Affine.translation(-BUFFER, -BUFFER)),
            rtol=0,
            atol=1e-7
        )
    ):
        raise RuntimeError("DEM alignment mismatch.")

    original = ref.read(1, masked=True)

    buffered = src.read(
        1,
        window=Window(
            BUFFER,
            BUFFER,
            ref.width,
            ref.height
        ),
        masked=True
    )

    districts = gpd.read_file(BOUNDARY)

    selected = districts[
        (districts["state"].str.upper() == "UK")
        & (districts["district"].str.lower() == "haridwar")
    ].to_crs(ref.crs)

    if len(selected) != 1:
        raise RuntimeError("Haridwar boundary match failed.")

    geometry = selected.geometry.iloc[0]

    if not geometry.is_valid:
        geometry = geometry.buffer(0)

    inside = geometry_mask(
        [mapping(geometry)],
        out_shape=(ref.height, ref.width),
        transform=ref.transform,
        invert=True,
        all_touched=False
    )

    original_valid = (
        ~np.ma.getmaskarray(original)
        & np.isfinite(original.data)
    )

    buffered_valid = (
        ~np.ma.getmaskarray(buffered)
        & np.isfinite(buffered.data)
    )

    common = inside & original_valid & buffered_valid

    difference = np.full(
        original.shape,
        NODATA,
        dtype=np.float32
    )

    # Positive difference means buffered elevation is higher.
    difference[common] = (
        buffered.data[common] - original.data[common]
    )

    absolute = np.abs(difference[common])

    if absolute.size == 0:
        raise RuntimeError("No comparable elevation pixels.")

    print("\n1. DIFFERENCE STATISTICS")
    print("-" * 50)

    print(f"Compared pixels: {absolute.size:,}")
    print(f"Mean absolute difference: {absolute.mean():.6f} m")
    print(f"Median absolute difference: {np.median(absolute):.6f} m")
    print(f"Mean signed difference: {difference[common].mean():.6f} m")
    print(f"Maximum difference: {absolute.max():.6f} m")

    print("\nDifference percentiles:")

    for percentile in [50, 90, 95, 99, 99.9]:
        value = np.percentile(absolute, percentile)
        print(f"  P{percentile}: {value:.4f} m")


    print("\n2. DIFFERENCE DISTRIBUTION")
    print("-" * 50)

    for threshold in [1, 2, 5, 10, 20, 30]:

        count = np.count_nonzero(absolute > threshold)
        percentage = count / absolute.size * 100

        print(
            f"Difference > {threshold:2} m: "
            f"{count:>10,} pixels ({percentage:.4f}%)"
        )


    print("\n3. BOUNDARY VS INTERIOR")
    print("-" * 50)

    # Pad with outside pixels so raster edges are handled.
    padded = np.pad(
        inside, 1, mode="constant", constant_values=False
    )

    distance = distance_transform_edt(
        padded,
        sampling=(ref.res[1], ref.res[0])
    )[1:-1, 1:-1]

    edge = common & (distance <= 90)
    interior = common & (distance > 90)

    for name, mask in [
        ("Boundary zone (within 90 m)", edge),
        ("District interior", interior)
    ]:

        values = np.abs(difference[mask])

        print(f"\n{name}")

        if values.size > 0:
            print(f"  Pixels: {values.size:,}")
            print(f"  Mean difference: {values.mean():.6f} m")
            print(f"  Maximum difference: {values.max():.6f} m")


    print("\n4. ELEVATION RANGES INSIDE DISTRICT")
    print("-" * 50)

    for name, data, mask in [
        ("Original", original, inside & original_valid),
        ("Buffered", buffered, inside & buffered_valid)
    ]:

        values = data.data[mask]

        if values.size > 0:
            print(f"\n{name} DEM:")
            print(f"  Minimum: {values.min():.4f} m")
            print(f"  Maximum: {values.max():.4f} m")
            print(f"  Mean: {values.mean():.4f} m")


    print("\n5. RECOVERED ELEVATION PIXELS")
    print("-" * 50)

    recovered = inside & ~original_valid & buffered_valid
    lost = inside & original_valid & ~buffered_valid

    print(f"Recovered pixels: {np.count_nonzero(recovered):,}")
    print(f"Lost pixels: {np.count_nonzero(lost):,}")


    profile = ref.profile.copy()

    profile.update(
        dtype="float32",
        nodata=NODATA,
        count=1,
        compress="deflate",
        predictor=3
    )

    with rasterio.open(OUTPUT, "w", **profile) as dst:
        dst.write(difference, 1)
        dst.set_band_description(
            1, "Buffered minus original elevation (metres)"
        )


print("\n" + "=" * 65)
print(f"Difference raster saved to:\n{OUTPUT}")
print("Elevation consistency audit complete.")
print("=" * 65)