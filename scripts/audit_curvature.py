from pathlib import Path

import numpy as np
import rasterio


ROOT = Path(__file__).resolve().parent.parent

TERRAIN = ROOT / "data/processed/terrain_v2/UK/Haridwar"

SLOPE = TERRAIN / "slope.tif"
PLAN = TERRAIN / "plan_curvature.tif"
PROFILE = TERRAIN / "profile_curvature.tif"

OUTPUT = TERRAIN / "tangential_curvature.tif"

NODATA = -9999.0


def describe(name, values):

    print(f"\n{name}")
    print("-" * 50)

    if values.size == 0:
        print("No valid pixels.")
        return

    print(f"Valid pixels: {values.size:,}")
    print(f"Minimum: {values.min():.8f}")
    print(f"Maximum: {values.max():.8f}")
    print(f"Mean: {values.mean():.8f}")
    print(f"Standard deviation: {values.std():.8f}")

    absolute = np.abs(values)

    print("\nAbsolute curvature percentiles:")

    for p in [50, 90, 95, 99, 99.9]:

        print(
            f"  P{p}: "
            f"{np.percentile(absolute, p):.8f}"
        )


print("=" * 65)
print("HARIDWAR CURVATURE AUDIT")
print("=" * 65)


with rasterio.open(SLOPE) as slope_src, \
     rasterio.open(PLAN) as plan_src, \
     rasterio.open(PROFILE) as profile_src:

    for src in [plan_src, profile_src]:

        if (
            src.crs != slope_src.crs
            or src.transform != slope_src.transform
            or src.shape != slope_src.shape
        ):
            raise RuntimeError("Raster alignment mismatch.")

    print("\nRaster alignment: PASS")

    slope = slope_src.read(1, masked=True)
    plan = plan_src.read(1, masked=True)
    profile = profile_src.read(1, masked=True)

    valid = (
        ~np.ma.getmaskarray(slope)
        & ~np.ma.getmaskarray(plan)
        & np.isfinite(slope.data)
        & np.isfinite(plan.data)
    )

    profile_valid = (
        ~np.ma.getmaskarray(profile)
        & np.isfinite(profile.data)
    )

    describe(
        "PLAN CURVATURE",
        plan.data[valid]
    )

    describe(
        "PROFILE CURVATURE",
        profile.data[profile_valid]
    )

    print("\n" + "=" * 65)
    print("PLAN CURVATURE BY SLOPE CATEGORY")
    print("=" * 65)

    categories = [
        (0, 0.1),
        (0.1, 0.5),
        (0.5, 1),
        (1, 5),
        (5, 15),
        (15, 30),
        (30, 90.01)
    ]

    for lower, upper in categories:

        mask = (
            valid
            & (slope.data >= lower)
            & (slope.data < upper)
        )

        values = np.abs(plan.data[mask])

        print(f"\nSlope: {lower} to {upper} degrees")
        print(f"  Pixels: {values.size:,}")

        if values.size:

            print(
                f"  Median absolute curvature: "
                f"{np.median(values):.8f}"
            )

            print(
                f"  P99 absolute curvature: "
                f"{np.percentile(values, 99):.8f}"
            )

            print(
                f"  Maximum absolute curvature: "
                f"{values.max():.8f}"
            )

    # Generate tangential curvature.

    tangential = np.full(
        slope.shape,
        NODATA,
        dtype=np.float32
    )

    slope_radians = np.radians(
        slope.data[valid].astype(np.float64)
    )

    tangential[valid] = (
        plan.data[valid].astype(np.float64)
        * np.sin(slope_radians)
    )

    describe(
        "TANGENTIAL CURVATURE",
        tangential[valid]
    )

    metadata = slope_src.profile.copy()

    metadata.update(
        dtype="float32",
        nodata=NODATA,
        count=1,
        compress="deflate",
        predictor=3
    )

    with rasterio.open(
        OUTPUT, "w", **metadata
    ) as dst:

        dst.write(tangential, 1)

        dst.set_band_description(
            1, "Tangential curvature"
        )

        dst.update_tags(
            units="1/metre",
            source="Buffered DEM V2",
            algorithm="Plan curvature multiplied by sin(slope)",
            sign_convention="Positive convex",
            neighbourhood="3x3"
        )


print("\n" + "=" * 65)
print("CURVATURE AUDIT COMPLETE")
print("=" * 65)

print(f"\nTangential curvature saved to:\n{OUTPUT}")