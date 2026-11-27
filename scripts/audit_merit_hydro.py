from pathlib import Path
from contextlib import ExitStack
from collections import Counter
import math

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds
from rasterio.transform import from_origin
from shapely.geometry import mapping


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

RAW = ROOT / "data/raw/hydrology/merit_hydro"

BOUNDARY = ROOT / "data/processed/boundary/ihr_districts.gpkg"

OUTPUT = ROOT / "data/processed/hydrology"

LAYERS = ["upa", "hnd", "elv", "dir"]

RES = 1 / 1200

BLOCK = 512

MIN_COVERAGE = 95.0

VALID_DIRECTIONS = [
    -1, 0, 1, 2, 4, 8, 16, 32, 64, 128
]

OUTPUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# PIXEL VALIDATION
# ============================================================

def validate_pixels(data, layer):

    values = data.data.astype(np.float64)

    missing = np.ma.getmaskarray(data).copy()

    if layer == "dir":

        # Normalize signed/unsigned byte representations.
        values[values == -128] = 128
        values[values == 255] = -1
        values[values == 247] = -9

        missing |= values == -9

        invalid = (
            ~missing
            & ~np.isin(values, VALID_DIRECTIONS)
        )

    else:

        # MERIT Hydro documented undefined value.
        missing |= values == -9999

        invalid = (
            ~missing
            & ~np.isfinite(values)
        )

        if layer in ["upa", "hnd"]:

            invalid |= (
                ~missing
                & (values < 0)
            )

        elif layer == "elv":

            # Broad diagnostic elevation limits.
            invalid |= (
                ~missing
                & (
                    (values < -500)
                    | (values > 9000)
                )
            )

    valid = ~missing & ~invalid

    return values, valid, missing, invalid


# ============================================================
# TILE-LEVEL AUDIT
# ============================================================

def audit_tiles():

    records = []
    inventory = {}

    print("\n" + "=" * 65)
    print("PART 1: HYDROLOGICAL TILE PIXEL AUDIT")
    print("=" * 65)

    for layer in LAYERS:

        files = sorted((RAW / layer).rglob("*.tif"))

        if len(files) != 12:
            raise RuntimeError(
                f"Expected 12 {layer} tiles, found {len(files)}"
            )

        for path in files:

            tile = path.stem.removesuffix(f"_{layer}")

            key = (tile, layer)

            if key in inventory:
                raise RuntimeError(f"Duplicate tile: {key}")

            inventory[key] = path

            counts = Counter()
            direction_counts = Counter()

            minimum = np.inf
            maximum = -np.inf

            with rasterio.open(path) as src:

                total = src.width * src.height

                for _, window in src.block_windows(1):

                    data = src.read(
                        1, window=window, masked=True
                    )

                    values, valid, missing, invalid = (
                        validate_pixels(data, layer)
                    )

                    counts["valid"] += int(
                        np.count_nonzero(valid)
                    )

                    counts["missing"] += int(
                        np.count_nonzero(missing)
                    )

                    counts["invalid"] += int(
                        np.count_nonzero(invalid)
                    )

                    selected = values[valid]

                    if selected.size:

                        minimum = min(
                            minimum, selected.min()
                        )

                        maximum = max(
                            maximum, selected.max()
                        )

                        if layer == "dir":

                            codes, frequencies = np.unique(
                                selected,
                                return_counts=True
                            )

                            for code, count in zip(
                                codes, frequencies
                            ):

                                direction_counts[int(code)] += (
                                    int(count)
                                )

                status = (
                    "PASS"
                    if counts["invalid"] == 0
                    and counts["valid"] > 0
                    else "REVIEW"
                )

                record = {
                    "tile": tile,
                    "layer": layer,
                    "dtype": src.dtypes[0],
                    "nodata": src.nodata,
                    "total_pixels": total,
                    "valid_pixels": counts["valid"],
                    "missing_pixels": counts["missing"],
                    "invalid_pixels": counts["invalid"],
                    "valid_percentage":
                        counts["valid"] / total * 100,
                    "minimum":
                        minimum if counts["valid"] else np.nan,
                    "maximum":
                        maximum if counts["valid"] else np.nan,
                    "status": status
                }

                if layer == "dir":

                    for code in VALID_DIRECTIONS:
                        record[f"direction_{code}"] = (
                            direction_counts[code]
                        )

                records.append(record)

                print(
                    f"{path.name}: {status} | "
                    f"Valid: {record['valid_percentage']:.2f}% | "
                    f"Invalid: {counts['invalid']:,}",
                    flush=True
                )

    report = pd.DataFrame(records)

    report.to_csv(
        OUTPUT / "merit_pixel_audit.csv",
        index=False
    )

    return report, inventory


# ============================================================
# DISTRICT COVERAGE AUDIT
# ============================================================

def audit_districts(inventory):

    print("\n" + "=" * 65)
    print("PART 2: DISTRICT HYDROLOGICAL COVERAGE")
    print("=" * 65)

    districts = gpd.read_file(BOUNDARY).to_crs("EPSG:4326")

    results = []

    for index, row in enumerate(
        districts.itertuples(), start=1
    ):

        geom = row.geometry

        if not geom.is_valid:
            geom = geom.buffer(0)

        if geom.is_empty:
            raise RuntimeError(f"Empty geometry: {row.district}")

        geometry = mapping(geom)

        west, south, east, north = geom.bounds

        total_inside = 0
        common_count = 0
        uncovered = 0

        valid_counts = Counter()
        invalid_counts = Counter()

        # Include the source grid's half-pixel offset.
        lon_start = math.floor((west + RES / 2) / 5) * 5
        lon_end = math.floor((east + RES / 2) / 5) * 5

        lat_start = math.floor((south + RES / 2) / 5) * 5
        lat_end = math.floor((north + RES / 2) / 5) * 5

        for lat in range(lat_start, lat_end + 1, 5):

            for lon in range(lon_start, lon_end + 1, 5):

                tile = f"n{lat:02d}e{lon:03d}"

                transform = from_origin(
                    lon - RES / 2,
                    lat + 5 - RES / 2,
                    RES,
                    RES
                )

                requested = from_bounds(
                    west, south, east, north,
                    transform=transform
                )

                c0 = max(0, math.floor(requested.col_off))
                r0 = max(0, math.floor(requested.row_off))

                c1 = min(
                    6000,
                    math.ceil(
                        requested.col_off + requested.width
                    )
                )

                r1 = min(
                    6000,
                    math.ceil(
                        requested.row_off + requested.height
                    )
                )

                if c1 <= c0 or r1 <= r0:
                    continue

                with ExitStack() as stack:

                    sources = {}

                    for layer in LAYERS:

                        path = inventory.get((tile, layer))

                        if path is None:
                            continue

                        src = stack.enter_context(
                            rasterio.open(path)
                        )

                        if not np.allclose(
                            tuple(src.transform),
                            tuple(transform),
                            rtol=0,
                            atol=1e-9
                        ):

                            raise RuntimeError(
                                f"Grid mismatch: {path.name}"
                            )

                        sources[layer] = src

                    for r in range(r0, r1, BLOCK):

                        for c in range(c0, c1, BLOCK):

                            height = min(BLOCK, r1 - r)
                            width = min(BLOCK, c1 - c)

                            window = Window(
                                c, r, width, height
                            )

                            inside = geometry_mask(
                                [geometry],
                                out_shape=(height, width),
                                transform=(
                                    transform
                                    * rasterio.Affine.translation(c, r)
                                ),
                                invert=True,
                                all_touched=False
                            )

                            inside_count = int(
                                np.count_nonzero(inside)
                            )

                            if inside_count == 0:
                                continue

                            total_inside += inside_count

                            common = inside.copy()

                            if len(sources) != len(LAYERS):
                                uncovered += inside_count

                            for layer in LAYERS:

                                if layer not in sources:

                                    common[:] = False
                                    continue

                                data = sources[layer].read(
                                    1,
                                    window=window,
                                    masked=True
                                )

                                _, valid, _, invalid = (
                                    validate_pixels(data, layer)
                                )

                                valid_counts[layer] += int(
                                    np.count_nonzero(
                                        inside & valid
                                    )
                                )

                                invalid_counts[layer] += int(
                                    np.count_nonzero(
                                        inside & invalid
                                    )
                                )

                                common &= valid

                            common_count += int(
                                np.count_nonzero(common)
                            )

        record = {
            "state_code": row.state,
            "district": row.district,
            "district_pixels": total_inside,
            "uncovered_pixels": uncovered
        }

        for layer in LAYERS:

            record[f"{layer}_valid_pixels"] = (
                valid_counts[layer]
            )

            record[f"{layer}_invalid_pixels"] = (
                invalid_counts[layer]
            )

            record[f"{layer}_coverage"] = (
                valid_counts[layer] / total_inside * 100
                if total_inside else 0
            )

        common_coverage = (
            common_count / total_inside * 100
            if total_inside else 0
        )

        record["common_valid_pixels"] = common_count
        record["common_coverage"] = common_coverage

        record["status"] = (
            "PASS"
            if common_coverage >= MIN_COVERAGE
            and sum(invalid_counts.values()) == 0
            and uncovered == 0
            else "REVIEW"
        )

        results.append(record)

        print(
            f"[{index}/{len(districts)}] "
            f"{row.state}/{row.district} | "
            f"Common coverage: {common_coverage:.4f}% | "
            f"{record['status']}",
            flush=True
        )

    report = pd.DataFrame(results)

    report.to_csv(
        OUTPUT / "merit_district_coverage_audit.csv",
        index=False
    )

    return report


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("MERIT HYDRO PIXEL AND DISTRICT COVERAGE AUDIT")
    print("=" * 65)

    tiles, inventory = audit_tiles()

    districts = audit_districts(inventory)

    print("\n" + "=" * 65)
    print("FINAL AUDIT SUMMARY")
    print("=" * 65)

    print("\nTILE STATUS")
    print(tiles["status"].value_counts().to_string())

    print("\nDISTRICT STATUS")
    print(districts["status"].value_counts().to_string())

    columns = [
        f"{layer}_coverage"
        for layer in LAYERS
    ] + ["common_coverage"]

    print("\nCOVERAGE STATISTICS (%)")

    print(
        districts[columns]
        .agg(["min", "mean", "max"])
        .T.to_string()
    )

    print("\nFIVE DISTRICTS WITH LOWEST COMMON COVERAGE")

    print(
        districts.nsmallest(5, "common_coverage")[
            [
                "state_code",
                "district",
                "common_coverage",
                "status"
            ]
        ].to_string(index=False)
    )

    print("\nReports saved to:")

    print(OUTPUT / "merit_pixel_audit.csv")
    print(OUTPUT / "merit_district_coverage_audit.csv")

    print("=" * 65)


if __name__ == "__main__":
    main()