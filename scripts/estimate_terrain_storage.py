from pathlib import Path
import shutil
import rasterio

ROOT = Path(__file__).resolve().parent.parent

DEM_ROOT = (
    ROOT / "data/processed/DEM/district_dem_utm"
)

files = sorted(DEM_ROOT.rglob("*.tif"))

FEATURE_COUNT = 8
BYTES_PER_PIXEL = 4
BUFFER = 10

total_pixels = 0
buffered_pixels = 0

largest = []

for path in files:

    with rasterio.open(path) as src:

        pixels = src.width * src.height

        buffered = (
            (src.width + 2 * BUFFER)
            * (src.height + 2 * BUFFER)
        )

        total_pixels += pixels
        buffered_pixels += buffered

        largest.append((
            path.stem,
            pixels,
            src.width,
            src.height
        ))


def gb(size):
    return size / (1024**3)


terrain_size = (
    total_pixels * FEATURE_COUNT * BYTES_PER_PIXEL
)

buffered_size = (
    buffered_pixels * BYTES_PER_PIXEL
)

disk = shutil.disk_usage(ROOT)

print("=" * 65)
print("TERRAIN PROCESSING STORAGE ESTIMATE")
print("=" * 65)

print(f"\nDistricts: {len(files)}")
print(f"Total raster pixels: {total_pixels:,}")
print(f"Buffered raster pixels: {buffered_pixels:,}")

print("\nUNCOMPRESSED STORAGE ESTIMATES")
print("-" * 50)

print(f"Eight terrain features: {gb(terrain_size):.2f} GiB")
print(f"Buffered DEMs: {gb(buffered_size):.2f} GiB")

print(
    f"Combined storage: "
    f"{gb(terrain_size + buffered_size):.2f} GiB"
)

print("\nDISK INFORMATION")
print("-" * 50)

print(f"Total capacity: {gb(disk.total):.2f} GiB")
print(f"Used space: {gb(disk.used):.2f} GiB")
print(f"Available space: {gb(disk.free):.2f} GiB")

print("\nFIVE LARGEST DISTRICT RASTERS")
print("-" * 50)

for name, pixels, width, height in sorted(
    largest,
    key=lambda item: item[1],
    reverse=True
)[:5]:

    print(f"\n{name}")
    print(f"  Dimensions: {width} x {height}")
    print(f"  Pixels: {pixels:,}")

    print(
        f"  Eight feature rasters: "
        f"{gb(pixels * FEATURE_COUNT * 4):.2f} GiB"
    )

print("\nStorage estimation complete.")
print("=" * 65)