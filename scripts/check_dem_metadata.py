import rasterio
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parents[1]
DEM_DIR = BASE_DIR / "data" / "raw" / "dem"

files = sorted(DEM_DIR.glob("*.tif"))

crs_count = Counter()
resolution_count = Counter()
dimensions_count = Counter()
dtype_count = Counter()
nodata_count = Counter()

print(f"Total DEM files: {len(files)}")

for file in files:
    try:
        with rasterio.open(file) as src:
            crs_count[str(src.crs)] += 1
            resolution_count[str(src.res)] += 1
            dimensions_count[f"{src.width} x {src.height}"] += 1
            dtype_count[str(src.dtypes)] += 1
            nodata_count[str(src.nodata)] += 1

    except Exception as e:
        print(f"ERROR: {file.name}: {e}")

print("\n========== CRS ==========")
for key, value in crs_count.items():
    print(key, ":", value)

print("\n========== RESOLUTION ==========")
for key, value in resolution_count.items():
    print(key, ":", value)

print("\n========== DIMENSIONS ==========")
for key, value in dimensions_count.items():
    print(key, ":", value)

print("\n========== DATA TYPES ==========")
for key, value in dtype_count.items():
    print(key, ":", value)

print("\n========== NODATA ==========")
for key, value in nodata_count.items():
    print(key, ":", value)