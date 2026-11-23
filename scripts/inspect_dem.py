
from pathlib import Path
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEM_DIR = PROJECT_ROOT / "data" / "raw" / "DEM"

print("=" * 60)
print("DEM DATA INSPECTION")
print("=" * 60)

if not DEM_DIR.exists():
    raise FileNotFoundError(f"DEM directory not found: {DEM_DIR}")

dem_files = sorted(DEM_DIR.glob("*.tif"))

print(f"\nTotal DEM tiles: {len(dem_files)}")

for index, dem_file in enumerate(dem_files, start=1):
    print(f"\n[{index}/{len(dem_files)}] {dem_file.name}")

    try:
        with rasterio.open(dem_file) as src:
            print("  CRS:", src.crs)
            print("  Dimensions:", src.width, "x", src.height)
            print("  Resolution:", src.res)
            print("  Bounds:", src.bounds)
            print("  NoData:", src.nodata)
            print("  Data type:", src.dtypes)

    except Exception as e:
        print("  ERROR:", e)

print("\nDEM inspection completed.")
