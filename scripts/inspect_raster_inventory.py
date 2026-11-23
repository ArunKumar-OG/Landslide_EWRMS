
import rasterio
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path("data")

extensions = {".tif", ".tiff", ".img", ".vrt"}

files = [
    p for p in DATA_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in extensions
]

groups = defaultdict(list)

for path in files:
    groups[str(path.parent)].append(path)

print("=" * 75)
print("RASTER DATA INVENTORY")
print("=" * 75)

for folder, paths in sorted(groups.items()):

    print(f"\nFOLDER: {folder}")
    print("Number of files:", len(paths))

    for path in sorted(paths):

        try:
            with rasterio.open(path) as src:

                print(f"\n  File: {path.name}")
                print(f"  Dimensions: {src.width} x {src.height}")
                print(f"  CRS: {src.crs}")
                print(f"  Resolution: {src.res}")
                print(f"  Bands: {src.count}")
                print(f"  Data type: {src.dtypes}")
                print(f"  NoData: {src.nodata}")
                print(f"  Bounds: {src.bounds}")

        except Exception as e:
            print(f"\n  ERROR: {path.name}")
            print(f"  {e}")

print("\n" + "=" * 75)
print("TOTAL RASTER FILES:", len(files))
print("=" * 75)
