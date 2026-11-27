
import rasterio
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DEM_DIR = BASE_DIR / "data" / "raw" / "dem"

files = sorted(DEM_DIR.glob("*.tif"))

print(f"Inspecting {len(files)} DEM tiles...\n")

for file in files:
    try:
        with rasterio.open(file) as src:
            data = src.read(1, masked=True)

            valid = data.compressed()

            if valid.size == 0:
                print(f"{file.name}: No valid pixels")
                continue

            print(f"\n{file.name}")
            print(f"  Min elevation: {np.min(valid):.2f} m")
            print(f"  Max elevation: {np.max(valid):.2f} m")
            print(f"  Mean elevation: {np.mean(valid):.2f} m")
            print(f"  NaN count: {np.isnan(valid).sum()}")
            print(f"  Infinite count: {np.isinf(valid).sum()}")
            print(f"  Valid pixels: {valid.size:,}")

    except Exception as e:
        print(f"ERROR: {file.name}: {e}")
