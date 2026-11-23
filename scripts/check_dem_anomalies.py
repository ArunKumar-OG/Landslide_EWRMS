
import rasterio
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DEM_DIR = BASE_DIR / "data" / "raw" / "dem"

files = sorted(DEM_DIR.glob("*.tif"))

print("Checking elevation anomalies...\n")

for file in files:
    with rasterio.open(file) as src:
        data = src.read(1)

        negative = np.sum(data < 0)
        below_minus_50 = np.sum(data < -50)
        above_8500 = np.sum(data > 8500)

        if negative > 0 or above_8500 > 0:
            print(f"\n{file.name}")
            print(f"  Negative pixels: {negative:,}")
            print(f"  Pixels below -50m: {below_minus_50:,}")
            print(f"  Pixels above 8500m: {above_8500:,}")
            print(f"  Min: {np.min(data):.2f}")
            print(f"  Max: {np.max(data):.2f}")

print("\nAnomaly inspection complete.")
