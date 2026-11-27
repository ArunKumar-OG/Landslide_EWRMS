
import rasterio
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DEM_DIR = BASE_DIR / "data" / "raw" / "dem"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

checks = {
    "Copernicus_DSM_COG_10_N12_00_E080_00_DEM.tif": ("below", -50),
    "Copernicus_DSM_COG_10_N27_00_E088_00_DEM.tif": ("above", 8500),
    "Copernicus_DSM_COG_10_N35_00_E076_00_DEM.tif": ("above", 8500),
}

records = []

for filename, (condition, threshold) in checks.items():
    path = DEM_DIR / filename

    with rasterio.open(path) as src:
        data = src.read(1)

        if condition == "below":
            rows, cols = np.where(data < threshold)
        else:
            rows, cols = np.where(data > threshold)

        print(f"\n{filename}")
        print(f"Flagged pixels: {len(rows)}")

        for row, col in zip(rows, cols):
            x, y = rasterio.transform.xy(
                src.transform, int(row), int(col)
            )

            records.append({
                "tile": filename,
                "longitude": x,
                "latitude": y,
                "elevation_m": float(data[row, col]),
                "row": int(row),
                "column": int(col),
            })

        if len(rows):
            values = data[rows, cols]
            idx = np.argmin(values) if condition == "below" else np.argmax(values)

            print("Most extreme pixel:")
            print(f"  Elevation: {values[idx]:.2f} m")
            print(f"  Longitude: {records[-len(rows) + idx]['longitude']:.6f}")
            print(f"  Latitude: {records[-len(rows) + idx]['latitude']:.6f}")

output = OUTPUT_DIR / "dem_anomaly_coordinates.csv"
pd.DataFrame(records).to_csv(output, index=False)

print(f"\nSaved anomaly coordinates to: {output}")
