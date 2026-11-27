import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


# ==========================================
# CONFIGURATION
# ==========================================

# Change this to your actual raw data directory
DATA_ROOT = Path("../Landslide_ML/data/raw")


# ==========================================
# GENERAL FILE INFORMATION
# ==========================================

def inspect_files():

    print("\n" + "=" * 60)
    print("       LANDSLIDE DATASET - FILE INSPECTION")
    print("=" * 60)

    if not DATA_ROOT.exists():
        print(f"\nDataset directory not found: {DATA_ROOT.resolve()}")
        return

    for folder in sorted(DATA_ROOT.iterdir()):

        if not folder.is_dir():
            continue

        print(f"\n\nFOLDER: {folder.name}")
        print("-" * 60)

        files = list(folder.rglob("*"))

        for file in files:

            if file.is_file():

                size_mb = file.stat().st_size / (1024 * 1024)

                print(f"File: {file.name}")
                print(f"Extension: {file.suffix}")
                print(f"Size: {size_mb:.2f} MB")


# ==========================================
# RASTER INSPECTION
# ==========================================

def inspect_raster(file_path):

    print("\n" + "=" * 60)
    print("RASTER INSPECTION:", file_path.name)
    print("=" * 60)

    try:

        with rasterio.open(file_path) as src:

            print("Driver:", src.driver)
            print("CRS:", src.crs)
            print("Width:", src.width)
            print("Height:", src.height)
            print("Bands:", src.count)
            print("Resolution:", src.res)
            print("Bounds:", src.bounds)
            print("NoData:", src.nodata)
            print("Data type:", src.dtypes)

            # Read a reduced sample to avoid loading
            # extremely large rasters into memory.
            band = src.read(
                1,
                out_shape=(
                    min(src.height, 512),
                    min(src.width, 512)
                ),
                masked=True
            )

            values = band.compressed()

            if len(values) > 0:

                print("Sample minimum:", np.min(values))
                print("Sample maximum:", np.max(values))
                print("Sample mean:", np.mean(values))
                print("Sample missing pixels:", np.ma.count_masked(band))

                if len(values) < 100000:
                    unique = np.unique(values)

                    if len(unique) <= 30:
                        print("Unique values:", unique)

            else:
                print("No valid sample pixels found.")

    except Exception as e:

        print("Could not read as raster.")
        print("Reason:", str(e))


# ==========================================
# CSV INSPECTION
# ==========================================

def inspect_csv(file_path):

    print("\n" + "=" * 60)
    print("CSV INSPECTION:", file_path.name)
    print("=" * 60)

    try:

        df = pd.read_csv(file_path)

        print("\nRows:", len(df))
        print("Columns:", len(df.columns))

        print("\nColumn names:")
        print(df.columns.tolist())

        print("\nData types:")
        print(df.dtypes)

        print("\nMissing values:")
        print(df.isnull().sum())

        print("\nFirst 5 records:")
        print(df.head())

        print("\nStatistical summary:")
        print(df.describe(include="all").to_string())

    except Exception as e:

        print("Could not read CSV.")
        print("Reason:", str(e))


# ==========================================
# MAIN FUNCTION
# ==========================================

def main():

    print("Python:", sys.version)

    inspect_files()

    for file_path in DATA_ROOT.rglob("*"):

        if not file_path.is_file():
            continue

        extension = file_path.suffix.lower()

        if extension in [".tif", ".tiff", ".grd"]:
            inspect_raster(file_path)

        elif extension == ".csv":
            inspect_csv(file_path)

        elif extension == ".gbindexes":
            print("\n" + "=" * 60)
            print("UNKNOWN / AUXILIARY FILE:", file_path.name)
            print("Size:", file_path.stat().st_size, "bytes")
            print("We need to identify its associated dataset.")
            print("=" * 60)


if __name__ == "__main__":
    main()