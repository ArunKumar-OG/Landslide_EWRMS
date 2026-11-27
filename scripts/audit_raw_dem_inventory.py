from pathlib import Path
import pandas as pd
import rasterio

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data/raw/dem"
OUTPUT = ROOT / "data/processed/DEM/raw_dem_inventory.csv"

files = sorted(RAW_DIR.rglob("*.tif"))
records = []

print("=" * 65)
print("RAW DEM INVENTORY AUDIT")
print("=" * 65)

print(f"\nTotal TIFF files: {len(files)}")

for index, path in enumerate(files, start=1):

    try:
        with rasterio.open(path) as src:

            records.append({
                "filename": path.name,
                "relative_path": str(path.relative_to(RAW_DIR)),
                "size_mb": path.stat().st_size / (1024 ** 2),
                "crs": str(src.crs),
                "width": src.width,
                "height": src.height,
                "resolution_x": src.res[0],
                "resolution_y": src.res[1],
                "left": src.bounds.left,
                "bottom": src.bounds.bottom,
                "right": src.bounds.right,
                "top": src.bounds.top,
                "origin_x": src.transform.c,
                "origin_y": src.transform.f,
                "nodata": src.nodata,
                "dtype": src.dtypes[0],
                "status": "OK"
            })

    except Exception as error:

        records.append({
            "filename": path.name,
            "relative_path": str(path.relative_to(RAW_DIR)),
            "status": "ERROR",
            "error": str(error)
        })

    if index % 50 == 0:
        print(f"Inspected {index}/{len(files)} files")


df = pd.DataFrame(records)
valid = df[df["status"] == "OK"].copy()

print("\n1. FILE SUMMARY")
print("-" * 50)

print(f"Total files: {len(df)}")
print(f"Unique filenames: {df['filename'].nunique()}")
print(f"Readable files: {len(valid)}")
print(f"Unreadable files: {len(df) - len(valid)}")

copernicus = df["filename"].str.startswith("Copernicus_DSM_")

print(f"Copernicus files: {copernicus.sum()}")
print(f"Other TIFF files: {(~copernicus).sum()}")

print("\n2. DUPLICATE FILENAMES")
print("-" * 50)

duplicates = df[
    df.duplicated("filename", keep=False)
]

if duplicates.empty:
    print("No duplicate filenames.")
else:
    print(
        duplicates[
            ["filename", "relative_path"]
        ].to_string(index=False)
    )

print("\n3. CRS DISTRIBUTION")
print("-" * 50)

print(valid["crs"].value_counts().to_string())

print("\n4. RESOLUTION DISTRIBUTION")
print("-" * 50)

print(
    valid.groupby(
        ["resolution_x", "resolution_y"]
    ).size().to_string()
)

print("\n5. DIMENSION DISTRIBUTION")
print("-" * 50)

print(
    valid.groupby(
        ["width", "height"]
    ).size().to_string()
)

print("\n6. DUPLICATE SPATIAL EXTENTS")
print("-" * 50)

extent_duplicates = valid[
    valid.duplicated(
        ["crs", "left", "bottom", "right", "top"],
        keep=False
    )
]

print(f"Files sharing spatial extents: {len(extent_duplicates)}")

if not extent_duplicates.empty:
    print(
        extent_duplicates[
            ["filename", "relative_path"]
        ].to_string(index=False)
    )

print("\n7. HARIDWAR SOURCE TILES")
print("-" * 50)

tile_ids = [
    "N29_00_E077_00",
    "N29_00_E078_00",
    "N30_00_E077_00",
    "N30_00_E078_00"
]

haridwar = valid[
    valid["filename"].str.contains(
        "|".join(tile_ids),
        regex=True
    )
]

for _, row in haridwar.iterrows():

    print(f"\n{row['filename']}")
    print(f"  Path: {row['relative_path']}")
    print(f"  CRS: {row['crs']}")
    print(f"  Dimensions: {row['width']} x {row['height']}")
    print(f"  X resolution: {row['resolution_x']:.12f}")
    print(f"  Y resolution: {row['resolution_y']:.12f}")
    print(f"  X origin: {row['origin_x']:.12f}")
    print(f"  Y origin: {row['origin_y']:.12f}")
    print(f"  NoData: {row['nodata']}")
    print(f"  Data type: {row['dtype']}")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

print("\n" + "=" * 65)
print(f"Inventory saved to:\n{OUTPUT}")
print("Raw DEM inventory audit complete.")
print("=" * 65)