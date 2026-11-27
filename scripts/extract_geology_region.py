from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio

ROOT = Path(__file__).resolve().parent.parent

GDB = ROOT / "data/raw/geology/LiMW_GIS 2015.gdb"
BOUNDARY = ROOT / "data/processed/boundary/ihr_districts.gpkg"
OUTPUT = ROOT / "data/processed/geology"

LAYER = "GLiM_export"
BUFFER = 20000

VALID_CODES = {
    "su", "ss", "py", "sm",
    "sc", "ev", "va", "vi",
    "vb", "pa", "pi", "pb",
    "mt", "wb", "ig", "nd"
}

OUTPUT.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("GLIM REGIONAL EXTRACTION AND ATTRIBUTE AUDIT")
print("=" * 65)

# ----------------------------------------------------------
# 1. Read source metadata
# ----------------------------------------------------------

info = pyogrio.read_info(GDB, layer=LAYER)

source_crs = info["crs"]

print("\nGlobal features:", info["features"])
print("Source CRS:", source_crs)

# ----------------------------------------------------------
# 2. Load district boundaries
# ----------------------------------------------------------

districts = gpd.read_file(BOUNDARY)

print("\nDistricts:", len(districts))

if len(districts) != 137:
    raise RuntimeError("Unexpected district count.")

districts = districts.to_crs(source_crs)

xmin, ymin, xmax, ymax = districts.total_bounds

bbox = (
    xmin - BUFFER,
    ymin - BUFFER,
    xmax + BUFFER,
    ymax + BUFFER
)

print("\nBuffered regional bounding box:")
print(bbox)

# ----------------------------------------------------------
# 3. Extract regional geology
# ----------------------------------------------------------

print("\nReading regional geology polygons...", flush=True)

geology = pyogrio.read_dataframe(
    GDB,
    layer=LAYER,
    bbox=bbox,
    columns=[
        "IDENTITY_",
        "Litho",
        "xx"
    ],
    fid_as_index=True
)

geology.index.name = "source_fid"
geology = geology.reset_index()

print("Regional features:", len(geology))

if geology.empty:
    raise RuntimeError("Regional extraction returned no features.")

# ----------------------------------------------------------
# 4. Geometry audit
# ----------------------------------------------------------

null_geometry = geology.geometry.isna()
empty_geometry = geology.geometry.is_empty

usable = ~(null_geometry | empty_geometry)

invalid_geometry = usable & ~geology.geometry.is_valid

print("\nGEOMETRY AUDIT")
print("-" * 50)

print("Total polygons:", len(geology))
print("Null geometries:", null_geometry.sum())
print("Empty geometries:", empty_geometry.sum())
print("Invalid geometries:", invalid_geometry.sum())

print("\nGeometry types:")
print(geology.geom_type.value_counts(dropna=False))

# ----------------------------------------------------------
# 5. Lithology attribute audit
# ----------------------------------------------------------

xx = geology["xx"].astype("string").str.strip().str.lower()

litho = (
    geology["Litho"]
    .astype("string")
    .str.strip()
    .str.lower()
)

unknown = ~xx.isin(VALID_CODES)

comparable = xx.notna() & litho.notna()

mismatch = comparable & (
    litho.str[:2] != xx
).fillna(False)

print("\nATTRIBUTE AUDIT")
print("-" * 50)

print("Missing xx values:", xx.isna().sum())
print("Missing Litho values:", litho.isna().sum())
print("Unknown or missing class codes:", unknown.sum())

print(
    "Litho and xx mismatches:",
    mismatch.sum()
)

print(
    "Explicit no-data polygons:",
    (xx == "nd").sum()
)

print(
    "Duplicate source identifiers:",
    geology["IDENTITY_"].duplicated().sum()
)

# ----------------------------------------------------------
# 6. Class distributions
# ----------------------------------------------------------

print("\nLEVEL 1 LITHOLOGY DISTRIBUTION")
print("-" * 50)

print(xx.value_counts(dropna=False).to_string())

print("\nDETAILED LITHOLOGY DISTRIBUTION")
print("-" * 50)

print(litho.value_counts(dropna=False).to_string())

print("\nUnique level 1 classes:", xx.nunique())
print("Unique detailed classes:", litho.nunique())

summary = pd.DataFrame({
    "xx": xx,
    "Litho": litho
})

summary = (
    summary
    .groupby(["xx", "Litho"], dropna=False)
    .size()
    .reset_index(name="polygon_count")
)

# ----------------------------------------------------------
# 7. Save regional dataset
# ----------------------------------------------------------

geology["xx_normalized"] = xx
geology["litho_normalized"] = litho

geology["geometry_invalid"] = invalid_geometry
geology["class_unknown"] = unknown
geology["class_mismatch"] = mismatch

output_file = OUTPUT / "glim_regional_candidates.gpkg"

print("\nSaving regional geology dataset...", flush=True)

pyogrio.write_dataframe(
    geology,
    output_file,
    layer="geology",
    driver="GPKG"
)

summary.to_csv(
    OUTPUT / "glim_class_inventory.csv",
    index=False
)

# ----------------------------------------------------------
# 8. Final summary
# ----------------------------------------------------------

print("\n" + "=" * 65)
print("FINAL REGIONAL EXTRACTION SUMMARY")
print("=" * 65)

print("Global polygons:", info["features"])
print("Extracted polygons:", len(geology))
print("Districts:", len(districts))

print(
    "Retained percentage:",
    round(len(geology) / info["features"] * 100, 4)
)

print("Invalid geometries:", invalid_geometry.sum())
print("Unknown classes:", unknown.sum())
print("Attribute mismatches:", mismatch.sum())

print("\nOutput files:")
print(output_file)
print(OUTPUT / "glim_class_inventory.csv")

print("\nExtraction complete.")
print("Original geology dataset unchanged.")
print("=" * 65)