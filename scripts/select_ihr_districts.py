
import geopandas as gpd
import pandas as pd
from pathlib import Path
from shapely.validation import explain_validity

SOURCE = Path("data/raw/district_nwic.GeoJSON")

# States and UTs fully included in the working NMHS definition
FULL_COVERAGE_CODES = {
    "AR",  # Arunachal Pradesh
    "HP",  # Himachal Pradesh
    "MN",  # Manipur
    "ML",  # Meghalaya
    "MZ",  # Mizoram
    "NL",  # Nagaland
    "SK",  # Sikkim
    "TR",  # Tripura
    "UK",  # Uttarakhand
    "JK",  # Jammu & Kashmir
    "LK",  # Ladakh
}

# Only these districts are included from partially covered states
PARTIAL_DISTRICTS = {
    "AS": {
        "dima hasao",
        "karbi anglong",
        "west karbi anglong",
    },
    "WB": {
        "darjeeling",
        "kalimpong",
    },
}

EXPECTED_TOTAL = 137

def normalize(value):
    return str(value).strip().casefold()

# Read original data
gdf = gpd.read_file(SOURCE)

print("=" * 60)
print("IHR DISTRICT SELECTION AND AUDIT")
print("=" * 60)

# Normalize attribute values for matching
gdf["_state"] = gdf["state"].astype(str).str.strip().str.upper()
gdf["_district"] = gdf["district"].apply(normalize)

# Select fully covered states/UTs
full_mask = gdf["_state"].isin(FULL_COVERAGE_CODES)

# Select only specified districts from Assam and West Bengal
partial_mask = pd.Series(False, index=gdf.index)

for state_code, districts in PARTIAL_DISTRICTS.items():
    partial_mask |= (
        (gdf["_state"] == state_code)
        & gdf["_district"].isin(districts)
    )

selected = gdf[full_mask | partial_mask].copy()

# Remove temporary columns
selected = selected.drop(columns=["_state", "_district"])

print("\n1. SELECTION SUMMARY")
print("Fully covered state/UT features:", int(full_mask.sum()))
print("Partially covered features:", int(partial_mask.sum()))
print("Total selected features:", len(selected))
print("Expected features:", EXPECTED_TOTAL)

if len(selected) == EXPECTED_TOTAL:
    print("PASS: Expected district count achieved.")
else:
    print("WARNING: District count does not match expectation.")

# State-wise breakdown
print("\n2. STATE-WISE DISTRICT COUNTS")

print(
    selected.groupby("state_name")["district"]
    .count()
    .sort_index()
    .to_string()
)

# Check expected partial districts
print("\n3. PARTIAL DISTRICT VERIFICATION")

for state_code, expected in PARTIAL_DISTRICTS.items():
    actual = set(
        selected.loc[
            selected["state"] == state_code, "district"
        ].apply(normalize)
    )

    print(f"\nState code: {state_code}")
    print("Expected:", sorted(expected))
    print("Found:", sorted(actual))
    print("Missing:", sorted(expected - actual))
    print("Unexpected:", sorted(actual - expected))

# Duplicate district check
print("\n4. DUPLICATE DISTRICT CHECK")

duplicates = selected[
    selected.duplicated(
        subset=["state", "district"], keep=False
    )
]

if duplicates.empty:
    print("PASS: No duplicate state-district records.")
else:
    print(duplicates[["state_name", "district"]].to_string(index=False))

# Geometry validation
print("\n5. SELECTED GEOMETRY VALIDITY")

invalid = selected[~selected.geometry.is_valid]

print("Selected geometries:", len(selected))
print("Invalid geometries:", len(invalid))
print("Valid geometries:", len(selected) - len(invalid))

if not invalid.empty:
    print("\nInvalid district details:")

    for _, row in invalid.iterrows():
        print(
            f"{row['state_name']} | {row['district']} | "
            f"{explain_validity(row.geometry)}"
        )

print("\n6. CRS")
print(selected.crs)

print("\nAudit completed. No files were modified or exported.")
