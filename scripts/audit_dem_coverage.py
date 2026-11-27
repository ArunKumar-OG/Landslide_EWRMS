
import pandas as pd
from pathlib import Path
from collections import Counter

TILE_REPORT = Path("data/processed/DEM/dem_tile_coverage.csv")
DISTRICT_REPORT = Path("data/processed/DEM/district_dem_coverage.csv")

tiles = pd.read_csv(TILE_REPORT)
districts = pd.read_csv(DISTRICT_REPORT)

print("=" * 65)
print("COMPLETE DEM COVERAGE AUDIT")
print("=" * 65)

# -----------------------------------
# 1. TILE SUMMARY
# -----------------------------------

print("\n1. DEM TILE SUMMARY")

relevant = tiles[
    tiles["intersecting_district_count"] > 0
]

irrelevant = tiles[
    tiles["intersecting_district_count"] == 0
]

print("Total DEM tiles:", len(tiles))
print("Relevant tiles:", len(relevant))
print("Irrelevant tiles:", len(irrelevant))

# -----------------------------------
# 2. DISTRICT COVERAGE
# -----------------------------------

print("\n2. DISTRICT COVERAGE SUMMARY")

print("Total district records:", len(districts))

print("\nCoverage status:")
print(districts["coverage_status"].value_counts().to_string())

covered = districts[
    districts["coverage_status"] == "COVERED"
]

not_covered = districts[
    districts["coverage_status"] != "COVERED"
]

print("\nCovered districts:", len(covered))
print("Not covered:", len(not_covered))

if not not_covered.empty:
    print("\nDistricts requiring attention:")
    print(
        not_covered[
            ["state", "district", "intersecting_tile_count",
             "coverage_status"]
        ].to_string(index=False)
    )

# -----------------------------------
# 3. UNIQUE REQUIRED DEM TILES
# -----------------------------------

print("\n3. UNIQUE REQUIRED DEM TILES")

tile_counter = Counter()

for value in districts["tiles"].dropna():
    for name in str(value).split(";"):
        name = name.strip()

        if name:
            tile_counter[name] += 1

required_tiles = set(tile_counter)

print("Unique required tiles:", len(required_tiles))

print(
    "Tiles shared by multiple districts:",
    sum(1 for count in tile_counter.values() if count > 1)
)

# -----------------------------------
# 4. CONSISTENCY CHECK
# -----------------------------------

print("\n4. TILE REPORT CONSISTENCY")

reported_tiles = set(relevant["tile"].astype(str))

missing_from_tile_report = required_tiles - reported_tiles
missing_from_district_report = reported_tiles - required_tiles

print(
    "District-required tiles absent from tile report:",
    len(missing_from_tile_report)
)

print(
    "Relevant tile-report entries absent from district report:",
    len(missing_from_district_report)
)

if missing_from_tile_report:
    print("\nMissing tile names:")
    for name in sorted(missing_from_tile_report):
        print(name)

if missing_from_district_report:
    print("\nExtra tile names:")
    for name in sorted(missing_from_district_report):
        print(name)

# -----------------------------------
# 5. DUPLICATE DISTRICT CHECK
# -----------------------------------

print("\n5. DISTRICT RECORD CHECK")

duplicates = districts.duplicated(
    subset=["state", "district"]
)

print("Duplicate district records:", duplicates.sum())

print("\n6. FINAL SUMMARY")

print("All districts covered:", len(not_covered) == 0)
print("Tile reports consistent:",
      len(missing_from_tile_report) == 0
      and len(missing_from_district_report) == 0)

print("\nAudit completed.")
