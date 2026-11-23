
import geopandas as gpd
import pandas as pd

DISTRICT_PATH = "data/processed/boundary/ihr_districts.gpkg"
COVERAGE_PATH = "data/processed/DEM/district_dem_coverage.csv"

districts = gpd.read_file(DISTRICT_PATH)
coverage = pd.read_csv(COVERAGE_PATH)

def normalize(value):
    return str(value).strip().casefold()

district_keys = set(
    zip(
        districts["state_name"].apply(normalize),
        districts["district"].apply(normalize)
    )
)

coverage_keys = set(
    zip(
        coverage["state"].apply(normalize),
        coverage["district"].apply(normalize)
    )
)

missing = district_keys - coverage_keys
extra = coverage_keys - district_keys

print("=" * 60)
print("DISTRICT MATCHING VALIDATION")
print("=" * 60)

print("GeoPackage districts:", len(district_keys))
print("Coverage districts:", len(coverage_keys))
print("Missing matches:", len(missing))
print("Extra coverage records:", len(extra))

if missing:
    print("\nMissing districts:")
    for state, district in sorted(missing):
        print(state, "-", district)

if extra:
    print("\nExtra coverage records:")
    for state, district in sorted(extra):
        print(state, "-", district)

if not missing and not extra:
    print("\nALL 137 DISTRICTS MATCH SUCCESSFULLY.")
