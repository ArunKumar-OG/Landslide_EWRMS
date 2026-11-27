from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

ROOT = Path(__file__).resolve().parent.parent

BOUNDARY = (
    ROOT / "data/processed/boundary/ihr_districts.gpkg"
)

districts = gpd.read_file(BOUNDARY).to_crs("EPSG:4326")

# Approximate acquisition margin in degrees.
BUFFER = 0.1

LAYERS = ["upa", "hnd", "elv", "dir"]


def coordinate_name(lat, lon):

    ns = "n" if lat >= 0 else "s"
    ew = "e" if lon >= 0 else "w"

    return f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}"


print("=" * 65)
print("MERIT HYDRO DOWNLOAD PACKAGE IDENTIFICATION")
print("=" * 65)

print(f"\nDistricts: {len(districts)}")

print("\nStudy area bounds:")

west, south, east, north = districts.total_bounds

print(f"West:  {west:.6f}")
print(f"South: {south:.6f}")
print(f"East:  {east:.6f}")
print(f"North: {north:.6f}")

packages = []
tiles = []

print("\nREQUIRED 30-DEGREE PACKAGES")
print("-" * 50)

for lat in range(-60, 90, 30):

    for lon in range(-180, 180, 30):

        extent = box(
            lon - BUFFER,
            lat - BUFFER,
            lon + 30 + BUFFER,
            lat + 30 + BUFFER
        )

        matches = districts.intersects(extent)

        if matches.any():

            name = coordinate_name(lat, lon)
            packages.append(name)

            print(
                f"{name}: "
                f"{matches.sum()} intersecting districts"
            )

print("\nREQUIRED DOWNLOAD ARCHIVES")
print("-" * 50)

for layer in LAYERS:

    print(f"\nLayer: {layer}")

    for package in packages:
        print(f"  {layer}_{package}.tar")


print("\nREQUIRED 5-DEGREE TILES")
print("-" * 50)

for lat in range(-60, 90, 5):

    for lon in range(-180, 180, 5):

        extent = box(
            lon - BUFFER,
            lat - BUFFER,
            lon + 5 + BUFFER,
            lat + 5 + BUFFER
        )

        if districts.intersects(extent).any():

            name = coordinate_name(lat, lon)

            tiles.append(name)
            print(name)


print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)

print(f"Regional packages: {len(packages)}")
print(f"Required layers: {len(LAYERS)}")
print(f"Expected archives: {len(packages) * len(LAYERS)}")
print(f"Required tiles per layer: {len(tiles)}")

print("=" * 65)