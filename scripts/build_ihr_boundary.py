
import geopandas as gpd
import pandas as pd
import shapely

from pathlib import Path
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

SOURCE = Path("data/raw/district_nwic.GeoJSON")
OUTPUT_DIR = Path("data/processed/boundary")

FULL_COVERAGE_CODES = {
    "AR", "HP", "MN", "ML", "MZ",
    "NL", "SK", "TR", "UK", "JK", "LK"
}

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


# --------------------------------------------------
# GEOMETRY REPAIR
# --------------------------------------------------

def extract_polygons(geom):
    """Extract polygon components recursively."""

    if geom is None or geom.is_empty:
        return []

    if isinstance(geom, Polygon):
        return [geom]

    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)

    if isinstance(geom, GeometryCollection):
        polygons = []

        for part in geom.geoms:
            polygons.extend(extract_polygons(part))

        return polygons

    return []


def repair_geometry(geom):
    """Repair invalid geometry and retain polygonal components."""

    if geom is None or geom.is_empty:
        raise ValueError("Empty geometry encountered.")

    original_area = geom.area

    if not geom.is_valid:
        geom = shapely.make_valid(geom)

    polygons = extract_polygons(geom)

    if not polygons:
        raise ValueError("No polygonal geometry remains after repair.")

    repaired = unary_union(polygons)

    if isinstance(repaired, Polygon):
        repaired = MultiPolygon([repaired])

    if not isinstance(repaired, MultiPolygon):
        raise ValueError(
            f"Unexpected geometry type: {repaired.geom_type}"
        )

    if repaired.is_empty or not repaired.is_valid:
        raise ValueError("Geometry remains invalid after repair.")

    return repaired, original_area, repaired.area


# --------------------------------------------------
# LOAD AND SELECT DISTRICTS
# --------------------------------------------------

print("=" * 60)
print("IHR BOUNDARY BUILD PROCESS")
print("=" * 60)

gdf = gpd.read_file(SOURCE)

gdf["_state"] = gdf["state"].astype(str).str.strip().str.upper()
gdf["_district"] = (
    gdf["district"].astype(str).str.strip().str.casefold()
)

full_mask = gdf["_state"].isin(FULL_COVERAGE_CODES)

partial_mask = pd.Series(False, index=gdf.index)

for state, districts in PARTIAL_DISTRICTS.items():
    partial_mask |= (
        (gdf["_state"] == state)
        & gdf["_district"].isin(districts)
    )

selected = gdf.loc[full_mask | partial_mask].copy()

selected = selected.drop(
    columns=["_state", "_district"]
)

print("\nSelected districts:", len(selected))

if len(selected) != EXPECTED_TOTAL:
    raise ValueError(
        f"Expected {EXPECTED_TOTAL} districts, got {len(selected)}"
    )

if selected.duplicated(
    subset=["state", "district"]
).any():
    raise ValueError("Duplicate district records found.")

# --------------------------------------------------
# REPAIR GEOMETRIES
# --------------------------------------------------

print("\nRepairing geometries...")

repair_log = []

repaired_geometries = []

for index, row in selected.iterrows():

    original = row.geometry

    repaired, old_area, new_area = repair_geometry(original)

    repaired_geometries.append(repaired)

    if not original.is_valid:
        repair_log.append({
            "district": row["district"],
            "state": row["state_name"],
            "old_area": old_area,
            "new_area": new_area,
            "area_change_percent": (
                (new_area - old_area) / old_area * 100
                if old_area else None
            ),
        })

selected = selected.set_geometry(
    gpd.GeoSeries(
        repaired_geometries,
        index=selected.index,
        crs=gdf.crs
    )
)

print("Repaired geometries:", len(repair_log))

for item in repair_log:
    print(
        f"{item['state']} - {item['district']}: "
        f"area change = {item['area_change_percent']:.6f}%"
    )

# --------------------------------------------------
# FINAL VALIDATION
# --------------------------------------------------

invalid_count = (~selected.geometry.is_valid).sum()
empty_count = selected.geometry.is_empty.sum()

print("\nFINAL VALIDATION")
print("Districts:", len(selected))
print("Invalid geometries:", invalid_count)
print("Empty geometries:", empty_count)
print("CRS:", selected.crs)

if invalid_count != 0 or empty_count != 0:
    raise ValueError("Geometry validation failed.")

# --------------------------------------------------
# DISSOLVE DISTRICTS
# --------------------------------------------------

print("\nDissolving district boundaries...")

boundary_geometry = selected.geometry.union_all()

if not boundary_geometry.is_valid:
    boundary_geometry = shapely.make_valid(boundary_geometry)

polygons = extract_polygons(boundary_geometry)
boundary_geometry = unary_union(polygons)

if not boundary_geometry.is_valid:
    raise ValueError("Dissolved boundary is invalid.")

boundary = gpd.GeoDataFrame(
    {
        "name": ["Indian Himalayan Region"],
        "district_count": [len(selected)],
        "definition": ["NMHS administrative approximation"],
    },
    geometry=[boundary_geometry],
    crs=selected.crs
)

# --------------------------------------------------
# EXPORT
# --------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

district_path = OUTPUT_DIR / "ihr_districts.gpkg"
boundary_path = OUTPUT_DIR / "ihr_boundary.gpkg"

# Replace only these generated output files
for path in [district_path, boundary_path]:
    if path.exists():
        path.unlink()

selected.to_file(
    district_path,
    layer="ihr_districts",
    driver="GPKG"
)

boundary.to_file(
    boundary_path,
    layer="ihr_boundary",
    driver="GPKG"
)

print("\nEXPORT SUCCESSFUL")
print("District file:", district_path)
print("Boundary file:", boundary_path)

print("\nFinal boundary geometry:", boundary_geometry.geom_type)
print("Boundary valid:", boundary_geometry.is_valid)

print("\nProcess completed successfully.")
