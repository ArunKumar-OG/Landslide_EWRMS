
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path

# Input files
RAW_PATH = Path("data/raw/district_nwic.GeoJSON")
DISTRICT_PATH = Path("data/processed/boundary/ihr_districts.gpkg")
BOUNDARY_PATH = Path("data/processed/boundary/ihr_boundary.gpkg")

# Output
OUTPUT_DIR = Path("outputs/maps")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load datasets
print("Loading geographical datasets...")

india = gpd.read_file(RAW_PATH)
districts = gpd.read_file(DISTRICT_PATH)
boundary = gpd.read_file(BOUNDARY_PATH)

# Ensure all datasets use the same CRS
districts = districts.to_crs(india.crs)
boundary = boundary.to_crs(india.crs)

print("India districts:", len(india))
print("IHR districts:", len(districts))
print("Boundary features:", len(boundary))

# Create figure
fig, axes = plt.subplots(
    1, 2,
    figsize=(18, 11),
    gridspec_kw={"width_ratios": [1, 1.2]}
)

# -----------------------------------------
# MAP 1: INDIA OVERVIEW
# -----------------------------------------

ax = axes[0]

india.plot(
    ax=ax,
    color="whitesmoke",
    edgecolor="gray",
    linewidth=0.25
)

districts.plot(
    ax=ax,
    color="steelblue",
    edgecolor="black",
    linewidth=0.35
)

boundary.boundary.plot(
    ax=ax,
    color="red",
    linewidth=1.3
)

ax.set_title(
    "Indian Himalayan Region - India Overview",
    fontsize=13,
    fontweight="bold"
)

ax.set_axis_off()

# -----------------------------------------
# MAP 2: IHR CLOSE-UP
# -----------------------------------------

ax = axes[1]

districts.plot(
    ax=ax,
    color="lightblue",
    edgecolor="dimgray",
    linewidth=0.55
)

boundary.boundary.plot(
    ax=ax,
    color="red",
    linewidth=1.5
)

ax.set_title(
    "Selected IHR Districts (137)",
    fontsize=13,
    fontweight="bold"
)

ax.set_axis_off()

# -----------------------------------------
# FIGURE DETAILS
# -----------------------------------------

fig.suptitle(
    "Indian Himalayan Region - Administrative Study Boundary",
    fontsize=17,
    fontweight="bold"
)

fig.text(
    0.5, 0.025,
    "Source: Survey of India district administrative boundaries | "
    "Working definition: NMHS administrative approximation",
    ha="center",
    fontsize=9
)

plt.tight_layout(rect=[0, 0.05, 1, 0.94])

output_file = OUTPUT_DIR / "ihr_boundary_overview.png"

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nMap saved successfully:")
print(output_file)
