
import importlib.util
import rasterio
import geopandas as gpd
import scipy
import numpy as np

print("=" * 60)
print("TERRAIN PROCESSING ENVIRONMENT")
print("=" * 60)

packages = [
    "rasterio",
    "geopandas",
    "scipy",
    "numpy",
    "osgeo",
    "richdem",
    "whitebox",
    "pyproj"
]

for package in packages:
    available = importlib.util.find_spec(package) is not None
    print(f"{package:15} : {'AVAILABLE' if available else 'NOT INSTALLED'}")

print("\nRasterio version:", rasterio.__version__)
print("GeoPandas version:", gpd.__version__)
print("SciPy version:", scipy.__version__)
print("NumPy version:", np.__version__)

print("\nEnvironment inspection complete.")
