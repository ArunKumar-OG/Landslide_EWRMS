import sys

print("Python version:", sys.version)
print("\nChecking libraries...\n")

libraries = [
    "numpy",
    "pandas",
    "matplotlib",
    "rasterio",
    "geopandas",
    "pyproj",
    "shapely",
    "xarray",
    "netCDF4",
    "sklearn",
    "scipy",
    "jupyter"
]

for library in libraries:
    try:
        module = __import__(library)
        version = getattr(module, "__version__", "Installed")
        print(f"[OK] {library}: {version}")
    except ImportError as e:
        print(f"[ERROR] {library}: {e}")