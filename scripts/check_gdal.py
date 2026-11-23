import rasterio
import fiona

print("Rasterio GDAL version:", rasterio.__gdal_version__)
print("Fiona GDAL version:", fiona.__gdal_version__)

drivers = fiona.supported_drivers

print("\nFileGDB:", drivers.get("FileGDB"))
print("OpenFileGDB:", drivers.get("OpenFileGDB"))

print("\nAvailable geodatabase drivers:")

for driver in drivers:
    if "GDB" in driver.upper():
        print(driver, drivers[driver])