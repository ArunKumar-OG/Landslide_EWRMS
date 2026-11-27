import rasterio
import pyogrio

print("Rasterio GDAL version:", rasterio.__gdal_version__)
print("Pyogrio GDAL version:", pyogrio.__gdal_version_string__)

drivers = pyogrio.list_drivers()

print("\nFileGDB:", drivers.get("FileGDB"))
print("OpenFileGDB:", drivers.get("OpenFileGDB"))

print("\nAvailable geodatabase drivers:")

for driver in drivers:
    if "GDB" in driver.upper():
        print(driver, drivers[driver])
