
import rasterio
from rasterio.warp import transform_bounds, transform

path = (
    "data/processed/DEM/district_dem_utm/"
    "UK/Haridwar.tif"
)

with rasterio.open(path) as src:

    print("=" * 55)
    print("HARIDWAR DEM SPATIAL INSPECTION")
    print("=" * 55)

    print("CRS:", src.crs)
    print("EPSG:", src.crs.to_epsg())
    print("Dimensions:", src.width, "x", src.height)
    print("Resolution:", src.res)

    print("\nUTM BOUNDS:")
    print(src.bounds)

    # Convert raster bounds to longitude/latitude
    geographic_bounds = transform_bounds(
        src.crs,
        "EPSG:4326",
        *src.bounds
    )

    print("\nGEOGRAPHIC BOUNDS:")
    print(geographic_bounds)

    # Convert raster centre to longitude/latitude
    center_x = (src.bounds.left + src.bounds.right) / 2
    center_y = (src.bounds.bottom + src.bounds.top) / 2

    lon, lat = transform(
        src.crs,
        "EPSG:4326",
        [center_x],
        [center_y]
    )

    print("\nRASTER CENTRE:")
    print("Longitude:", lon[0])
    print("Latitude:", lat[0])

    print("\nEXPECTED UTM ZONE FROM CENTRE:")
    print(int((lon[0] + 180) // 6) + 1)

    print("=" * 55)
