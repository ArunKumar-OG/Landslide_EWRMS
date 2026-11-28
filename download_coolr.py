import requests
import json
from pathlib import Path

BASE_URL = (
    "https://gis.earthdata.nasa.gov/portal/rest/services/Landslides"
)

OUTPUT_DIR = Path("data/raw/landslides/COOLR")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LAYERS = [
    "COOLR_Reports_Points",
    "COOLR_Events_Points",
    "COOLR_Reports_Polygons",
    "COOLR_Events_Polygons"
]

session = requests.Session()

def query(url, params):
    response = session.get(url, params=params, timeout=180)
    response.raise_for_status()
    data = response.json()

    if "error" in data:
        raise RuntimeError(data["error"])

    return data

for layer in LAYERS:
    print(f"\nDownloading {layer}...")

    url = f"{BASE_URL}/{layer}/FeatureServer/0/query"

    try:
        result = query(url, {
            "where": "1=1",
            "geometry": "65,5,100,40",
            "geometryType": "esriGeometryEnvelope",
            "spatialRel": "esriSpatialRelIntersects",
            "inSR": 4326,
            "returnIdsOnly": "true",
            "f": "json"
        })

        ids = sorted(set(result["objectIds"] or []))
        print(f"Records found: {len(ids)}")

        features = []

        for i in range(0, len(ids), 100):
            batch = ids[i:i + 100]

            data = query(url, {
                "objectIds": ",".join(map(str, batch)),
                "outFields": "*",
                "returnGeometry": "true",
                "outSR": 4326,
                "f": "geojson"
            })

            records = data["features"]

            if len(records) != len(batch):
                raise RuntimeError("Incomplete batch received")

            features.extend(records)

            print(f"Downloaded {len(features)}/{len(ids)}")

        output = OUTPUT_DIR / f"{layer}.geojson"

        with output.open("w", encoding="utf-8") as f:
            json.dump({
                "type": "FeatureCollection",
                "features": features
            }, f)

        print(f"SUCCESS: {output}")

    except Exception as e:
        print(f"FAILED: {layer}: {e}")

print("\nDownload process completed.")