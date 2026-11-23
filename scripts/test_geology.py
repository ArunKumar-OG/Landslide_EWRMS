import fiona
from pathlib import Path

gdb_path = Path(
    "data/raw/geology/LiMW_GIS 2015.gdb"
).resolve()

print("Geodatabase:", gdb_path)

try:
    layers = fiona.listlayers(
        str(gdb_path),
        driver="OpenFileGDB"
    )

    print("\nAvailable geological layers:")
    for layer in layers:
        print("-", layer)

except Exception as e:
    print("\nError:", repr(e))