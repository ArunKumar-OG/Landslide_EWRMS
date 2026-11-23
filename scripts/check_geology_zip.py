from zipfile import ZipFile
from pathlib import Path

zip_path = Path("data/raw/geology/LiMW_GIS_2015.gdb.zip")

with ZipFile(zip_path, "r") as archive:
    names = archive.namelist()

    print("Total entries:", len(names))
    print("\nFirst 30 entries:")

    for name in names[:30]:
        print(name)

    print("\nPossible geodatabase roots:")
    roots = sorted({
        name.split("/")[0]
        for name in names
        if "/" in name
    })

    for root in roots:
        print(root)