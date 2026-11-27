from pathlib import Path, PurePosixPath
import csv
import math
import shutil
import tarfile
import os

import rasterio

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / 'data/raw/hydrology/merit_hydro'
ARCHIVES = BASE / 'archives'
REPORT = ROOT / 'data/processed/hydrology/merit_tile_inventory.csv'
LAYERS = ('upa', 'hnd', 'elv', 'dir')
TILES = ('n20e090', 'n25e075', 'n25e080', 'n25e085', 'n25e090',
         'n25e095', 'n30e070', 'n30e075', 'n30e080', 'n35e070',
         'n35e075', 'n35e080')
RES = 1 / 1200
TOL = 1e-8


def package_for(tile):
    lat, lon = int(tile[1:3]), int(tile[4:7])
    return f'n{lat // 30 * 30:02d}e{lon // 30 * 30:03d}'


def archive_plan():
    plan = {}
    for layer in LAYERS:
        for tile in TILES:
            key = f'{layer}_{package_for(tile)}'
            plan.setdefault(key, []).append((tile, layer))
    return plan


def locate_archive(key):
    options = [ARCHIVES / (key + ext) for ext in ('.tar', '.tar.gz', '.tgz')]
    found = [p for p in options if p.is_file()]
    if len(found) != 1:
        raise RuntimeError(f'Expected exactly one archive for {key}; found {found}')
    return found[0]


def extract_archive(path, targets):
    # Never use extractall: write regular files only, to known destination names.
    wanted = {f'{tile}_{layer}.tif': layer for tile, layer in targets}
    with tarfile.open(path, 'r:*') as tar:
        members = {}
        for member in tar:
            name = PurePosixPath(member.name.replace('\\', '/')).name.lower()
            if name in wanted:
                if not member.isfile():
                    raise RuntimeError(f'Not a regular file: {member.name}')
                if name in members:
                    raise RuntimeError(f'Duplicate archive member: {name}')
                members[name] = member
        missing = set(wanted) - set(members)
        if missing:
            raise RuntimeError(f'Missing members in {path.name}: {sorted(missing)}')
        for name, layer in wanted.items():
            member = members[name]
            output = BASE / layer / name
            output.parent.mkdir(parents=True, exist_ok=True)
            temp = output.with_suffix('.tif.part')
            print(f'  Extracting {name}', flush=True)
            try:
                with tar.extractfile(member) as source, temp.open('wb') as dest:
                    shutil.copyfileobj(source, dest, length=1024 * 1024)
                if temp.stat().st_size != member.size:
                    raise RuntimeError(f'Incomplete extraction: {name}')
                os.replace(temp, output)
            finally:
                temp.unlink(missing_ok=True)


def close(a, b):
    return math.isclose(a, b, rel_tol=0, abs_tol=TOL)


def inspect(tile, layer):
    path = BASE / layer / f'{tile}_{layer}.tif'
    row = dict(tile=tile, layer=layer, file=str(path.relative_to(ROOT)))
    issues = []
    grid = None
    try:
        with rasterio.open(path) as src:
            t = src.transform
            lat, lon = int(tile[1:3]), int(tile[4:7])
            expected = (lon - RES/2, lat - RES/2, lon + 5 - RES/2, lat + 5 - RES/2)
            row.update(crs=str(src.crs), width=src.width, height=src.height,
                       res_x=src.res[0], res_y=src.res[1], dtype=src.dtypes[0],
                       nodata=src.nodata, left=src.bounds.left, bottom=src.bounds.bottom,
                       right=src.bounds.right, top=src.bounds.top,
                       size_bytes=path.stat().st_size)
            if src.crs is None or src.crs.to_epsg() != 4326:
                issues.append('Unexpected CRS')
            if (src.width, src.height, src.count) != (6000, 6000, 1):
                issues.append('Unexpected dimensions or band count')
            if not all(close(a,b) for a,b in zip((t.a,t.b,t.d,t.e),(RES,0,0,-RES))):
                issues.append('Unexpected pixel grid')
            if not all(close(a,b) for a,b in zip(src.bounds, expected)):
                issues.append('Unexpected tile extent')
            allowed = ('int8','uint8') if layer == 'dir' else ('float32',)
            if src.dtypes[0] not in allowed:
                issues.append('Unexpected data type')
            # Check that every raster block can be decoded, without loading whole tile.
            for _, window in src.block_windows(1):
                src.read(1, window=window)
            grid = (src.crs, src.width, src.height, tuple(t))
    except Exception as exc:
        issues.append(str(exc))
    row.update(status='PASS' if not issues else 'REVIEW', issues='; '.join(issues))
    return row, grid


def main():
    print('=' * 65)
    print('MERIT HYDRO SELECTIVE EXTRACTION AND INVENTORY AUDIT')
    print('=' * 65)
    plan = archive_plan()
    archives = {key: locate_archive(key) for key in plan}
    print(f'Archives found: {len(archives)} / 12')
    for i, (key, targets) in enumerate(plan.items(), 1):
        print(f'\n[{i}/{len(plan)}] {archives[key].name} | Required tiles: {len(targets)}', flush=True)
        extract_archive(archives[key], targets)
    rows = []
    print('\nValidating extracted rasters...')
    for tile in TILES:
        reference = None
        for layer in LAYERS:
            row, grid = inspect(tile, layer)
            if grid is not None:
                if reference is None:
                    reference = grid
                elif grid[:3] != reference[:3] or not all(close(a,b) for a,b in zip(grid[3],reference[3])):
                    row['status'] = 'REVIEW'
                    row['issues'] += '; Cross-layer grid mismatch'
            rows.append(row)
            print(f"  {tile}_{layer}.tif: {row['status']} {row['issues']}", flush=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with REPORT.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    passed = sum(row['status'] == 'PASS' for row in rows)
    print('\n' + '=' * 65)
    print('FINAL SUMMARY')
    print(f'Archives: {len(archives)} / 12')
    for layer in LAYERS:
        subset = [r for r in rows if r['layer'] == layer]
        print(f"{layer}: {sum(r['status'] == 'PASS' for r in subset)} / 12 PASS")
    print(f'Tiles audited: {len(rows)} / 48')
    print(f'PASS: {passed} | REVIEW: {len(rows)-passed}')
    print(f'Extracted size: {sum(r.get("size_bytes",0) for r in rows)/1024**3:.2f} GiB')
    print(f'Report: {REPORT}')
    print('Original archives retained. No reprojection or resampling performed.')
    if passed != 48:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
