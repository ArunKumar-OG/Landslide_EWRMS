"""Read-only recursive inspection of GSI downloads. Original ZIPs stay untouched.
Requires inspect_multisource_landslides.py in the same scripts directory.
"""
from pathlib import Path
import argparse
import json
import shutil
import tempfile
import zipfile

import pandas as pd
import pyogrio
from inspect_multisource_landslides import inspect_vector, inspect_csv, VECTOR

ROOT = Path(__file__).resolve().parent.parent
MAX_DEPTH = 5
MAX_MEMBER = 2 * 1024**3
MAX_ARCHIVE_UNCOMPRESSED = 8 * 1024**3


def inspect(raw, out):
    archives = sorted(p for p in raw.rglob('*') if p.is_file() and p.suffix.lower() == '.zip')
    if not archives:
        raise RuntimeError(f'No GSI ZIPs found in {raw}')
    out.mkdir(parents=True, exist_ok=True)
    members, layers, fields, details = [], [], [], []
    inner_count = 0

    def add(outer, chain, result):
        base = dict(source='GSI', file=outer, member=chain, layer=result.get('layer', ''))
        details.append({**base, **result})
        layers.append({**base, **{k:v for k,v in result.items() if k not in ('fields','dtypes','samples','bounds','layer')},
                       'field_count':len(result.get('fields', [])), 'bounds':json.dumps(result.get('bounds'))})
        for i, (name, dtype) in enumerate(zip(result.get('fields', []), result.get('dtypes', []))):
            fields.append({**base, 'position':i+1, 'field':name, 'dtype':dtype,
                           'sample_values':json.dumps([r[i] for r in result.get('samples', [])], ensure_ascii=False)})
        print(f"  {chain} | {result.get('layer','')} | {result.get('status')} | records={result.get('records','?')} | geometry={result.get('geometry_type','?')} | CRS={result.get('crs','')}", flush=True)
        if result.get('fields'):
            print('    Fields:', ', '.join(result['fields']))
        if result.get('error'):
            print('    Error:', result['error'])

    def review(outer, chain, error):
        add(outer, chain, dict(status='REVIEW', error=str(error)))

    def walk(path, outer, prefix='', depth=0):
        nonlocal inner_count
        if depth > MAX_DEPTH:
            review(outer, prefix, 'Maximum archive nesting exceeded')
            return
        try:
            with zipfile.ZipFile(path) as z:
                entries = [x for x in z.infolist() if not x.is_dir()]
                names = [x.filename for x in entries]
                if len(set(names)) != len(names):
                    raise ValueError('Duplicate ZIP member names')
                if sum(x.file_size for x in entries) > MAX_ARCHIVE_UNCOMPRESSED:
                    raise ValueError('Archive exceeds uncompressed size safety limit')
                candidates = 0
                for entry in entries:
                    name = entry.filename
                    chain = f'{prefix}!/{name}' if prefix else name
                    ext = Path(name).suffix.lower()
                    candidate = ext in VECTOR | {'.csv','.zip','.7z','.rar','.gz'}
                    members.append(dict(source='GSI', archive=outer, member=chain, depth=depth,
                                        bytes=entry.file_size, compressed_bytes=entry.compress_size,
                                        candidate=candidate, crc32=f'{entry.CRC:08x}'))
                    if not candidate:
                        continue
                    candidates += 1
                    try:
                        if entry.file_size > MAX_MEMBER:
                            raise ValueError('Member exceeds size safety limit')
                        if ext == '.zip':
                            inner_count += 1
                            # Temporary filename is controlled, never derived from an archive path.
                            with tempfile.TemporaryDirectory(prefix='gsi_inspect_') as tmp:
                                nested = Path(tmp) / 'inner.zip'
                                with z.open(entry) as src, nested.open('wb') as dst:
                                    shutil.copyfileobj(src, dst, length=1024*1024)
                                walk(nested, outer, chain, depth+1)
                        elif ext == '.csv':
                            add(outer, chain, inspect_csv(path, name))
                        elif ext in VECTOR:
                            uri = f'/vsizip/{{{path.resolve().as_posix()}}}/{name}'
                            found = pyogrio.list_layers(uri)
                            if not len(found):
                                raise ValueError('No readable vector layers')
                            for layer, _ in found:
                                try:
                                    add(outer, chain, inspect_vector(uri, str(layer)))
                                except Exception as e:
                                    review(outer, chain, f'{layer}: {e}')
                        else:
                            review(outer, chain, 'Unsupported archive type')
                    except Exception as e:
                        review(outer, chain, e)
                if not candidates:
                    review(outer, prefix, 'No supported vector or CSV members')
        except Exception as e:
            review(outer, prefix, e)

    for i, path in enumerate(archives, 1):
        outer = path.relative_to(raw.parent).as_posix()
        print(f'\n[{i}/{len(archives)}] {outer}', flush=True)
        walk(path, outer)
    for name, rows in [('archive_members',members),('layers',layers),('fields',fields)]:
        pd.DataFrame(rows).to_csv(out / f'gsi_{name}.csv', index=False, encoding='utf-8-sig')
    (out/'gsi_schema_details.json').write_text(json.dumps(details, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    df = pd.DataFrame(layers)
    print('\nFINAL GSI NESTED ARCHIVE INSPECTION')
    print(f'Outer ZIPs: {len(archives)} | Inner ZIPs encountered: {inner_count} | Layer results: {len(df)}')
    print(df.status.value_counts().to_string())
    passed = df[df.status.eq('PASS')]
    if 'records' in passed and 'geometry_type' in passed:
        print('\nPASS record counts by declared geometry type:')
        print(passed.groupby('geometry_type').records.sum().to_string())
        print('Total readable PASS records:', int(passed.records.sum()))
    print('Reports:', out)
    print('Counts are not unique events. Original archives unchanged.')
    return df, inner_count


def self_test():
    import io
    import geopandas as gpd
    from shapely.geometry import Point, Polygon
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp); raw = base/'GSI'; raw.mkdir()
        point = gpd.GeoDataFrame({'id':[1]},geometry=[Point(78,30)],crs='EPSG:4326')
        poly = gpd.GeoDataFrame({'id':[2]},geometry=[Polygon([(78,30),(79,30),(79,31),(78,30)])],crs='EPSG:4326')
        shp = base/'polygon.shp'; poly.to_file(shp, driver='ESRI Shapefile')
        b1 = io.BytesIO(); b2 = io.BytesIO()
        with zipfile.ZipFile(b1,'w') as z:
            z.writestr('sub/point.geojson',point.to_json())
        with zipfile.ZipFile(b2,'w') as z:
            for p in base.glob('polygon.*'):
                z.write(p, 'sub/'+p.name)
        outer = raw/'outer.zip'
        with zipfile.ZipFile(outer,'w') as z:
            z.writestr('points.zip',b1.getvalue()); z.writestr('polygons.zip',b2.getvalue())
        before = outer.read_bytes()
        df, n = inspect(raw, base/'out')
        assert n == 2 and len(df) == 2 and df.status.eq('PASS').all()
        assert df.records.sum() == 2 and outer.read_bytes() == before
        assert set(df.geometry_type) == {'Point','Polygon'}
        with zipfile.ZipFile(raw/'broken.zip','w') as z:
            z.writestr('broken_inner.zip',b'not a zip')
        df, n = inspect(raw, base/'out_bad')
        assert n == 3 and df.status.eq('REVIEW').sum() == 1
    print('Synthetic nested GeoJSON, shapefile, corrupt inner ZIP and raw preservation tests: PASS')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        raw = ROOT/'data/raw/landslides'
        matches = [p for p in raw.iterdir() if p.is_dir() and p.name.casefold() == 'gsi']
        if len(matches) != 1:
            raise RuntimeError('Expected one GSI folder')
        df, n = inspect(matches[0], ROOT/'data/processed/landslides/inspection_gsi_v1')
        if n != 32:
            print(f'REVIEW: Expected 32 inner ZIPs from previous log, encountered {n}')
        raise SystemExit(0 if n == 32 and df.status.eq('PASS').all() else 2)
