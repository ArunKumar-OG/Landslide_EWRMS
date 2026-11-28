"""Read-only inventory of raw landslide files and ZIP members.
Run from project scripts/; outputs reports to data/processed/landslides/inspection_v1.
"""
from pathlib import Path
import argparse
import csv
import io
import json
import tempfile
import zipfile
from contextlib import contextmanager

import pandas as pd
import pyogrio

ROOT = Path(__file__).resolve().parent.parent
VECTOR = {'.geojson', '.json', '.shp', '.gpkg', '.kml', '.gml', '.fgb'}
ARCHIVE = {'.zip', '.7z', '.rar', '.gz'}
csv.field_size_limit(128 * 1024 * 1024)


def short(value):
    if value is None:
        return ''
    s = str(value).replace('\n', ' ').replace('\r', ' ')
    return s if len(s) <= 180 else s[:180] + '...'


def source_name(rel):
    return 'GLC' if rel.lower() == 'glc_global.csv' else rel.split('/')[0].upper()


@contextmanager
def open_csv(path, member=None):
    if member is None:
        with path.open('r', encoding='utf-8-sig', newline='') as f:
            yield f
    else:
        with zipfile.ZipFile(path) as z, z.open(member) as raw:
            with io.TextIOWrapper(raw, encoding='utf-8-sig', newline='') as f:
                yield f


def inspect_csv(path, member=None):
    # Infer delimiter from header only, without splitting WKT or quoted values.
    with open_csv(path, member) as f:
        header = f.readline()
        if not header.strip():
            raise ValueError('Empty CSV header')
        delimiter = max([',', '|', ';', '\t'], key=lambda d: len(next(csv.reader([header], delimiter=d))))
    with open_csv(path, member) as f:
        reader = csv.reader(f, delimiter=delimiter, strict=True)
        fields = next(reader)
        rows = 0
        malformed = 0
        samples = []
        blank = 0
        for row in reader:
            if not row or all(not v.strip() for v in row):
                blank += 1
                continue
            rows += 1
            if len(row) != len(fields):
                malformed += 1
            elif len(samples) < 3:
                samples.append([short(v) for v in row])
            if rows % 250000 == 0:
                print(f'    CSV records scanned: {rows:,}', flush=True)
    normalized = [v.strip().casefold() for v in fields]
    review = malformed or len(fields) < 2 or len(set(normalized)) != len(fields) or '' in normalized or rows == 0
    return dict(layer='', driver='CSV', geometry_type='not verified', crs='',
                records=rows, fields=fields, dtypes=['text (unparsed)'] * len(fields),
                samples=samples, delimiter=repr(delimiter), malformed_rows=malformed,
                blank_rows=blank, bounds=None, status='REVIEW' if review else 'PASS')


def inspect_vector(uri, layer):
    info = pyogrio.read_info(uri, layer=layer, force_feature_count=True)
    fields = [str(x) for x in info['fields']]
    types = [str(x) for x in info['dtypes']]
    sample = pyogrio.read_dataframe(uri, layer=layer, read_geometry=False, max_features=3)
    samples = [[short(x) for x in row] for row in sample[fields].itertuples(index=False, name=None)]
    geom = info.get('geometry_type')
    count = int(info.get('features', -1))
    crs = info.get('crs')
    bounds = info.get('total_bounds')
    return dict(layer=layer, driver=info.get('driver'), geometry_type=geom or 'none',
                crs=str(crs or ''), records=count, fields=fields, dtypes=types,
                samples=samples, delimiter='', malformed_rows=None, blank_rows=None,
                bounds=list(bounds) if bounds is not None else None,
                status='PASS' if count > 0 and geom and crs else 'REVIEW')


def inspect(raw, out):
    out.mkdir(parents=True, exist_ok=True)
    files, members, layers, fields = [], [], [], []
    detailed = []

    def add_result(rel, member, result):
        base = {'source': source_name(rel), 'file': rel, 'member': member, 'layer': result.get('layer', '')}
        details = {**base, **result}
        detailed.append(details)
        layers.append({**base, **{k: v for k, v in result.items() if k not in ('fields','dtypes','samples','bounds','layer')},
                       'field_count': len(result.get('fields', [])), 'bounds': json.dumps(result.get('bounds'))})
        for i, (name, dtype) in enumerate(zip(result.get('fields', []), result.get('dtypes', []))):
            fields.append({**base, 'position': i+1, 'field': name, 'dtype': dtype,
                           'sample_values': json.dumps([r[i] for r in result.get('samples', [])], ensure_ascii=False)})
        print(f"  {member or rel} | {result.get('layer','')} | {result.get('status')} | records={result.get('records','?')} | geometry={result.get('geometry_type','?')} | CRS={result.get('crs','')}")
        if result.get('fields'):
            print('    Fields:', ', '.join(result['fields']))
        if result.get('error'):
            print('    Error:', result['error'])

    def inspect_item(path, rel, member=''):
        suffix = Path(member or path.name).suffix.lower()
        try:
            if suffix == '.csv':
                add_result(rel, member, inspect_csv(path, member or None))
            elif suffix in VECTOR:
                uri = f'/vsizip/{{{path.resolve().as_posix()}}}/{member}' if member else str(path)
                for name, _ in pyogrio.list_layers(uri):
                    try:
                        add_result(rel, member, inspect_vector(uri, str(name)))
                    except Exception as e:
                        add_result(rel, member, dict(layer=str(name), status='REVIEW', error=str(e)))
            else:
                add_result(rel, member, dict(status='REVIEW', error='Nested or unsupported archive; not opened'))
        except Exception as e:
            add_result(rel, member, dict(status='REVIEW', error=str(e)))

    paths = sorted(p for p in raw.rglob('*') if p.is_file())
    if not paths:
        raise RuntimeError(f'No files found: {raw}')
    for i, path in enumerate(paths, 1):
        rel = path.relative_to(raw).as_posix()
        suffix = path.suffix.lower()
        files.append(dict(source=source_name(rel), file=rel, bytes=path.stat().st_size, extension=suffix))
        print(f'\n[{i}/{len(paths)}] {rel} | {path.stat().st_size / 1024**2:.2f} MiB', flush=True)
        if suffix == '.zip':
            try:
                with zipfile.ZipFile(path) as z:
                    entries = [x for x in z.infolist() if not x.is_dir()]
                    names = [x.filename for x in entries]
                    for entry in entries:
                        ext = Path(entry.filename).suffix.lower()
                        candidate = ext in VECTOR or ext == '.csv' or ext in ARCHIVE
                        members.append(dict(source=source_name(rel), archive=rel, member=entry.filename,
                                            bytes=entry.file_size, compressed_bytes=entry.compress_size,
                                            candidate=candidate, crc32=f'{entry.CRC:08x}'))
                        print(f'  Member: {entry.filename} ({entry.file_size:,} bytes)')
                    if len(set(names)) != len(names):
                        raise ValueError('Duplicate member names; archive requires review')
                    candidates = [n for n in names if Path(n).suffix.lower() in VECTOR | ARCHIVE | {'.csv'}]
                    if not candidates:
                        add_result(rel, '', dict(status='REVIEW', error='No supported vector or CSV members found'))
                    for name in candidates:
                        inspect_item(path, rel, name)
            except Exception as e:
                add_result(rel, '', dict(status='REVIEW', error=str(e)))
        elif suffix in VECTOR | ARCHIVE | {'.csv'}:
            inspect_item(path, rel)
        else:
            print('  Auxiliary file: listed only')
    for name, data in [('raw_files',files), ('archive_members',members), ('layers',layers), ('fields',fields)]:
        pd.DataFrame(data).to_csv(out / f'landslide_{name}.csv', index=False, encoding='utf-8-sig')
    (out / 'landslide_schema_details.json').write_text(json.dumps(detailed, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    df = pd.DataFrame(layers)
    print('\nFINAL MULTI-SOURCE INSPECTION')
    print('Raw files:', len(files), '| ZIP members:', len(members), '| inspected layers/tables:', len(layers))
    if not df.empty:
        print(df[['source','file','member','layer','records','geometry_type','crs','status']].to_string(index=False) if {'records','geometry_type','crs'}.issubset(df.columns) else df.to_string(index=False))
        print('\nStatus counts:')
        print(df['status'].value_counts().to_string())
    print('\nReports:', out)
    print('Record counts are not unique landslide counts. Raw files unchanged.')
    return df


def self_test():
    with tempfile.TemporaryDirectory() as t:
        base = Path(t)
        raw = base / 'raw'; raw.mkdir()
        (raw / 'glc_global.csv').write_text('event_id,latitude,longitude\n1,30,78\n', encoding='utf-8')
        (raw / 'UGLC_poly.csv').write_text('ID|WKT_GEOM\n1|"POLYGON ((0 0, 1 0, 1 1, 0 0))"\n', encoding='utf-8')
        geo = {'type':'FeatureCollection','features':[{'type':'Feature','properties':{'date':'2023-08-01'},'geometry':{'type':'Point','coordinates':[78,30]}}]}
        with zipfile.ZipFile(raw / 'test.zip','w') as z:
            z.writestr('nested/points.geojson',json.dumps(geo))
            z.writestr('nested/records.csv','a;b\n1;2\n')
        result = inspect(raw, base / 'out')
        assert len(result) == 4 and (result.status == 'PASS').all()
        assert result.records.sum() == 4
        assert inspect_csv(raw / 'UGLC_poly.csv')['delimiter'] == repr('|')
        (raw / 'bad.csv').write_text('a,b\n1,2,3\n', encoding='utf-8')
        assert inspect_csv(raw / 'bad.csv')['status'] == 'REVIEW'
    print('Synthetic CSV delimiter, malformed row and ZIP GeoJSON inspection tests: PASS')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        inspect(ROOT / 'data/raw/landslides', ROOT / 'data/processed/landslides/inspection_v1')
