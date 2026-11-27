"""Read-only audit of persisted terrain rasters; writes CSV diagnostics only."""
from pathlib import Path
from contextlib import ExitStack
import argparse
import json
import re
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window
from rasterio.transform import Affine
from shapely.geometry import mapping

ROOT = Path(__file__).resolve().parent.parent
FEATURES = ('elevation', 'slope', 'aspect', 'plan_curvature',
            'profile_curvature', 'tangential_curvature', 'tri', 'tpi')
CURVATURE = ('plan_curvature', 'profile_curvature', 'tangential_curvature')
NODATA = -9999.0
BUFFER = 10
VERSION = 'source-aligned-v2-batch-1'


def norm(s):
    return re.sub(r'[^a-z0-9]', '', str(s).lower())


def grid_check(src, ref, buffered=False):
    b = BUFFER if buffered else 0
    return (src.crs == ref.crs and src.shape == (ref.height+2*b, ref.width+2*b)
            and np.allclose(tuple(src.transform), tuple(ref.transform * Affine.translation(-b,-b)), rtol=0, atol=1e-7)
            and src.count == 1 and src.dtypes[0] == 'float32' and src.nodata == NODATA)


def valid(a, mask):
    return (mask != 0) & np.isfinite(a) & (a != NODATA)


def audit(refpath, output, geom, crs):
    state, district = refpath.parent.name, refpath.stem
    base = output/'terrain_v2'/state/district
    marker = json.loads((base/'complete.json').read_text(encoding='utf-8'))
    if marker['pipeline'] != VERSION or (marker['state'], marker['district']) != (state,district):
        raise ValueError('Completion marker identity/version mismatch')
    files = {n:base/f'{n}.tif' for n in FEATURES}
    bufferpath = output/'district_dem_buffered_v2'/state/f'{district}.tif'
    stats = {n:dict(valid=0, outside=0, invalid_numeric=0, mask_mismatch=0,
                    range_errors=0, minimum=np.inf, maximum=-np.inf) for n in FEATURES}
    pixels = flat = elevation_mismatch = tangential_mismatch = 0
    with ExitStack() as stack:
        ref = stack.enter_context(rasterio.open(refpath))
        buf = stack.enter_context(rasterio.open(bufferpath))
        if (not ref.crs or not ref.crs.is_projected or not np.allclose(ref.res,(30,30))
                or ref.transform.b != 0 or ref.transform.d != 0
                or ref.transform.a <= 0 or ref.transform.e >= 0 or not grid_check(buf, ref, True)):
            raise ValueError('Reference or buffered grid metadata mismatch')
        srcs = {n:stack.enter_context(rasterio.open(p)) for n,p in files.items()}
        for n,src in srcs.items():
            if not grid_check(src,ref) or src.tags().get('pipeline') != VERSION:
                raise ValueError(f'{n}: metadata mismatch')
        geom = gpd.GeoSeries([geom], crs=crs).to_crs(ref.crs).iloc[0]
        if not geom.is_valid: geom = geom.buffer(0)
        if geom.is_empty: raise ValueError('Empty district geometry')
        geo = mapping(geom)
        blocks = 0
        for row in range(0,ref.height,256):
            for col in range(0,ref.width,256):
                w,h = min(256,ref.width-col),min(256,ref.height-row)
                win = Window(col,row,w,h)
                inside = geometry_mask([geo],out_shape=(h,w),transform=ref.window_transform(win),invert=True,all_touched=False)
                pixels += int(inside.sum())
                vals, masks = {}, {}
                for n,src in srcs.items():
                    a = src.read(1,window=win)
                    m = src.read_masks(1,window=win)
                    good = valid(a,m)
                    vals[n], masks[n] = a,good
                    s = stats[n]
                    s['outside'] += int(np.count_nonzero(~inside & (a != NODATA)))
                    s['invalid_numeric'] += int(np.count_nonzero(~np.isfinite(a) | ((m != 0) != (a != NODATA))))
                    v = a[inside & good]
                    s['valid'] += v.size
                    if v.size:
                        s['minimum'] = min(s['minimum'],float(v.min()))
                        s['maximum'] = max(s['maximum'],float(v.max()))
                        if n=='slope': s['range_errors'] += int(np.count_nonzero((v<0)|(v>90)))
                        if n=='aspect': s['range_errors'] += int(np.count_nonzero((v<0)|(v>=360)))
                        if n=='tri': s['range_errors'] += int(np.count_nonzero(v<0))
                if inside.any():
                    bw = Window(col+BUFFER-1,row+BUFFER-1,w+2,h+2)
                    b = buf.read(1,window=bw)
                    bv = valid(b,buf.read_masks(1,window=bw))
                    expected_elev = inside & bv[1:-1,1:-1]
                    expected_local = inside.copy()
                    for i in range(3):
                        for j in range(3): expected_local &= bv[i:i+h,j:j+w]
                    for n in ('elevation','slope','tri','tpi'):
                        expected = expected_elev if n=='elevation' else expected_local
                        stats[n]['mask_mismatch'] += int(np.count_nonzero(masks[n] != expected))
                    common = expected_elev & masks['elevation']
                    elevation_mismatch += int(np.count_nonzero(vals['elevation'][common] != b[1:-1,1:-1][common]))
                    slope = vals['slope']
                    flat += int(np.count_nonzero(masks['slope'] & (slope==0)))
                    stats['aspect']['mask_mismatch'] += int(np.count_nonzero(masks['aspect'] != (masks['slope'] & (slope>0))))
                    # Float32 slope rounding can affect classification at the exact threshold.
                    # Only flag differences clearly away from the threshold.
                    threshold = np.degrees(np.arctan(0.0001))
                    away = np.abs(slope.astype(np.float64)-threshold)>1e-8
                    expected_curv = masks['slope'] & (slope>=threshold)
                    for n in CURVATURE:
                        stats[n]['mask_mismatch'] += int(np.count_nonzero((masks[n] != expected_curv) & away))
                    common = masks['plan_curvature'] & masks['tangential_curvature'] & masks['slope']
                    predicted = vals['plan_curvature'][common].astype(np.float64)*np.sin(np.radians(slope[common].astype(np.float64)))
                    tangential_mismatch += int(np.count_nonzero(~np.isclose(vals['tangential_curvature'][common],predicted,rtol=1e-5,atol=1e-7)))
                blocks += 1
                if blocks%1000==0: print(f'  Audited {blocks} blocks',flush=True)
    if not pixels: raise ValueError('No district pixels')
    rows=[]
    for n,s in stats.items():
        old=marker['features'][n]
        count=s['valid']
        s['minimum']=s['minimum'] if count else None
        s['maximum']=s['maximum'] if count else None
        marker_match=(pixels==marker['district_pixels'] and flat==marker['flat_pixels']
                      and count==old['valid'] and pixels-count==old['missing']
                      and np.isclose(100*count/pixels,old['coverage'],rtol=0,atol=1e-8)
                      and (not count or (s['minimum']==old['min'] and s['maximum']==old['max'])))
        extra=(elevation_mismatch if n=='elevation' else tangential_mismatch if n=='tangential_curvature' else 0)
        issues=sum(s[k] for k in ('outside','invalid_numeric','mask_mismatch','range_errors'))+extra
        status='FAIL' if issues or not marker_match or not count else ('REVIEW' if n in ('elevation','slope','tri','tpi') and count/pixels<0.95 else 'PASS')
        rows.append(dict(state=state,district=district,feature=n,status=status,district_pixels=pixels,
                         missing=pixels-count,coverage=100*count/pixels,marker_match=marker_match,
                         consistency_errors=extra,**s))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root',type=Path,default=ROOT/'data/processed/terrain_batch')
    p.add_argument('--district',help='Optional pilot e.g. UK/Haridwar')
    args=p.parse_args()
    output=args.output_root.resolve()
    refs=sorted((ROOT/'data/processed/DEM/district_dem_utm').rglob('*.tif'))
    if len(refs)!=137: raise RuntimeError(f'Expected 137 reference rasters, found {len(refs)}')
    keys={(r.parent.name,r.stem) for r in refs}
    if len(keys)!=137: raise RuntimeError('Duplicate reference keys')
    expected={output/'terrain_v2'/s/d/f'{n}.tif' for s,d in keys for n in FEATURES}
    actual=set((output/'terrain_v2').rglob('*.tif'))
    expected_buf={output/'district_dem_buffered_v2'/s/f'{d}.tif' for s,d in keys}
    actual_buf=set((output/'district_dem_buffered_v2').rglob('*.tif'))
    print(f'Terrain rasters: {len(actual)} / {len(expected)}; missing {len(expected-actual)}, unexpected {len(actual-expected)}')
    print(f'Buffered DEMs: {len(actual_buf)} / {len(expected_buf)}; missing {len(expected_buf-actual_buf)}, unexpected {len(actual_buf-expected_buf)}')
    report=pd.read_csv(output/'terrain_batch_report.csv')
    report_keys=set(zip(report['state'],report['district']))
    report_ok=(len(report)==137 and report_keys==keys and report['status'].isin(['PASS','REVIEW']).all())
    print(f'Batch report inventory: {"PASS" if report_ok else "FAIL"}')
    if args.district:
        refs=[r for r in refs if f'{r.parent.name}/{r.stem}'.lower()==args.district.lower()]
        if not refs: p.error('Unknown district')
    gdf=gpd.read_file(ROOT/'data/processed/boundary/ihr_districts.gpkg').to_crs('EPSG:4326')
    boundaries={}
    for _,r in gdf.iterrows():
        key=(norm(r['state']),norm(r['district']))
        if key in boundaries: raise RuntimeError(f'Duplicate boundary key {key}')
        boundaries[key]=r.geometry
    rows=[]
    failures=[]
    with rasterio.Env(GDAL_CACHEMAX=128*1024**2):
        for i,r in enumerate(refs,1):
            s,d=r.parent.name,r.stem
            print(f'[{i}/{len(refs)}] {s}/{d}',flush=True)
            try:
                result=audit(r,output,boundaries[(norm(s),norm(d))],gdf.crs)
                rows.extend(result)
                print('  '+', '.join(f'{n}: {sum(x["status"]==n for x in result)}' for n in ('PASS','REVIEW','FAIL')),flush=True)
            except Exception as e:
                failures.append(dict(state=s,district=d,feature='ALL',status='ERROR',error=str(e)))
                print(f'  ERROR: {e}',flush=True)
    df=pd.DataFrame(rows+failures)
    dest=output/('terrain_quality_audit_pilot.csv' if args.district else 'terrain_quality_audit.csv')
    df.to_csv(dest,index=False)
    print('\nFINAL TERRAIN QUALITY AUDIT')
    print(f'Districts selected: {len(refs)} | Features audited: {len(rows)}')
    print(df['status'].value_counts().to_string())
    if rows:
        print('\nCoverage by feature (district-level minimum and mean):')
        print(pd.DataFrame(rows).groupby('feature')['coverage'].agg(['min','mean']).to_string())
    inventory_ok=(expected==actual and expected_buf==actual_buf and report_ok)
    print(f'Inventory: {"PASS" if inventory_ok else "FAIL"}')
    print(f'Report: {dest}')
    if not inventory_ok or df['status'].isin(['FAIL','ERROR','REVIEW']).any():
        print('Review flagged rows and inventory before proceeding.')
    else: print('All selected checks passed; physical accuracy is not independently established.')


if __name__=='__main__':
    main()
