"""Read-only final WorldCover production audit. Writes only new audit CSVs.
Keep batch_landcover_alignment.py, test_landcover_resampling.py and
 audit_landcover_pixels.py beside this script.
"""
import argparse
import json
import tempfile
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import batch_landcover_alignment as b
import test_landcover_resampling as p

DISTRICTS = 137
DELTA_LIMIT_PP = 0.5  # diagnostic, not a physical accuracy threshold


def indexed(df):
    keys = [b.key(s, d) for s, d in zip(df.state, df.district)]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate district identifiers')
    return dict(zip(keys, range(len(keys))))


def compare_record(expected, saved):
    for field, value in expected.items():
        if field not in saved:
            raise ValueError(f'Missing report field: {field}')
        if isinstance(value, (int, np.integer)):
            ok = float(saved[field]) == int(value)
        elif isinstance(value, (float, np.floating)):
            ok = np.isclose(float(saved[field]), value, atol=1e-7, rtol=0)
        else:
            ok = str(saved[field]) == str(value)
        if not ok:
            raise ValueError(f'Report mismatch: {field}')


def compare_classes(expected, saved):
    if len(saved) != len(expected) or saved.code.duplicated().any():
        raise ValueError('Saved class inventory mismatch')
    for a, z in zip(expected.sort_values('code').to_dict('records'),
                    saved.sort_values('code').to_dict('records')):
        compare_record(a, z)


def native_counts(group, cov):
    if len(group) != len(p.CLASSES) or set(group.code) != set(p.CLASSES):
        raise ValueError('Native class inventory mismatch')
    counts = {}
    for row in group.itertuples(index=False):
        if (not np.isfinite(row.pixels) or row.pixels < 0 or
                int(row.pixels) != row.pixels or
                row.landcover != p.NAMES[list(p.CLASSES).index(row.code)]):
            raise ValueError('Invalid native class record')
        counts[int(row.code)] = int(row.pixels)
    total = sum(counts.values())
    if (total <= 0 or cov['status'] != 'PASS' or
            cov['pixels'] != total or cov['valid_pixels'] != total or
            cov['nodata_pixels'] != 0 or cov['unexpected_pixels'] != 0 or
            cov['coverage_pct'] != 100):
        raise ValueError('Native coverage/count mismatch')
    return counts


def audit_one(folder, ref, geom, k, nc, report):
    marker = json.loads((folder/'complete.json').read_text(encoding='utf-8'))
    identity = dict(state=k[0], district=k[1], method=b.METHOD)
    compare_record(identity, marker)
    compare_record(identity, report)
    if marker.get('origin') not in ('validated_pilot', 'warp'):
        raise ValueError('Unexpected output origin')
    compare_record({'origin':marker['origin']}, report)
    with rasterio.open(folder/b.RASTER) as ds:
        if ds.tags().get('source') != 'ESA WorldCover 2021 v200':
            raise ValueError('Unexpected source metadata')
    result = b.scan(folder/b.RASTER, ref, geom, mask=False)
    stats, frame = b.metrics(result, nc)
    compare_record(stats, marker)
    compare_record(stats, report)
    compare_classes(frame, pd.read_csv(folder/'landcover_class_pixels.csv'))
    frame['difference_pp'] = frame.aligned_pct-frame.native_pct
    max_delta = float(frame.difference_pp.abs().max())
    status = 'PASS'
    if stats['status'] != 'PASS':
        status = 'REVIEW_COVERAGE'
    elif stats['distribution_delta_pp'] > DELTA_LIMIT_PP:
        status = 'REVIEW_DISTRIBUTION'
    return dict(**identity, origin=marker['origin'], **stats,
                max_class_difference_pp=max_delta, audit_status=status), frame


def self_test():
    from rasterio.transform import from_origin
    from shapely.geometry import box
    b.self_test()
    with tempfile.TemporaryDirectory() as td:
        folder = Path(td)
        profile = dict(driver='GTiff', width=3, height=2, count=1,
                       crs='EPSG:32643', transform=from_origin(0,60,30,30))
        refpath = folder/'ref.tif'
        with rasterio.open(refpath,'w',**profile,dtype='float32',nodata=-9999) as ds:
            ds.write(np.array([[1,-9999,1],[1,1,1]],dtype='float32'),1)
        with rasterio.open(folder/b.RASTER,'w',**profile,dtype='uint8',nodata=0) as ds:
            ds.write(np.array([[10,40,0],[10,40,0]],dtype='uint8'),1)
            ds.update_tags(source='ESA WorldCover 2021 v200',resampling='nearest')
        nc = {c:0 for c in p.CLASSES}; nc[10]=2; nc[40]=2
        k=('UK','Test'); geom=box(0,0,60,60)
        with rasterio.open(refpath) as ref:
            stats, frame = b.metrics(b.scan(folder/b.RASTER,ref,geom),nc)
            rec=dict(state=k[0],district=k[1],method=b.METHOD,origin='warp',**stats)
            (folder/'complete.json').write_text(json.dumps(rec),encoding='utf-8')
            csv=folder/'landcover_class_pixels.csv'; frame.to_csv(csv,index=False)
            out, _ = audit_one(folder,ref,geom,k,nc,rec)
            assert out['audit_status']=='PASS' and out['joint_elevation_pct']==75
            bad = dict(rec,valid_pixels=3)
            try: audit_one(folder,ref,geom,k,nc,bad)
            except ValueError: pass
            else: raise AssertionError('Bad report accepted')
            frame.loc[0,'aligned_pct']+=1; frame.to_csv(csv,index=False)
            try: audit_one(folder,ref,geom,k,nc,rec)
            except ValueError: pass
            else: raise AssertionError('Bad class percentage accepted')
            group=pd.DataFrame([dict(code=c,landcover=n,pixels=nc[c]) for c,n in zip(p.CLASSES,p.NAMES)])
            cov=dict(status='PASS',pixels=4,valid_pixels=4,nodata_pixels=0,unexpected_pixels=0,coverage_pct=100)
            assert native_counts(group,cov)==nc
            try: native_counts(group,dict(cov,pixels=5))
            except ValueError: pass
            else: raise AssertionError('Bad native count accepted')
    print('Synthetic final audit: readback, report, class percentage and native count checks: PASS')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    modes=ap.add_mutually_exclusive_group()
    modes.add_argument('--district',help='STATE/District, optional')
    modes.add_argument('--self-test',action='store_true')
    args=ap.parse_args(); self_test()
    if args.self_test: return 0
    districts=gpd.read_file(p.BOUNDARY)
    report=pd.read_csv(b.OUT/'landcover_batch_report.csv')
    cov=pd.read_csv(p.OUT/'landcover_district_pixel_coverage.csv')
    native=pd.read_csv(p.OUT/'landcover_district_class_counts.csv')
    di,ri,ci=map(indexed,(districts,report,cov))
    keys=set(di)
    refs={b.key(f.parent.parent.name,f.parent.name):f for f in p.REFS.glob('*/*/elevation.tif')}
    if len(keys)!=DISTRICTS or any(set(x)!=keys for x in (ri,ci,refs)):
        raise ValueError('District inventories differ')
    expected={b.OUT/s/d/b.RASTER for s,d in keys}
    actual=set(b.OUT.glob('*/*/landcover.tif'))
    markers=set(b.OUT.glob('*/*/complete.json'))
    tables=set(b.OUT.glob('*/*/landcover_class_pixels.csv'))
    print(f'Rasters: {len(actual)}/{DISTRICTS} | missing {len(expected-actual)} | unexpected {len(actual-expected)}')
    print(f'Completion markers: {len(markers)}/{DISTRICTS} | Class tables: {len(tables)}/{DISTRICTS}')
    if (actual!=expected or markers!={f.parent/'complete.json' for f in expected}
            or tables!={f.parent/'landcover_class_pixels.csv' for f in expected}):
        raise ValueError('Production file inventory mismatch')
    if len(pd.read_csv(b.OUT/'landcover_batch_errors.csv')):
        raise ValueError('Batch error report not empty')
    native['_key']=[b.key(s,d) for s,d in zip(native.state,native.district)]
    groups={k:g for k,g in native.groupby('_key')}
    if set(groups)!=keys: raise ValueError('Native class district inventory differs')
    selected=sorted(keys)
    if args.district:
        bits=args.district.replace('\\','/').split('/')
        if len(bits)!=2 or b.key(*bits) not in keys: raise ValueError('Unknown district')
        selected=[b.key(*bits)]
    rows=[]; classes=[]
    with rasterio.Env(GDAL_CACHEMAX=128*1024**2):
        for i,k in enumerate(selected,1):
            s,d=k; folder=b.OUT/s/d
            try:
                nc=native_counts(groups[k],cov.iloc[ci[k]].to_dict())
                with rasterio.open(refs[k]) as ref:
                    if not ref.crs.is_projected or not np.allclose(ref.res,(30,30),rtol=0,atol=1e-8) or ref.transform.b or ref.transform.d:
                        raise ValueError('Reference grid invalid')
                    geom=districts.iloc[[di[k]]].to_crs(ref.crs).geometry.iloc[0]
                    if geom.is_empty or not geom.is_valid: raise ValueError('Invalid boundary')
                    rec,frame=audit_one(folder,ref,geom,k,nc,report.iloc[ri[k]].to_dict())
                rows.append(dict(**rec,error=''))
                classes.append(frame.assign(state=s,district=d))
                print(f'[{i}/{len(selected)}] {s}/{d} | {rec["audit_status"]} | coverage {rec["coverage_pct"]:.6f}% | distribution delta {rec["distribution_delta_pp"]:.6f} pp',flush=True)
            except Exception as exc:
                rows.append(dict(state=s,district=d,audit_status='FAIL',error=str(exc)))
                print(f'[{i}/{len(selected)}] {s}/{d} | FAIL: {exc}',flush=True)
    final=pd.DataFrame(rows); suffix='_pilot' if args.district else ''
    final.to_csv(b.OUT/f'landcover_final_quality_districts{suffix}.csv',index=False)
    detail=pd.concat(classes,ignore_index=True) if classes else pd.DataFrame(columns=['state','district','code','landcover','native_pixels','native_pct','aligned_pixels','aligned_pct','difference_pp'])
    detail.to_csv(b.OUT/f'landcover_final_quality_classes{suffix}.csv',index=False)
    print('\nFINAL LAND COVER QUALITY AUDIT')
    print(f'Districts audited: {len(final)}/{len(selected)}')
    print(final.audit_status.value_counts().to_string())
    good=final.audit_status.eq('PASS')
    if good.any():
        print(final.loc[good,['coverage_pct','joint_elevation_pct','distribution_delta_pp','max_class_difference_pp']].agg(['min','mean','max']).T.to_string())
    if len(detail):
        totals=detail.groupby(['code','landcover']).aligned_pixels.sum()
        print('\nAligned class pixel counts (successfully read districts):')
        print(totals.to_string()); print('Total valid aligned class pixels:',int(totals.sum()))
    print('Production files unchanged. Audit reports:',b.OUT)
    return 0 if good.all() else 1

if __name__=='__main__': raise SystemExit(main())
