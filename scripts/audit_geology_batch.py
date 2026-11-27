"""Final production geology audit. Reads rasters and reports; writes new audit CSVs only."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import test_geology_rasterization as p
from batch_geology_rasterization import readback, key, OUT


def indexed(df, state_col, name_col='district'):
    keys = [key(a,b) for a,b in zip(df[state_col], df[name_col])]
    if len(set(keys)) != len(keys):
        raise RuntimeError('Duplicate district keys in inventory')
    return dict(zip(keys, range(len(df))))


def compare_record(actual, saved):
    for field, value in actual.items():
        if field not in saved:
            raise RuntimeError(f'Missing report field: {field}')
        if isinstance(value, (int,float,np.number)):
            if not np.isclose(float(saved[field]), float(value), atol=1e-7, rtol=0):
                raise RuntimeError(f'Report mismatch: {field}')
        elif str(saved[field]) != str(value):
            raise RuntimeError(f'Report mismatch: {field}')


def class_diagnostics(classes, vector_classes, district_area):
    if not np.isfinite(district_area) or district_area <= 0:
        raise RuntimeError('Invalid vector district area')
    if vector_classes.class_id.duplicated().any():
        raise RuntimeError('Duplicate vector class IDs')
    for row in vector_classes.itertuples(index=False):
        if row.class_id not in range(1,17) or row.code != p.LABELS[row.class_id] or not np.isfinite(row.area_km2) or row.area_km2 < 0:
            raise RuntimeError('Invalid vector class record')
    area = dict(zip(vector_classes.class_id, vector_classes.area_km2))
    rows = []
    for row in classes.itertuples(index=False):
        if row.class_id > 16:
            continue
        vp = 100 * area.get(row.class_id,0) / district_area
        rows.append(dict(class_id=row.class_id,code=row.code,pixels=int(row.pixels),
                         raster_pct=float(row.percentage),vector_pct=vp,
                         difference_pp=float(row.percentage)-vp))
    return rows


def smoke_test():
    compare_record({'n':10,'status':'PASS'},{'n':10.,'status':'PASS'})
    try:
        compare_record({'n':10},{'n':11})
    except RuntimeError:
        pass
    else:
        raise AssertionError('Incorrect report accepted')
    cls=pd.DataFrame([dict(class_id=i,code=p.LABELS[i],pixels=80 if i==1 else 20 if i==16 else 0,
                           percentage=80. if i==1 else 20. if i==16 else 0.) for i in range(1,19)])
    v=pd.DataFrame({'class_id':[1,16],'code':['su','nd'],'area_km2':[8.,2.]})
    assert all(r['difference_pp']==0 for r in class_diagnostics(cls,v,10.))
    v.loc[0,'area_km2']=7.9
    assert np.isclose(class_diagnostics(cls,v,10.)[0]['difference_pp'],1.)
    print('Synthetic report and class-area checks: PASS')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--district',help='STATE/District, optional pilot audit')
    ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args()
    smoke_test()
    if args.self_test: return 0
    d=gpd.read_file(p.BOUNDARY,engine='pyogrio')
    v=pd.read_csv(p.GEO/'glim_district_coverage_audit.csv')
    vc=pd.read_csv(p.GEO/'glim_district_class_area.csv')
    r=pd.read_csv(OUT/'geology_batch_report.csv')
    lookup=pd.read_csv(p.GEO/'glim_class_lookup.csv')
    if lookup.class_id.tolist()!=list(range(1,17)) or lookup.code.tolist()!=p.CODES:
        raise RuntimeError('Class lookup differs from validated encoding')
    di=indexed(d,'state'); vi=indexed(v,'state_code'); ri=indexed(r,'state')
    refs=sorted(p.REFS.glob('*/*/elevation.tif'))
    rk=[key(f.parent.parent.name,f.parent.name) for f in refs]
    if len(refs)!=137 or len(set(rk))!=137 or any(set(x)!=set(rk) for x in (di,vi,ri)):
        raise RuntimeError('District inventories do not match')
    expected={OUT/f.parent.parent.name/f.parent.name/'lithology.tif' for f in refs}
    actual=set(OUT.glob('*/*/lithology.tif'))
    markers=set(OUT.glob('*/*/complete.json'))
    expected_markers={f.parent/'complete.json' for f in expected}
    print(f'Rasters: {len(actual)}/137 | missing {len(expected-actual)} | unexpected {len(actual-expected)}')
    print(f'Completion markers: {len(markers)}/137')
    if actual!=expected or markers!=expected_markers:
        raise RuntimeError('Production file inventory mismatch')
    errors=pd.read_csv(OUT/'geology_batch_errors.csv')
    if len(errors): raise RuntimeError('Batch error report is not empty')
    vc=vc.assign(_key=[key(a,b) for a,b in zip(vc.state_code,vc.district)])
    if set(vc._key)!=set(rk): raise RuntimeError('Vector class inventory mismatch')
    vg={k:g for k,g in vc.groupby('_key')}
    if args.district:
        bits=args.district.replace('\\','/').split('/')
        if len(bits)!=2 or key(*bits) not in di: raise RuntimeError('District not found')
        refs=[f for f in refs if key(f.parent.parent.name,f.parent.name)==key(*bits)]
    rows=[]; detail=[]
    with rasterio.Env(GDAL_CACHEMAX=128*1024**2):
        for i,f in enumerate(refs,1):
            state,name=f.parent.parent.name,f.parent.name
            k=key(state,name); folder=OUT/state/name
            try:
                vector=v.iloc[vi[k]]
                rec=readback(folder,f,d.iloc[[di[k]]].geometry,vector)
                marker=json.loads((folder/'complete.json').read_text(encoding='utf-8'))
                compare_record(rec,marker)
                compare_record(rec,r.iloc[ri[k]].to_dict())
                cls=pd.read_csv(folder/'lithology_class_pixels.csv')
                cr=class_diagnostics(cls,vg[k],float(vector.district_area_km2))
                maxdiff=max(abs(x['difference_pp']) for x in cr)
                # A diagnostic, not evidence of corruption on its own: projected
                # pixel counts and equal-area vector measurements are different.
                audit='PASS' if rec['disposition']=='PASS' else ('EXPECTED_SOURCE_ND' if rec['disposition']=='EXPECTED_SOURCE_ND' else 'REVIEW')
                if maxdiff>.5: audit='REVIEW_CLASS_AREA'
                rows.append({**rec,'max_class_difference_pp':maxdiff,'audit_status':audit,'error':''})
                detail.extend(dict(state=state,district=name,**x) for x in cr)
                print(f'[{i}/{len(refs)}] {state}/{name} | {audit} | known {rec["known_coverage_pct"]:.4f}% | max class delta {maxdiff:.6f} pp',flush=True)
            except Exception as exc:
                rows.append(dict(state=state,district=name,audit_status='FAIL',error=str(exc)))
                print(f'[{i}/{len(refs)}] {state}/{name} | FAIL: {exc}',flush=True)
    report=pd.DataFrame(rows)
    suffix='_pilot' if args.district else ''
    report.to_csv(OUT/f'geology_final_quality_districts{suffix}.csv',index=False)
    pd.DataFrame(detail).to_csv(OUT/f'geology_final_quality_classes{suffix}.csv',index=False)
    print('\nFINAL GEOLOGY QUALITY AUDIT')
    print(f'Districts audited: {len(report)} / {len(refs)}')
    print(report.audit_status.value_counts().to_string())
    ok=report.audit_status.isin(['PASS','EXPECTED_SOURCE_ND'])
    if ok.any():
        cols=['known_coverage_pct','source_nd_pct','joint_elevation_pct','max_class_difference_pp']
        print(report.loc[ok,cols].agg(['min','mean','max']).T.to_string())
    if detail:
        c=pd.DataFrame(detail)
        print('\nPixel counts by source class (excluding gaps/conflicts):')
        print(c.groupby(['class_id','code']).pixels.sum().to_string())
    print('Production files unchanged. Audit reports:',OUT)
    return 0 if ok.all() else 1

if __name__=='__main__': raise SystemExit(main())
