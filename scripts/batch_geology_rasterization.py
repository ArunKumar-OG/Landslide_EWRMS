"""Batch wrapper for the validated test_geology_rasterization.py pilot.
Run from the project's scripts directory. Never writes source files.
Existing pilots are adopted only after readback checks and report agreement.
"""
import argparse
import json
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
import test_geology_rasterization as p

OUT = p.GEO / 'batch_v1'
PILOT = p.GEO / 'pilot'
AUDIT = 'lithology_pilot_audit.csv'
CLASSES = 'lithology_class_pixels.csv'
RASTER = 'lithology.tif'
REVIEW_KEY = ('LK', 'Leh_Ladakh')


def key(a, b):
    return p.norm(a), p.norm(b)


def readback(folder, refpath, geometry, vec):
    """Validate stored output against reference, district, counts and audit."""
    report = pd.read_csv(folder / AUDIT)
    classes = pd.read_csv(folder / CLASSES)
    if len(report) != 1 or classes.class_id.tolist() != list(range(1,19)):
        raise RuntimeError('Invalid audit/class table inventory')
    if classes.code.tolist() != [p.LABELS[i] for i in range(1,19)]:
        raise RuntimeError('Class lookup mismatch')
    rec = report.iloc[0].to_dict()
    if key(rec['state'],rec['district']) != key(refpath.parent.parent.name,refpath.parent.name):
        raise RuntimeError('Audit district mismatch')
    counts = np.zeros(19, dtype=np.int64)
    joint = 0
    with rasterio.open(refpath) as ref, rasterio.open(folder/RASTER) as src:
        if src.count != 1 or (src.shape,src.transform,src.crs,src.dtypes[0],src.nodata) != (ref.shape,ref.transform,ref.crs,'uint8',0):
            raise RuntimeError('Raster metadata mismatch')
        if ref.crs is None or not ref.crs.is_projected or not np.allclose(ref.res,(30,30)):
            raise RuntimeError('Unexpected reference grid')
        geom = p.polygonal(geometry.to_crs(ref.crs).iloc[0])
        for win in p.windows(ref.width,ref.height):
            a = src.read(1,window=win)
            inside = geometry_mask([geom],out_shape=a.shape,transform=ref.window_transform(win),invert=True,all_touched=False)
            if (a>18).any() or not np.array_equal(a!=0,inside):
                raise RuntimeError('Raster class or mask mismatch')
            counts += np.bincount(a.ravel(),minlength=19)
            elev = ref.read(1,window=win,masked=True)
            valid = ~np.ma.getmaskarray(elev)&np.isfinite(elev.data)&(elev.data!=-9999)
            joint += int(((a>=1)&(a<=15)&valid).sum())
    n = int(counts[1:].sum())
    if not n or not np.array_equal(counts[1:],classes.pixels.to_numpy()):
        raise RuntimeError('Class pixel counts disagree')
    pct = lambda c: 100*int(c)/n
    known = pct(counts[1:16].sum())
    delta = known-float(vec.known_coverage_pct)
    expected = dict(district_pixels=n,known_pixels=int(counts[1:16].sum()),known_coverage_pct=known,
                    source_nd_pct=pct(counts[16]),unmapped_pct=pct(counts[17]),conflict_pct=pct(counts[18]),
                    joint_elevation_pct=pct(joint),vector_known_pct=float(vec.known_coverage_pct),known_difference_pp=delta)
    for field, value in expected.items():
        if not np.isclose(float(rec[field]),value,rtol=0,atol=1e-7):
            raise RuntimeError(f'Audit mismatch: {field}')
    if not np.allclose(classes.percentage.to_numpy(),100*counts[1:]/n,rtol=0,atol=1e-7):
        raise RuntimeError('Class percentages disagree')
    status = 'PASS' if known>=95 and expected['unmapped_pct']<=.1 and expected['conflict_pct']<=.1 and abs(delta)<=.5 else 'REVIEW'
    if rec['status'] != status:
        raise RuntimeError('Status mismatch')
    # Preserve REVIEW, but distinguish the documented source gap from other issues.
    expected_review = (status=='REVIEW' and key(rec['state'],rec['district'])==key(*REVIEW_KEY)
                       and vec.known_coverage_pct<95 and abs(delta)<=.5
                       and expected['unmapped_pct']<=.1 and expected['conflict_pct']<=.1
                       and abs(expected['source_nd_pct']-float(vec.nd_only_pct))<=.5)
    rec.update(expected)
    rec['disposition'] = 'PASS' if status=='PASS' else ('EXPECTED_SOURCE_ND' if expected_review else 'UNEXPECTED_REVIEW')
    return rec


def save_report():
    records=[]
    for marker in sorted(OUT.glob('*/*/complete.json')):
        records.append(json.loads(marker.read_text(encoding='utf-8')))
    report=pd.DataFrame(records)
    if len(report):
        tmp=OUT/'geology_batch_report.tmp.csv'
        report.to_csv(tmp,index=False); tmp.replace(OUT/'geology_batch_report.csv')
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    mode=ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--all',action='store_true'); mode.add_argument('--district'); mode.add_argument('--self-test',action='store_true')
    args=ap.parse_args()
    p.smoke_tests()
    if args.self_test: return 0
    d=gpd.read_file(p.BOUNDARY,engine='pyogrio')
    geo=gpd.read_file(p.GEO/'glim_regional_repaired.gpkg',layer='geology',engine='pyogrio')
    vec=pd.read_csv(p.GEO/'glim_district_coverage_audit.csv')
    lookup=pd.read_csv(p.GEO/'glim_class_lookup.csv')
    mapping={i+1:c for i,c in enumerate(p.CODES)}
    if dict(zip(lookup.class_id,lookup.code))!=mapping or not (geo.class_id.map(mapping)==geo.xx_normalized).all():
        raise RuntimeError('Class mapping mismatch')
    if len(geo)!=10815 or geo.source_fid.isna().any() or geo.source_fid.duplicated().any() or not geo.geometry.is_valid.all() or geo.geometry.is_empty.any():
        raise RuntimeError('Unexpected repaired geology inventory')
    if len(d)!=137 or len(vec)!=137: raise RuntimeError('Unexpected district count')
    dkeys=[key(a,b) for a,b in zip(d.state,d.district)]
    vkeys=[key(a,b) for a,b in zip(vec.state_code,vec.district)]
    refs=sorted(p.REFS.glob('*/*/elevation.tif'))
    rkeys=[key(r.parent.parent.name,r.parent.name) for r in refs]
    if any(len(set(x))!=137 for x in (dkeys,vkeys,rkeys)) or len(refs)!=137 or set(dkeys)!=set(vkeys) or set(dkeys)!=set(rkeys):
        raise RuntimeError('Boundary, vector audit and reference inventories disagree')
    di=dict(zip(dkeys,range(len(d)))); vi=dict(zip(vkeys,range(len(vec))))
    if not args.all:
        bits=args.district.replace('\\','/').split('/')
        if len(bits)!=2: raise RuntimeError('Use --district STATE/District')
        selected=[r for r,k in zip(refs,rkeys) if k==key(*bits)]
        if len(selected)!=1: raise RuntimeError('District not found')
    else: selected=refs
    OUT.mkdir(parents=True,exist_ok=True)
    p.OUT=OUT
    errors=[]; unexpected=[]
    with rasterio.Env(GDAL_CACHEMAX=128*1024**2):
        for i,ref in enumerate(selected,1):
            state,name=ref.parent.parent.name,ref.parent.name
            k=key(state,name); match=d.iloc[[di[k]]]; v=vec.iloc[vi[k]]
            dest=OUT/state/name; marker=dest/'complete.json'
            print(f'\n[{i}/{len(selected)}] {state}/{name}',flush=True)
            try:
                # Resume only if existing output can be read and agrees with audit.
                if marker.exists():
                    marker.unlink()
                    rec=readback(dest,ref,match.geometry,v)
                    print('Existing output revalidated; skipping generation')
                    origin='existing'
                elif k in (key('UK','Haridwar'),key(*REVIEW_KEY)) and all((PILOT/state/name/f).is_file() for f in (RASTER,AUDIT,CLASSES)):
                    pilot=PILOT/state/name
                    rec=readback(pilot,ref,match.geometry,v)
                    dest.mkdir(parents=True,exist_ok=True)
                    # Copy bytes, not resample. Recheck the actual production copy.
                    with tempfile.TemporaryDirectory(dir=dest) as td:
                        tmp=Path(td)
                        for f in (RASTER,AUDIT,CLASSES): shutil.copy2(pilot/f,tmp/f)
                        rec=readback(tmp,ref,match.geometry,v)
                        for f in (RASTER,AUDIT,CLASSES): (tmp/f).replace(dest/f)
                    print('Validated pilot adopted without rasterization')
                    origin='pilot_copy'
                else:
                    p.run(ref,match.geometry,geo,p.polygonal(match.to_crs(geo.crs).geometry.iloc[0]),v)
                    rec=readback(dest,ref,match.geometry,v)
                    origin='batch'
                rec['origin']=origin
                tmp=dest/'complete.tmp.json'
                tmp.write_text(json.dumps(rec,indent=2,default=lambda v:v.item() if hasattr(v,'item') else str(v)),encoding='utf-8')
                tmp.replace(marker)
                if rec['disposition']=='UNEXPECTED_REVIEW': unexpected.append(f'{state}/{name}')
                print(f"{rec['status']} | {rec['disposition']} | known {rec['known_coverage_pct']:.4f}% | nd {rec['source_nd_pct']:.4f}% | joint elevation {rec['joint_elevation_pct']:.4f}%",flush=True)
            except Exception as exc:
                errors.append(dict(state=state,district=name,error=str(exc)))
                print(f'ERROR: {exc}',flush=True)
            report=save_report()
            print(f'Report: {len(report)} completed districts',flush=True)
    pd.DataFrame(errors,columns=['state','district','error']).to_csv(OUT/'geology_batch_errors.csv',index=False)
    print('\nFINAL GEOLOGY BATCH')
    print(f'Errors: {len(errors)} | Unexpected reviews in this run: {len(unexpected)}')
    if len(report):
        print(report.status.value_counts().to_string())
        print(report.disposition.value_counts().to_string())
        cols=['known_coverage_pct','source_nd_pct','unmapped_pct','conflict_pct','joint_elevation_pct','known_difference_pp']
        print(report[cols].agg(['min','mean','max']).T.to_string())
    print(f'Completed: {len(report)}/137 | Output: {OUT}')
    return 1 if errors or unexpected or (args.all and (len(report)!=137 or (report.disposition=='UNEXPECTED_REVIEW').any())) else 0

if __name__=='__main__': raise SystemExit(main())
