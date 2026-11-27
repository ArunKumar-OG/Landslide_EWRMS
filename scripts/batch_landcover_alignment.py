"""Production WorldCover 2021 nearest-neighbor alignment.
Requires test_landcover_resampling.py and audit_landcover_pixels.py beside this file.
Sources and pilots are read-only. Resume uses readback and saved class counts.
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
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.warp import transform_bounds
from rasterio.transform import from_origin
from shapely.geometry import box
from shapely.ops import unary_union
import test_landcover_resampling as p

OUT = p.OUT / 'batch_v1'
RASTER = 'landcover.tif'
METHOD = 'nearest'


def key(state, district):
    return str(state).upper(), str(district).replace(' ', '_')


def scan(path, ref, geom, mask=False):
    counts = np.zeros(256, dtype=np.int64)
    n = joint = 0
    with rasterio.open(path, 'r+' if mask else 'r') as ds:
        if (ds.shape, ds.crs, ds.transform, ds.count, ds.dtypes[0], ds.nodata) != (ref.shape, ref.crs, ref.transform, 1, 'uint8', 0):
            raise ValueError('Raster metadata mismatch')
        if ds.tags().get('resampling') != METHOD:
            raise ValueError('Unexpected resampling metadata')
        for i, (_, win) in enumerate(ds.block_windows(1), 1):
            a = ds.read(1, window=win)
            inside = geometry_mask([geom], out_shape=a.shape, transform=ds.window_transform(win), invert=True, all_touched=False)
            if mask:
                a[~inside] = 0
                ds.write(a, 1, window=win)
            elif np.any(a[~inside] != 0):
                raise ValueError('Nonzero outside district')
            if not np.array_equal(ds.read_masks(1, window=win) != 0, a != 0) and not mask:
                raise ValueError('Raster validity mask mismatch')
            h = np.bincount(a[inside], minlength=256)
            if h.sum() != h[list(p.ALLOWED)].sum():
                raise ValueError('Unexpected class values')
            counts += h
            n += int(inside.sum())
            elev = ref.read(1, window=win, masked=True)
            ev = ~np.ma.getmaskarray(elev) & np.isfinite(elev.data)
            joint += int(np.count_nonzero(inside & ev & (a != 0)))
            if i % 1000 == 0:
                print(f'  Audited blocks: {i}', flush=True)
    if not n or counts.sum() != n:
        raise ValueError('District pixel accounting failed')
    return n, counts, joint


def metrics(result, native):
    n,c,j = result
    total = sum(native.values())
    rows = [dict(code=code, landcover=label, native_pixels=int(native[code]),
                 native_pct=100*native[code]/total, aligned_pixels=int(c[code]), aligned_pct=100*c[code]/n)
            for code,label in zip(p.CLASSES,p.NAMES)]
    frame = pd.DataFrame(rows)
    delta = float((frame.aligned_pct-frame.native_pct).abs().sum()/2+50*c[0]/n)
    stats = dict(district_pixels=int(n), valid_pixels=int(n-c[0]), nodata_pixels=int(c[0]),
                 coverage_pct=100*int(n-c[0])/n, joint_elevation_pct=100*j/n,
                 distribution_delta_pp=delta, status='PASS' if c[0]==0 else 'REVIEW')
    return stats, frame


def check_pilot(folder, result, stats):
    r = pd.read_csv(folder/'landcover_pilot_audit.csv')
    f = pd.read_csv(folder/'landcover_class_comparison.csv')
    if len(r)!=1 or r.status.iloc[0]!='PASS' or f.code.tolist()!=list(p.CLASSES):
        raise ValueError('Invalid pilot report')
    for a,b in [('district_pixels','district_pixels'),('nearest_coverage_pct','coverage_pct'),('nearest_joint_elevation_pct','joint_elevation_pct'),('nearest_distribution_delta_pp','distribution_delta_pp')]:
        if not np.isclose(float(r[a].iloc[0]),stats[b],rtol=0,atol=1e-7):
            raise ValueError(f'Pilot statistic mismatch: {a}')
    if not np.array_equal(f.nearest_pixels.to_numpy(),result[1][list(p.CLASSES)]):
        raise ValueError('Pilot class counts mismatch')


def check_saved(folder, rec, stats, frame):
    for field,value in stats.items():
        if isinstance(value,str):
            ok = rec.get(field)==value
        else:
            ok = np.isclose(float(rec[field]),value,rtol=0,atol=1e-7)
        if not ok: raise ValueError(f'Saved statistic mismatch: {field}')
    old=pd.read_csv(folder/'landcover_class_pixels.csv')
    for field in ('code','native_pixels','aligned_pixels'):
        if not np.array_equal(old[field].to_numpy(),frame[field].to_numpy()):
            raise ValueError(f'Saved class table mismatch: {field}')


def write_report(keys):
    records=[]
    for state,name in keys:
        marker=OUT/state/name/'complete.json'
        if marker.is_file():
            r=json.loads(marker.read_text(encoding='utf-8'))
            if key(r['state'],r['district'])!=(state,name): raise ValueError('Marker identity mismatch')
            records.append(r)
    df=pd.DataFrame(records)
    tmp=OUT/'landcover_batch_report.tmp.csv'
    df.to_csv(tmp,index=False); tmp.replace(OUT/'landcover_batch_report.csv')
    return df


def self_test():
    with tempfile.TemporaryDirectory() as td:
        td=Path(td); rp=td/'ref.tif'; out=td/'landcover.tif'
        profile=dict(driver='GTiff',width=3,height=2,count=1,crs='EPSG:32643',transform=from_origin(0,60,30,30))
        with rasterio.open(rp,'w',**profile,dtype='float32',nodata=-9999) as ds:
            ds.write(np.array([[1,-9999,1],[1,1,1]],dtype='float32'),1)
        with rasterio.open(out,'w',**profile,dtype='uint8',nodata=0) as ds:
            ds.write(np.array([[10,40,10],[10,40,10]],dtype='uint8'),1); ds.update_tags(resampling='nearest')
        geom=box(0,0,60,60)
        native={c:0 for c in p.CLASSES}; native[10]=2; native[40]=2
        with rasterio.open(rp) as ref:
            a=scan(out,ref,geom,True); b=scan(out,ref,geom)
            assert a[0]==b[0]==4 and np.array_equal(a[1],b[1]) and a[2]==b[2]==3
            stats,frame=metrics(b,native)
            assert stats['coverage_pct']==100 and stats['joint_elevation_pct']==75 and stats['distribution_delta_pp']==0
            frame.to_csv(td/'landcover_class_pixels.csv',index=False); check_saved(td,stats,stats,frame)
            with rasterio.open(out,'r+') as ds:
                arr=ds.read(1); arr[0,0]=123; ds.write(arr,1)
            try: scan(out,ref,geom)
            except ValueError: pass
            else: raise AssertionError('Illegal class accepted')
    print('Synthetic single-raster masking, class accounting, readback and invalid-code tests: PASS',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    m=ap.add_mutually_exclusive_group(required=True)
    m.add_argument('--all',action='store_true'); m.add_argument('--district'); m.add_argument('--self-test',action='store_true')
    args=ap.parse_args(); p.self_test(); self_test()
    if args.self_test: return 0
    OUT.mkdir(parents=True,exist_ok=True)
    districts=gpd.read_file(p.BOUNDARY)
    keys=[key(a,b) for a,b in zip(districts.state,districts.district)]
    if len(keys)!=137 or len(set(keys))!=137: raise ValueError('District inventory mismatch')
    lookup=dict(zip(keys,range(len(keys))))
    selected=sorted(keys) if args.all else [key(*args.district.split('/'))]
    if any(k not in lookup for k in selected): raise ValueError('Unknown district')
    inv=pd.read_csv(p.OUT/'landcover_tile_inventory.csv'); aud=pd.read_csv(p.OUT/'landcover_pixel_audit.csv')
    rel=inv[inv.relevant.astype(str).str.lower().eq('true')]
    if len(rel)!=20 or rel.file.duplicated().any() or aud.file.duplicated().any() or set(rel.file)!=set(aud.file) or not rel.status.eq('PASS').all() or not aud.status.eq('PASS').all():
        raise ValueError('Source audit mismatch')
    source=[]
    for file in rel.file:
        path=p.RAW/file
        with rasterio.open(path) as src: _,fp=p.validate_tile(src,path.name)
        source.append((path,fp))
    native=pd.read_csv(p.OUT/'landcover_district_class_counts.csv'); cov=pd.read_csv(p.OUT/'landcover_district_pixel_coverage.csv')
    native['key']=[key(a,b) for a,b in zip(native.state,native.district)]; cov['key']=[key(a,b) for a,b in zip(cov.state,cov.district)]
    if set(native.key)!=set(keys) or set(cov.key)!=set(keys) or len(cov)!=137 or not cov.status.eq('PASS').all(): raise ValueError('Native district audit mismatch')
    errors=[]; reviews=0
    for i,k in enumerate(selected,1):
        state,name=k; print(f'\n[{i}/{len(selected)}] {state}/{name}',flush=True)
        try:
            sel=districts.iloc[[lookup[k]]]
            nf=native[native.key.map(lambda x:x==k)]; cf=cov[cov.key.map(lambda x:x==k)]
            if len(nf)!=len(p.CLASSES) or set(nf.code)!=set(p.CLASSES) or len(cf)!=1: raise ValueError('Native class inventory mismatch')
            nc=nf.set_index('code').pixels.to_dict()
            if sum(nc.values())<=0 or any(v<0 for v in nc.values()) or sum(nc.values())!=int(cf.pixels.iloc[0]): raise ValueError('Native count mismatch')
            refpath=p.REFS/state/name/'elevation.tif'; dest=OUT/state/name; dest.mkdir(parents=True,exist_ok=True)
            marker=dest/'complete.json'; pilot=p.OUT/'pilot'/state/name
            with rasterio.open(refpath) as ref:
                if not ref.crs.is_projected or not np.allclose(ref.res,(30,30)) or ref.transform.b or ref.transform.d: raise ValueError('Invalid reference grid')
                geom=sel.to_crs(ref.crs).geometry.iloc[0]
                if not geom.is_valid or geom.is_empty: raise ValueError('Invalid district geometry')
                if marker.exists():
                    rec=json.loads(marker.read_text(encoding='utf-8'))
                    if key(rec['state'],rec['district'])!=k or rec['method']!=METHOD: raise ValueError('Marker identity/method mismatch')
                    result=scan(dest/RASTER,ref,geom); stats,frame=metrics(result,nc); check_saved(dest,rec,stats,frame)
                    print('Existing output validated; skipping warp',flush=True)
                else:
                    with tempfile.TemporaryDirectory(dir=dest) as td:
                        td=Path(td); path=td/RASTER
                        if (pilot/'landcover_nearest.tif').is_file():
                            shutil.copy2(pilot/'landcover_nearest.tif',path)
                            result=scan(path,ref,geom); stats,frame=metrics(result,nc); check_pilot(pilot,result,stats)
                            origin='validated_pilot'; print('Pilot reused and validated',flush=True)
                        else:
                            bbox=box(*transform_bounds(ref.crs,'EPSG:4326',*ref.bounds,densify_pts=41)).buffer(.01)
                            paths=[path for path,fp in source if fp.intersects(bbox)]
                            footprints=[fp for path,fp in source if fp.intersects(bbox)]
                            geo=sel.to_crs('EPSG:4326').geometry.iloc[0]
                            if geo.difference(unary_union(footprints)).area>geo.area*1e-10: raise ValueError('Selected tile footprint gap')
                            vrt=td/'mosaic.vrt'; p.make_mosaic(paths,vrt)
                            p.warp(vrt,path,ref,Resampling.nearest)
                            first=scan(path,ref,geom,True); result=scan(path,ref,geom)
                            if first[0]!=result[0] or not np.array_equal(first[1],result[1]) or first[2]!=result[2]: raise ValueError('Readback mismatch')
                            stats,frame=metrics(result,nc); origin='warp'
                        rec=dict(state=state,district=name,method=METHOD,origin=origin,**stats)
                        frame.to_csv(td/'landcover_class_pixels.csv',index=False)
                        path.replace(dest/RASTER); (td/'landcover_class_pixels.csv').replace(dest/'landcover_class_pixels.csv')
                        (td/'complete.json').write_text(json.dumps(rec,indent=2),encoding='utf-8'); (td/'complete.json').replace(marker)
                reviews+=stats['status']!='PASS'
                print(f"  {stats['status']} | coverage {stats['coverage_pct']:.6f}% | joint elevation {stats['joint_elevation_pct']:.6f}% | distribution delta {stats['distribution_delta_pp']:.6f} pp",flush=True)
        except Exception as e:
            errors.append(dict(state=state,district=name,error=str(e)))
            print(f'  ERROR: {e}',flush=True)
        df=write_report(sorted(keys))
        print(f'Report: {len(df)} completed districts',flush=True)
    pd.DataFrame(errors,columns=['state','district','error']).to_csv(OUT/'landcover_batch_errors.csv',index=False)
    print('\nFINAL LAND COVER BATCH')
    print(f'Errors: {len(errors)} | REVIEW in this run: {reviews}')
    if len(df):
        print(df.status.value_counts().to_string()); print(df[['coverage_pct','joint_elevation_pct','distribution_delta_pp']].agg(['min','mean','max']).T.to_string())
    print(f'Completed: {len(df)}/137 | Output: {OUT}')
    return 2 if errors or reviews or (args.all and len(df)!=137) else 0

if __name__=='__main__': raise SystemExit(main())
