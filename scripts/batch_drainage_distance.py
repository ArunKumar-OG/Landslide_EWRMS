"""Capped distance to MERIT Hydro UPA >= 5 km² stream cell centres.
Run from project/scripts beside test_drainage_distance.py and test_hydrology_features.py.
Independent buffered output windows bound memory; original UPA is never resampled.
"""
import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
import traceback

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from pyproj import Transformer
from rasterio.features import geometry_mask
from rasterio.windows import Window, bounds
from rasterio.warp import transform_bounds
import test_drainage_distance as pilot

ROOT = Path(__file__).resolve().parent.parent
REFS = ROOT / 'data/processed/terrain_batch/terrain_v2'
RAW = ROOT / 'data/raw/hydrology/merit_hydro/upa'
BOUNDARY = ROOT / 'data/processed/boundary/ihr_districts.gpkg'
OUT = ROOT / 'data/processed/hydrology/drainage_batch_v1'
NAME = 'distance_upa_5km2.tif'
THRESHOLD = 5.
CAP, BUFFER, RES, NODATA = 10000., 12000., 1/1200, -9999.
VERSION = 'drainage_5km2_cap10000_buffer12000_v1'
BLOCK = 1024
TILES = {'n20e090','n25e075','n25e080','n25e085','n25e090','n25e095','n30e070','n30e075','n30e080','n35e070','n35e075','n35e080'}


def norm(s): return re.sub(r'[^a-z0-9]', '', str(s).lower())

def windows(w,h,block=BLOCK):
    for r in range(0,h,block):
        for c in range(0,w,block):
            yield Window(c,r,min(block,w-c),min(block,h-r))

def offset(x):
    i=round(x)
    if abs(x-i)>1e-5: raise ValueError(f'Nonaligned native grid: {x}')
    return i

def inventory(stack):
    paths=sorted(RAW.rglob('*.tif'))
    if len(paths)!=12 or {p.stem.removesuffix('_upa') for p in paths}!=TILES:
        raise RuntimeError('UPA inventory mismatch')
    sources=[]
    anchor=None
    for p in paths:
        s=stack.enter_context(rasterio.open(p)); t=s.transform
        if s.crs is None or s.crs.to_epsg()!=4326 or s.shape!=(6000,6000) or s.count!=1 or t.b!=0 or t.d!=0 or not np.allclose((t.a,t.e),(RES,-RES),rtol=0,atol=1e-12):
            raise RuntimeError(f'Unexpected UPA grid: {p}')
        if anchor is None: anchor=t
        c=offset((t.c-anchor.c)/RES); r=offset((anchor.f-t.f)/RES)
        for old,oc,orr in sources:
            if c<oc+old.width and c+s.width>oc and r<orr+old.height and r+s.height>orr:
                raise RuntimeError('Overlapping native tile footprints')
        sources.append((s,c,r))
    return sources,anchor

def native_window(b,crs,anchor):
    l,bot,r,t=b
    west,south,east,north=transform_bounds(crs,'EPSG:4326',l-BUFFER,bot-BUFFER,r+BUFFER,t+BUFFER,densify_pts=101)
    c0=math.floor((west-anchor.c)/RES)-2; c1=math.ceil((east-anchor.c)/RES)+2
    r0=math.floor((anchor.f-north)/RES)-2; r1=math.ceil((anchor.f-south)/RES)+2
    return c0,r0,c1,r1

def read_native(sources,box):
    c0,r0,c1,r1=box
    if (c1-c0)*(r1-r0)>3_000_000: raise RuntimeError('Unexpectedly large native window')
    a=np.full((r1-r0,c1-c0),np.nan,dtype=np.float64)
    for s,c,r in sources:
        l,t=max(c,c0),max(r,r0); rr,b=min(c+s.width,c1),min(r+s.height,r1)
        if rr<=l or b<=t: continue
        data=s.read(1,window=Window(l-c,t-r,rr-l,b-t),masked=True).astype(np.float64).filled(np.nan)
        data[(data==NODATA)|~np.isfinite(data)|(data<0)]=np.nan
        a[t-r0:b-r0,l-c0:rr-c0]=data
    return a

def calculate(inside, tr, crs, sources, anchor, project):
    out=np.full(inside.shape,NODATA,dtype=np.float32)
    if not inside.any(): return out,False
    h,w=inside.shape
    b=rasterio.transform.array_bounds(h,w,tr)
    box=native_window(b,crs,anchor)
    upa=read_native(sources,box)
    native_tr=anchor*Affine.translation(box[0],box[1])
    valid=np.isfinite(upa)
    stream=pilot.tree_for(valid&(upa>=THRESHOLD),native_tr,project)
    unknown=pilot.tree_for(~valid,native_tr,project)
    rr,cc=np.nonzero(inside)
    # Bound query memory separately from output block dimensions.
    for start in range(0,len(rr),131072):
        r=rr[start:start+131072]; c=cc[start:start+131072]
        x=tr.c+(c+.5)*tr.a; y=tr.f+(r+.5)*tr.e
        d=pilot.capped_distance(stream,unknown,np.column_stack((x,y)))
        ok=np.isfinite(d)
        out[r[ok],c[ok]]=d[ok].astype('float32')
    return out,not valid.all()

def check_ref(ref):
    if ref.crs is None or not ref.crs.is_projected or ref.crs.linear_units.lower() not in ('metre','meter','m') or ref.res!=(30.,30.) or ref.transform.b!=0 or ref.transform.d!=0 or ref.transform.a<=0 or ref.transform.e>=0:
        raise RuntimeError('Unexpected terrain reference grid')

def compare(a_path,b_path):
    n=changed=mask_diff=0; maximum=err=0.
    with rasterio.open(a_path) as a,rasterio.open(b_path) as b:
        if a.shape!=b.shape or a.crs!=b.crs or a.transform!=b.transform or a.nodata!=NODATA or b.nodata!=NODATA or a.dtypes[0]!='float32' or b.dtypes[0]!='float32': raise RuntimeError('Pilot grid or metadata mismatch')
        for win in windows(a.width,a.height):
            x=a.read(1,window=win); y=b.read(1,window=win)
            if not np.isfinite(x).all() or not np.isfinite(y).all(): raise RuntimeError('Nonfinite stored pixel')
            vx=x!=NODATA; vy=y!=NODATA; mask_diff+=int((vx!=vy).sum()); both=vx&vy
            d=np.abs(x[both].astype('float64')-y[both]); n+=d.size; changed+=int((d!=0).sum()); err+=float(d.sum())
            if d.size: maximum=max(maximum,float(d.max()))
    ok=n>0 and mask_diff==0 and maximum<=0.002
    print(f'  Pilot comparison: {"PASS" if ok else "REVIEW"} | common={n:,} | mask differences={mask_diff} | changed={changed} | MAE={err/max(n,1):.9g} m | max={maximum:.9g} m',flush=True)
    return ok

def scan(path,ref,geom):
    total=valid=capped=joint=0; minval=float('inf'); maxval=-float('inf'); sumval=0.
    with rasterio.open(path) as s:
        if s.shape!=ref.shape or s.transform!=ref.transform or s.crs!=ref.crs or s.nodata!=NODATA or s.dtypes[0]!='float32': raise RuntimeError('Output grid mismatch')
        for win in windows(ref.width,ref.height):
            a=s.read(1,window=win); inside=geometry_mask([geom],out_shape=a.shape,transform=ref.window_transform(win),invert=True)
            if not np.isfinite(a).all(): raise RuntimeError('Nonfinite output pixel')
            ok=a!=NODATA; v=a[ok]
            if (ok&~inside).any() or (v<0).any() or (v>CAP).any(): raise RuntimeError('Invalid distance or outside-boundary pixel')
            elev=ref.read(1,window=win,masked=True); ev=~np.ma.getmaskarray(elev)&np.isfinite(elev.data)&(elev.data!=NODATA)
            total+=int(inside.sum()); valid+=len(v); capped+=int((v>=CAP).sum()); joint+=int((ok&ev).sum())
            if v.size: minval=min(minval,float(v.min())); maxval=max(maxval,float(v.max())); sumval+=float(v.sum(dtype=np.float64))
    if not total: raise RuntimeError('Empty district raster mask')
    return dict(district_pixels=total,valid_pixels=valid,missing_pixels=total-valid,coverage_percent=100*valid/total,capped_pixels=capped,capped_percent=100*capped/valid if valid else None,joint_elevation_coverage=100*joint/total,min_m=minval if valid else None,mean_m=sumval/valid if valid else None,max_m=maxval if valid else None)

def process(refpath,geometry,geom_crs,sources,anchor):
    state,district=refpath.parent.parent.name,refpath.parent.name
    dest=OUT/state/district; marker=dest/'complete.json'
    if marker.exists():
        record=json.loads(marker.read_text(encoding='utf-8'))
        if record.get('version')!=VERSION or record.get('reference')!=str(refpath) or not (dest/NAME).is_file(): raise RuntimeError('Stale/incomplete output marker')
        print('  Already processed; skipping',flush=True); return record
    if dest.exists(): raise RuntimeError(f'Unmarked output exists: {dest}')
    with rasterio.open(refpath) as ref:
        check_ref(ref)
        geom=gpd.GeoSeries([geometry],crs=geom_crs).to_crs(ref.crs).iloc[0]
        if not geom.is_valid: geom=geom.buffer(0)
        if geom.is_empty: raise RuntimeError('Empty projected geometry')
        if shutil.disk_usage(OUT).free<ref.width*ref.height*4+2*1024**3: raise RuntimeError('Insufficient conservative storage estimate')
        project=Transformer.from_crs('EPSG:4326',ref.crs,always_xy=True)
        with tempfile.TemporaryDirectory(prefix=f'{state}_{district}_',dir=OUT/'_work') as td:
            work=Path(td)/'result'; work.mkdir(); path=work/NAME
            profile=ref.profile.copy(); profile.update(dtype='float32',count=1,nodata=NODATA,compress='deflate',predictor=3,tiled=True,blockxsize=256,blockysize=256,BIGTIFF='IF_SAFER')
            total=math.ceil(ref.width/BLOCK)*math.ceil(ref.height/BLOCK); missing_blocks=0
            with rasterio.open(path,'w',**profile) as dst:
                dst.set_band_description(1,'Distance to modeled drainage cell centre (m)')
                dst.update_tags(source='MERIT Hydro v1.0.1 UPA',threshold_km2=str(THRESHOLD),cap_m=str(CAP),buffer_m=str(BUFFER),units='m',native_resolution='3 arc-seconds',method='UTM cell-centre KDTree, buffered windows')
                for i,win in enumerate(windows(ref.width,ref.height),1):
                    tr=ref.window_transform(win)
                    inside=geometry_mask([geom],out_shape=(int(win.height),int(win.width)),transform=tr,invert=True)
                    a,missing=calculate(inside,tr,ref.crs,sources,anchor,project)
                    missing_blocks+=int(missing); dst.write(a,1,window=win)
                    if i%50==0 or i==total: print(f'  Distance blocks: {i}/{total}',flush=True)
            stats=scan(path,ref,geom)
            record=dict(version=VERSION,reference=str(refpath),state=state,district=district,status='PASS' if stats['coverage_percent']>=95 else 'REVIEW',windows_with_unknown_source=missing_blocks,**stats)
            (work/'complete.json').write_text(json.dumps(record,indent=2,allow_nan=False),encoding='utf-8')
            dest.parent.mkdir(parents=True,exist_ok=True); work.rename(dest)
    print(f"  {record['status']} | coverage {record['coverage_percent']:.4f}% | capped {record['capped_percent']}% | unknown-source windows {missing_blocks}",flush=True)
    return record

def report(failures):
    rows=[json.loads(p.read_text(encoding='utf-8')) for p in sorted(OUT.glob('*/*/complete.json')) if p.parts[-3]!='_work']
    temp=OUT/'drainage_batch_report.tmp'; pd.DataFrame(rows+failures).to_csv(temp,index=False); temp.replace(OUT/'drainage_batch_report.csv')
    print(f'  Report: {len(rows)} completed districts',flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__); sel=parser.add_mutually_exclusive_group(required=True); sel.add_argument('--district',nargs='+'); sel.add_argument('--all',action='store_true'); args=parser.parse_args()
    if (pilot.CAP,pilot.BUFFER,pilot.RES,pilot.NODATA,pilot.UNKNOWN_MARGIN)!=(CAP,BUFFER,RES,NODATA,150.): raise RuntimeError('Pilot constants changed')
    pilot.self_test()
    refs=sorted(REFS.glob('*/*/elevation.tif')); pairs={(norm(r.parent.parent.name),norm(r.parent.name)):r for r in refs}
    gdf=gpd.read_file(BOUNDARY); geoms={}
    for row in gdf.itertuples():
        key=(norm(row.state),norm(row.district))
        if key in geoms: raise RuntimeError('Duplicate boundary')
        geoms[key]=row.geometry
    if len(refs)!=137 or len(pairs)!=137 or set(pairs)!=set(geoms): raise RuntimeError('District inventory mismatch')
    selected=list(pairs) if args.all else [tuple(map(norm,s.replace('\\','/').split('/'))) for s in args.district]
    if not all(k in pairs for k in selected): parser.error('Unknown district; use STATE/District')
    OUT.mkdir(parents=True,exist_ok=True); (OUT/'_work').mkdir(exist_ok=True)
    baseline=ROOT/'data/processed/hydrology/drainage_pilot/UK/Haridwar'/NAME
    hkey=('uk','haridwar'); lkey=('lk','lehladakh')
    def dest_for(k):
        r=pairs[k]; return OUT/r.parent.parent.name/r.parent.name
    if args.all:
        for k in (hkey,lkey):
            m=dest_for(k)/'complete.json'
            if not m.is_file() or json.loads(m.read_text(encoding='utf-8')).get('status')!='PASS': raise RuntimeError('Run both single-district pilots first; inspect any REVIEW')
        if not compare(baseline,dest_for(hkey)/NAME): raise RuntimeError('Haridwar comparison requires review')
    failures=[]; review=0; comparison=True
    with rasterio.Env(GDAL_CACHEMAX=128*1024**2), ExitStack() as stack:
        sources,anchor=inventory(stack)
        for i,k in enumerate(selected,1):
            r=pairs[k]; print(f'\n[{i}/{len(selected)}] {r.parent.parent.name}/{r.parent.name}',flush=True)
            try:
                rec=process(r,geoms[k],gdf.crs,sources,anchor); review+=rec['status']=='REVIEW'
                if k==hkey and not args.all: comparison &= compare(baseline,dest_for(k)/NAME)
            except Exception as e:
                failures.append(dict(state=r.parent.parent.name,district=r.parent.name,status='ERROR',error=str(e)))
                print(f'  ERROR: {e}',flush=True)
                with (OUT/'drainage_errors.log').open('a',encoding='utf-8') as f: f.write(traceback.format_exc()+'\n')
            report(failures)
    print(f'Finished | Errors: {len(failures)} | REVIEW: {review} | Pilot comparison OK: {comparison}',flush=True)
    return int(bool(failures or review or not comparison))

if __name__=='__main__': raise SystemExit(main())
