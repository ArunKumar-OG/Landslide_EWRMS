"""Block-wise MERIT Hydro batch. Keep test_hydrology_features.py beside this script."""
import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
import traceback

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from affine import Affine
from rasterio.features import geometry_mask
from rasterio.windows import Window
from rasterio.warp import transform_bounds, reproject, Resampling
import test_hydrology_features as pilot

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / 'data/raw/hydrology/merit_hydro'
REFS = ROOT / 'data/processed/terrain_batch/terrain_v2'
BOUNDARY = ROOT / 'data/processed/boundary/ihr_districts.gpkg'
OUT = ROOT / 'data/processed/hydrology/batch_v1'
FEATURES = tuple(pilot.UNITS)
LAYERS = ('upa', 'hnd', 'elv')
RES = pilot.RES
NODATA = pilot.NODATA
VERSION = 'merit_batch_v1_floor_0.1'
OPTIONS = dict(driver='GTiff', count=1, dtype='float32', nodata=NODATA,
               compress='deflate', predictor=3, tiled=True, blockxsize=256,
               blockysize=256, BIGTIFF='IF_SAFER')


def norm(s):
    return re.sub(r'[^a-z0-9]', '', str(s).lower())


def windows(width, height, block=512):
    for r in range(0, height, block):
        for c in range(0, width, block):
            yield Window(c, r, min(block,width-c), min(block,height-r))


def inventory():
    paths = {k: sorted((RAW/k).rglob('*.tif')) for k in LAYERS}
    if any(len(v)!=12 for v in paths.values()):
        raise RuntimeError('Expected 12 source tiles per used layer')
    expected = {p.stem.removesuffix('_elv') for p in paths['elv']}
    with rasterio.open(paths['elv'][0]) as a:
        anchor = a.transform
    grids = {}
    for k, files in paths.items():
        if {p.stem.removesuffix('_'+k) for p in files} != expected:
            raise RuntimeError(f'Tile inventory mismatch: {k}')
        for p in files:
            with rasterio.open(p) as s:
                t = s.transform
                if s.crs.to_epsg()!=4326 or s.shape!=(6000,6000) or not np.allclose(s.res,(RES,RES),rtol=0,atol=1e-12) or t.b!=0 or t.d!=0 or t.a<=0 or t.e>=0:
                    raise RuntimeError(f'Bad source grid: {p}')
                tile = p.stem.removesuffix('_'+k)
                key = (pilot.integer_offset((t.c-anchor.c)/RES),pilot.integer_offset((anchor.f-t.f)/RES))
                if tile in grids and grids[tile]!=key: raise RuntimeError(f'Layer grid mismatch: {p}')
                grids[tile]=key
    if len(set(grids.values()))!=len(grids): raise RuntimeError('Duplicate source grid positions')
    return paths, anchor


def native_grid(ref, anchor):
    west,south,east,north = transform_bounds(ref.crs,'EPSG:4326',*ref.bounds,densify_pts=21)
    ox,oy = anchor.c,anchor.f
    c0=math.floor((west-ox)/RES)-2; c1=math.ceil((east-ox)/RES)+2
    r0=math.floor((oy-north)/RES)-2; r1=math.ceil((oy-south)/RES)+2
    # The outermost cell is the Horn halo; output matches pilot's inner grid.
    return Affine(RES,0,ox+(c0+1)*RES,0,-RES,oy-(r0+1)*RES),c1-c0-2,r1-r0-2


def sources_for_grid(stack, paths, tr, width, height):
    result={}
    for k, files in paths.items():
        result[k]=[]
        for p in files:
            with rasterio.open(p) as s:
                c=pilot.integer_offset((s.transform.c-tr.c)/RES)
                r=pilot.integer_offset((tr.f-s.transform.f)/RES)
                if c>=width+1 or r>=height+1 or c+s.width<=-1 or r+s.height<=-1: continue
            s=stack.enter_context(rasterio.open(p))
            result[k].append((s,c,r))
        if not result[k]: raise RuntimeError(f'No intersecting tiles for {k}')
    return result


def read_halo(sources, w):
    c0,r0=int(w.col_off)-1,int(w.row_off)-1
    width,height=int(w.width)+2,int(w.height)+2
    out=np.full((height,width),np.nan,dtype=np.float64)
    for s,c,r in sources:
        l,t=max(c,c0),max(r,r0)
        rr,b=min(c+s.width,c0+width),min(r+s.height,r0+height)
        if rr<=l or b<=t: continue
        arr=s.read(1,window=Window(l-c,t-r,rr-l,b-t),masked=True).astype(np.float64).filled(np.nan)
        arr[(arr==NODATA)|~np.isfinite(arr)]=np.nan
        dest=out[t-r0:b-r0,l-c0:rr-c0]
        if np.any(np.isfinite(dest)&np.isfinite(arr)): raise RuntimeError('Overlapping source pixels')
        dest[np.isfinite(arr)]=arr[np.isfinite(arr)]
    return out


def writer(stack, path, name, width, height, crs, tr, grid):
    dst=stack.enter_context(rasterio.open(path,'w',width=width,height=height,crs=crs,transform=tr,**OPTIONS))
    dst.set_band_description(1,name)
    dst.update_tags(source='MERIT Hydro v1.0.1',native_resolution='3 arc-seconds',units=pilot.UNITS[name],grid=grid,alignment_resampling='nearest' if grid=='district UTM' else 'none',twi_slope_floor_deg=str(pilot.FLOOR_DEG),sca_width='sqrt(dx*dy), approximate')
    return dst


def valid(a):
    return np.isfinite(a)&(a!=NODATA)


def build_native(ref, paths, anchor, geom, directory):
    tr,width,height=native_grid(ref,anchor)
    lat=tr.f-(np.arange(height)+.5)*RES
    zero=np.zeros_like(lat)
    _,_,dx=pilot.GEOD.inv(zero,lat,zero+RES,lat)
    _,_,dy=pilot.GEOD.inv(zero,lat-RES/2,zero,lat+RES/2)
    count=0; floor=0; zeros=0; counts={k:0 for k in FEATURES}
    blocks=math.ceil(width/512)*math.ceil(height/512)
    print(f'  Native grid: {width} x {height}; {blocks} blocks',flush=True)
    directory.mkdir()
    with ExitStack() as stack:
        sources=sources_for_grid(stack,paths,tr,width,height)
        dest={k:writer(stack,directory/f'{k}.tif',k,width,height,'EPSG:4326',tr,'native geographic') for k in FEATURES}
        for i,w in enumerate(windows(width,height),1):
            a={k:read_halo(sources[k],w) for k in LAYERS}
            r=int(w.row_off); h=int(w.height)
            f,d=pilot.native_features(a['upa'],a['hnd'],a['elv'],dx[r:r+h,None],dy[r:r+h,None])
            inside=geometry_mask([geom],out_shape=(h,int(w.width)),transform=tr*Affine.translation(w.col_off,w.row_off),invert=True)
            count+=int(inside.sum()); floor+=int((inside&d['twi_floor_applied']).sum()); zeros+=int((inside&d['zero_upa']).sum())
            for k in FEATURES:
                v=f[k]; counts[k]+=int((inside&np.isfinite(v)).sum())
                dest[k].write(np.where(np.isfinite(v),v,NODATA).astype('float32'),1,window=w)
            if i%100==0 or i==blocks: print(f'  Native blocks: {i}/{blocks}',flush=True)
    if not count: raise RuntimeError('No native district pixels')
    return dict(native_pixels=count,twi_floor_pixels=floor,zero_upa_pixels=zeros,**{f'native_{k}_coverage':100*n/count for k,n in counts.items()})


def align(ref, geom, native, aligned):
    aligned.mkdir()
    # Warp one feature at a time directly to a disk-backed raster, then mask
    # and audit blockwise. Native features retain surroundings until the warp.
    for k in FEATURES:
        print(f'  Aligning {k}',flush=True)
        with ExitStack() as stack:
            src=stack.enter_context(rasterio.open(native/f'{k}.tif'))
            dst=writer(stack,aligned/f'{k}.tif',k,ref.width,ref.height,ref.crs,ref.transform,'district UTM')
            reproject(rasterio.band(src,1),rasterio.band(dst,1),src_nodata=NODATA,dst_nodata=NODATA,resampling=Resampling.nearest,num_threads=2,warp_mem_limit=128)
    counts={k:0 for k in FEATURES}; minimum={k:np.inf for k in FEATURES}; maximum={k:-np.inf for k in FEATURES}
    total=0; common=0; terrain_count=0; hydro_common=0
    with ExitStack() as stack:
        files={k:stack.enter_context(rasterio.open(aligned/f'{k}.tif','r+')) for k in FEATURES}
        for k,s in files.items():
            if s.crs!=ref.crs or s.transform!=ref.transform or s.shape!=ref.shape: raise RuntimeError(f'Grid mismatch {k}')
        blocks=math.ceil(ref.width/512)*math.ceil(ref.height/512)
        for i,w in enumerate(windows(ref.width,ref.height),1):
            inside=geometry_mask([geom],out_shape=(int(w.height),int(w.width)),transform=ref.window_transform(w),invert=True)
            total+=int(inside.sum()); both=inside.copy()
            elev=ref.read(1,window=w,masked=True)
            ev=~np.ma.getmaskarray(elev)&valid(elev.data)
            terrain_count+=int((inside&ev).sum())
            for k,s in files.items():
                a=s.read(1,window=w)
                a[~inside]=NODATA
                s.write(a,1,window=w)
                ok=inside&valid(a); both&=ok
                counts[k]+=int(ok.sum())
                if ok.any():
                    v=a[ok]; minimum[k]=min(minimum[k],float(v.min())); maximum[k]=max(maximum[k],float(v.max()))
                    if k in ('upa','log_upa','hand','hydro_slope','spi','log_spi') and (v<0).any(): raise RuntimeError(f'Negative values in {k}')
                    if k=='hydro_slope' and (v>=90).any(): raise RuntimeError('Invalid slope')
            hydro_common+=int(both.sum()); common+=int((both&ev).sum())
            if i%500==0 or i==blocks: print(f'  Mask/audit blocks: {i}/{blocks}',flush=True)
    if not total: raise RuntimeError('No district pixels on target grid')
    result=dict(aligned_pixels=total,hydro_common_coverage=100*hydro_common/total,terrain_elevation_coverage=100*terrain_count/total,common_with_terrain_coverage=100*common/total)
    for k,n in counts.items():
        result.update({f'{k}_coverage':100*n/total,f'{k}_min':minimum[k] if n else None,f'{k}_max':maximum[k] if n else None})
    return result


def compare_pilot(directory):
    base=ROOT/'data/processed/hydrology/pilot/UK/Haridwar'
    if not base.exists(): raise RuntimeError('Haridwar pilot missing; cannot verify batch reproduction')
    # Only derived floating-point features may differ by tiny rounding amounts.
    # Absolute limits in each feature's own units; no relative tolerance.
    tolerances = {'hydro_slope': 2e-6, 'spi': 1e-5, 'log_spi': 5e-7}
    accepted=True
    exact=True
    for grid in ('native','aligned'):
        for k in FEATURES:
            n=0; diffmask=0; changed=0; err=0.; maximum=0.
            with rasterio.open(base/grid/f'{k}.tif') as a,rasterio.open(directory/grid/f'{k}.tif') as b:
                if a.shape!=b.shape or a.crs!=b.crs or not np.allclose(tuple(a.transform),tuple(b.transform),rtol=0,atol=1e-10): raise RuntimeError(f'Pilot grid mismatch {grid}/{k}')
                if a.nodata!=NODATA or b.nodata!=NODATA or a.dtypes[0]!='float32' or b.dtypes[0]!='float32': raise RuntimeError(f'Pilot raster metadata mismatch {grid}/{k}')
                for w in windows(a.width,a.height):
                    x=a.read(1,window=w); y=b.read(1,window=w); vx=valid(x); vy=valid(y)
                    if not np.isfinite(x).all() or not np.isfinite(y).all(): raise RuntimeError(f'Nonfinite stored values in {grid}/{k}')
                    diffmask+=int((vx!=vy).sum()); both=vx&vy
                    d=np.abs(x[both].astype('float64')-y[both].astype('float64'))
                    n+=d.size; changed+=int((d!=0).sum()); err+=float(d.sum())
                    if d.size: maximum=max(maximum,float(d.max()))
            print(f'  {grid}/{k}: common={n:,}, mask differences={diffmask}, changed={changed:,}, MAE={err/max(n,1):.8g}, max={maximum:.8g}',flush=True)
            tolerance = tolerances.get(k, 0.0)
            ok = n>0 and diffmask==0 and maximum<=tolerance
            accepted &= ok
            exact &= ok and changed==0
            print(f'    Absolute tolerance: {tolerance:.8g} | '+('PASS' if ok else 'REVIEW'),flush=True)
    print('  Pilot comparison: '+('EXACT MATCH' if exact else 'PASS (within floating-point tolerance)' if accepted else 'REVIEW (inspect differences before full batch)'),flush=True)
    return accepted


def process(refpath, geom, paths, anchor, out):
    state,district=refpath.parent.parent.name,refpath.parent.name
    dest=out/state/district
    marker=dest/'complete.json'
    if marker.exists():
        record=json.loads(marker.read_text(encoding='utf-8'))
        if record.get('version')!=VERSION or record.get('reference')!=str(refpath) or not all((dest/g/f'{k}.tif').is_file() for g in ('native','aligned') for k in FEATURES): raise RuntimeError(f'Incomplete/stale marker: {dest}; inspect manually')
        print('  Already processed; skipping',flush=True)
        return record
    if dest.exists(): raise RuntimeError(f'Unmarked output exists: {dest}; inspect manually')
    with rasterio.open(refpath) as ref:
        if not ref.crs.is_projected or not np.allclose(ref.res,(30,30)): raise RuntimeError('Unexpected reference grid')
        ng=gpd.GeoSeries([geom],crs='EPSG:4326').to_crs(ref.crs).iloc[0]
        if not ng.is_valid: ng=ng.buffer(0)
        _,nw,nh=native_grid(ref,anchor)
        need=7*4*(nw*nh+ref.width*ref.height)+5*1024**3
        if shutil.disk_usage(out).free<need: raise RuntimeError('Insufficient free space for conservative per-district estimate')
        with tempfile.TemporaryDirectory(prefix=f'{state}_{district}_',dir=out/'_work') as td:
            work=Path(td)/'result'; work.mkdir()
            result=build_native(ref,paths,anchor,geom,work/'native')
            result.update(align(ref,ng,work/'native',work/'aligned'))
            status='PASS' if result['hydro_common_coverage']>=95 and min(result[f'native_{k}_coverage'] for k in FEATURES)>=95 else 'REVIEW'
            record=dict(version=VERSION,reference=str(refpath),state=state,district=district,status=status,**result)
            (work/'complete.json').write_text(json.dumps(record,indent=2,allow_nan=False),encoding='utf-8')
            dest.parent.mkdir(parents=True,exist_ok=True); work.rename(dest)
    print(f"  {status}: hydro common {result['hydro_common_coverage']:.4f}% | with terrain elevation {result['common_with_terrain_coverage']:.4f}%",flush=True)
    return record


def report(out, failures):
    records=[json.loads(p.read_text(encoding='utf-8')) for p in sorted(out.glob('*/*/complete.json')) if p.parts[-3]!='_work']
    tmp=out/'hydrology_batch_report.tmp'; final=out/'hydrology_batch_report.csv'
    pd.DataFrame(records+failures).to_csv(tmp,index=False); tmp.replace(final)
    print(f'  Report: {len(records)} completed districts',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True); group.add_argument('--district',nargs='+'); group.add_argument('--all',action='store_true')
    args=p.parse_args()
    if abs(pilot.FLOOR_DEG-.1)>1e-12: raise RuntimeError('Pilot slope floor changed')
    pilot.self_test()
    refs=sorted(REFS.glob('*/*/elevation.tif'))
    pairs={(norm(r.parent.parent.name),norm(r.parent.name)):r for r in refs}
    gdf=gpd.read_file(BOUNDARY).to_crs('EPSG:4326'); geoms={}
    for _,row in gdf.iterrows():
        key=(norm(row['state']),norm(row['district']))
        if key in geoms: raise RuntimeError(f'Duplicate boundary {key}')
        g=row.geometry
        if not g.is_valid: g=g.buffer(0)
        if g.is_empty: raise RuntimeError(f'Empty boundary {key}')
        geoms[key]=g
    if len(refs)!=137 or len(pairs)!=137 or set(pairs)!=set(geoms): raise RuntimeError('Reference/boundary inventory mismatch')
    if args.all: selected=list(pairs)
    else:
        selected=[tuple(map(norm,s.replace('\\','/').split('/'))) for s in args.district]
        if any(k not in pairs for k in selected): p.error('Unknown district; use STATE/District')
    OUT.mkdir(parents=True,exist_ok=True); (OUT/'_work').mkdir(exist_ok=True)
    failures=[]; review=0; comparisons_ok=True
    with rasterio.Env(GDAL_CACHEMAX=128*1024**2):
        paths,anchor=inventory()
        print(f'Source tiles: 36 | Selected districts: {len(selected)} | Output: {OUT}',flush=True)
        if args.all:
            # Require both staged pilot outputs before processing all districts.
            for k in (('uk','haridwar'),('lk','lehladakh')):
                r=pairs[k]; m=OUT/r.parent.parent.name/r.parent.name/'complete.json'
                if not m.exists() or json.loads(m.read_text(encoding='utf-8')).get('status')!='PASS': raise RuntimeError('Run both single-district pilots first')
            r=pairs[('uk','haridwar')]
            if not compare_pilot(OUT/r.parent.parent.name/r.parent.name): raise RuntimeError('Haridwar comparison requires review')
        for i,key in enumerate(selected,1):
            r=pairs[key]; state,district=r.parent.parent.name,r.parent.name
            print(f'\n[{i}/{len(selected)}] {state}/{district}',flush=True)
            try:
                record=process(r,geoms[key],paths,anchor,OUT)
                review+=record['status']=='REVIEW'
                if key==('uk','haridwar') and not args.all: comparisons_ok &= compare_pilot(OUT/state/district)
            except Exception as e:
                failures.append(dict(state=state,district=district,status='ERROR',error=str(e)))
                print(f'ERROR: {e}',flush=True)
                with (OUT/'hydrology_errors.log').open('a',encoding='utf-8') as f: f.write(traceback.format_exc()+'\n')
            report(OUT,failures)
    print(f'Finished | Errors: {len(failures)} | REVIEW: {review} | Pilot comparison OK: {comparisons_ok}',flush=True)
    return int(bool(failures or review or not comparisons_ok))

if __name__=='__main__': raise SystemExit(main())
