"""District-wise Copernicus terrain processing; run from the Landslide_ML project.
Requires rasterio >= 1.4, numpy, pandas, geopandas, shapely.
Outputs are separate from previous pilots. A completed district has a JSON marker.
"""
import argparse
import csv
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
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.merge import merge
from rasterio.transform import Affine, array_bounds
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window
from shapely.geometry import mapping

ROOT = Path(__file__).resolve().parent.parent
NODATA = -9999.0
BUFFER = 10
RES = 1 / 3600
MIN_GRADIENT = 0.0001
VERSION = 'source-aligned-v2-batch-1'
FEATURES = {
    'elevation': ('metres', 'Source-aligned Copernicus DSM'),
    'slope': ('degrees', 'Horn'),
    'aspect': ('degrees', 'Horn; clockwise from grid north; flat NoData'),
    'plan_curvature': ('1/metre', 'Projected contour curvature'),
    'profile_curvature': ('1/metre', 'Profile curvature'),
    'tangential_curvature': ('1/metre', 'Plan curvature times sin(slope)'),
    'tri': ('metres', 'Riley root sum of squared differences'),
    'tpi': ('metres', 'Centre minus eight-neighbour mean'),
}
OPTIONS = dict(driver='GTiff', count=1, dtype='float32', nodata=NODATA,
               compress='deflate', predictor=3, tiled=True, blockxsize=256,
               blockysize=256, BIGTIFF='IF_SAFER')


def norm(value):
    return re.sub(r'[^a-z0-9]', '', str(value).lower())


def integer_offset(value):
    return abs(value - round(value)) < 1e-6


def aligned_bounds(bounds, ox, oy, rx, ry):
    l, b, r, t = bounds
    return (ox + math.floor((l-ox)/rx)*rx,
            oy + math.floor((b-oy)/ry)*ry,
            ox + math.ceil((r-ox)/rx)*rx,
            oy + math.ceil((t-oy)/ry)*ry)


def inventory(raw_dir):
    all_paths = sorted(raw_dir.rglob('*.tif'))
    tiles = []
    anchor = None
    for path in all_paths:
        with rasterio.open(path) as src:
            if not (path.name.startswith('Copernicus_DSM_') and src.crs
                    and src.crs.to_epsg() == 4326 and src.shape == (3600, 3600)
                    and np.allclose(src.res, (RES, RES), rtol=0, atol=1e-10)):
                continue
            if (src.count != 1 or src.dtypes[0] != 'float32'
                    or src.transform.b != 0 or src.transform.d != 0
                    or src.transform.a <= 0 or src.transform.e >= 0):
                raise RuntimeError(f'Unexpected source metadata: {path}')
            ox, oy = src.transform.c, src.transform.f
            if anchor is None:
                anchor = (ox, oy)
            if not (integer_offset((ox-anchor[0])/RES)
                    and integer_offset((oy-anchor[1])/RES)):
                raise RuntimeError(f'Native grid mismatch: {path}')
            tiles.append((path, src.bounds))
    if not tiles:
        raise RuntimeError('No eligible 1 arc-second DEMs found')
    print(f'Raw files: {len(all_paths)} | Accepted: {len(tiles)} | Excluded: {len(all_paths)-len(tiles)}', flush=True)
    return tiles, anchor


def build_buffer(ref, tiles, anchor, mosaic_path, buffered_path):
    if not (ref.crs and ref.crs.is_projected and np.allclose(ref.res, (30, 30))
            and ref.transform.b == 0 and ref.transform.d == 0
            and ref.transform.a > 0 and ref.transform.e < 0):
        raise RuntimeError('Reference grid is not a north-up projected 30 m raster')
    height, width = ref.height + 2*BUFFER, ref.width + 2*BUFFER
    transform = ref.transform * Affine.translation(-BUFFER, -BUFFER)
    bounds = array_bounds(height, width, transform)
    l, b, r, t = transform_bounds(ref.crs, 'EPSG:4326', *bounds, densify_pts=41)
    requested = (l-2*RES, b-2*RES, r+2*RES, t+2*RES)
    bounds = aligned_bounds(requested, *anchor, RES, RES)
    l, b, r, t = bounds
    paths = [p for p, z in tiles if l < z.right and r > z.left and b < z.top and t > z.bottom]
    if not paths:
        raise RuntimeError('No intersecting 30 m source tiles')
    print(f'  Source tiles: {len(paths)}; building disk-backed mosaic', flush=True)
    # merge returns no full array when dst_path is provided; mem_limit chunks the merge.
    merge(paths, bounds=bounds, res=(RES, RES), nodata=NODATA, dtype='float32',
          indexes=[1], method='first', resampling=Resampling.nearest,
          target_aligned_pixels=False, mem_limit=64, dst_path=str(mosaic_path),
          dst_kwds=OPTIONS)
    # Match the V2 pilot: replace non-finite source samples before interpolation.
    with rasterio.open(mosaic_path, 'r+') as mosaic:
        for _, win in mosaic.block_windows(1):
            arr = mosaic.read(1, window=win)
            bad = ~np.isfinite(arr)
            if bad.any():
                arr[bad] = NODATA
                mosaic.write(arr, 1, window=win)
    print('  Reprojecting to buffered grid', flush=True)
    with rasterio.open(mosaic_path) as src:
        if not (integer_offset((src.transform.c-anchor[0])/RES)
                and integer_offset((src.transform.f-anchor[1])/RES)):
            raise RuntimeError('Intermediate mosaic alignment failed')
        with rasterio.open(buffered_path, 'w', **OPTIONS, width=width, height=height,
                           crs=ref.crs, transform=transform) as dst:
            reproject(rasterio.band(src, 1), rasterio.band(dst, 1),
                      src_nodata=NODATA, dst_nodata=NODATA,
                      resampling=Resampling.bilinear, num_threads=2,
                      warp_mem_limit=128, init_dest_nodata=True)
            dst.update_tags(pipeline=VERSION, source='Copernicus 1 arc-second DSM',
                            mosaic_alignment='Native source grid', resampling='Bilinear',
                            buffer_pixels=str(BUFFER), buffer_metres='300')
    return len(paths)


def calculate(z, valid, dx, dy):
    a,b,c = z[:-2,:-2], z[:-2,1:-1], z[:-2,2:]
    d,e,f = z[1:-1,:-2], z[1:-1,1:-1], z[1:-1,2:]
    g,h,i = z[2:,:-2], z[2:,1:-1], z[2:,2:]
    near = [a,b,c,d,f,g,h,i]
    full = np.ones(e.shape, dtype=bool)
    for row in range(3):
        for col in range(3):
            full &= valid[row:row+e.shape[0], col:col+e.shape[1]]
    p = ((c-a)+2*(f-d)+(i-g))/(8*dx)
    q = ((g-a)+2*(h-b)+(i-c))/(8*dy)
    gs = p*p+q*q
    grad = np.sqrt(gs)
    slope = np.degrees(np.arctan(grad))
    aspect = (np.degrees(np.arctan2(-p,q))+360)%360
    r = (d-2*e+f)/dx**2
    t = (b-2*e+h)/dy**2
    s = (a-c-g+i)/(4*dx*dy)
    ok = full & (grad >= MIN_GRADIENT)
    plan = np.full(e.shape, NODATA)
    profile = np.full(e.shape, NODATA)
    tang = np.full(e.shape, NODATA)
    cross = -(q*q*r-2*p*q*s+p*p*t)
    np.divide(cross, gs**1.5, out=plan, where=ok)
    np.divide(-(p*p*r+2*p*q*s+q*q*t), gs*(1+gs)**1.5, out=profile, where=ok)
    # Algebraically equivalent to plan * sin(slope), without the inverse gradient amplification.
    np.divide(cross, gs*np.sqrt(1+gs), out=tang, where=ok)
    tri = np.sqrt(sum((n-e)**2 for n in near))
    tpi = e-sum(near)/8
    elevation = e.copy()
    elevation[~valid[1:-1,1:-1]] = NODATA
    for arr in (slope, aspect, tri, tpi):
        arr[~full] = NODATA
    aspect[grad == 0] = NODATA
    return dict(elevation=elevation, slope=slope, aspect=aspect, plan_curvature=plan,
                profile_curvature=profile, tangential_curvature=tang, tri=tri, tpi=tpi)


def algorithm_tests():
    x,y = np.meshgrid(np.arange(-2,3)*30., np.arange(-2,3)*30.)
    good = np.ones((5,5), bool)
    f = calculate(x*0+100, good, 30, 30)
    for key in ('slope','tri','tpi'):
        np.testing.assert_allclose(f[key], 0)
    for key in ('aspect','plan_curvature','profile_curvature','tangential_curvature'):
        assert np.all(f[key] == NODATA)
    f = calculate(x+100, good, 30, 30)
    np.testing.assert_allclose(f['slope'], 45)
    np.testing.assert_allclose(f['aspect'], 270)
    np.testing.assert_allclose(f['tri'], np.sqrt(6*30**2))
    np.testing.assert_allclose(f['tpi'], 0)
    np.testing.assert_allclose(f['plan_curvature'], 0)
    bad = good.copy(); bad[2,2] = False
    f = calculate(x+100, bad, 30, 30)
    assert f['elevation'][1,1] == NODATA and np.all(f['slope']==NODATA)
    for k in (.002,-.002):
        f = calculate(100+.5*x-.5*k*x*x, good, 30, 30)
        np.testing.assert_allclose(f['profile_curvature'][1,1], k/(1.25**1.5), atol=1e-10)
        f = calculate(100+.5*x-.5*k*y*y, good, 30, 30)
        np.testing.assert_allclose(f['plan_curvature'][1,1], k/.5, atol=1e-10)
        np.testing.assert_allclose(f['tangential_curvature'][1,1], k/np.sqrt(1.25), atol=1e-10)
    p,q,r,s,t = .4,.3,.002,.001,-.003
    f = calculate(100+p*x+q*y+.5*r*x*x+s*x*y+.5*t*y*y, good, 30, 30)
    grad = np.hypot(p,q); H = np.array([[r,s],[s,t]])
    along = np.array([p,q])/grad; across = np.array([-q,p])/grad
    np.testing.assert_allclose(f['profile_curvature'][1,1], -(along@H@along)/(1+grad**2)**1.5, atol=1e-10)
    np.testing.assert_allclose(f['tangential_curvature'][1,1], -(across@H@across)/np.sqrt(1+grad**2), atol=1e-10)
    f = calculate(100+1e-6*x, good, 30, 30)
    assert f['plan_curvature'][1,1] == NODATA and f['tangential_curvature'][1,1] == NODATA
    print('Batch algorithm smoke tests: PASS', flush=True)


def extract(ref, buffered_path, geom, out_dir):
    if not geom.is_valid:
        geom = geom.buffer(0)
    if geom.is_empty:
        raise RuntimeError('Empty district geometry')
    geo = mapping(geom)
    counts = dict.fromkeys(FEATURES, 0)
    mins = dict.fromkeys(FEATURES, float('inf'))
    maxs = dict.fromkeys(FEATURES, float('-inf'))
    district_pixels = flat = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(buffered_path) as src, ExitStack() as stack:
        expected = ref.transform * Affine.translation(-BUFFER,-BUFFER)
        if (src.crs != ref.crs or src.shape != (ref.height+2*BUFFER,ref.width+2*BUFFER)
                or not np.allclose(tuple(src.transform),tuple(expected),rtol=0,atol=1e-7)):
            raise RuntimeError('Buffered grid mismatch')
        writers = {}
        for name,(units,algorithm) in FEATURES.items():
            writer = stack.enter_context(rasterio.open(out_dir/f'{name}.tif','w', **OPTIONS,
                width=ref.width,height=ref.height,crs=ref.crs,transform=ref.transform))
            writer.set_band_description(1,name)
            writer.update_tags(pipeline=VERSION, units=units, algorithm=algorithm,
                               neighbourhood='3x3' if name!='elevation' else 'centre',
                               source='Buffered V2 native-grid DEM')
            if 'curvature' in name:
                writer.update_tags(sign_convention='Positive convex',min_gradient=str(MIN_GRADIENT))
            writers[name] = writer
        # Always use fixed-size output windows, regardless of reference file strip layout.
        total = math.ceil(ref.height/256)*math.ceil(ref.width/256)
        idx=0
        for row in range(0,ref.height,256):
            for col in range(0,ref.width,256):
                idx+=1
                w,h = min(256,ref.width-col),min(256,ref.height-row)
                win = Window(col,row,w,h)
                inside = geometry_mask([geo],out_shape=(h,w),transform=ref.window_transform(win),
                                       invert=True,all_touched=False)
                district_pixels += int(inside.sum())
                if inside.any():
                    data=src.read(1,window=Window(col+BUFFER-1,row+BUFFER-1,w+2,h+2),masked=True)
                    z=data.data.astype(np.float64)
                    valid=~np.ma.getmaskarray(data)&np.isfinite(z)&(z!=NODATA)
                    z[~valid]=0
                    results=calculate(z,valid,*ref.res)
                    flat += int(np.count_nonzero(inside & (results['slope']==0)))
                else:
                    results={name:np.full((h,w),NODATA,dtype=np.float32) for name in FEATURES}
                for name,arr in results.items():
                    arr[~inside | ~np.isfinite(arr)]=NODATA
                    arr=arr.astype(np.float32)
                    if name=='aspect': arr[arr>=360]=0
                    values=arr[arr!=NODATA]
                    if values.size:
                        counts[name]+=int(values.size)
                        mins[name]=min(mins[name],float(values.min()))
                        maxs[name]=max(maxs[name],float(values.max()))
                    writers[name].write(arr,1,window=win)
                if idx%250==0 or idx==total:
                    print(f'  Terrain blocks: {idx}/{total}',flush=True)
    if not district_pixels:
        raise RuntimeError('No rasterized district pixels')
    stats={}
    for name in FEATURES:
        stats[name]={'valid':counts[name],'missing':district_pixels-counts[name],
                     'coverage':100*counts[name]/district_pixels,
                     'min':mins[name] if counts[name] else None,
                     'max':maxs[name] if counts[name] else None}
        # Validate persisted metadata before marking a district complete.
        with rasterio.open(out_dir/f'{name}.tif') as chk:
            if chk.crs!=ref.crs or chk.shape!=ref.shape or chk.transform!=ref.transform or chk.nodata!=NODATA:
                raise RuntimeError(f'Output metadata mismatch: {name}')
    for name,lo,hi in [('slope',0,90),('aspect',0,360),('tri',0,float('inf'))]:
        a=stats[name]
        if a['valid'] and (a['min']<lo or a['max']>hi or (name=='aspect' and a['max']>=360)):
            raise RuntimeError(f'Invalid {name} range')
    return district_pixels, flat, stats


def compare_pilot(output):
    """Compare the new batch implementation with the existing Haridwar V2 pilot."""
    old_root=ROOT/'data/processed/terrain_v2/UK/Haridwar'
    new_root=output/'terrain_v2/UK/Haridwar'
    print('  Haridwar V2 pilot comparison:',flush=True)
    for name in FEATURES:
        old,new=old_root/f'{name}.tif',new_root/f'{name}.tif'
        if not old.exists() or not new.exists() or old.resolve()==new.resolve(): continue
        n=0; total=0.; maximum=0.; changed_masks=0
        with rasterio.open(old) as a,rasterio.open(new) as b:
            if (a.crs,a.shape,a.transform)!=(b.crs,b.shape,b.transform):
                print(f'    {name}: grid mismatch'); continue
            for _,win in b.block_windows(1):
                aa=a.read(1,window=win,masked=True); bb=b.read(1,window=win,masked=True)
                av=~np.ma.getmaskarray(aa)&np.isfinite(aa.data)
                bv=~np.ma.getmaskarray(bb)&np.isfinite(bb.data)
                changed_masks+=int(np.count_nonzero(av^bv))
                common=av&bv
                if not common.any(): continue
                d=np.abs(aa.data[common].astype(np.float64)-bb.data[common].astype(np.float64))
                if name=='aspect': d=np.minimum(d,360-d)
                n+=d.size; total+=float(d.sum()); maximum=max(maximum,float(d.max()))
        print(f'    {name}: common={n:,}, mask differences={changed_masks:,}, MAE={total/n if n else float("nan"):.8f}, max={maximum:.8f}',flush=True)


def marker_valid(marker, ref_path, feature_dir, buffered):
    try:
        data=json.loads(marker.read_text(encoding='utf-8'))
        return (data['pipeline']==VERSION and data['reference']==str(ref_path)
                and data['status'] in ('PASS','REVIEW') and buffered.is_file()
                and all((feature_dir/f'{n}.tif').is_file() for n in FEATURES))
    except (OSError,ValueError,KeyError):
        return False


def process(ref_path, boundaries, tiles, anchor, output):
    state,district=ref_path.parent.name,ref_path.stem
    key=(norm(state),norm(district))
    if key not in boundaries:
        raise RuntimeError('District boundary not found')
    feature_dir=output/'terrain_v2'/state/district
    buffered=output/'district_dem_buffered_v2'/state/f'{district}.tif'
    marker=feature_dir/'complete.json'
    if marker_valid(marker,ref_path,feature_dir,buffered):
        print('  Already processed; skipping',flush=True)
        return json.loads(marker.read_text(encoding='utf-8'))
    if marker.exists():
        raise RuntimeError(f'Existing incompatible completion marker: {marker}; choose a fresh output directory')
    if buffered.exists() or any((feature_dir/f'{n}.tif').exists() for n in FEATURES):
        raise RuntimeError('Existing outputs without completion marker; inspect or move them before retrying')
    # Conservative per-district free-space check: outputs + temporary mosaic + margin.
    with rasterio.open(ref_path) as ref:
        approx=(ref.width+2*BUFFER)*(ref.height+2*BUFFER)*4
        if shutil.disk_usage(output).free < 12*approx + 5*1024**3:
            raise RuntimeError('Insufficient free space for conservative per-district estimate')
        geom=gpd.GeoSeries([boundaries[key][0]],crs=boundaries[key][1]).to_crs(ref.crs).iloc[0]
        # Only temporary files are automatically deleted. Prior pilot files are never modified.
        with tempfile.TemporaryDirectory(prefix='terrain_',dir=output/'_work') as temp:
            temp=Path(temp)
            count=build_buffer(ref,tiles,anchor,temp/'mosaic.tif',temp/'buffered.tif')
            (temp/'mosaic.tif').unlink()
            pixels,flat,stats=extract(ref,temp/'buffered.tif',geom,temp/'features')
            # A review flag is deliberately not a failed extraction.
            status='PASS' if min(stats['elevation']['coverage'],stats['slope']['coverage'])>=95 else 'REVIEW'
            result=dict(pipeline=VERSION,state=state,district=district,reference=str(ref_path),
                        status=status,source_tiles=count,district_pixels=pixels,flat_pixels=flat,
                        features=stats)
            feature_dir.mkdir(parents=True,exist_ok=True)
            buffered.parent.mkdir(parents=True,exist_ok=True)
            (temp/'buffered.tif').replace(buffered)
            for name in FEATURES:
                (temp/'features'/f'{name}.tif').replace(feature_dir/f'{name}.tif')
            tmp_marker=feature_dir/'complete.json.tmp'
            tmp_marker.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
            tmp_marker.replace(marker)
    print(f'  {status}: elevation {stats["elevation"]["coverage"]:.4f}% | slope {stats["slope"]["coverage"]:.4f}%',flush=True)
    return result


def report(output, failures):
    rows=[]
    for marker in sorted((output/'terrain_v2').rglob('complete.json')):
        data=json.loads(marker.read_text(encoding='utf-8'))
        if data.get('pipeline')!=VERSION: continue
        row={k:data[k] for k in ('state','district','status','source_tiles','district_pixels','flat_pixels')}
        for n,st in data['features'].items():
            row.update({f'{n}_{k}':v for k,v in st.items()})
        rows.append(row)
    existing={(r['state'],r['district']) for r in rows}
    rows.extend(r for r in failures if (r['state'],r['district']) not in existing)
    if not rows: return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    dest=output/'terrain_batch_report.csv'
    tmp=output/'terrain_batch_report.csv.tmp'
    with tmp.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    tmp.replace(dest)
    print(f'Report: {dest} | Completed: {sum(r["status"] in ("PASS","REVIEW") for r in rows)}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root',required=True,type=Path)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--district',nargs='+',help='State/district pairs, e.g. UK/Haridwar LK/Leh_Ladakh')
    group.add_argument('--all',action='store_true')
    args=p.parse_args()
    output=args.output_root.resolve()
    # Require the drive to exist, rather than silently creating a relative path after a typo.
    if not args.output_root.is_absolute() or not Path(output.anchor).exists():
        p.error('--output-root must be an absolute path on an existing drive')
    output.mkdir(parents=True,exist_ok=True)
    (output/'_work').mkdir(exist_ok=True)
    refs=sorted((ROOT/'data/processed/DEM/district_dem_utm').rglob('*.tif'))
    if not refs: raise RuntimeError('No reference DEMs found')
    if len(refs)!=137: print(f'WARNING: expected 137 reference DEMs, found {len(refs)}')
    pairs={(norm(x.parent.name),norm(x.stem)):x for x in refs}
    if len(pairs)!=len(refs): raise RuntimeError('Duplicate reference district keys')
    if args.district:
        selected=[]
        for name in args.district:
            parts=name.replace('\\','/').split('/')
            if len(parts)!=2 or tuple(map(norm,parts)) not in pairs:
                p.error(f'Unknown district: {name}')
            selected.append(pairs[tuple(map(norm,parts))])
    else: selected=refs
    algorithm_tests()
    gdf=gpd.read_file(ROOT/'data/processed/boundary/ihr_districts.gpkg').to_crs('EPSG:4326')
    boundaries={}
    for _,r in gdf.iterrows():
        key=(norm(r['state']),norm(r['district']))
        if key in boundaries: raise RuntimeError(f'Duplicate boundary key: {key}')
        boundaries[key]=(r.geometry,gdf.crs)
    failures=[]
    with rasterio.Env(GDAL_CACHEMAX=128*1024**2):
        tiles,anchor=inventory(ROOT/'data/raw/dem')
        print(f'Output: {output} | Selected districts: {len(selected)}',flush=True)
        for idx,ref in enumerate(selected,1):
            print(f'\n[{idx}/{len(selected)}] {ref.parent.name}/{ref.stem}',flush=True)
            try:
                process(ref,boundaries,tiles,anchor,output)
                if (norm(ref.parent.name),norm(ref.stem)) == ('uk','haridwar'):
                    compare_pilot(output)
            except Exception as exc:
                print(f'ERROR: {exc}',flush=True)
                failures.append(dict(state=ref.parent.name,district=ref.stem,status='ERROR',error=str(exc)))
                with (output/'terrain_errors.log').open('a',encoding='utf-8') as f:
                    f.write(f'\n{ref}:\n{traceback.format_exc()}\n')
            report(output,failures)
    print(f'\nSelected districts: {len(selected)} | Errors in this run: {len(failures)}',flush=True)
    return 1 if failures else 0


if __name__=='__main__':
    raise SystemExit(main())
