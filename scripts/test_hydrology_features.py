"""Haridwar pilot: native MERIT Hydro derivatives and 30 m grid alignment.
Run from project/scripts; reads raw data and writes only hydrology/pilot outputs.
"""
from pathlib import Path
import math
import numpy as np
import geopandas as gpd
import rasterio
from affine import Affine
from pyproj import Geod
from rasterio.features import geometry_mask
from rasterio.windows import Window
from rasterio.warp import transform_bounds, reproject, Resampling
from rasterio.transform import array_bounds

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / 'data/raw/hydrology/merit_hydro'
REF = ROOT / 'data/processed/terrain_batch/terrain_v2/UK/Haridwar/elevation.tif'
BOUNDARY = ROOT / 'data/processed/boundary/ihr_districts.gpkg'
OUT = ROOT / 'data/processed/hydrology/pilot/UK/Haridwar'
NODATA = -9999.0
RES = 1/1200
FLOOR_DEG = 0.1
MAX_NATIVE_PIXELS = 10_000_000  # this is a pilot, not the batch implementation
GEOD = Geod(ellps='WGS84')
UNITS = {'upa':'km2', 'log_upa':'ln(1+UPA in km2)', 'hand':'m',
         'hydro_slope':'degrees', 'twi':'ln(SCA in m / tan(slope))',
         'spi':'m (SCA * tan(slope))', 'log_spi':'ln(1+SPI in m)'}


def native_features(upa, hand, elv, dx, dy):
    """Arrays include a one-pixel halo; dx/dy are metres by centre row."""
    a,b,c = elv[:-2,:-2],elv[:-2,1:-1],elv[:-2,2:]
    d,e,f = elv[1:-1,:-2],elv[1:-1,1:-1],elv[1:-1,2:]
    g,h,i = elv[2:,:-2],elv[2:,1:-1],elv[2:,2:]
    valid = np.ones(e.shape, dtype=bool)
    for z in (a,b,c,d,e,f,g,h,i):
        valid &= np.isfinite(z)
    with np.errstate(invalid='ignore'):
        p = ((c-a)+2*(f-d)+(i-g))/(8*dx)
        q = ((g-a)+2*(h-b)+(i-c))/(8*dy)
        gradient = np.hypot(p,q)
    gradient[~valid] = np.nan
    slope = np.degrees(np.arctan(gradient))
    u = upa[1:-1,1:-1].copy()
    hnd = hand[1:-1,1:-1].copy()
    u[~np.isfinite(u) | (u<0)] = np.nan
    hnd[~np.isfinite(hnd) | (hnd<0)] = np.nan
    sca = u*1e6 / np.sqrt(dx*dy)  # equivalent-square contour width approximation
    twi = np.full(e.shape, np.nan)
    ok = np.isfinite(sca) & (sca>0) & np.isfinite(gradient)
    twi[ok] = np.log(sca[ok]/np.maximum(gradient[ok],np.tan(np.radians(FLOOR_DEG))))
    spi = sca*gradient  # true zero slope stays zero; floor applies only to TWI
    features = {'upa':u, 'log_upa':np.log1p(u), 'hand':hnd,
                'hydro_slope':slope, 'twi':twi, 'spi':spi, 'log_spi':np.log1p(spi)}
    diagnostics = {'zero_upa':np.isfinite(u)&(u==0),
                   'twi_floor_applied':ok&(slope<FLOOR_DEG)}
    return features, diagnostics


def self_test():
    z = np.tile(np.arange(5)*90.,(5,1))
    u = np.full((5,5),.81); h = np.full((5,5),5.)
    dx = dy = np.full((3,1),90.)
    f,_ = native_features(u,h,z,dx,dy)
    np.testing.assert_allclose(f['hydro_slope'],45)
    np.testing.assert_allclose(f['twi'],np.log(9000.))
    np.testing.assert_allclose(f['spi'],9000.)
    f,diag = native_features(u,h,np.ones((5,5)),dx,dy)
    np.testing.assert_allclose(f['spi'],0)
    np.testing.assert_allclose(f['twi'],np.log(9000/np.tan(np.radians(FLOOR_DEG))))
    assert diag['twi_floor_applied'].all()
    z[2,2] = np.nan
    f,_ = native_features(u,h,z,dx,dy)
    assert np.isnan(f['hydro_slope']).all() and np.isnan(f['twi']).all()
    assert np.isfinite(f['upa']).all()
    u[2,2]=0
    f,_=native_features(u,h,np.ones((5,5)),dx,dy)
    assert np.isnan(f['twi'][1,1]) and f['spi'][1,1]==0
    print('Synthetic hydrology tests: PASS', flush=True)


def integer_offset(value):
    i=round(value)
    if abs(value-i)>1e-5:
        raise ValueError(f'Native grid alignment error: {value}')
    return i


def read_mosaic(paths, tr, width, height):
    out=np.full((height,width),np.nan,dtype=np.float64)
    used=[]
    for path in paths:
        with rasterio.open(path) as src:
            if src.crs.to_epsg()!=4326 or not np.allclose(src.res,(RES,RES),rtol=0,atol=1e-12) or src.transform.b!=0 or src.transform.d!=0 or src.transform.a<=0 or src.transform.e>=0:
                raise ValueError(f'Unexpected source grid: {path}')
            c0=integer_offset((src.transform.c-tr.c)/RES)
            r0=integer_offset((tr.f-src.transform.f)/RES)
            c1=max(c0,0); r1=max(r0,0)
            c2=min(c0+src.width,width); r2=min(r0+src.height,height)
            if c2<=c1 or r2<=r1: continue
            arr=src.read(1,window=Window(c1-c0,r1-r0,c2-c1,r2-r1),masked=True).astype(np.float64).filled(np.nan)
            arr[(arr==NODATA)|~np.isfinite(arr)]=np.nan
            dest=out[r1:r2,c1:c2]
            if np.any(np.isfinite(dest)&np.isfinite(arr)):
                raise ValueError(f'Overlapping valid source pixels: {path}')
            dest[np.isfinite(arr)]=arr[np.isfinite(arr)]
            used.append(path.name)
    if not used: raise RuntimeError('No intersecting tiles')
    print('  '+', '.join(used),flush=True)
    return out


def save(path, data, crs, tr, feature, grid):
    path.parent.mkdir(parents=True,exist_ok=True)
    arr=np.where(np.isfinite(data),data,NODATA).astype('float32')
    with rasterio.open(path,'w',driver='GTiff',width=arr.shape[1],height=arr.shape[0],count=1,dtype='float32',crs=crs,transform=tr,nodata=NODATA,compress='deflate',predictor=3,tiled=True,blockxsize=256,blockysize=256,BIGTIFF='IF_SAFER') as dst:
        dst.write(arr,1)
        dst.set_band_description(1,feature)
        dst.update_tags(source='MERIT Hydro v1.0.1',native_resolution='3 arc-seconds',units=UNITS[feature],grid=grid,alignment_resampling='nearest' if grid=='district UTM' else 'none',twi_slope_floor_deg=str(FLOOR_DEG),sca_width='sqrt(dx*dy), approximate')


def describe(name,arr,inside):
    ok=inside&np.isfinite(arr)&(arr!=NODATA)
    vals=arr[ok]
    if vals.size:
        p=np.percentile(vals,[50,99])
        print(f'{name:14s} | valid={vals.size:,} | coverage={100*vals.size/inside.sum():.4f}% | min={vals.min():.6g} | median={p[0]:.6g} | p99={p[1]:.6g} | max={vals.max():.6g}')
    else: print(f'{name}: NO VALID PIXELS')
    return vals.size


def main():
    self_test()
    districts=gpd.read_file(BOUNDARY)
    selected=districts[(districts['state'].str.upper()=='UK') & (districts['district'].str.lower()=='haridwar')]
    if len(selected)!=1: raise RuntimeError('Haridwar boundary match failed')
    paths={k:sorted((RAW/k).rglob('*.tif')) for k in ('upa','hnd','elv')}
    if any(len(v)!=12 for v in paths.values()): raise RuntimeError('Expected 12 tiles per layer')
    with rasterio.open(REF) as ref, rasterio.open(paths['elv'][0]) as anchor:
        if not ref.crs.is_projected or not np.allclose(ref.res,(30,30)): raise RuntimeError('Unexpected reference grid')
        west,south,east,north=transform_bounds(ref.crs,'EPSG:4326',*ref.bounds,densify_pts=21)
        ox,oy=anchor.transform.c,anchor.transform.f
        c0=math.floor((west-ox)/RES)-2; c1=math.ceil((east-ox)/RES)+2
        r0=math.floor((oy-north)/RES)-2; r1=math.ceil((oy-south)/RES)+2
        width,height=c1-c0,r1-r0
        if width*height>MAX_NATIVE_PIXELS: raise RuntimeError('Pilot window too large; use a block-wise implementation')
        tr=Affine(RES,0,ox+c0*RES,0,-RES,oy-r0*RES)
        print(f'Native window with halo: {width} x {height}')
        arrays={}
        for layer in paths:
            print(f'Reading {layer}:',flush=True)
            arrays[layer]=read_mosaic(paths[layer],tr,width,height)
        lat=tr.f-(np.arange(1,height-1)+.5)*RES
        zeros=np.zeros_like(lat)
        _,_,dx=GEOD.inv(zeros,lat,zeros+RES,lat)
        _,_,dy=GEOD.inv(zeros,lat-RES/2,zeros,lat+RES/2)
        dx=dx[:,None]; dy=dy[:,None]
        print(f'Native cell size: east-west {dx.min():.2f}–{dx.max():.2f} m; north-south {dy.min():.2f}–{dy.max():.2f} m')
        features,diag=native_features(arrays['upa'],arrays['hnd'],arrays['elv'],dx,dy)
        nt=tr*Affine.translation(1,1)
        g=selected.to_crs('EPSG:4326').geometry.iloc[0]
        if not g.is_valid: g=g.buffer(0)
        inside_native=geometry_mask([g],out_shape=(height-2,width-2),transform=nt,invert=True)
        if not inside_native.any(): raise RuntimeError('No district pixels on native grid')
        print('\nNATIVE DISTRICT STATISTICS')
        print(f'District pixels: {inside_native.sum():,}')
        for k,v in diag.items(): print(f'{k}: {np.count_nonzero(v&inside_native):,}')
        # Retain unmasked surrounding native data for nearest-neighbor alignment.
        for k,arr in features.items():
            describe(k,arr,inside_native)
            save(OUT/'native'/f'{k}.tif',arr,'EPSG:4326',nt,k,'native geographic')
        g=selected.to_crs(ref.crs).geometry.iloc[0]
        if not g.is_valid: g=g.buffer(0)
        inside=geometry_mask([g],out_shape=ref.shape,transform=ref.transform,invert=True)
        if not inside.any(): raise RuntimeError('No district pixels on target grid')
        terrain=ref.read(1,masked=True)
        terrain_valid=~np.ma.getmaskarray(terrain)&np.isfinite(terrain.data)
        common=inside&terrain_valid
        print('\nALIGNED 30 M DISTRICT STATISTICS')
        print(f'District pixels: {inside.sum():,}')
        for k,arr in features.items():
            dest=np.full(ref.shape,NODATA,dtype='float32')
            reproject(np.where(np.isfinite(arr),arr,NODATA).astype('float32'),dest,src_transform=nt,src_crs='EPSG:4326',src_nodata=NODATA,dst_transform=ref.transform,dst_crs=ref.crs,dst_nodata=NODATA,resampling=Resampling.nearest,num_threads=2)
            dest[~inside]=NODATA
            describe(k,dest,inside)
            common &= np.isfinite(dest)&(dest!=NODATA)
            save(OUT/'aligned'/f'{k}.tif',dest,ref.crs,ref.transform,k,'district UTM')
            with rasterio.open(OUT/'aligned'/f'{k}.tif') as check:
                if check.crs!=ref.crs or check.transform!=ref.transform or check.shape!=ref.shape: raise RuntimeError(f'Output alignment failed: {k}')
        print(f'Common validity with Copernicus elevation: {100*common.sum()/inside.sum():.4f}%')
        print('Aligned output metadata: PASS')
        print(f'Output: {OUT}')
        print('Pilot complete; coverage and index distributions require review.')

if __name__=='__main__': main()
