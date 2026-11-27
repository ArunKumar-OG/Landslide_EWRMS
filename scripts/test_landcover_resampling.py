"""Compare WorldCover nearest and mode on the existing district UTM grid.
Read-only source files; disk-backed warp, block-wise district mask and audit.
Requires audit_landcover_pixels.py in the same scripts directory.
"""
from pathlib import Path
import argparse
import tempfile
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.warp import reproject, transform_bounds
from rasterio.transform import from_origin
from shapely.geometry import box
from shapely.ops import unary_union
from audit_landcover_pixels import CLASSES, NAMES, validate_tile

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / 'data/raw/landcover'
OUT = ROOT / 'data/processed/landcover'
BOUNDARY = ROOT / 'data/processed/boundary/ihr_districts.gpkg'
REFS = ROOT / 'data/processed/terrain_batch/terrain_v2'
ALLOWED = (0, *CLASSES)


def make_mosaic(paths, dest):
    """Native-grid VRT, not separately warped tiles. Missing areas are zero."""
    records=[]
    for path in paths:
        with rasterio.open(path) as src:
            if src.crs.to_epsg()!=4326 or src.count!=1 or src.dtypes[0]!='uint8' or src.nodata!=0:
                raise ValueError(f'Invalid source: {path}')
            t=src.transform
            if t.a<=0 or t.e>=0 or t.b or t.d:
                raise ValueError('Non-north-up grid')
            records.append((path.resolve(),src.bounds,src.width,src.height,t.a,-t.e))
    if not records: raise ValueError('Empty mosaic')
    rx,ry=records[0][-2:]
    west=min(r[1].left for r in records); north=max(r[1].top for r in records)
    east=max(r[1].right for r in records); south=min(r[1].bottom for r in records)
    def integer(x):
        n=round(x)
        if abs(x-n)>1e-5: raise ValueError('Source grid not aligned')
        return n
    w=integer((east-west)/rx); h=integer((north-south)/ry)
    root=ET.Element('VRTDataset',rasterXSize=str(w),rasterYSize=str(h))
    ET.SubElement(root,'SRS').text='EPSG:4326'
    ET.SubElement(root,'GeoTransform').text=','.join(str(v) for v in (west,rx,0,north,0,-ry))
    band=ET.SubElement(root,'VRTRasterBand',dataType='Byte',band='1')
    ET.SubElement(band,'NoDataValue').text='0'
    used=[]
    for path,b,nx,ny,dx,dy in records:
        if not np.allclose([dx,dy],[rx,ry],atol=1e-12,rtol=0): raise ValueError('Resolution mismatch')
        x=integer((b.left-west)/rx); y=integer((north-b.top)/ry)
        if any(x<xx+ww and xx<x+nx and y<yy+hh and yy<y+ny for xx,yy,ww,hh in used):
            raise ValueError('Overlapping tile grids')
        used.append((x,y,nx,ny))
        s=ET.SubElement(band,'ComplexSource')
        ET.SubElement(s,'SourceFilename',relativeToVRT='0').text=path.as_posix()
        ET.SubElement(s,'SourceBand').text='1'
        ET.SubElement(s,'SrcRect',xOff='0',yOff='0',xSize=str(nx),ySize=str(ny))
        ET.SubElement(s,'DstRect',xOff=str(x),yOff=str(y),xSize=str(nx),ySize=str(ny))
        ET.SubElement(s,'UseMaskBand').text='true'
    ET.ElementTree(root).write(dest,encoding='UTF-8',xml_declaration=True)


def warp(mosaic, path, ref, method):
    profile=dict(driver='GTiff',width=ref.width,height=ref.height,count=1,dtype='uint8',
                 crs=ref.crs,transform=ref.transform,nodata=0,tiled=True,blockxsize=256,
                 blockysize=256,compress='deflate',BIGTIFF='IF_SAFER')
    with rasterio.Env(GDAL_CACHEMAX=128*1024**2), rasterio.open(mosaic) as src, rasterio.open(path,'w',**profile) as dst:
        reproject(source=rasterio.band(src,1),destination=rasterio.band(dst,1),
                  src_nodata=0,dst_nodata=0,resampling=method,init_dest_nodata=True,
                  warp_mem_limit=128,num_threads=1)
        dst.update_tags(source='ESA WorldCover 2021 v200',resampling=method.name)


def audit_pair(near,mode,ref,geom,mask_outputs=False,progress=False):
    """Compare outputs; only mutate the pilot rasters during masking pass."""
    counts=np.zeros((2,256),dtype=np.int64)
    transitions=np.zeros((256,256),dtype=np.int64)
    joint=np.zeros(2,dtype=np.int64); n=0; both=0; agree=0
    with rasterio.open(near,'r+' if mask_outputs else 'r') as a, rasterio.open(mode,'r+' if mask_outputs else 'r') as b:
        for ds in (a,b):
            if (ds.shape!=ref.shape or ds.crs!=ref.crs or ds.transform!=ref.transform or
                ds.dtypes[0]!='uint8' or ds.count!=1 or ds.nodata!=0):
                raise ValueError('Output grid or metadata mismatch')
        for i,(_,win) in enumerate(a.block_windows(1),1):
            inside=geometry_mask([geom],out_shape=(int(win.height),int(win.width)),
                                 transform=a.window_transform(win),invert=True,all_touched=False)
            arrays=[ds.read(1,window=win) for ds in (a,b)]
            elev=ref.read(1,window=win,masked=True)
            ev=~np.ma.getmaskarray(elev)&np.isfinite(elev.data)
            n+=int(inside.sum())
            for k,(ds,arr) in enumerate(zip((a,b),arrays)):
                if mask_outputs:
                    arr[~inside]=0; ds.write(arr,1,window=win)
                elif np.any(arr[~inside]!=0): raise ValueError('Outside-boundary nonzero pixels')
                if not np.isin(arr,ALLOWED).all(): raise ValueError('Unexpected output class')
                counts[k]+=np.bincount(arr[inside],minlength=256)
                joint[k]+=np.count_nonzero(inside&ev&(arr!=0))
            av,bv=(arr[inside] for arr in arrays)
            transitions+=np.bincount(av.astype(np.int32)*256+bv,minlength=65536).reshape(256,256)
            valid=(av!=0)&(bv!=0); both+=int(valid.sum()); agree+=int(np.count_nonzero(valid&(av==bv)))
            if progress and i%500==0: print(f'  Audited blocks: {i}',flush=True)
    if n==0 or np.any(counts.sum(axis=1)!=n) or transitions.sum()!=n: raise ValueError('Pixel accounting failed')
    return n,counts,transitions,joint,both,agree


def self_test():
    # 6x3 mosaic, target 2x1; second target crosses tile seam (tile width 4).
    with tempfile.TemporaryDirectory() as td:
        td=Path(td); paths=[]
        data=np.array([[10,10,10,40,40,40],[10,50,10,40,10,40],[10,10,10,40,40,40]],dtype='uint8')
        for k,(x,nx) in enumerate(((0,4),(4,2))):
            p=td/f'{k}.tif'; paths.append(p)
            with rasterio.open(p,'w',driver='GTiff',width=nx,height=3,count=1,dtype='uint8',crs='EPSG:4326',transform=from_origin(x,3,1,1),nodata=0) as ds:
                ds.write(data[:,x:x+nx],1)
                if k==1:
                    mask=np.full((3,nx),255,dtype='uint8'); mask[0,0]=0; ds.write_mask(mask)
        mosaic=td/'mosaic.vrt'; make_mosaic(paths,mosaic)
        with rasterio.open(mosaic) as ds:
            expected=data.copy(); expected[0,4]=0
            assert np.array_equal(ds.read(1),expected), ds.read(1)
        rp=td/'ref.tif'
        with rasterio.open(rp,'w',driver='GTiff',width=2,height=1,count=1,dtype='float32',crs='EPSG:4326',transform=from_origin(0,3,3,3),nodata=-9999) as ds: ds.write(np.ones((1,2),dtype='float32'),1)
        near=td/'near.tif'; mode=td/'mode.tif'
        with rasterio.open(rp) as ref:
            warp(mosaic,near,ref,Resampling.nearest); warp(mosaic,mode,ref,Resampling.mode)
            with rasterio.open(near) as ds: assert ds.read(1).tolist()==[[50,10]]
            with rasterio.open(mode) as ds: assert ds.read(1).tolist()==[[10,40]]
            first=audit_pair(near,mode,ref,box(0,0,3,3),True)
            second=audit_pair(near,mode,ref,box(0,0,3,3))
            assert first[0]==second[0]==1 and all(np.array_equal(x,y) for x,y in zip(first,second))
    print('Synthetic mosaic seam, mask, mode/nearest and readback tests: PASS',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--district',default='UK/Haridwar')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args(); self_test()
    if args.self_test: return 0
    state,name=args.district.split('/')
    districts=gpd.read_file(BOUNDARY)
    sel=districts[(districts.state.str.upper()==state.upper()) & (districts.district.str.replace(' ','_',regex=False).str.lower()==name.lower())]
    if len(sel)!=1: raise ValueError('District match failed')
    refpath=REFS/state/name/'elevation.tif'
    inv=pd.read_csv(OUT/'landcover_tile_inventory.csv')
    audit=pd.read_csv(OUT/'landcover_pixel_audit.csv')
    relevant=inv[inv.relevant.astype(str).str.lower().eq('true')]
    if len(relevant)!=20 or not relevant.status.eq('PASS').all() or set(relevant.file)!=set(audit.file) or not audit.status.eq('PASS').all():
        raise ValueError('Source audits incomplete or inconsistent')
    if audit.file.duplicated().any() or relevant.file.duplicated().any(): raise ValueError('Duplicate file inventory')
    native=pd.read_csv(OUT/'landcover_district_class_counts.csv')
    native=native[(native.state.str.upper()==state.upper()) & (native.district.str.replace(' ','_',regex=False).str.lower()==name.lower())]
    if len(native)!=len(CLASSES) or set(native.code)!=set(CLASSES): raise ValueError('Native class inventory mismatch')
    native_counts=native.set_index('code').pixels.to_dict()
    coverage=pd.read_csv(OUT/'landcover_district_pixel_coverage.csv')
    coverage=coverage[(coverage.state.str.upper()==state.upper()) & (coverage.district.str.replace(' ','_',regex=False).str.lower()==name.lower())]
    if len(coverage)!=1 or coverage.status.iloc[0]!='PASS' or int(coverage.pixels.iloc[0])!=sum(native_counts.values()) or sum(native_counts.values())<=0:
        raise ValueError('Native district pixel coverage or class counts inconsistent')
    dest=OUT/'pilot'/state/name; dest.mkdir(parents=True,exist_ok=True)
    with rasterio.open(refpath) as ref:
        if not ref.crs.is_projected or not np.allclose(ref.res,(30,30)) or ref.transform.b or ref.transform.d: raise ValueError('Invalid terrain reference grid')
        geom=sel.to_crs(ref.crs).geometry.iloc[0]
        if not geom.is_valid or geom.is_empty: raise ValueError('Invalid district geometry')
        bbox=box(*transform_bounds(ref.crs,'EPSG:4326',*ref.bounds,densify_pts=41)).buffer(.01)
        paths=[]; footprints=[]
        for file in relevant.file:
            p=RAW/file
            with rasterio.open(p) as src: _,fp=validate_tile(src,p.name)
            if fp.intersects(bbox): paths.append(p); footprints.append(fp)
        district_geo=sel.to_crs('EPSG:4326').geometry.iloc[0]
        if district_geo.difference(unary_union(footprints)).area>district_geo.area*1e-10: raise ValueError('Selected tile footprint gap')
        print(f'Grid: {ref.width} x {ref.height} | {ref.crs} | 30 m')
        print('Mosaic sources:',', '.join(p.name for p in paths))
        with tempfile.TemporaryDirectory(dir=dest) as td:
            td=Path(td); vrt=td/'mosaic.vrt'; make_mosaic(paths,vrt)
            near=td/'landcover_nearest.tif'; mode=td/'landcover_mode.tif'
            for p,method in ((near,Resampling.nearest),(mode,Resampling.mode)):
                print(f'Warping: {method.name}',flush=True); warp(vrt,p,ref,method)
            print('Masking and auditing outputs...',flush=True)
            first=audit_pair(near,mode,ref,geom,True,True)
            print('Readback validation...',flush=True)
            result=audit_pair(near,mode,ref,geom,False,True)
            if not all(np.array_equal(x,y) for x,y in zip(first,result)): raise ValueError('Readback differs')
            for p in (near,mode): p.replace(dest/p.name)
    n,c,t,j,both,agree=result
    rows=[]; native_n=sum(native_counts.values())
    for code,label in zip(CLASSES,NAMES):
        rows.append(dict(code=code,landcover=label,native_pixels=native_counts[code],native_pct=100*native_counts[code]/native_n,
                         nearest_pixels=int(c[0,code]),nearest_pct=100*c[0,code]/n,
                         mode_pixels=int(c[1,code]),mode_pct=100*c[1,code]/n,
                         mode_minus_nearest_pp=100*(int(c[1,code])-int(c[0,code]))/n))
    frame=pd.DataFrame(rows); frame.to_csv(dest/'landcover_class_comparison.csv',index=False)
    trans=pd.DataFrame(t[np.ix_(ALLOWED,ALLOWED)],index=ALLOWED,columns=ALLOWED); trans.index.name='nearest_code'; trans.to_csv(dest/'landcover_transition_counts.csv')
    stats=dict(state=state,district=name,district_pixels=n,nearest_coverage_pct=100*(n-c[0,0])/n,mode_coverage_pct=100*(n-c[1,0])/n,
               nearest_joint_elevation_pct=100*j[0]/n,mode_joint_elevation_pct=100*j[1]/n,common_valid_pixels=both,
               mask_difference_pixels=int(t[0,1:].sum()+t[1:,0].sum()),agreement_pct=100*agree/both if both else 0,
               changed_common_pixels=both-agree,nearest_distribution_delta_pp=float((frame.nearest_pct-frame.native_pct).abs().sum()/2+50*c[0,0]/n),
               mode_distribution_delta_pp=float((frame.mode_pct-frame.native_pct).abs().sum()/2+50*c[1,0]/n),status='PASS' if c[:,0].sum()==0 else 'REVIEW')
    pd.DataFrame([stats]).to_csv(dest/'landcover_pilot_audit.csv',index=False)
    print('\nPILOT SUMMARY'); print('\n'.join(f'{k}: {v}' for k,v in stats.items()))
    print('\nCLASS COMPARISON'); print(frame.to_string(index=False,float_format=lambda x:f'{x:.6f}'))
    print('Readback metadata, mask and class checks: PASS'); print('Output:',dest)
    return 0 if stats['status']=='PASS' else 2

if __name__=='__main__': raise SystemExit(main())
