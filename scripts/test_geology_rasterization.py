"""Pilot categorical GLiM rasterization on existing district UTM grids.
Source files are read only. 0 outside, 1..15 mapped classes, 16 source nd,
17 unmapped district pixel, 18 ambiguous/conflicting district pixel.
"""
from pathlib import Path
import argparse
import re
import tempfile
import math
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize, geometry_mask
from rasterio.windows import Window, bounds
from shapely.geometry import box
from shapely import make_valid, union_all

ROOT = Path(__file__).resolve().parent.parent
GEO = ROOT / 'data/processed/geology'
REFS = ROOT / 'data/processed/terrain_batch/terrain_v2'
BOUNDARY = ROOT / 'data/processed/boundary/ihr_districts.gpkg'
OUT = GEO / 'pilot'
CODES = ['su','ss','py','sm','sc','ev','va','vi','vb','pa','pi','pb','mt','wb','ig','nd']
LABELS = {i+1:c for i,c in enumerate(CODES)} | {0:'outside',17:'unmapped',18:'conflict'}
BLOCK = 512


def norm(s):
    return re.sub(r'[^a-z0-9]', '', str(s).lower())


def windows(w,h):
    for r in range(0,h,BLOCK):
        for c in range(0,w,BLOCK):
            yield Window(c,r,min(BLOCK,w-c),min(BLOCK,h-r))


def parts(g):
    if g is None or g.is_empty: return []
    if g.geom_type=='Polygon': return [g]
    if g.geom_type in ('MultiPolygon','GeometryCollection'):
        return [p for child in g.geoms for p in parts(child)]
    return []


def polygonal(g):
    if g is not None and not g.is_empty and g.is_valid and g.geom_type in ('Polygon','MultiPolygon'):
        return g
    p=parts(make_valid(g)) if g is not None else []
    if not p: raise ValueError('No polygonal geometry')
    result=union_all(p)
    if not result.is_valid or result.is_empty: raise ValueError('Invalid repaired geometry')
    return result


def classify(inside,tr,polygons):
    """Group by class so same-class overlaps never create conflicts."""
    out=np.zeros(inside.shape,dtype='uint8')
    out[inside]=17
    if not inside.any(): return out
    for cid, geoms in sorted(polygons.items()):
        if not geoms: continue
        covered=rasterize([(g,1) for g in geoms],out_shape=inside.shape,
                          transform=tr,fill=0,dtype='uint8',all_touched=False).astype(bool)&inside
        fresh=covered & (out==17)
        conflict=covered & (out!=17) & (out!=cid)
        out[fresh]=cid
        out[conflict]=18
    return out


def smoke_tests():
    tr=rasterio.transform.from_origin(0,4,1,1)
    mask=np.ones((4,6),bool); mask[0,0]=False
    polys={1:[box(0,0,3,4),box(0,0,2,4)], 13:[box(2,0,4,4)],16:[box(4,0,5,4)]}
    a=classify(mask,tr,polys)
    assert np.array_equal(a[1], [1,1,18,13,16,17]) and a[0,0]==0
    for start in (0,3):
        b=classify(mask[:,start:start+3],tr*rasterio.Affine.translation(start,0),polys)
        assert np.array_equal(b,a[:,start:start+3])
    assert np.all(classify(mask,tr,{})[mask]==17)
    # Known vs explicit NoData is also a conflict, not silently prioritized.
    assert classify(np.ones((4,6),bool),tr,{1:[box(0,0,2,4)],16:[box(1,0,3,4)]})[1,1]==18
    print('Synthetic categorical rasterization tests: PASS',flush=True)


def run(refpath,district,source,source_geom,vector_row):
    state,name=refpath.parent.parent.name,refpath.parent.name
    dest=OUT/state/name; dest.mkdir(parents=True,exist_ok=True)
    with rasterio.open(refpath) as ref:
        if ref.crs is None or not ref.crs.is_projected or not np.allclose(ref.res,(30,30)) or ref.transform.b!=0 or ref.transform.d!=0:
            raise RuntimeError('Unexpected reference grid')
        geom=polygonal(district.to_crs(ref.crs).iloc[0])
        # Select regional candidates in their original CRS, retain full geometries.
        # Include a small margin for boundary/reprojection effects (projected units).
        hits=source.sindex.query(source_geom.buffer(1000),predicate='intersects')
        selected=source.iloc[hits].to_crs(ref.crs).copy()
        repairs=int((~selected.geometry.is_valid).sum())
        selected.geometry=[polygonal(g) for g in selected.geometry]
        si=selected.sindex
        print(f'Candidates: {len(selected)} | projected geometry repairs: {repairs}')
        print(f'Grid: {ref.width} x {ref.height} | {ref.crs} | 30 m')
        path=dest/'lithology.tif'
        profile=dict(driver='GTiff',height=ref.height,width=ref.width,count=1,dtype='uint8',crs=ref.crs,transform=ref.transform,nodata=0,compress='deflate',tiled=True,blockxsize=256,blockysize=256,BIGTIFF='IF_SAFER')
        generated=np.zeros(19,dtype=np.int64)
        total_blocks=math.ceil(ref.width/BLOCK)*math.ceil(ref.height/BLOCK)
        with tempfile.TemporaryDirectory(dir=dest) as td:
            tmp=Path(td)/'lithology.tif'
            with rasterio.open(tmp,'w',**profile) as dst:
                dst.set_band_description(1,'GLiM level-1 lithology categorical IDs')
                dst.update_tags(source='GLiM v1.1',method='pixel centre; distinct class overlaps flagged',code_16='source nd; unknown lithology',code_17='unmapped inside district',code_18='conflicting classes',code_0='outside district; nodata',class_lookup='glim_class_lookup.csv')
                for i,win in enumerate(windows(ref.width,ref.height),1):
                    tr=ref.window_transform(win); shape=(int(win.height),int(win.width))
                    inside=geometry_mask([geom],out_shape=shape,transform=tr,invert=True,all_touched=False)
                    groups={}
                    if inside.any():
                        for j in si.query(box(*bounds(win,ref.transform)),predicate='intersects'):
                            row=selected.iloc[j]
                            groups.setdefault(int(row.class_id),[]).append(row.geometry)
                    a=classify(inside,tr,groups)
                    generated+=np.bincount(a.ravel(),minlength=19)
                    dst.write(a,1,window=win)
                    if i%250==0 or i==total_blocks: print(f'  Raster blocks: {i}/{total_blocks}',flush=True)
            # Independent readback of masks, codes and joint elevation validity.
            counts=np.zeros(19,dtype=np.int64); joint=0; inside_total=0
            with rasterio.open(tmp) as src:
                if (src.shape,src.transform,src.crs,src.dtypes[0],src.nodata)!=(ref.shape,ref.transform,ref.crs,'uint8',0):
                    raise RuntimeError('Output metadata mismatch')
                for win in windows(ref.width,ref.height):
                    a=src.read(1,window=win)
                    inside=geometry_mask([geom],out_shape=a.shape,transform=ref.window_transform(win),invert=True,all_touched=False)
                    if (a>18).any() or not np.array_equal(a!=0,inside): raise RuntimeError('Invalid class or boundary mask')
                    counts+=np.bincount(a.ravel(),minlength=19)
                    elev=ref.read(1,window=win,masked=True)
                    ev=~np.ma.getmaskarray(elev)&np.isfinite(elev.data)&(elev.data!=-9999)
                    joint+=int(((a>=1)&(a<=15)&ev).sum())
                    inside_total+=int(inside.sum())
            if not np.array_equal(counts,generated) or not inside_total: raise RuntimeError('Pixel count mismatch or empty district')
            pct=lambda n: 100*int(n)/inside_total
            known=pct(counts[1:16].sum()); nd=pct(counts[16]); gap=pct(counts[17]); conflict=pct(counts[18])
            vector=float(vector_row.known_coverage_pct)
            delta=known-vector
            # Area vs centre-pixel percent is diagnostic, not exact equality.
            status='PASS' if known>=95 and gap<=0.1 and conflict<=0.1 and abs(delta)<=0.5 else 'REVIEW'
            rec=dict(state=state,district=name,district_pixels=inside_total,known_pixels=int(counts[1:16].sum()),known_coverage_pct=known,source_nd_pct=nd,unmapped_pct=gap,conflict_pct=conflict,joint_elevation_pct=pct(joint),vector_known_pct=vector,known_difference_pp=delta,projected_geometry_repairs=repairs,status=status)
            for k,v in rec.items(): print(f'{k}: {v}')
            print('\nCLASS DISTRIBUTION (inside district)')
            classes=pd.DataFrame([dict(class_id=i,code=LABELS[i],pixels=int(counts[i]),percentage=pct(counts[i])) for i in range(1,19)])
            print(classes[classes.pixels>0].to_string(index=False))
            # Commit only after successful readback; original sources unchanged.
            tmp.replace(path)
            classes.to_csv(dest/'lithology_class_pixels.csv',index=False)
            pd.DataFrame([rec]).to_csv(dest/'lithology_pilot_audit.csv',index=False)
            print(f'Output: {dest}\nReadback metadata and mask checks: PASS\nPilot coverage status: {status}')
            return status


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--district',default='UK/Haridwar'); p.add_argument('--self-test',action='store_true'); args=p.parse_args()
    smoke_tests()
    if args.self_test: return 0
    key=tuple(norm(x) for x in args.district.replace('\\','/').split('/'))
    refs=[r for r in REFS.glob('*/*/elevation.tif') if (norm(r.parent.parent.name),norm(r.parent.name))==key]
    d=gpd.read_file(BOUNDARY,engine='pyogrio')
    match=d[(d.state.map(norm)==key[0])&(d.district.map(norm)==key[1])] if len(key)==2 else d.iloc[:0]
    if len(refs)!=1 or len(match)!=1: raise RuntimeError('Reference or district match failed')
    geo=gpd.read_file(GEO/'glim_regional_repaired.gpkg',layer='geology',engine='pyogrio')
    lookup=pd.read_csv(GEO/'glim_class_lookup.csv')
    expected={i+1:c for i,c in enumerate(CODES)}
    if dict(zip(lookup.class_id,lookup.code))!=expected or not (geo.class_id.map(expected)==geo.xx_normalized).all(): raise RuntimeError('Class mapping mismatch')
    if len(geo)!=10815 or geo.source_fid.duplicated().any() or not geo.geometry.is_valid.all() or geo.geometry.is_empty.any(): raise RuntimeError('Unexpected repaired source inventory')
    vec=pd.read_csv(GEO/'glim_district_coverage_audit.csv')
    v=vec[(vec.state_code.map(norm)==key[0])&(vec.district.map(norm)==key[1])]
    if len(v)!=1: raise RuntimeError('Vector audit match failed')
    with rasterio.Env(GDAL_CACHEMAX=128*1024**2):
        status=run(refs[0],match.geometry,geo,polygonal(match.to_crs(geo.crs).geometry.iloc[0]),v.iloc[0])
    return 0 if status=='PASS' else 2

if __name__=='__main__': raise SystemExit(main())
