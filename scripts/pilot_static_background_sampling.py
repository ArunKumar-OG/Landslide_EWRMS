"""Pilot only: district-matched background candidates, never confirmed absences.
Uses previous audit helpers. Original data is read-only. No final labels.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import rasterio
from rasterio.windows import Window
from pyproj import Transformer
import shapely
from shapely.geometry import box, Point
from audit_footprint_grid_batch import ROOT, paths, masks, FEATURES, load_districts, read, export, marker, require

PILOTS=[('UK','Haridwar'),('HP','Chamba'),('LK','Leh Ladakh'),('AS','West Karbi Anglong')]
BUFFER=300.0
SEED=42
DIST=['state','district']
COLS=['background_id','state','district','row','col','epsg','x','y','longitude','latitude','exclusion_buffer_m','sample_kind','approved_training_label']


def district_seed(s,d,seed):
    return int.from_bytes(hashlib.sha256(f'{seed}|{s}|{d}'.encode()).digest()[:8],'little')


def local_observations(gpkg, repaired, positives, district, crs, buffer):
    # Do not clip the inventory to a district: adjacent observations matter.
    to_ll=Transformer.from_crs(crs,4326,always_xy=True)
    a,b,c,d=district.bounds
    ll=to_ll.transform_bounds(a-buffer,b-buffer,c+buffer,d+buffer,densify_pts=101)
    bbox=(ll[0]-.01,ll[1]-.01,ll[2]+.01,ll[3]+.01)
    geoms=[]; counts={}
    for p,layer in [(gpkg,'inventory_points'),(gpkg,'inventory_polygons'),(repaired,'review_candidates')]:
        g=pyogrio.read_dataframe(p,layer=layer,bbox=bbox,columns=[])
        require(g.crs is not None and g.crs.to_epsg()==4326,f'{layer}: CRS mismatch')
        require(g.geometry.notna().all() and not g.geometry.is_empty.any() and g.geometry.is_valid.all(),f'{layer}: invalid source geometry')
        g=g.to_crs(crs); require(g.geometry.is_valid.all() and not g.geometry.is_empty.any(),f'{layer}: invalid projected geometry')
        geoms.extend(g.geometry.tolist()); counts[layer]=len(g)
    p=positives[positives.longitude.between(bbox[0],bbox[2]) & positives.latitude.between(bbox[1],bbox[3])]
    if len(p):
        x,y=Transformer.from_crs(4326,crs,always_xy=True).transform(p.longitude.to_numpy(),p.latitude.to_numpy())
        geoms.extend(shapely.points(x,y).tolist())
    counts['selected_positive_centres']=len(p)
    require(all(g.is_valid and not g.is_empty and np.isfinite(g.bounds).all() for g in geoms),'Bad local exclusion geometry')
    return shapely.STRtree(geoms),counts


def excluded(tree,x,y,buffer):
    result=np.zeros(len(x),dtype=bool)
    if len(tree.geometries) and len(x):
        pairs=tree.query(shapely.points(x,y),predicate='dwithin',distance=buffer)
        result[pairs[0]]=True
    return result


def feature_ok(srcs,rr,cc,height,width,block=256):
    # Read small windows only for spatially admissible random proposals.
    groups=defaultdict(list)
    for i,(r,c) in enumerate(zip(rr,cc)):
        groups[(int(r)//block,int(c)//block)].append(i)
    ok=np.zeros(len(rr),bool)
    for (br,bc),idx in groups.items():
        r0=br*block; c0=bc*block
        w=Window(c0,r0,min(block,width-c0),min(block,height-r0))
        inside=np.ones((int(w.height),int(w.width)),bool)
        _,joint=masks(srcs,w,inside)
        eligible=joint['all18_known'] & joint['core14_land']
        ii=np.asarray(idx); ok[ii]=eligible[rr[ii]-r0,cc[ii]-c0]
    return ok


def sample_grid(ref,srcs,district,tree,target,s,d,buffer=BUFFER,seed=SEED,max_draws=None):
    require(target>=0 and buffer>0,'Invalid sampling parameters')
    require(ref.crs is not None and ref.crs.is_projected and ref.crs.linear_units.lower() in ('metre','meter','metres','meters') and np.allclose(ref.res,(30,30)) and ref.transform.b==ref.transform.d==0 and ref.transform.a>0 and ref.transform.e<0,'Expected metric north-up 30 m grid')
    require(set(srcs)==set(FEATURES),'All 18 features required')
    for k,a in srcs.items():
        require(a.crs==ref.crs and a.shape==ref.shape and a.transform.almost_equals(ref.transform,precision=1e-7) and a.count==1,f'{k}: grid mismatch')
    epsg=ref.crs.to_epsg(); require(epsg is not None,'Missing EPSG')
    rng=np.random.default_rng(district_seed(s,d,seed)); total=ref.width*ref.height
    limit=min(total,max_draws if max_draws is not None else max(10000,target*100))
    # A uniformly random sequence of distinct flat cell indices; rejection sampling.
    seen=set(); rows=[]; stats=Counter(); trans=ref.transform
    to_ll=Transformer.from_crs(ref.crs,4326,always_xy=True)
    while len(rows)<target and len(seen)<limit:
        n=min(512,limit-len(seen)); batch=[]
        # As grid occupancy rises, explicitly draw from remaining cells (small-grid tests).
        if len(seen)>total//2:
            remaining=np.fromiter((i for i in range(total) if i not in seen),dtype=np.int64)
            batch=rng.choice(remaining,size=min(n,len(remaining)),replace=False).tolist(); seen.update(batch)
        else:
            while len(batch)<n:
                for v in rng.integers(0,total,size=max(16,n-len(batch))):
                    v=int(v)
                    if v not in seen:
                        seen.add(v); batch.append(v)
                        if len(batch)==n: break
        flat=np.asarray(batch,dtype=np.int64); rr=flat//ref.width; cc=flat%ref.width
        x=trans.c+(cc+.5)*30; y=trans.f-(rr+.5)*30
        inside=shapely.contains_xy(district,x,y)
        blocked=np.zeros(len(rr),bool); blocked[inside]=excluded(tree,x[inside],y[inside],buffer)
        idx=np.flatnonzero(inside & ~blocked); ok=np.zeros(len(rr),bool)
        ok[idx]=feature_ok(srcs,rr[idx],cc[idx],ref.height,ref.width)
        for i in range(len(batch)):
            if len(rows)>=target: break
            stats['proposals_evaluated']+=1
            if not inside[i]: stats['outside_district']+=1; continue
            if blocked[i]: stats['inventory_buffer_rejected']+=1; continue
            if not ok[i]: stats['feature_or_land_rejected']+=1; continue
            lon,lat=to_ll.transform(float(x[i]),float(y[i]))
            rows.append(dict(background_id=f'BG_{s}_{d.replace(" ","_")}_{rr[i]}_{cc[i]}',state=s,district=d,row=int(rr[i]),col=int(cc[i]),epsg=epsg,x=float(x[i]),y=float(y[i]),longitude=lon,latitude=lat,exclusion_buffer_m=buffer,sample_kind='BACKGROUND_UNVERIFIED',approved_training_label=False))
    f=pd.DataFrame(rows,columns=COLS)
    require(not f.duplicated(['row','col']).any(),'Duplicate sampled cell')
    if len(f):
        require(shapely.contains_xy(district,f.x.to_numpy(),f.y.to_numpy()).all(),'Outside selected')
        require(not excluded(tree,f.x.to_numpy(),f.y.to_numpy(),buffer).any(),'Buffer violation')
        require(feature_ok(srcs,f.row.to_numpy(),f.col.to_numpy(),ref.height,ref.width).all(),'Feature readback failure')
    require(stats['proposals_evaluated']==len(f)+sum(stats[k] for k in ('outside_district','inventory_buffer_rejected','feature_or_land_rejected')),'Proposal accounting mismatch')
    report=dict(state=s,district=d,target=target,selected=len(f),shortfall=target-len(f),status='SKIP_NO_BASELINE' if target==0 else ('PASS' if len(f)==target else 'REVIEW_SHORTFALL'),**{k:int(stats[k]) for k in ('proposals_evaluated','outside_district','inventory_buffer_rejected','feature_or_land_rejected')})
    return f,report


def run(root,out):
    base=root/'data/processed/landslides'; p=base/'static_positive_sampling_batch_v1'; u=base/'unified_inventory_v1'
    m=marker(p/'complete.json'); um=marker(u/'complete.json')
    require(m['status']=='PASS' and m['scope']=='ALL_DISTRICTS' and m['center_baseline_candidates']==41884 and m['overlap25_sensitivity_candidates']==6133 and m['approved_training_labels']==0,'Unexpected positive batch')
    require(um['status']=='PASS' and um['records']==128846,'Unified inventory mismatch')
    a=read(p/'center_baseline_candidates.csv'); b=read(p/'overlap25_sensitivity_candidates.csv'); allp=pd.concat([a,b],ignore_index=True)
    require(len(a)==41884 and len(b)==6133 and not allp.duplicated(['source','qa_record_id']).any() and not allp.duplicated(DIST+['row','col']).any(),'Positive count or uniqueness mismatch')
    require(a.method.eq('CENTER_BASELINE').all() and b.method.eq('OVERLAP_25_SENSITIVITY').all() and allp.approved_training_label.str.lower().eq('false').all(),'Positive flag mismatch')
    for k in ('longitude','latitude'): allp[k]=pd.to_numeric(allp[k],errors='raise')
    require(np.isfinite(allp[['longitude','latitude']]).all().all() and allp.longitude.between(-180,180).all() and allp.latitude.between(-90,90).all(),'Invalid positive coordinates')
    gpkg=u/'unified_landslide_inventory.gpkg'; repaired=base/'uglc_quality_v1/polygons/repaired_review_candidates.gpkg'
    for path,layer,n in [(gpkg,'inventory_points',35785),(gpkg,'inventory_polygons',93061),(repaired,'review_candidates',6)]:
        require(pyogrio.read_info(path,layer=layer)['features']==n,f'{layer}: source count mismatch')
    ds=load_districts(root); frames=[]; reports=[]
    for s,d in PILOTS:
        district=ds[(ds.state==s)&(ds.district==d)]; require(len(district)==1,'Missing district')
        target=int(((a.state==s)&(a.district==d)).sum())
        refpath,files=paths(root,s,d)
        with ExitStack() as stack:
            ref=stack.enter_context(rasterio.open(refpath)); srcs={k:stack.enter_context(rasterio.open(v)) for k,v in files.items()}
            projected=district.to_crs(ref.crs).geometry.iloc[0]; require(projected.is_valid,'Invalid projected district')
            tree,counts=local_observations(gpkg,repaired,allp,projected,ref.crs,BUFFER)
            f,r=sample_grid(ref,srcs,projected,tree,target,s,d); r.update(counts); frames.append(f); reports.append(r)
            print(json.dumps(r),flush=True)
    samples=pd.concat(frames,ignore_index=True); report=pd.DataFrame(reports)
    require(samples.background_id.is_unique and not samples.duplicated(['epsg','x','y']).any(),'Cross-district cell collision')
    require(int(report.selected.sum())==len(samples),'Selected accounting mismatch')
    export(out,'background_candidates.csv',samples); export(out,'district_background_audit.csv',report)
    meta=dict(status='PASS' if not report.shortfall.any() else 'REVIEW',scope='PILOT_ONLY',districts=len(PILOTS),target=int(report.target.sum()),selected=len(samples),shortfall=int(report.shortfall.sum()),seed=SEED,exclusion_buffer_m=BUFFER,approved_training_labels=0,confirmed_absences=0)
    (out/'complete.json').write_text(json.dumps(meta,indent=2),encoding='utf-8'); print(json.dumps(meta,indent=2))


def self_test():
    with TemporaryDirectory() as td:
        p=Path(td); transform=rasterio.transform.from_origin(500000,3300000,30,30); shape=(20,20)
        files={}
        for k in FEATURES:
            data=np.ones(shape,dtype='float32')
            if k=='landcover': data[:]=10; data[0,:]=80
            if k=='lithology': data[1,:]=16
            if k=='slope': data[2,:]=-9999
            path=p/(k+'.tif'); files[k]=path
            with rasterio.open(path,'w',driver='GTiff',height=20,width=20,count=1,dtype='float32',crs='EPSG:32643',transform=transform,nodata=-9999) as dst: dst.write(data,1)
        district=box(500000,3299400,500600,3300000); tree=shapely.STRtree([Point(500300,3299700),box(500000,3299400,500100,3299500)])
        with ExitStack() as stack:
            srcs={k:stack.enter_context(rasterio.open(v)) for k,v in files.items()}; ref=srcs['elevation']
            f,r=sample_grid(ref,srcs,district,tree,50,'T','Test',buffer=60)
            g,s=sample_grid(ref,srcs,district,tree,50,'T','Test',buffer=60)
            require(f.equals(g) and r==s and len(f)==50,'Determinism or target failure')
            require((f.row>=3).all() and not excluded(tree,f.x.to_numpy(),f.y.to_numpy(),60).any(),'Invalid sample')
            f0,r0=sample_grid(ref,srcs,district,tree,0,'T','Test',buffer=60); require(f0.empty and r0['status']=='SKIP_NO_BASELINE','Zero target failed')
            f1,r1=sample_grid(ref,srcs,district,shapely.STRtree([district]),5,'T','Test',buffer=60); require(f1.empty and r1['shortfall']==5 and r1['status']=='REVIEW_SHORTFALL','Shortfall test failed')
            export(p,'test.csv',f)
    print('Synthetic deterministic sampling, geometry buffer, land/geology/NoData, zero target, shortfall and CSV readback: PASS')


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=ROOT); ap.add_argument('--self-test',action='store_true'); args=ap.parse_args()
    if args.self_test: self_test(); return
    out=args.root/'data/processed/landslides/static_background_sampling_pilot_v1'; stage=out.with_name(out.name+'.staging')
    require(not out.exists() and not stage.exists(),'Output or staging exists; preserve or rename it before rerunning')
    stage.mkdir(parents=True); run(args.root,stage); stage.rename(out); print('Output:',out,'| Inputs unchanged; no confirmed absences or approved labels.')

if __name__=='__main__': main()
