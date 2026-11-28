"""Propose one footprint-supported cell per priority footprint; no approved labels.
Requires audit_footprint_grid_batch.py and its existing dependencies.
All districts. Separate centre baseline from small-footprint overlap sensitivity.
Uses exactly the pilot collection and deterministic assignment rules.
"""
import argparse
from collections import defaultdict
from contextlib import ExitStack
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import rasterio
from rasterio.features import geometry_mask, rasterize
from rasterio.windows import Window, bounds
from pyproj import Transformer
import shapely
from shapely.geometry import box
from audit_footprint_grid_batch import (ROOT, PRIORITY, FIELDS, DIST, KEY, polygon_parts,
    load_districts, read, export, marker, require, paths, masks, FEATURES)

PILOTS=[('UK','Haridwar'),('HP','Chamba'),('LK','Leh Ladakh'),('AS','West Karbi Anglong')]
THRESHOLD=0.25


def collect(refpath, files, district, f, state, name, global_no_center, block=512):
    """Candidate rows: key -> [(fraction,state,district,row,col,epsg,x,y,lon,lat)]."""
    require(not f.duplicated(KEY).any() and f.crs.to_epsg()==4326,'Footprint key or CRS mismatch')
    require(set(files)==set(FEATURES),'Missing feature paths')
    center=defaultdict(list); fallback=defaultdict(list)
    counts=np.zeros((len(f),4),dtype=np.int64)
    keys=list(f[KEY].itertuples(index=False,name=None))
    with ExitStack() as stack:
        ref=stack.enter_context(rasterio.open(refpath)); srcs={k:stack.enter_context(rasterio.open(p)) for k,p in files.items()}
        require(ref.crs is not None and ref.crs.is_projected and ref.crs.linear_units.lower() in ('metre','meter','metres','meters') and np.allclose(ref.res,(30,30)) and ref.transform.b==ref.transform.d==0 and ref.transform.a>0 and ref.transform.e<0,'Expected metric north-up 30 m grid')
        for k,s in srcs.items():
            require(s.crs==ref.crs and s.shape==ref.shape and s.transform.almost_equals(ref.transform,precision=1e-7) and s.count==1,f'{k} grid mismatch')
        epsg=ref.crs.to_epsg(); require(epsg is not None,'Missing EPSG')
        to_ll=Transformer.from_crs(ref.crs,4326,always_xy=True)
        d=gpd.GeoSeries([district],crs=4326).to_crs(ref.crs).iloc[0]
        fproj=f.to_crs(ref.crs); require(d.is_valid and not d.is_empty and fproj.geometry.is_valid.all(),'Invalid projected geometry')
        geoms=[polygon_parts(g.intersection(d)) for g in fproj.geometry]; require(all(g.is_valid for g in geoms),'Invalid clipped geometry')
        tree=shapely.STRtree(geoms)
        for r in range(0,ref.height,block):
            for c in range(0,ref.width,block):
                w=Window(c,r,min(block,ref.width-c),min(block,ref.height-r)); tr=ref.window_transform(w); shape=(int(w.height),int(w.width))
                hits=tree.query(box(*bounds(w,ref.transform)),predicate='intersects')
                if not len(hits): continue
                inside=geometry_mask([d],out_shape=shape,transform=tr,invert=True,all_touched=False)
                if not inside.any(): continue
                _,joint=masks(srcs,w,inside)
                land_full=joint['all18_known']&joint['core14_land']
                for i in hits:
                    g=geoms[i]; key=keys[i]
                    if g.is_empty or g.area<=0: continue
                    x0,y0,x1,y1=g.bounds; inv=~tr; a,b=inv*(x0,y1); z,t=inv*(x1,y0)
                    c0=max(0,int(np.floor(a))-1); r0=max(0,int(np.floor(b))-1); c1=min(shape[1],int(np.ceil(z))+1); r1=min(shape[0],int(np.ceil(t))+1)
                    if c1<=c0 or r1<=r0: continue
                    sl=(slice(r0,r1),slice(c0,c1)); local=tr*rasterio.Affine.translation(c0,r0); sh=(r1-r0,c1-c0)
                    ce=rasterize([(g,1)],out_shape=sh,transform=local,dtype='uint8',all_touched=False).astype(bool)&inside[sl]
                    counts[i,0]+=int(ce.sum()); counts[i,1]+=int((ce&joint['all18_known'][sl]).sum()); counts[i,2]+=int((ce&joint['core14_land'][sl]).sum()); counts[i,3]+=int((ce&land_full[sl]).sum())
                    options=ce&land_full[sl]; target=center
                    if key in global_no_center:
                        require(not ce.any(),'Global no-centre flag contradicted by geometry')
                        options=rasterize([(g,1)],out_shape=sh,transform=local,dtype='uint8',all_touched=True).astype(bool)&inside[sl]&land_full[sl]; target=fallback
                    rr,cc=np.nonzero(options)
                    if not len(rr): continue
                    left=local.c+cc*30; top=local.f-rr*30
                    cells=shapely.box(left,top-30,left+30,top)
                    fractions=np.clip(shapely.area(shapely.intersection(cells,g))/900,0,1)
                    xx=left+15; yy=top-15; lon,lat=to_ll.transform(xx,yy)
                    for j in range(len(rr)):
                        frac=float(fractions[j]);
                        if frac<=0 or (target is fallback and frac<THRESHOLD): continue
                        target[key].append((frac,state,name,int(r+r0+rr[j]),int(c+c0+cc[j]),epsg,float(xx[j]),float(yy[j]),float(lon[j]),float(lat[j])))
    detail=f[KEY].copy()
    for j,k in enumerate(('center_pixels','all18_known_pixels','core14_land_pixels','all18_land_pixels')): detail[k]=counts[:,j]
    return center,fallback,detail


def choose(center, fallback, keys):
    """Deterministic greedy assignment, constrained footprints first. No optimality claim."""
    used=set(); selections=[]; flags=[]
    for method,options in [('CENTER_BASELINE',center),('OVERLAP_25_SENSITIVITY',fallback)]:
        for key in sorted(options,key=lambda k:(len(options[k]),k)):
            require(key in keys,'Unknown candidate key')
            opts=sorted(options[key],key=lambda o:(-o[0],o[1],o[2],o[3],o[4]))
            # district grid coordinates are the identity; exact projected centre collision is also prevented.
            for o in opts:
                frac,s,d,r,c,epsg,x,y,lon,lat=o
                identifiers=((s,d,r,c),('metric',epsg,round(x,6),round(y,6)))
                if any(i in used for i in identifiers): continue
                used.update(identifiers)
                selections.append(dict(zip(KEY,key),method=method,overlap_fraction=frac,state=s,district=d,row=r,col=c,epsg=epsg,x=x,y=y,longitude=lon,latitude=lat,eligible_cell_options=len(opts),approved_training_label=False))
                break
    selected={tuple(s[k] for k in KEY):s['method'] for s in selections}
    require(len(selected)==len(selections),'More than one sample per footprint')
    for key in sorted(keys):
        status=selected.get(key,'CELL_COLLISION_REVIEW' if key in center or key in fallback else 'NO_ELIGIBLE_CELL')
        flags.append(dict(zip(KEY,key),status=status,center_options=len(center.get(key,[])),fallback_options=len(fallback.get(key,[]))))
    return selections,flags


def run(root,out,chosen):
    base=root/'data/processed/landslides'; u=base/'unified_inventory_v1'; a=base/'footprint_feature_support_batch_v1'
    m=marker(a/'complete.json'); require(m['status']=='PASS' and m['scope']=='ALL_DISTRICTS' and m['priority_distinct_footprints']==62609 and m['approved_training_labels']==0,'Full audit required')
    distinct=read(a/'distinct_footprint_support.csv'); detail=read(a/'footprint_feature_support.csv'); districts=load_districts(root)
    require(len(distinct)==72659 and not distinct.duplicated(KEY).any() and len(detail)==m['footprint_district_links'] and not detail.duplicated(KEY+DIST).any(),'Audit key/count mismatch')
    pri=distinct[distinct.static_queue.eq(PRIORITY)]; require(len(pri)==62609,'Priority count mismatch')
    global_no_center=set(pri.loc[pri.center_pixels.astype(int).eq(0),KEY].itertuples(index=False,name=None))
    fps=pyogrio.read_dataframe(u/'unified_landslide_inventory.gpkg',layer='static_footprints',columns=FIELDS)
    fps=fps[fps.static_queue.eq(PRIORITY)]; require(len(fps)==62609 and not fps.duplicated(KEY).any() and set(fps[KEY].itertuples(index=False,name=None))==set(pri[KEY].itertuples(index=False,name=None)),'Geometry inventory mismatch')
    require(len(chosen)==len(set(chosen)) and set(chosen)<=set(districts[DIST].itertuples(index=False,name=None)),'Unknown/repeated district')
    require(len(chosen)==137,'All 137 districts required')
    pilot=base/'static_positive_sampling_pilot_v1'
    pm=marker(pilot/'complete.json')
    require(pm['status']=='PASS' and pm['scope']=='PILOT_ONLY' and pm['districts']==4 and pm['priority_distinct_footprints']==3196 and pm['center_baseline_candidates']==2220 and pm['overlap25_sensitivity_candidates']==218 and pm['approved_training_labels']==0 and pm['threshold_fraction']==THRESHOLD,'Expected successful pilot required')
    previous=read(pilot/'district_sampling_audit.csv')
    require(len(previous)==4 and not previous.duplicated(DIST).any() and set(previous[DIST].itertuples(index=False,name=None))==set(PILOTS) and previous.audit_crosscheck.eq('PASS').all(),'Pilot audit mismatch')
    center=defaultdict(list); fallback=defaultdict(list); keys=set(); checks=[]
    for ix,(s,d) in enumerate(sorted(chosen),1):
        expected=detail[(detail.state==s)&(detail.district==d)&detail.static_queue.eq(PRIORITY)]
        f=fps.merge(expected[KEY],on=KEY,validate='one_to_one'); require(len(f)==len(expected),'District geometry mismatch')
        ds=districts[(districts.state==s)&(districts.district==d)]; require(len(ds)==1,'District missing')
        ref,files=paths(root,s,d); ce,fb,actual=collect(ref,files,ds.geometry.iloc[0],f,s,d,global_no_center)
        compared=actual.merge(expected[KEY+['center_pixels','all18_known_pixels','core14_land_pixels']],on=KEY,validate='one_to_one',suffixes=('_new','_old'))
        require(len(compared)==len(expected),'Audit join mismatch')
        for k in ('center_pixels','all18_known_pixels','core14_land_pixels'):
            require(compared[k+'_new'].eq(compared[k+'_old'].astype(int)).all(),f'{s}/{d}: audit mismatch {k}')
        for k,v in ce.items(): center[k].extend(v)
        for k,v in fb.items(): fallback[k].extend(v)
        keys.update(f[KEY].itertuples(index=False,name=None))
        checks.append(dict(state=s,district=d,priority_links=len(f),center_eligible_links=len(ce),fallback_eligible_links=len(fb),all18_land_pixel_memberships=int(actual.all18_land_pixels.sum()),audit_crosscheck='PASS'))
        pr=previous[(previous.state==s)&(previous.district==d)]
        checks[-1]['pilot_crosscheck']='NOT_PILOT'
        if len(pr):
            for k in ('priority_links','center_eligible_links','fallback_eligible_links','all18_land_pixel_memberships'):
                require(checks[-1][k]==int(pr.iloc[0][k]),f'{s}/{d}: sampling pilot discrepancy {k}')
            checks[-1]['pilot_crosscheck']='PASS'
        print(f'[{ix}/{len(chosen)}] {s}/{d}',checks[-1],flush=True)
    require(keys==set(pri[KEY].itertuples(index=False,name=None)), 'Regional footprint accounting mismatch')
    require(sum(c['priority_links'] for c in checks)==int(detail.static_queue.eq(PRIORITY).sum()),'Priority district link accounting mismatch')
    require(sum(c['pilot_crosscheck']=='PASS' for c in checks)==4,'Pilot comparison incomplete')
    require(not (set(center)&set(fallback)) and set(fallback)<=global_no_center,'Baseline and fallback footprint overlap or incorrect fallback')
    # These audited input reports showed identical all18-known and core14-known membership counts.
    require(pri.all18_known_pixels.astype(int).eq(pri.core14_known_pixels.astype(int)).all(),'Full/core mask discrepancy; revisit expected support check')
    require(set(center)==set(pri.loc[pri.core14_land_pixels.astype(int).gt(0),KEY].itertuples(index=False,name=None)), 'Centre eligible footprints disagree with regional audit')
    selections,flags=choose(center,fallback,keys); cols=KEY+['method','overlap_fraction','state','district','row','col','epsg','x','y','longitude','latitude','eligible_cell_options','approved_training_label']
    samples=pd.DataFrame(selections,columns=cols); fl=pd.DataFrame(flags,columns=KEY+['status','center_options','fallback_options'])
    require(not samples.duplicated(DIST+['row','col']).any() and not samples.duplicated(KEY).any(),'Duplicate sample')
    require(not samples.assign(x=samples.x.round(6),y=samples.y.round(6)).duplicated(['epsg','x','y']).any(),'Duplicate metric centre')
    require(np.isfinite(samples[['overlap_fraction','x','y','longitude','latitude']].to_numpy(dtype=float)).all(),'Nonfinite candidate values')
    require(samples.overlap_fraction.gt(0).all() and samples.overlap_fraction.le(1).all() and samples.loc[samples.method.eq('OVERLAP_25_SENSITIVITY'),'overlap_fraction'].ge(THRESHOLD).all(),'Overlap fraction mismatch')
    require(len(fl)==62609 and not fl.duplicated(KEY).any() and not samples.approved_training_label.any(),'Final accounting mismatch')
    district_summary=pd.DataFrame(checks)
    for method,col in [('CENTER_BASELINE','baseline_selected'),('OVERLAP_25_SENSITIVITY','overlap25_selected')]:
        counts=samples[samples.method.eq(method)].groupby(DIST).size().rename(col).reset_index()
        district_summary=district_summary.merge(counts,on=DIST,how='left',validate='one_to_one')
        district_summary[col]=district_summary[col].fillna(0).astype(int)
    state_summary=district_summary.groupby('state',as_index=False).agg(districts=('district','size'),districts_with_baseline=('baseline_selected',lambda v:int(v.gt(0).sum())),districts_with_any_selected=('baseline_selected','size'),baseline_selected=('baseline_selected','sum'),overlap25_selected=('overlap25_selected','sum'))
    any_counts=district_summary.assign(any_selected=(district_summary.baseline_selected+district_summary.overlap25_selected).gt(0)).groupby('state').any_selected.sum()
    state_summary['districts_with_any_selected']=state_summary.state.map(any_counts).astype(int)
    export(out,'state_sampling_summary.csv',state_summary)
    # CSV export helper compares stringified values; reset indices after selection.
    for method,fn in [('CENTER_BASELINE','center_baseline_candidates.csv'),('OVERLAP_25_SENSITIVITY','overlap25_sensitivity_candidates.csv')]:
        export(out,fn,samples[samples.method.eq(method)].reset_index(drop=True))
    export(out,'footprint_sampling_status.csv',fl); export(out,'district_sampling_audit.csv',district_summary)
    result=dict(status='PASS',scope='ALL_DISTRICTS',districts=len(chosen),priority_distinct_footprints=len(keys),priority_no_center=len(global_no_center),center_eligible_distinct=len(center),fallback_eligible_distinct=len(fallback),pilot_crosschecks=4,districts_with_baseline=int(district_summary.baseline_selected.gt(0).sum()),districts_with_any_candidates=int((district_summary.baseline_selected+district_summary.overlap25_selected).gt(0).sum()),center_baseline_candidates=int(samples.method.eq('CENTER_BASELINE').sum()),overlap25_sensitivity_candidates=int(samples.method.eq('OVERLAP_25_SENSITIVITY').sum()),status_counts=fl.status.value_counts().to_dict(),approved_training_labels=0,threshold_fraction=THRESHOLD)
    print(json.dumps(result,indent=2)); (out/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


def self_test():
    from rasterio.transform import from_origin
    with TemporaryDirectory() as td:
        p=Path(td); tr=from_origin(500000,120,30,30); crs='EPSG:32643'; files={k:p/(k+'.tif') for k in FEATURES}
        for k,fn in files.items():
            arr=np.ones((4,4),np.float32)
            if k=='landcover': arr[:]=10; arr[0,0]=80
            if k=='lithology': arr[0,1]=16
            if k=='aspect': arr[1,0]=-9999
            with rasterio.open(fn,'w',driver='GTiff',height=4,width=4,count=1,dtype='float32',crs=crs,transform=tr,nodata=-9999) as dst: dst.write(arr,1)
        d=gpd.GeoSeries([box(500000,0,500120,120)],crs=crs).to_crs(4326).iloc[0]
        # Footprints 0 and 1 overlap the same valid cell; 2 has no centre but >=25% overlap, 3 below threshold.
        geoms=[box(500031,31,500059,59),box(500032,32,500058,58),box(500061,1,500074,21),box(500091,1,500099,9)]
        f=gpd.GeoDataFrame([dict(source='test',qa_record_id=str(i)) for i in range(4)],geometry=geoms,crs=crs).to_crs(4326)
        keys=set(f[KEY].itertuples(index=False,name=None)); ce,fb,a=collect(files['elevation'],files,d,f,'T','Test',{('test','2'),('test','3')},block=2)
        assert a.center_pixels.tolist()==[1,1,0,0] and set(ce)=={('test','0'),('test','1')} and set(fb)=={('test','2')}
        ss,flags=choose(ce,fb,keys); assert len(ss)==2 and {x['method'] for x in ss}=={'CENTER_BASELINE','OVERLAP_25_SENSITIVITY'} and sum(x['status']=='CELL_COLLISION_REVIEW' for x in flags)==1
        assert choose(ce,fb,keys)==(ss,flags)
        export(p,'test.csv',pd.DataFrame(ss))
        # Explicit water, unknown lithology and aspect NoData are excluded.
        f2=gpd.GeoDataFrame([dict(source='test',qa_record_id=str(i)) for i in range(3)],geometry=[box(500001,91,500029,119),box(500031,91,500059,119),box(500001,61,500029,89)],crs=crs).to_crs(4326)
        ce,fb,a=collect(files['elevation'],files,d,f2,'T','Test',set(),block=2)
        assert not ce and not fb and a.center_pixels.tolist()==[1,1,1] and a.all18_land_pixels.tolist()==[0,0,0]
    print('Synthetic centre, overlap threshold, collisions, deterministic selection, water, unknown geology, NoData and CSV readback: PASS')


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--root',type=Path,default=ROOT); ap.add_argument('--self-test',action='store_true'); ap.add_argument('--district',action='append',help='STATE/District (repeatable)'); args=ap.parse_args()
    if args.self_test: self_test(); return
    require(not args.district,'This batch always runs all districts; use the original pilot script for subsets')
    chosen=list(load_districts(args.root)[DIST].itertuples(index=False,name=None))
    out=args.root/'data/processed/landslides/static_positive_sampling_batch_v1'; stage=out.with_name(out.name+'.staging')
    require(not out.exists() and not stage.exists(),'Output or staging exists; preserve/inspect before retrying')
    stage.mkdir(); run(args.root,stage,chosen); stage.rename(out); print('Output:',out,'| No training labels approved; original files unchanged.')

if __name__=='__main__': main()
