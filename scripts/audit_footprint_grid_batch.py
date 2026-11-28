"""Read-only all-district footprint and static-feature compatibility audit.
No labels, feature samples or raster changes. Requires audit_footprint_grid_pilot.py.
"""
import argparse
import json
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import rasterio
from rasterio.features import geometry_mask, rasterize
from rasterio.windows import Window, bounds
import shapely
from shapely.geometry import box
from audit_footprint_grid_pilot import ROOT, PRIORITY, FIELDS, DIST, KEY, valid, polygon_parts, load_districts, read, export, marker, require

TERRAIN = ('elevation','slope','aspect','plan_curvature','profile_curvature','tangential_curvature','tri','tpi')
HYDRO = ('upa','log_upa','hand','hydro_slope','twi','spi','log_spi')
FEATURES = TERRAIN + HYDRO + ('distance_upa_5km2','lithology','landcover')
OPTIONAL = {'aspect','plan_curvature','profile_curvature','tangential_curvature'}
CORE = tuple(k for k in FEATURES if k not in OPTIONAL)
LC_CODES = [10,20,30,40,50,60,70,80,90,95,100]
METRICS = ('center','touch','elev_slope','all18_raw','all18_known','core14_known','core14_land')


def paths(root,s,d):
    p=root/'data/processed'; stem=d.replace(' ','_'); t=p/'terrain_batch/terrain_v2'/s/stem; h=p/'hydrology/batch_v1'/s/stem/'aligned'
    f={k:t/(k+'.tif') for k in TERRAIN}; f.update({k:h/(k+'.tif') for k in HYDRO})
    f.update(distance_upa_5km2=p/'hydrology/drainage_batch_v1'/s/stem/'distance_upa_5km2.tif',lithology=p/'geology/batch_v1'/s/stem/'lithology.tif',landcover=p/'landcover/batch_v1'/s/stem/'landcover.tif')
    return p/'DEM/district_dem_utm'/s/(stem+'.tif'),f


def masks(srcs,w,inside):
    vs={}; rawgeo=None; land=None
    for k,src in srcs.items():
        a=src.read(1,window=w); v=(src.read_masks(1,window=w)>0)&np.isfinite(a)&inside
        if src.nodata is not None: v &= a!=src.nodata
        if k=='lithology':
            require(not (v & ~np.isin(a,np.arange(1,19))).any(),'Unexpected geology class')
            rawgeo=v.copy(); v &= np.isin(a,np.arange(1,16))
        if k=='landcover':
            require(not (v & ~np.isin(a,LC_CODES)).any(),'Unexpected landcover class')
            land=v & ~np.isin(a,[70,80])
        vs[k]=v
    full=np.logical_and.reduce([vs[k] for k in FEATURES]); core=np.logical_and.reduce([vs[k] for k in CORE]); raw=np.logical_and.reduce([rawgeo if k=='lithology' else vs[k] for k in FEATURES])
    return vs,dict(elev_slope=vs['elevation']&vs['slope'],all18_raw=raw,all18_known=full,core14_known=core,core14_land=core&land)


def audit(refpath,files,district,fps,state,name,block=512):
    require(set(files)==set(FEATURES),'18 feature paths required')
    require(not fps.duplicated(KEY).any() and fps.crs is not None,'Bad footprint keys/CRS')
    with ExitStack() as stack:
        ref=stack.enter_context(rasterio.open(refpath)); srcs={k:stack.enter_context(rasterio.open(files[k])) for k in FEATURES}
        require(ref.crs is not None and ref.crs.is_projected and ref.crs.linear_units.lower() in ('metre','meter','metres','meters') and np.allclose(ref.res,(30,30)) and ref.transform.b==ref.transform.d==0 and ref.transform.a>0 and ref.transform.e<0,'Not north-up metric 30 m grid')
        for k,src in srcs.items():
            require(src.crs==ref.crs and src.shape==ref.shape and src.transform.almost_equals(ref.transform,precision=1e-7) and src.count==1,f'{state}/{name}: {k} grid mismatch')
        d=gpd.GeoSeries([district],crs=4326).to_crs(ref.crs).iloc[0]; f=fps.to_crs(ref.crs)
        require(d.is_valid and not d.is_empty and f.geometry.is_valid.all(),'Reprojection invalid')
        clipped=[polygon_parts(g.intersection(d)) for g in f.geometry]; require(all(g.is_valid for g in clipped),'Clipping invalid')
        area=np.array([g.area for g in clipped]); tree=shapely.STRtree(clipped); n=len(f); pri=f.static_queue.eq(PRIORITY).to_numpy()
        counts={k:np.zeros(n,dtype=np.int64) for k in METRICS}; maxima={k:np.zeros(n,dtype=float) for k in ('any','core14_known','all18_known')}
        # Maxima are computed only in blocks where this footprint has no centre.
        # They are reported ONLY for footprints with zero centres over the entire district.
        totals={p+'_'+k:0 for p in ('all','priority') for k in METRICS+('overlap_center',)}
        feature_counts={p:{k:0 for k in FEATURES} for p in ('all','priority')}; blocks=0
        for r in range(0,ref.height,block):
            for c in range(0,ref.width,block):
                w=Window(c,r,min(block,ref.width-c),min(block,ref.height-r)); tr=ref.window_transform(w); shape=(int(w.height),int(w.width))
                hits=tree.query(box(*bounds(w,ref.transform)),predicate='intersects')
                if not len(hits): continue
                inside=geometry_mask([d],out_shape=shape,transform=tr,invert=True,all_touched=False)
                if not inside.any(): continue
                vs,joint=masks(srcs,w,inside)
                an=np.zeros(shape,np.uint32); pn=an.copy(); at=np.zeros(shape,bool); pt=at.copy()
                for i in hits:
                    if area[i]<=0: continue
                    g=clipped[i]; x0,y0,x1,y1=g.bounds; inv=~tr; a,b=inv*(x0,y1); z,t=inv*(x1,y0)
                    c0=max(0,int(np.floor(a))-1); r0=max(0,int(np.floor(b))-1); c1=min(shape[1],int(np.ceil(z))+1); r1=min(shape[0],int(np.ceil(t))+1)
                    if c1<=c0 or r1<=r0: continue
                    sl=(slice(r0,r1),slice(c0,c1)); local=tr*rasterio.Affine.translation(c0,r0); sh=(r1-r0,c1-c0)
                    ce=rasterize([(g,1)],out_shape=sh,transform=local,dtype='uint8',all_touched=False).astype(bool)&inside[sl]
                    to=rasterize([(g,1)],out_shape=sh,transform=local,dtype='uint8',all_touched=True).astype(bool)&inside[sl]
                    counts['center'][i]+=int(ce.sum()); counts['touch'][i]+=int(to.sum())
                    for k,mask in joint.items(): counts[k][i]+=int((ce&mask[sl]).sum())
                    an[sl]+=ce; at[sl]|=to
                    if pri[i]: pn[sl]+=ce; pt[sl]|=to
                    if not ce.any() and to.any():
                        rr,cc=np.nonzero(to)
                        x=local.c+cc*30; y=local.f-rr*30
                        cells=shapely.box(x,y-30,x+30,y)
                        frac=np.clip(shapely.area(shapely.intersection(cells,g))/900,0,1)
                        maxima['any'][i]=max(maxima['any'][i],float(frac.max()))
                        for k in ('core14_known','all18_known'):
                            ok=joint[k][sl][rr,cc]
                            if ok.any(): maxima[k][i]=max(maxima[k][i],float(frac[ok].max()))
                for p,nm,tm in [('all',an,at),('priority',pn,pt)]:
                    u=nm>0; totals[p+'_center']+=int(u.sum()); totals[p+'_touch']+=int(tm.sum()); totals[p+'_overlap_center']+=int((nm>1).sum())
                    for k,mask in joint.items(): totals[p+'_'+k]+=int((u&mask).sum())
                    for k,mask in vs.items(): feature_counts[p][k]+=int((u&mask).sum())
                blocks+=1
        for k in METRICS[2:]: require(np.all(counts[k]<=counts['center']),f'{k} count exceeds centre')
        require(np.all(counts['all18_known']<=counts['core14_known']) and np.all(counts['core14_land']<=counts['core14_known']) and np.all(counts['all18_known']<=counts['all18_raw']),'Joint mask order')
        rows=f[FIELDS].copy(); rows.insert(0,'district',name); rows.insert(0,'state',state); rows['clipped_area_m2']=area
        for k,v in counts.items(): rows[k+'_pixels']=v
        for k,v in maxima.items(): rows['no_center_max_fraction_'+k]=np.where(counts['center']==0,v,np.nan)
        s=dict(state=state,district=name,epsg=ref.crs.to_epsg(),footprint_district_links=n,priority_footprint_links=int(pri.sum()),blocks_with_footprints=blocks)
        for p,sel in [('all',np.ones(n,bool)),('priority',pri)]:
            for k in ('center','elev_slope','all18_known','core14_known','core14_land'):
                s[p+'_no_'+k]=int((sel&(counts[k]==0)).sum())
            for k in ('any','core14_known','all18_known'):
                for thresh in (0.10,0.25,0.50): s[f'{p}_no_center_{k}_ge_{int(thresh*100)}pct']=int((sel&(counts['center']==0)&(maxima[k]>=thresh)).sum())
            require(totals[p+'_center']<=int(counts['center'][sel].sum()),'Union count mismatch')
        s.update(totals)
        fc=[dict(state=state,district=name,subset=p,feature=k,valid_center_pixels=v,union_center_pixels=totals[p+'_center']) for p,fs in feature_counts.items() for k,v in fs.items()]
        return rows,s,fc


def run(root,out,chosen):
    base=root/'data/processed/landslides'; u=base/'unified_inventory_v1'; pm=marker(base/'footprint_grid_pilot_v1/complete.json'); um=marker(u/'complete.json')
    require(pm['status']=='PASS' and um['status']=='PASS' and um['approved_training_labels']==0,'Prior audit not complete')
    cov=read(base/'footprint_grid_pilot_v1/district_coverage_diagnostics.csv'); links=read(u/'unified_district_links.csv'); require(len(cov)==137 and not cov.duplicated(DIST).any() and len(links)==um['district_links'] and not links.duplicated(KEY+DIST).any(),'District/link accounting')
    allfp=pyogrio.read_dataframe(u/'unified_landslide_inventory.gpkg',layer='static_footprints',columns=FIELDS)
    require(len(allfp)==um['static_footprint_representatives']==72659 and not allfp.duplicated(KEY).any() and allfp.crs.to_epsg()==4326 and allfp.geometry.is_valid.all(),'Footprint inventory mismatch')
    require(int(allfp.static_queue.eq(PRIORITY).sum())==um['priority_footprints']==62609,'Priority mismatch')
    links=links[links.boundary_touch_only.eq('False')].merge(allfp[KEY],on=KEY,validate='many_to_one'); districts=load_districts(root)
    prior=read(base/'footprint_grid_pilot_v1/pilot_district_grid_summary.csv'); requested=set(chosen or [tuple(x) for x in cov[DIST].itertuples(index=False,name=None)])
    require(requested<=set(cov[DIST].itertuples(index=False,name=None)),'Unknown district requested')
    frames=[]; summaries=[]; features=[]
    for ix,(s,d) in enumerate(sorted(requested),1):
        cv=cov[(cov.state==s)&(cov.district==d)].iloc[0]; ds=districts[(districts.state==s)&(districts.district==d)]; require(len(ds)==1,'District missing')
        li=links[(links.state==s)&(links.district==d)][KEY]; f=allfp.merge(li,on=KEY,validate='one_to_one')
        require(len(f)==len(li)==int(cv.mapped_representatives) and int(f.static_queue.eq(PRIORITY).sum())==int(cv.priority_footprints),'Footprint district counts mismatch')
        ref,files=paths(root,s,d); rows,su,fc=audit(ref,files,ds.geometry.iloc[0],f,s,d)
        pr=prior[(prior.state==s)&(prior.district==d)]; su['pilot_crosscheck']='NOT_PILOT'
        if len(pr):
            for new,old in [('all_center','all_union_center_pixels'),('priority_center','priority_union_center_pixels'),('all_touch','all_union_all_touched_pixels'),('priority_touch','priority_union_all_touched_pixels'),('all_elev_slope','all_union_elev_slope_valid_pixels'),('priority_elev_slope','priority_union_elev_slope_valid_pixels'),('priority_no_center','priority_no_center')]:
                require(su[new]==int(pr.iloc[0][old]),f'{s}/{d} pilot discrepancy: {new}')
            su['pilot_crosscheck']='PASS'
        frames.append(rows); summaries.append(su); features.extend(fc)
        print(f'[{ix}/{len(requested)}] {s}/{d} | footprint links {len(f)} | priority {su["priority_footprint_links"]} | centre {su["priority_center"]} | all18 known {su["priority_all18_known"]} | core14 known {su["priority_core14_known"]} | no centre {su["priority_no_center"]} | pilot {su["pilot_crosscheck"]}',flush=True)
    detail=pd.concat(frames,ignore_index=True); summ=pd.DataFrame(summaries)
    require(not detail.duplicated(KEY+DIST).any(),'Duplicate detail rows')
    if len(requested)==137: require(len(detail)==len(links),'Global district-link accounting mismatch')
    export(out,'footprint_feature_support.csv',detail); export(out,'district_feature_support.csv',summ); export(out,'feature_valid_center_counts.csv',pd.DataFrame(features))
    # Count distinct footprint keys with support in ANY audited district, not sums of links.
    agg=detail.groupby(KEY,as_index=False).agg(static_queue=('static_queue','first'),center_pixels=('center_pixels','sum'),all18_known_pixels=('all18_known_pixels','sum'),core14_known_pixels=('core14_known_pixels','sum'),core14_land_pixels=('core14_land_pixels','sum'))
    export(out,'distinct_footprint_support.csv',agg); pri=agg[agg.static_queue.eq(PRIORITY)]
    result=dict(status='PASS',scope='ALL_DISTRICTS' if len(requested)==137 else 'PILOT_SUBSET',districts=len(requested),footprint_district_links=len(detail),distinct_footprints=len(agg),priority_distinct_footprints=len(pri),priority_no_center=int(pri.center_pixels.eq(0).sum()),priority_with_all18_known=int(pri.all18_known_pixels.gt(0).sum()),priority_with_core14_known=int(pri.core14_known_pixels.gt(0).sum()),priority_with_core14_land=int(pri.core14_land_pixels.gt(0).sum()),pilot_crosschecks=int(summ.pilot_crosscheck.eq('PASS').sum()),approved_training_labels=0)
    if len(requested)==137: require(len(agg)==72659 and len(pri)==62609,'Not every distinct footprint has a non-touch district link')
    print(json.dumps(result,indent=2)); (out/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


def self_test():
    from rasterio.transform import from_origin
    with TemporaryDirectory() as td:
        p=Path(td); tr=from_origin(500000,120,30,30); crs='EPSG:32643'; files={k:p/(k+'.tif') for k in FEATURES}
        for k,fn in files.items():
            a=np.ones((4,4),np.float32)
            if k=='landcover': a[:]=10; a[0,0]=80
            if k=='lithology': a[0,1]=16
            if k=='aspect': a[1,0]=-9999
            with rasterio.open(fn,'w',driver='GTiff',height=4,width=4,count=1,dtype='float32',crs=crs,transform=tr,nodata=-9999) as dst: dst.write(a,1)
        d=gpd.GeoSeries([box(500000,0,500120,120)],crs=crs).to_crs(4326).iloc[0]
        f=gpd.GeoDataFrame([dict(source='test',qa_record_id=str(i),static_geometry_group=str(i),static_queue=PRIORITY,spatial_status='MAPPED',type_group='SLIDE',mapping_method='test') for i in range(3)],geometry=[box(500001,61,500059,119),box(500031,61,500089,119),box(500002,2,500011,11)],crs=crs).to_crs(4326)
        a,s,fc=audit(files['elevation'],files,d,f,'T','Test',block=2)
        assert a.center_pixels.tolist()==[4,4,0] and s['priority_center']==6 and s['priority_overlap_center']==2
        assert s['priority_all18_raw']==5 and s['priority_all18_known']==4 and s['priority_core14_known']==5 and s['priority_core14_land']==4
        assert abs(a.no_center_max_fraction_any.iloc[2]-.09)<1e-6 and abs(a.no_center_max_fraction_all18_known.iloc[2]-.09)<1e-6
        export(p,'test.csv',a); a,s,fc=audit(files['elevation'],files,d,f.iloc[:0],'T','Empty',block=2); assert a.empty and s['all_center']==0
        bad=p/'bad.tif'
        with rasterio.open(bad,'w',driver='GTiff',height=4,width=4,count=1,dtype='float32',crs=crs,transform=from_origin(500001,120,30,30),nodata=-9999) as dst: dst.write(np.ones((4,4),np.float32),1)
        try: audit(files['elevation'],dict(files,aspect=bad),d,f,'T','Test',block=2)
        except ValueError as e: assert 'grid mismatch' in str(e)
        else: raise AssertionError('Shifted grid accepted')
    print('Synthetic overlap, 18-feature masks, unknown geology, water, optional NoData, subpixel overlap, empty district, shifted-grid rejection and CSV readback: PASS')


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--root',type=Path,default=ROOT); ap.add_argument('--self-test',action='store_true'); ap.add_argument('--district',action='append',help='STATE/District (repeatable)'); a=ap.parse_args()
    if a.self_test: self_test(); return
    chosen=[tuple(x.split('/',1)) for x in a.district] if a.district else None
    require(chosen is None or all(len(x)==2 for x in chosen),'Use STATE/District')
    name='footprint_feature_support_batch_v1' if chosen is None else 'footprint_feature_support_pilot_v1'
    out=a.root/'data/processed/landslides'/name; stage=out.with_name(out.name+'.staging'); require(not out.exists() and not stage.exists(),'Output/staging exists; inspect before retrying')
    stage.mkdir(); run(a.root,stage,chosen); stage.rename(out); print('Output:',out,'| No labels or raster modifications.')
if __name__=='__main__': main()
