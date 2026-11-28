"""Read-only district coverage and blockwise footprint / 30 m terrain grid pilot.
No raster labels, geometry edits, or training samples are written.
"""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import rasterio
from rasterio.features import rasterize, geometry_mask
from rasterio.windows import Window, bounds
import shapely
from shapely.geometry import box
from extract_uglc_regional import load_districts
from resolve_landslide_duplicates import read, export, marker, require, KEY

ROOT=Path(__file__).resolve().parent.parent
PRIORITY='SLIDE_FOOTPRINT_PRIORITY_REVIEW'
FIELDS=KEY+['static_geometry_group','static_queue','spatial_status','type_group','mapping_method']
DIST=['state','district']


def coverage(root, out):
    base=root/'data/processed/landslides'; u=base/'unified_inventory_v1'; p=u/'unified_landslide_inventory.gpkg'
    m=marker(u/'complete.json'); require(m['status']=='PASS' and m['records']==128846 and m['approved_training_labels']==0,'Unexpected unified marker')
    f=read(base/'candidate_selection_v1/candidate_selection_flags.csv')
    q=read(base/'quality_standardization_v1/quality_candidates.csv')
    l=read(u/'unified_district_links.csv'); previous=read(u/'district_candidate_coverage.csv')
    require(len(f)==len(q)==m['records'] and not f.duplicated(KEY).any() and not q.duplicated(KEY).any(),'Candidate key/count mismatch')
    require(len(l)==m['district_links'] and not l.duplicated(KEY+DIST).any() and l.boundary_touch_only.isin(['True','False']).all(),'Link mismatch')
    attrs=f[KEY+['geometry_kind','static_group_representative','static_queue','spatial_status','type_group','mapping_method','representation_status']].merge(q[KEY+['type_raw','mapping_method_raw']],on=KEY,validate='one_to_one')
    require(len(attrs)==len(f),'Attribute join lost records')
    j=l.merge(attrs,on=KEY,validate='many_to_one'); require(len(j)==len(l),'Link join lost records')
    j=j[j.boundary_touch_only.eq('False')].copy()
    poly=j[j.geometry_kind.eq('POLYGON')]; reps=poly[poly.static_group_representative.eq('True')]
    rows=[]
    for s,d in previous[DIST].itertuples(index=False,name=None):
        a=j[(j.state==s)&(j.district==d)]; b=poly[(poly.state==s)&(poly.district==d)]; c=reps[(reps.state==s)&(reps.district==d)]
        n=int(c.static_queue.eq(PRIORITY).sum())
        reason=('PRIORITY_AVAILABLE' if n else 'MAPPED_NONPRIORITY_AVAILABLE' if len(c) else 'POLYGONS_NOT_MAPPED_REPRESENTATIVES' if len(b) else 'POINTS_ONLY' if len(a) else 'NO_NONTOUCH_RECORDS')
        rows.append(dict(state=s,district=d,all_records=len(a),point_records=int(a.geometry_kind.eq('POINT').sum()),polygon_records=len(b),mapped_representatives=len(c),priority_footprints=n,nonpriority_footprints=len(c)-n,coverage_status=reason))
    cov=pd.DataFrame(rows); require(len(cov)==137 and not cov.duplicated(DIST).any(),'District coverage mismatch')
    chk=cov.merge(previous,on=DIST,validate='one_to_one')
    require(len(chk)==137 and chk.priority_footprints.eq(chk.priority_footprint_representatives.astype(int)).all() and chk.mapped_representatives.eq(chk.distinct_footprint_representatives.astype(int)).all(),'Prior coverage discrepancy')
    require(int(cov.priority_footprints.gt(0).sum())==m['districts_with_priority_footprints'],'Priority district count mismatch')
    export(out,'district_coverage_diagnostics.csv',cov)
    no=cov[cov.priority_footprints.eq(0)][DIST]; b=poly.merge(no,on=DIST,validate='many_to_one')
    cols=DIST+['source','spatial_status','mapping_method','mapping_method_raw','type_group','type_raw','static_queue','static_group_representative']
    export(out,'no_priority_polygon_attribute_counts.csv',b.groupby(cols,dropna=False).size().rename('district_record_links').reset_index())
    export(out,'no_priority_point_attribute_counts.csv',j[j.geometry_kind.eq('POINT')].merge(no,on=DIST,validate='many_to_one').groupby(DIST+['source','spatial_status','type_group','type_raw'],dropna=False).size().rename('district_record_links').reset_index())
    export(out,'state_coverage_summary.csv',cov.groupby('state').agg(districts=('district','size'),districts_with_priority=('priority_footprints',lambda v:int(v.gt(0).sum())),districts_with_mapped=('mapped_representatives',lambda v:int(v.gt(0).sum())),priority_district_links=('priority_footprints','sum'),mapped_district_links=('mapped_representatives','sum')).reset_index())
    print('District coverage:',cov.coverage_status.value_counts().to_dict(),flush=True)
    return p,cov,reps


def valid(src,win):
    a=src.read(1,window=win); v=(src.read_masks(1,window=win)>0)&np.isfinite(a)
    if src.nodata is not None: v &= a!=src.nodata
    return v


def polygon_parts(g):
    if g.geom_type in ('Polygon','MultiPolygon'): return g
    if g.geom_type=='GeometryCollection':
        parts=[polygon_parts(a) for a in g.geoms]; return shapely.union_all([a for a in parts if not a.is_empty])
    return shapely.Polygon()


def grid_audit(refpath, elevpath, slopepath, district, fps, state, name, block=512):
    # Per-footprint counts are diagnostic, and may count an overlapping pixel repeatedly.
    # Union counts below count each district pixel only once.
    require(not fps.duplicated(KEY).any() and fps.crs is not None,'Invalid footprint input')
    with rasterio.open(refpath) as ref, rasterio.open(elevpath) as elev, rasterio.open(slopepath) as slope:
        require(ref.crs is not None and ref.crs.is_projected and ref.crs.linear_units.lower() in ('metre','meter','metres','meters') and np.allclose(ref.res,(30,30)) and ref.transform.b==ref.transform.d==0 and ref.transform.a>0 and ref.transform.e<0,'Not a north-up 30 m metric grid')
        for src in (elev,slope):
            require(src.crs==ref.crs and src.shape==ref.shape and src.transform.almost_equals(ref.transform,precision=1e-7) and src.count==1,'Terrain grid alignment failure')
        d=gpd.GeoSeries([district],crs=4326).to_crs(ref.crs).iloc[0]; require(d.is_valid and not d.is_empty,'District reprojection invalid')
        f=fps.to_crs(ref.crs); require(f.geometry.is_valid.all(),'Footprint reprojection invalid')
        clipped=[polygon_parts(g.intersection(d)) for g in f.geometry]
        require(all(g.is_valid for g in clipped),'Clipped geometry invalid')
        areas=np.array([g.area for g in clipped]); tree=shapely.STRtree(clipped)
        priority=f.static_queue.eq(PRIORITY).to_numpy(); n=len(f)
        centers=np.zeros(n,dtype=np.int64); touched=centers.copy(); supported=centers.copy()
        totals=dict(all_union_center_pixels=0,priority_union_center_pixels=0,all_union_all_touched_pixels=0,priority_union_all_touched_pixels=0,all_union_elev_slope_valid_pixels=0,priority_union_elev_slope_valid_pixels=0,all_overlap_center_pixels=0,priority_overlap_center_pixels=0,blocks_with_footprints=0)
        for r in range(0,ref.height,block):
            for c in range(0,ref.width,block):
                w=Window(c,r,min(block,ref.width-c),min(block,ref.height-r)); tr=ref.window_transform(w); shape=(int(w.height),int(w.width))
                hits=tree.query(box(*bounds(w,ref.transform)),predicate='intersects')
                if not len(hits): continue
                inside=geometry_mask([d],out_shape=shape,transform=tr,invert=True,all_touched=False)
                if not inside.any(): continue
                good=inside & valid(elev,w)&valid(slope,w)
                all_n=np.zeros(shape,dtype=np.uint32); pri_n=all_n.copy(); all_t=np.zeros(shape,dtype=bool); pri_t=all_t.copy()
                for i in hits:
                    if areas[i]<=0: continue
                    # Restrict per-record rasterization to its bounding pixel window, with a 1 pixel halo.
                    x0,y0,x1,y1=clipped[i].bounds; inv=~tr
                    a,b=inv*(x0,y1); z,t=inv*(x1,y0)
                    c0=max(0,int(np.floor(a))-1); r0=max(0,int(np.floor(b))-1); c1=min(shape[1],int(np.ceil(z))+1); r1=min(shape[0],int(np.ceil(t))+1)
                    if c1<=c0 or r1<=r0: continue
                    sl=(slice(r0,r1),slice(c0,c1)); local=tr*rasterio.Affine.translation(c0,r0); sh=(r1-r0,c1-c0)
                    center=rasterize([(clipped[i],1)],out_shape=sh,transform=local,dtype='uint8',all_touched=False).astype(bool)&inside[sl]
                    touch=rasterize([(clipped[i],1)],out_shape=sh,transform=local,dtype='uint8',all_touched=True).astype(bool)&inside[sl]
                    centers[i]+=int(center.sum()); touched[i]+=int(touch.sum()); supported[i]+=int((center&good[sl]).sum())
                    all_n[sl]+=center; all_t[sl]|=touch
                    if priority[i]: pri_n[sl]+=center; pri_t[sl]|=touch
                for prefix,counts,touch in [('all',all_n,all_t),('priority',pri_n,pri_t)]:
                    totals[prefix+'_union_center_pixels']+=int((counts>0).sum()); totals[prefix+'_union_all_touched_pixels']+=int(touch.sum()); totals[prefix+'_union_elev_slope_valid_pixels']+=int(((counts>0)&good).sum()); totals[prefix+'_overlap_center_pixels']+=int((counts>1).sum())
                totals['blocks_with_footprints']+=1
        require(np.all(supported<=centers),'Invalid support counts')
        require(totals['all_union_center_pixels']<=centers.sum() and totals['priority_union_center_pixels']<=centers[priority].sum(),'Union count mismatch')
        rows=f[FIELDS].copy(); rows.insert(0,'district',name); rows.insert(0,'state',state)
        rows['clipped_area_m2']=areas; rows['center_pixels']=centers; rows['all_touched_pixels']=touched; rows['elev_slope_valid_center_pixels']=supported
        rows['grid_status']=np.select([areas<=0,centers==0,supported==0,supported<centers],['NO_POSITIVE_CLIPPED_AREA','NO_CENTER_PIXEL','NO_ELEV_SLOPE_SUPPORT','PARTIAL_ELEV_SLOPE_SUPPORT'],default='CENTER_PIXELS_SUPPORTED')
        summary=dict(state=state,district=name,epsg=ref.crs.to_epsg(),footprint_district_links=n,priority_footprint_links=int(priority.sum()),priority_no_center=int((priority&(centers==0)).sum()),all_no_center=int((centers==0).sum()),priority_no_elev_slope_support=int((priority&(supported==0)).sum()),priority_sum_center_pixels=int(centers[priority].sum()),all_sum_center_pixels=int(centers.sum()),**totals)
        return rows,summary


def pilot(root,out,p,cov,reps,requested):
    districts=load_districts(root); allfp=pyogrio.read_dataframe(p,layer='static_footprints',columns=FIELDS)
    require(len(allfp)==72659 and not allfp.duplicated(KEY).any() and allfp.crs.to_epsg()==4326 and allfp.geometry.is_valid.all(),'Static footprint layer mismatch')
    if requested:
        chosen=[tuple(x.split('/',1)) for x in requested]; require(all(len(x)==2 for x in chosen),'Use STATE/District')
    else:
        chosen=[('UK','Haridwar')]
        for s,col in [('HP','priority_footprints'),('SK','mapped_representatives'),('TR','mapped_representatives')]:
            a=cov[cov.state.eq(s)].sort_values([col,'polygon_records','district'],ascending=[False,False,True]); require(len(a)>0,f'Missing pilot state {s}'); chosen.append(tuple(a[DIST].iloc[0]))
    require(len(set(chosen))==len(chosen),'Repeated pilot district')
    frames=[]; summaries=[]
    for s,n in chosen:
        d=districts[(districts.state==s)&(districts.district==n)]; require(len(d)==1,f'District not found: {s}/{n}')
        l=reps[(reps.state==s)&(reps.district==n)][KEY]; f=allfp.merge(l,on=KEY,validate='one_to_one'); require(len(f)==len(l),'Footprint link join mismatch')
        stem=n.replace(' ','_'); proc=root/'data/processed'; ref=proc/'DEM/district_dem_utm'/s/(stem+'.tif'); t=proc/'terrain_batch/terrain_v2'/s/stem
        rows,summary=grid_audit(ref,t/'elevation.tif',t/'slope.tif',d.geometry.iloc[0],f,s,n)
        frames.append(rows); summaries.append(summary); print(json.dumps(summary,indent=2),flush=True)
    export(out,'pilot_footprint_grid_counts.csv',pd.concat(frames,ignore_index=True)); export(out,'pilot_district_grid_summary.csv',pd.DataFrame(summaries))
    (out/'complete.json').write_text(json.dumps(dict(status='PASS',district_coverage=137,pilot_districts=[list(x) for x in chosen],pilot_footprint_district_links=sum(x['footprint_district_links'] for x in summaries),approved_training_labels=0),indent=2),encoding='utf-8')


def self_test():
    from rasterio.transform import from_origin
    with TemporaryDirectory() as td:
        p=Path(td); tr=from_origin(500000,120,30,30); crs='EPSG:32643'
        v=np.ones((4,4),np.float32); v[0,0]=-9999
        for name in ['ref','elev','slope']:
            with rasterio.open(p/(name+'.tif'),'w',driver='GTiff',height=4,width=4,count=1,dtype='float32',crs=crs,transform=tr,nodata=-9999) as dst: dst.write(v,1)
        d=gpd.GeoSeries([box(500000,0,500120,120)],crs=crs).to_crs(4326).iloc[0]
        f=gpd.GeoDataFrame([dict(source='test',qa_record_id=str(i),static_geometry_group=str(i),static_queue=PRIORITY,spatial_status='MAPPED',type_group='SLIDE',mapping_method='test') for i in range(3)],geometry=[box(500001,61,500059,119),box(500031,61,500089,119),box(500002,2,500005,5)],crs=crs).to_crs(4326)
        a,s=grid_audit(p/'ref.tif',p/'elev.tif',p/'slope.tif',d,f,'T','Test',block=2)
        assert a.center_pixels.tolist()==[4,4,0] and a.elev_slope_valid_center_pixels.tolist()==[3,4,0]
        assert s['priority_union_center_pixels']==6 and s['priority_overlap_center_pixels']==2 and s['priority_union_elev_slope_valid_pixels']==5 and s['priority_no_center']==1
        assert a.all_touched_pixels.iloc[2]==1
        export(p,'test.csv',a)
        # No-footprint district is legitimate, not a negative-label dataset.
        a,s=grid_audit(p/'ref.tif',p/'elev.tif',p/'slope.tif',d,f.iloc[:0],'T','Empty',block=2)
        assert a.empty and s['all_union_center_pixels']==0
    print('Synthetic block boundaries, overlapping footprints, tiny footprint, NoData, empty district and CSV readback: PASS')


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--root',type=Path,default=ROOT); ap.add_argument('--self-test',action='store_true'); ap.add_argument('--district',action='append',help='STATE/District, repeatable'); a=ap.parse_args()
    if a.self_test: self_test(); return
    out=a.root/'data/processed/landslides/footprint_grid_pilot_v1'; stage=out.with_name(out.name+'.staging'); require(not out.exists() and not stage.exists(),'Output/staging exists; inspect before retrying')
    stage.mkdir(); p,cov,reps=coverage(a.root,stage); pilot(a.root,stage,p,cov,reps,a.district); stage.rename(out); print('Output:',out,'| No labels or raster modifications.')
if __name__=='__main__': main()
