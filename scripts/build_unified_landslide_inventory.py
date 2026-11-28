"""Build an audited spatial inventory from completed review queues, not training labels.
Uses existing regional GeoPackages and CSV sidecars; no original data is modified.
"""
import argparse
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import geopandas as gpd
import pandas as pd
import pyogrio
import shapely
from audit_gsi_candidates import geometry_key
from extract_uglc_regional import load_districts
from standardize_landslide_quality import plan
from resolve_landslide_duplicates import read, marker, require, export, keys, KEY, EXPECTED, ASSOC, RAIN

ROOT = Path(__file__).resolve().parent.parent
FOOT = 'MAPPED_FOOTPRINT_CANDIDATE'
PRIORITY = 'SLIDE_FOOTPRINT_PRIORITY_REVIEW'
# Keep compact queryable attributes in the spatial inventory. The full selection CSV
# remains authoritative for all original and standardized metadata.
FIELDS = KEY + ['layer','geometry_kind','geometry_hash','original_dataset','original_id',
    'original_secondary_id','mapping_method','spatial_status','type_group','trigger_group',
    'accuracy_tier','date_status','date_precision','start_date','end_date',
    'representation_status','static_geometry_group','static_queue','static_group_representative',
    'event_review_group','event_review_status','dynamic_screen','dynamic_review_unit',
    'training_label_approved','record_deleted']
LINK = KEY + ['state','district','boundary_touch_only']

def load(base):
    p=base/'candidate_selection_v1'; m=marker(p/'complete.json')
    f=read(p/'candidate_selection_flags.csv'); s=read(p/'static_footprint_review_units.csv')
    u=read(p/'dynamic_review_units.csv'); r=read(p/'dynamic_rain_day_review_records.csv')
    require(m['status']=='PASS' and len(f)==m['records'] and Counter(f.source)==EXPECTED,'Candidate marker/count mismatch')
    require(not f.duplicated(KEY).any() and f.qa_record_id.ne('').all(),'Invalid record keys')
    require(f.training_label_approved.str.lower().eq('false').all() and f.record_deleted.str.lower().eq('false').all(),'Unexpected approvals or deletions')
    require(m['approved_training_labels']==m['records_deleted']==0,'Unexpected prior marker')
    require(f.geometry_hash.str.fullmatch('[0-9a-f]{64}').all(),'Invalid geometry hash')
    require(f.geometry_kind.isin(['POINT','POLYGON']).all(),'Invalid kind')
    require(f.static_group_representative.isin(['True','False']).all(),'Invalid representative flag')
    reps=f[f.static_group_representative.eq('True')]; fp=f[f.spatial_status.eq(FOOT)]
    require(len(fp)==m['mapped_footprint_records'] and len(reps)==len(s)==m['static_footprint_review_units'],'Static count mismatch')
    require(reps.geometry_kind.eq('POLYGON').all() and reps.spatial_status.eq(FOOT).all(),'Invalid static representative')
    require(reps.static_geometry_group.is_unique and s.static_geometry_group.is_unique and keys(reps)==keys(s),'Static representative mismatch')
    cols=['static_geometry_group','static_queue','layer','type_group']
    require(reps.set_index(KEY)[cols].sort_index().equals(s.set_index(KEY)[cols].sort_index()),'Static attributes mismatch')
    require(fp.groupby('static_geometry_group').size().to_dict()==s.set_index('static_geometry_group').members.astype(int).to_dict(),'Static membership mismatch')
    require(reps.static_queue.eq(PRIORITY).sum()==m['static_priority_review_units'],'Priority count mismatch')
    rain=f[f.dynamic_screen.eq(RAIN)]
    require(len(rain)==len(r)==m['rain_day_review_records'] and keys(rain)==keys(r) and not r.duplicated(KEY).any(),'Rain review record mismatch')
    require(rain.dynamic_review_unit.ne('').all() and f.loc[~f.dynamic_screen.eq(RAIN),'dynamic_review_unit'].eq('').all(),'Dynamic unit assignment mismatch')
    require(rain.set_index(KEY).dynamic_review_unit.sort_index().equals(r.set_index(KEY).review_unit.sort_index()),'Dynamic unit ID mismatch')
    require(u.review_unit.is_unique and len(u)==m['dynamic_review_units'] and rain.groupby('dynamic_review_unit').size().to_dict()==u.set_index('review_unit').rain_day_records.astype(int).to_dict(),'Dynamic unit count mismatch')
    require(f.representation_status.eq(ASSOC).sum()==3153 and not reps.representation_status.eq(ASSOC).any(),'Associated point mismatch')
    # Compare geometry fingerprints to the independent previous audit.
    w=read(base/'within_source_duplicates_v1/within_source_record_flags.csv')
    require(not w.duplicated(KEY).any() and keys(w)==keys(f),'Prior geometry inventory mismatch')
    require(f.set_index(KEY).geometry_hash.sort_index().equals(w.set_index(KEY).geometry_hash.sort_index()),'Prior geometry hashes differ')
    return f[FIELDS].copy()

def verify_geometry(frame):
    require(frame.crs is not None and frame.crs.to_epsg()==4326,'CRS mismatch')
    g=frame.geometry
    require(g.notna().all() and not g.is_empty.any() and g.is_valid.all(),'Invalid geometry')
    kind=g.geom_type.map({'Point':'POINT','Polygon':'POLYGON','MultiPolygon':'POLYGON'})
    require(kind.eq(frame.geometry_kind).all(),'Geometry kind mismatch')
    require(all(geometry_key(g)==h for g,h in zip(g,frame.geometry_hash)),'Geometry hash mismatch')

def verify_layer(path,layer,expected):
    info=pyogrio.read_info(path,layer=layer)
    require(info['features']==len(expected),f'{layer}: readback count mismatch')
    seen=set(); ix=expected.set_index(KEY)
    for off in range(0,len(expected),5000):
        g=pyogrio.read_dataframe(path,layer=layer,skip_features=off,max_features=5000)
        require(not g.duplicated(KEY).any(),f'{layer}: duplicate readback keys')
        k=keys(g); require(not (seen & k) and k<=set(ix.index),f'{layer}: unexpected readback keys'); seen.update(k)
        verify_geometry(g)
        attrs=pd.DataFrame(g.drop(columns=g.geometry.name)).set_index(KEY).sort_index()[ix.columns]
        require(attrs.fillna('').astype(str).equals(ix.loc[attrs.index].sort_index().fillna('').astype(str)),f'{layer}: attribute readback mismatch')
    require(seen==set(ix.index),f'{layer}: incomplete readback')

def run(base,out,f,districts):
    require(not f.duplicated(KEY).any() and len(f)>0,'Input key mismatch')
    require(districts.crs.to_epsg()==4326 and not districts.duplicated(['state','district']).any(),'District mismatch')
    dest=out/'unified_landslide_inventory.gpkg'; written=Counter(); seen=set(); link_frames=[]
    d=districts[['state','district','geometry']].reset_index(drop=True)
    for source,layer,path in plan(base,f[['source','layer']].to_dict('records')):
        attrs=f[(f.source==source)&(f.layer==layer)]
        g=pyogrio.read_dataframe(path,layer='candidates',columns=['qa_record_id'])
        require(g.qa_record_id.notna().all() and g.qa_record_id.is_unique and set(g.qa_record_id.astype(str))==set(attrs.qa_record_id),f'{path}: record mismatch')
        g['qa_record_id']=g.qa_record_id.astype(str)
        g=g.merge(attrs,on='qa_record_id',validate='one_to_one')
        g=gpd.GeoDataFrame(g[FIELDS+['geometry']],geometry='geometry',crs=g.crs)
        verify_geometry(g); k=keys(g); require(not (k&seen),'Repeated source record'); seen.update(k)
        # Recompute intersections; retain the entire geometry, including boundary-crossing records.
        j=gpd.sjoin(g[KEY+['geometry']],d,how='inner',predicate='intersects')
        require(keys(j)==k,'A regional candidate no longer intersects study districts')
        j['boundary_touch_only']=shapely.touches(j.geometry.array,d.geometry.loc[j.index_right.to_numpy()].array)
        link_frames.append(pd.DataFrame(j[LINK]))
        for name,part in [('inventory_points',g[g.geometry_kind.eq('POINT')]),('inventory_polygons',g[g.geometry_kind.eq('POLYGON')]),('static_footprints',g[g.static_group_representative.eq('True')])]:
            if part.empty: continue
            pyogrio.write_dataframe(part,dest,layer=name,driver='GPKG',append=written[name]>0,promote_to_multi=name!='inventory_points')
            written[name]+=len(part)
        print(f'{source} / {layer}: {len(g):,} records exported',flush=True)
    require(seen==keys(f),'Incomplete geometry export')
    for name,expected in [('inventory_points',f[f.geometry_kind.eq('POINT')]),('inventory_polygons',f[f.geometry_kind.eq('POLYGON')]),('static_footprints',f[f.static_group_representative.eq('True')])]:
        require(written[name]==len(expected),f'{name}: write count mismatch')
        if len(expected): verify_layer(dest,name,expected)
        print(f'{name}: {len(expected):,} readback PASS',flush=True)
    links=pd.concat(link_frames,ignore_index=True)
    require(not links.duplicated(KEY+['state','district']).any(),'Duplicate district links')
    export(out,'unified_district_links.csv',links)
    # Coverage only: not independent events, sampled pixels, or summed polygon area.
    j=links.merge(f[KEY+['static_group_representative','static_queue','dynamic_review_unit','representation_status']],on=KEY,validate='many_to_one')
    rows=[]
    for state,district in d[['state','district']].itertuples(index=False,name=None):
        a=j[(j.state==state)&(j.district==district)]; usable=a[~a.boundary_touch_only]
        static=usable[usable.static_group_representative.eq('True')]
        rows.append(dict(state=state,district=district,all_intersecting_records=len(a),non_touch_records=len(usable),
            associated_point_records=int(usable.representation_status.eq(ASSOC).sum()),
            distinct_footprint_representatives=len(static),priority_footprint_representatives=int(static.static_queue.eq(PRIORITY).sum()),
            rain_day_records=int(usable.dynamic_review_unit.ne('').sum()),dynamic_review_units=usable.loc[usable.dynamic_review_unit.ne(''),'dynamic_review_unit'].nunique()))
    coverage=pd.DataFrame(rows); export(out,'district_candidate_coverage.csv',coverage)
    export(out,'source_layer_counts.csv',f.groupby(['source','layer','geometry_kind']).size().rename('records').reset_index())
    report=dict(status='PASS',records=len(f),point_records=written['inventory_points'],polygon_records=written['inventory_polygons'],
        static_footprint_representatives=written['static_footprints'],priority_footprints=int(f.static_group_representative.eq('True').mul(f.static_queue.eq(PRIORITY)).sum()),
        dynamic_review_records=int(f.dynamic_review_unit.ne('').sum()),dynamic_review_units=f.loc[f.dynamic_review_unit.ne(''),'dynamic_review_unit'].nunique(),
        district_links=len(links),districts=len(d),districts_with_priority_footprints=int(coverage.priority_footprint_representatives.gt(0).sum()),
        districts_with_dynamic_review_records=int(coverage.rain_day_records.gt(0).sum()),
        records_deleted=0,approved_training_labels=0,excluded_repaired_review_polygons=6)
    # Full provenance and review-unit membership remain in candidate_selection_v1, not duplicated into the GPKG.
    (out/'complete.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    print('Districts with no priority footprints:',coverage.loc[coverage.priority_footprint_representatives.eq(0),['state','district']].to_dict('records'))

def self_test():
    from shapely.geometry import Point, box
    with TemporaryDirectory() as td:
        base=Path(td)/'landslides'; out=Path(td)/'out'; out.mkdir()
        d=gpd.GeoDataFrame({'state':['A','B'],'district':['West','East']},geometry=[box(0,0,1,1),box(1,0,2,1)],crs=4326)
        records=[]
        for s,l,ids,geoms in [('GLC','points',['a'],[Point(.2,.2)]),('UGLC','polygons',['b','c'],[box(.8,.3,1.2,.4),box(.2,.3,.3,.4)]),('COOLR','COOLR_Reports_Points',['d'],[Point(1,.2)])]:
            p=list(plan(base,[dict(source=s,layer=l)]))[0][2]; p.parent.mkdir(parents=True,exist_ok=True)
            pyogrio.write_dataframe(gpd.GeoDataFrame({'qa_record_id':ids},geometry=geoms,crs=4326),p,layer='candidates')
            for i,g in zip(ids,geoms):
                r=dict.fromkeys(FIELDS,''); r.update(source=s,layer=l,qa_record_id=i,geometry_kind='POINT' if g.geom_type=='Point' else 'POLYGON',geometry_hash=geometry_key(g),training_label_approved='False',record_deleted='False',static_group_representative='True' if i=='b' else 'False',static_queue=PRIORITY if i=='b' else '',dynamic_review_unit='E1' if i=='a' else '')
                records.append(r)
        f=pd.DataFrame(records); run(base,out,f,d)
        m=marker(out/'complete.json'); assert m['records']==4 and m['static_footprint_representatives']==1 and m['districts_with_priority_footprints']==2 and m['district_links']==6
        bad=f.copy(); bad.loc[0,'geometry_hash']='0'*64
        try: verify_layer(out/'unified_landslide_inventory.gpkg','inventory_points',bad[bad.geometry_kind.eq('POINT')])
        except ValueError: pass
        else: raise AssertionError('Modified hash was not rejected')
    print('Synthetic spatial export, append, district crossing, hash and attribute readback: PASS')

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',type=Path,default=ROOT); p.add_argument('--self-test',action='store_true'); a=p.parse_args()
    if a.self_test: self_test(); return
    base=a.root/'data/processed/landslides'; out=base/'unified_inventory_v1'; stage=out.with_name(out.name+'.staging')
    require(not out.exists() and not stage.exists(),'Output/staging exists; inspect before retrying; no overwrite')
    f=load(base); d=load_districts(a.root); stage.mkdir(); run(base,stage,f,d); stage.rename(out)
    print('Output:',out,'| Original data unchanged; no training labels approved.')
if __name__=='__main__': main()
