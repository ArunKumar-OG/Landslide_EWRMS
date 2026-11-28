"""Read-only source audit and regional extraction of Lesser Himalayas 2023.
Retain original attributes, distinguish GPS points, footprints and associated points.
No event date inference, deduplication or automatic geometry repair.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
import argparse, json, zipfile
import numpy as np
import pandas as pd
import geopandas as gpd
import pyogrio
import shapely
from shapely.geometry import Point, box
from extract_uglc_regional import load_districts

ROOT = Path(__file__).resolve().parent.parent
LAYERS = {'Field_GPS_landslides':(359,'gps'), 'landslides_shimla':(3176,'polygon'), 'landslides_shimla_points':(3176,'associated_point')}

def read_layer(archive,name,expected,kind):
    member=f'landslides/{name}.shp'
    uri=f'/vsizip/{{{archive.resolve().as_posix()}}}/{member}'
    g=pyogrio.read_dataframe(uri,fid_as_index=True,datetime_as_string=True)
    epsg=4326 if kind=='gps' else 32643
    if len(g)!=expected or g.crs is None or g.crs.to_epsg()!=epsg or not g.index.is_unique:
        raise ValueError(f'{name}: source count/CRS/FID mismatch')
    if any(c.lower().startswith('qa_') for c in g.columns): raise ValueError('Reserved QA column')
    g['qa_source_fid']=g.index.astype(str); g=g.reset_index(drop=True)
    g['qa_record_id']=name+':'+g.qa_source_fid
    g['qa_source_layer']=name; g['qa_kind']=kind
    g['qa_original_geom_type']=g.geometry.geom_type.fillna('NULL')
    g['qa_had_z']=shapely.has_z(g.geometry.array)
    g.geometry=shapely.force_2d(g.geometry.array)
    g=g.to_crs(4326)
    b=g.geometry.bounds.to_numpy(); reason=np.full(len(g),'',object)
    def flag(mask,label): reason[(reason=='')&np.asarray(mask,bool)]=label
    flag(g.geometry.isna(),'NULL'); flag(g.geometry.is_empty,'EMPTY')
    allowed=['Polygon','MultiPolygon'] if kind=='polygon' else ['Point']
    flag(~g.geometry.geom_type.isin(allowed),'UNEXPECTED_TYPE')
    flag(~np.isfinite(b).all(axis=1)|(b[:,0]<-180)|(b[:,2]>180)|(b[:,1]<-90)|(b[:,3]>90),'INVALID_BOUNDS')
    flag(~g.geometry.is_valid,'INVALID_GEOMETRY')
    g['qa_geometry_issue']=reason
    return g

def run(archive,dest,districts,layers=LAYERS):
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        for n in layers:
            for ext in ('shp','shx','dbf','prj'):
                if names.count(f'landslides/{n}.{ext}')!=1: raise ValueError(f'Missing or repeated archive member: {n}.{ext}')
    dest.mkdir(parents=True,exist_ok=False)
    tree=shapely.STRtree(districts.geometry.array)
    frames={}; audits=[]; links=[]; issues=[]; profiles=[]
    for name,(expected,kind) in layers.items():
        g=read_layer(archive,name,expected,kind); frames[kind]=g
        bad=g[g.qa_geometry_issue!='']; issues.append(bad.drop(columns='geometry'))
        good=g[g.qa_geometry_issue==''].copy()
        a,j=tree.query(good.geometry.array,predicate='intersects')
        l=pd.DataFrame({'qa_record_id':good.qa_record_id.iloc[a].to_numpy(),'qa_source_layer':name,'ihr_state':districts.state.iloc[j].to_numpy(),'ihr_district':districts.district.iloc[j].to_numpy(),'qa_boundary_touch_only':shapely.touches(good.geometry.array[a],districts.geometry.array[j])})
        r=good[good.qa_record_id.isin(l.qa_record_id)].copy(); links.append(l)
        cnt=l.groupby('qa_record_id').size(); touched=l.groupby('qa_record_id').qa_boundary_touch_only.all()
        r['qa_district_matches']=r.qa_record_id.map(cnt).astype(int); r['qa_touch_only']=r.qa_record_id.map(touched).astype(bool)
        if len(r):
            path=dest/(name+'.gpkg'); pyogrio.write_dataframe(r,path,layer='candidates',driver='GPKG',promote_to_multi=kind=='polygon')
            saved=pyogrio.read_dataframe(path,layer='candidates')
            if len(saved)!=len(r) or saved.crs.to_epsg()!=4326 or not saved.qa_record_id.is_unique or set(saved.qa_record_id)!=set(r.qa_record_id): raise ValueError('Readback IDs/CRS mismatch')
            saved=saved.set_index('qa_record_id').loc[r.qa_record_id]
            if not saved.geometry.is_valid.all() or not shapely.equals(saved.geometry.array,r.geometry.array).all(): raise ValueError('Readback geometry mismatch')
            aa,jj=tree.query(saved.geometry.array,predicate='intersects')
            actual=set(zip(saved.index[aa],districts.state.iloc[jj],districts.district.iloc[jj],shapely.touches(saved.geometry.array[aa],districts.geometry.array[jj])))
            if actual!=set(l[['qa_record_id','ihr_state','ihr_district','qa_boundary_touch_only']].itertuples(index=False,name=None)): raise ValueError('Readback district links mismatch')
        for field in ['Category','category','category_1','DateTimeS','DateTime','Type','qa_original_geom_type']:
            if field in r:
                for value,count in r[field].fillna('').astype(str).value_counts().items(): profiles.append(dict(layer=name,field=field,value=value or '<BLANK>',records=int(count)))
        arow=dict(layer=name,kind=kind,source_records=len(g),invalid_records=len(bad),regional_candidates=len(r),outside_valid_records=len(good)-len(r),districts=l[['ihr_state','ihr_district']].drop_duplicates().shape[0],multiple_district_records=int((cnt>1).sum()),touch_only_records=int(touched.sum()),status='PASS' if bad.empty else 'REVIEW')
        assert arow['invalid_records']+arow['regional_candidates']+arow['outside_valid_records']==expected
        audits.append(arow); print(name,arow,flush=True)
    # Evaluate ORIG_FID against actual shapefile FIDs, not guessed row numbering.
    p=frames['polygon']; q=frames['associated_point']; gps=frames['gps']
    if 'ORIG_FID' not in q: raise ValueError('Missing associated point ORIG_FID')
    lookup=p.set_index('qa_source_fid'); validp=p[p.qa_geometry_issue=='']; validq=q[q.qa_geometry_issue=='']; validgps=gps[gps.qa_geometry_issue=='']
    ptree=shapely.STRtree(validp.geometry.array)
    spatial_a,spatial_b=ptree.query(validq.geometry.array,predicate='intersects')
    spatial=pd.DataFrame({'point_id':validq.qa_record_id.iloc[spatial_a].to_numpy(),'polygon_id':validp.qa_record_id.iloc[spatial_b].to_numpy()})
    spatial.to_csv(dest/'associated_point_spatial_links.csv',index=False)
    matches=[]
    for _,row in q.iterrows():
        value=pd.to_numeric(row.ORIG_FID,errors='coerce'); key=str(int(value)) if pd.notna(value) and np.isfinite(value) and value==int(value) else ''
        found=key in lookup.index
        poly=lookup.loc[key] if found else None
        status='NO_FID_MATCH' if not found else ('INVALID_GEOMETRY' if row.qa_geometry_issue or poly.qa_geometry_issue else ('POINT_COVERED' if poly.geometry.covers(row.geometry) else 'POINT_OUTSIDE_REFERENCED_POLYGON'))
        matches.append(dict(point_id=row.qa_record_id,orig_fid=row.ORIG_FID,polygon_id=poly.qa_record_id if found else '',status=status,point_Id=row.get('Id'),polygon_Id=poly.get('Id') if found else None))
    m=pd.DataFrame(matches); m.to_csv(dest/'associated_point_fid_audit.csv',index=False)
    ga,gb=ptree.query(validgps.geometry.array,predicate='intersects')
    gl=pd.DataFrame({'gps_id':validgps.qa_record_id.iloc[ga].to_numpy(),'polygon_id':validp.qa_record_id.iloc[gb].to_numpy()}); gl.to_csv(dest/'gps_polygon_spatial_links.csv',index=False)
    s=pd.DataFrame(audits); l=pd.concat(links,ignore_index=True)
    s.to_csv(dest/'lesser_regional_audit.csv',index=False); l.to_csv(dest/'lesser_district_links.csv',index=False)
    pd.concat(issues,ignore_index=True).to_csv(dest/'lesser_geometry_issues.csv',index=False)
    pd.DataFrame(profiles,columns=['layer','field','value','records']).to_csv(dest/'lesser_attribute_counts.csv',index=False)
    gps.drop(columns='geometry').to_csv(dest/'gps_original_attributes.csv',index=False)
    summary=dict(source_records=int(s.source_records.sum()),regional_candidates=int(s.regional_candidates.sum()),districts=int(l[['ihr_state','ihr_district']].drop_duplicates().shape[0]),fid_status=m.status.value_counts().to_dict(),associated_points_intersecting_polygon=int(spatial.point_id.nunique()),gps_points_intersecting_polygon=int(gl.gps_id.nunique()),repeated_orig_fid_records=int(q.ORIG_FID.dropna().duplicated(keep=False).sum()))
    (dest/'complete.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('\nFINAL LESSER HIMALAYAS AUDIT\n'+s.to_string(index=False)); print(json.dumps(summary,indent=2)); return s,summary

def self_test():
    with TemporaryDirectory() as tmp:
        t=Path(tmp); folder=t/'landslides'; folder.mkdir(); zpath=t/'test.zip'
        d=gpd.GeoDataFrame({'state':['A'],'district':['One']},geometry=[box(76,30,78,32)],crs=4326)
        p=gpd.GeoDataFrame({'Id':[1,2]},geometry=[box(76.5,30.5,76.7,30.7),box(79,30,79.1,30.1)],crs=4326).to_crs(32643)
        q=gpd.GeoDataFrame({'Id':[1,2],'ORIG_FID':[0,1]},geometry=p.geometry.representative_point(),crs=32643)
        gps=gpd.GeoDataFrame({'DateTimeS':['2023-07-01']},geometry=[Point(76.6,30.6,10)],crs=4326)
        for name,g in zip(LAYERS,[gps,p,q]): pyogrio.write_dataframe(g,folder/(name+'.shp'))
        with zipfile.ZipFile(zpath,'w') as z:
            for f in folder.iterdir(): z.write(f,f.relative_to(t).as_posix())
        layers={n:(len(g),LAYERS[n][1]) for n,g in zip(LAYERS,[gps,p,q])}
        s,summary=run(zpath,t/'out',d,layers)
        assert s.regional_candidates.tolist()==[1,1,1] and summary['fid_status']=={'POINT_COVERED':2} and summary['gps_points_intersecting_polygon']==1
    print('Synthetic ZIP, CRS, point/polygon association and readback: PASS')

def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--self-test',action='store_true'); args=parser.parse_args()
    if args.self_test: self_test(); return
    archive=ROOT/'data/raw/landslides/lesser_himalayas_2023/landslides.zip'
    dest=ROOT/'data/processed/landslides/lesser_himalayas_regional_v1'; stage=dest.with_name(dest.name+'.staging')
    if dest.exists() or stage.exists(): raise RuntimeError('Output or staging exists; no overwrite.')
    run(archive,stage,load_districts(ROOT)); stage.rename(dest); print('Output:',dest,'| Original archive unchanged.')
if __name__=='__main__': main()
