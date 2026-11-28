"""COOLR regional extraction and preliminary quality audit; no raw data edits.
Keep reports and events, points and polygons separate. No deduplication or date imputation.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
from collections import Counter
import argparse
import json
import re
import numpy as np
import pandas as pd
import geopandas as gpd
import pyogrio
import shapely
from pyproj import Geod
from shapely.geometry import Point, box
from extract_uglc_regional import load_districts

ROOT = Path(__file__).resolve().parent.parent
LAYERS = {'COOLR_Events_Points': 10184, 'COOLR_Events_Polygons': 1758,
          'COOLR_Reports_Points': 4022, 'COOLR_Reports_Polygons': 8}
GEOD = Geod(ellps='WGS84')
MISSING = {'','nan','none','null','na','n/a','unknown','nat','-','<na>'}

def text(x):
    return '' if pd.isna(x) else str(x).strip()

def date_status(x):
    s=text(x)
    if s.casefold() in MISSING: return 'MISSING_OR_UNKNOWN'
    # ISO calendar date or ISO timestamp only; never interpret ambiguous numeric dates.
    if re.fullmatch(r'\d{4}-\d{2}-\d{2}(?:[ T].+)?', s):
        try:
            t=pd.Timestamp(s)
            if pd.isna(t): return 'INVALID_DATE'
            return 'PARSED_DAY_UNVERIFIED'
        except (ValueError, OverflowError, TypeError): return 'INVALID_DATE'
    return 'OTHER_FORMAT_REVIEW'

def process(path, name, expected, districts, dest):
    raw=pyogrio.read_dataframe(path, fid_as_index=True, datetime_as_string=True)
    if len(raw)!=expected or raw.crs is None or raw.crs.to_epsg()!=4326 or not raw.index.is_unique:
        raise ValueError(f'{name}: source count, CRS or feature identifier mismatch')
    if any(c.lower().startswith('qa_') for c in raw.columns): raise ValueError('Reserved QA field collision')
    if not {'event_id','event_date','source_name'}.issubset(raw.columns): raise ValueError('Unexpected schema')
    g=raw.copy(); g['qa_source_fid']=g.index.astype(str); g=g.reset_index(drop=True)
    g['qa_record_id']=name+':'+g.qa_source_fid; g['qa_source_layer']=name
    g['qa_record_kind']='report' if '_Reports_' in name else 'event'
    g['qa_original_geom_type']=g.geometry.geom_type.fillna('NULL')
    g['qa_had_z']=shapely.has_z(g.geometry.array)
    allowed=['Point'] if name.endswith('_Points') else ['Polygon','MultiPolygon']
    reason=np.full(len(g),'',dtype=object)
    def flag(mask,label): reason[(reason=='') & np.asarray(mask,dtype=bool)]=label
    flag(g.geometry.isna(),'NULL'); flag(g.geometry.is_empty,'EMPTY')
    flag(~g.qa_original_geom_type.isin(allowed),'UNEXPECTED_TYPE')
    b=g.geometry.bounds.to_numpy()
    flag(~np.isfinite(b).all(axis=1)|(b[:,0]<-180)|(b[:,2]>180)|(b[:,1]<-90)|(b[:,3]>90),'INVALID_BOUNDS')
    g.geometry=shapely.force_2d(g.geometry.array)
    flag(~g.geometry.is_valid,'INVALID_GEOMETRY')
    g['qa_geometry_issue']=reason
    bad=g.loc[reason!='',['qa_record_id','qa_source_fid','event_id','qa_original_geom_type','qa_geometry_issue']].copy()
    bad['qa_source_layer']=name
    good=g.loc[reason==''].copy()
    tree=shapely.STRtree(districts.geometry.array)
    pairs=tree.query(good.geometry.array,predicate='intersects')
    a,j=pairs
    touched=shapely.touches(good.geometry.array[a],districts.geometry.array[j])
    links=pd.DataFrame({'qa_record_id':good.qa_record_id.iloc[a].to_numpy(),
                        'ihr_state':districts.state.iloc[j].to_numpy(),
                        'ihr_district':districts.district.iloc[j].to_numpy(),
                        'qa_boundary_touch_only':touched,'qa_source_layer':name})
    regional=good[good.qa_record_id.isin(links.qa_record_id)].copy()
    count=links.groupby('qa_record_id').size(); touch=links.groupby('qa_record_id').qa_boundary_touch_only.all()
    regional['qa_district_matches']=regional.qa_record_id.map(count).astype(int)
    regional['qa_touch_only']=regional.qa_record_id.map(touch).astype(bool)
    regional['qa_date_status']=regional.event_date.map(date_status)
    regional['qa_coord_difference_m']=np.nan
    regional['qa_coord_status']='NOT_APPLICABLE_POLYGON' if name.endswith('_Polygons') else 'MISSING_OR_INVALID_ATTRIBUTE'
    if name.endswith('_Points') and {'latitude','longitude'}.issubset(regional.columns):
        lat=pd.to_numeric(regional.latitude,errors='coerce'); lon=pd.to_numeric(regional.longitude,errors='coerce')
        ok=lat.between(-90,90)&lon.between(-180,180)
        if ok.any():
            _,_,dist=GEOD.inv(regional.geometry.x[ok].to_numpy(),regional.geometry.y[ok].to_numpy(),lon[ok].to_numpy(),lat[ok].to_numpy())
            regional.loc[ok,'qa_coord_difference_m']=dist
            regional.loc[ok,'qa_coord_status']=np.select([dist<=30,dist<=100],['LE_30M','GT_30_LE_100M'],default='GT_100M')
    if 'location_accuracy' not in regional: regional['qa_location_accuracy_raw']='NO_FIELD'
    else: regional['qa_location_accuracy_raw']=regional.location_accuracy.map(text)
    target=dest/(name+'.gpkg')
    if len(regional):
        pyogrio.write_dataframe(regional,target,layer='candidates',driver='GPKG',promote_to_multi=name.endswith('_Polygons'))
        saved=pyogrio.read_dataframe(target,layer='candidates')
        if (len(saved)!=len(regional) or saved.crs.to_epsg()!=4326 or not saved.qa_record_id.is_unique
            or set(saved.qa_record_id)!=set(regional.qa_record_id) or not saved.geometry.is_valid.all()
            or saved.geometry.is_empty.any() or not saved.geometry.geom_type.isin(allowed).all()): raise ValueError('Readback mismatch')
        aligned=saved.set_index('qa_record_id').loc[regional.qa_record_id]
        if not shapely.equals(aligned.geometry.array,regional.geometry.array).all(): raise ValueError('Readback geometry changed')
        aa,jj=tree.query(saved.geometry.array,predicate='intersects')
        tt=shapely.touches(saved.geometry.array[aa],districts.geometry.array[jj])
        actual=set(zip(saved.qa_record_id.iloc[aa],districts.state.iloc[jj],districts.district.iloc[jj],tt))
        expected_links=set(links[['qa_record_id','ihr_state','ihr_district','qa_boundary_touch_only']].itertuples(index=False,name=None))
        if actual!=expected_links: raise ValueError('Readback district links mismatch')
    profiles=[]
    for field in ['source_name','landslide_category','landslide_trigger','method','qa_location_accuracy_raw','qa_date_status','qa_coord_status','qa_original_geom_type']:
        if field in regional:
            profiles.extend({'layer':name,'field':field,'value':v or '<BLANK>','records':n} for v,n in Counter(regional[field].map(text)).most_common())
    cols=['qa_record_id','qa_source_layer','qa_source_fid','qa_record_kind','event_id','event_date','qa_date_status','qa_location_accuracy_raw','qa_coord_status','qa_coord_difference_m','qa_district_matches','qa_touch_only','source_name','source_link','event_import_source','event_import_id','landslide_category','landslide_trigger','method','sat_date_before','sat_date_after','event_time']
    attrs=regional.reindex(columns=cols).copy()
    # Keep candidate dates raw, not verified occurrence dates.
    stats=dict(layer=name,source_records=len(g),invalid_records=len(bad),valid_records=len(good),regional_candidates=len(regional),outside_valid_records=len(good)-len(regional),districts=links[['ihr_state','ihr_district']].drop_duplicates().shape[0],district_links=len(links),multiple_district_records=int((count>1).sum()),touch_only_records=int(touch.sum()),status='PASS' if bad.empty else 'REVIEW')
    if stats['invalid_records']+stats['outside_valid_records']+stats['regional_candidates']!=expected: raise ValueError('Accounting mismatch')
    print(f'{name}: {stats["status"]} | regional {len(regional):,} | invalid {len(bad):,} | outside {stats["outside_valid_records"]:,} | districts {stats["districts"]}',flush=True)
    return stats, links, bad, attrs, profiles

def run(raw,dest,districts,layers):
    dest.mkdir(parents=True,exist_ok=False)
    stats=[]; links=[]; issues=[]; attrs=[]; profiles=[]
    for name,n in layers.items():
        r,l,b,a,p=process(raw/(name+'.geojson'),name,n,districts,dest)
        stats.append(r); links.append(l); issues.append(b); attrs.append(a); profiles.extend(p)
    s=pd.DataFrame(stats); l=pd.concat(links,ignore_index=True); a=pd.concat(attrs,ignore_index=True)
    if not a.qa_record_id.is_unique: raise ValueError('Duplicate QA record IDs')
    s.to_csv(dest/'coolr_regional_audit.csv',index=False); l.to_csv(dest/'coolr_district_links.csv',index=False)
    pd.concat(issues,ignore_index=True).to_csv(dest/'coolr_geometry_issues.csv',index=False)
    a.to_csv(dest/'coolr_candidate_attributes.csv',index=False)
    pd.DataFrame(profiles,columns=['layer','field','value','records']).to_csv(dest/'coolr_attribute_counts.csv',index=False)
    keys=districts[['state','district']].rename(columns={'state':'ihr_state','district':'ihr_district'})
    coverage=keys.copy()
    for name in layers:
        c=l[l.qa_source_layer==name].groupby(['ihr_state','ihr_district']).size().rename(name).reset_index()
        coverage=coverage.merge(c,how='left',on=['ihr_state','ihr_district']); coverage[name]=coverage[name].fillna(0).astype(int)
    coverage.to_csv(dest/'coolr_district_counts.csv',index=False)
    ids=a.event_id.map(text); nonmissing=~ids.str.casefold().isin(MISSING)
    repeat=nonmissing & ids.duplicated(keep=False)
    a.loc[repeat].to_csv(dest/'coolr_repeated_event_ids.csv',index=False)
    print('\nFINAL COOLR REGIONAL AUDIT\n'+s.to_string(index=False))
    print('\nTotal source:',s.source_records.sum(),'| regional:',len(a),'| district links:',len(l))
    print('Districts with any candidate:',l[['ihr_state','ihr_district']].drop_duplicates().shape[0],'/',len(districts))
    print('Repeated nonmissing event-ID records (candidate matches only):',int(repeat.sum()))
    for field in ['qa_date_status','qa_location_accuracy_raw','landslide_trigger','source_name','qa_coord_status']:
        print('\n'+field+':\n'+a.groupby('qa_source_layer')[field].value_counts(dropna=False).to_string())
    (dest/'complete.json').write_text(json.dumps({'layers':stats,'regional_candidates':len(a),'district_links':len(l),'repeated_event_id_records':int(repeat.sum())},indent=2),encoding='utf-8')
    return s,l,a

def self_test():
    d=gpd.GeoDataFrame({'state':['A','A'],'district':['One','Two']},geometry=[box(70,20,71,21),box(71,20,72,21)],crs=4326)
    with TemporaryDirectory() as t:
        p=Path(t); layers={'COOLR_Events_Points':4,'COOLR_Reports_Polygons':3}
        for name,geoms in [(list(layers)[0],[Point(70.5,20.5),Point(71,20.5),Point(80,20),None]),(list(layers)[1],[box(70.5,20.5,71.5,20.7),box(72,20,73,21),box(80,20,81,21)])]:
            n=len(geoms); g=gpd.GeoDataFrame({'event_id':['1']*n,'event_date':['2023-07-01']*n,'source_name':['test']*n,'latitude':[20.5]*n,'longitude':[70.5]*n},geometry=geoms,crs=4326)
            pyogrio.write_dataframe(g,p/(name+'.geojson'),driver='GeoJSON')
        s,l,a=run(p,p/'out',d,layers)
        assert s.regional_candidates.tolist()==[2,2] and s.invalid_records.tolist()==[1,0]
        assert s.multiple_district_records.tolist()==[1,1] and s.touch_only_records.tolist()==[1,1]
        assert len(l)==6 and a.qa_record_id.is_unique
        assert date_status('2023-02-30')=='INVALID_DATE' and date_status('01/02/2023')=='OTHER_FORMAT_REVIEW'
    print('Synthetic regional extraction, null geometry, boundaries, dates and readback: PASS')

def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--self-test',action='store_true'); args=parser.parse_args()
    if args.self_test: self_test(); return
    d=load_districts(ROOT)
    raw=ROOT/'data/raw/landslides/COOLR'; dest=ROOT/'data/processed/landslides/coolr_regional_v1'; stage=dest.with_name(dest.name+'.staging')
    if dest.exists() or stage.exists(): raise RuntimeError('Output or staging directory exists; inspect it before rerunning. No overwrite.')
    run(raw,stage,d,LAYERS); stage.rename(dest)
    print('Output:',dest,'| Original files unchanged.')
if __name__=='__main__': main()
