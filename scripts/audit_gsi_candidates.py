"""Read-only QA of GSI regional candidates; no deduplication or training labels."""
from pathlib import Path
from datetime import date, datetime
import argparse
import hashlib
import re
import tempfile
import geopandas as gpd
import pandas as pd
import numpy as np
import shapely
from pyproj import Geod
from extract_gsi_regional import history_kind, mapping_kind, clean_text, MISSING

ROOT = Path(__file__).resolve().parent.parent
GEOD = Geod(ellps='WGS84')
MONTHS = {s.lower(): i for i, s in enumerate('January February March April May June July August September October November December'.split(), 1)}
MONTHS.update({s[:3]: i for s,i in list(MONTHS.items())})

def require(condition, message):
    if not condition: raise ValueError(message)

def text_key(v):
    s=clean_text(v).casefold()
    return '' if s in MISSING else s

def day_candidate(value):
    """Only parse standalone unambiguous calendar dates; never infer occurrence."""
    s=clean_text(value).lower()
    kind=history_kind(value)
    if kind!='DAY_FORMAT_CANDIDATE': return kind, ''
    try:
        m=re.fullmatch(r'(\d{1,2})\s+([a-z]+)\s+(\d{4})',s)
        if m:
            d,mon,y=m.groups(); dt=date(int(y),MONTHS[mon],int(d))
        elif re.fullmatch(r'\d{4}-\d{2}-\d{2}',s):
            dt=date.fromisoformat(s)
        else:
            a,b,y=map(int,re.split(r'[-/]',s))
            if a<=12 and b<=12 and a!=b: return 'AMBIGUOUS_NUMERIC_DAY', ''
            if a>12: dt=date(y,b,a)
            else: dt=date(y,a,b)
        if dt>date.today(): return 'FUTURE_DATE', ''
        return 'PARSED_DAY_UNVERIFIED',dt.isoformat()
    except (ValueError,KeyError): return 'INVALID_DAY', ''

def geometry_key(geom):
    # Normalize ring orientation and component order; no coordinate rounding.
    # Treat a single polygon and a one-component multipolygon equivalently.
    if geom.geom_type=='MultiPolygon' and len(geom.geoms)==1: geom=geom.geoms[0]
    return hashlib.sha256(shapely.to_wkb(shapely.normalize(geom),byte_order=1,output_dimension=2)).hexdigest()

def compare_links(g,districts,expected):
    joined=gpd.sjoin(g[['qa_record_id','geometry']],districts,how='inner',predicate='intersects')
    joined['qa_boundary_touch_only']=shapely.touches(joined.geometry.array,districts.geometry.loc[joined.index_right.to_numpy()].array)
    cols=['qa_record_id','ihr_state','ihr_district','qa_boundary_touch_only']
    a=joined[cols].sort_values(cols).reset_index(drop=True)
    b=expected[cols].sort_values(cols).reset_index(drop=True)
    pd.testing.assert_frame_equal(a,b,check_dtype=False)
    counts=a.groupby('qa_record_id').size()
    require(g.qa_record_id.map(counts).eq(g.qa_district_matches).all(),'District match count mismatch')
    return a

def audit_layer(g, row, districts, links):
    name=row['layer']; point=name.startswith('LANDSLIDE_POINT_')
    require(g.crs is not None and g.crs.to_epsg()==4326,'Unexpected CRS')
    require(len(g)==int(row['regional_candidates']),'Readback count mismatch')
    require(g.qa_record_id.notna().all() and g.qa_record_id.is_unique,'Missing or duplicate record ID')
    require(g.qa_source_layer.eq(name).all(),'Source layer mismatch')
    require(g.qa_record_id.eq(name+':'+g.qa_source_fid.astype(str)).all(),'Record provenance mismatch')
    require(g.geometry.notna().all() and (~g.geometry.is_empty).all() and g.geometry.is_valid.all(),'Invalid readback geometry')
    require(g.geometry.geom_type.isin(['Point'] if point else ['Polygon','MultiPolygon']).all(),'Unexpected geometry type')
    require(not shapely.has_z(g.geometry.array).any(),'Unexpected Z in 2D output')
    bounds=g.geometry.bounds.to_numpy()
    require(np.isfinite(bounds).all() and (bounds[:,0]>=-180).all() and (bounds[:,2]<=180).all() and (bounds[:,1]>=-90).all() and (bounds[:,3]<=90).all(),'Invalid coordinate bounds')
    matched=compare_links(g,districts,links)
    require(int((g.qa_district_matches>1).sum())==int(row['multiple_district_records']),'Multiple district count mismatch')
    require(int(matched.groupby('qa_record_id').qa_boundary_touch_only.all().sum())==int(row['touch_only_records']),'Touch-only count mismatch')
    if point:
        require(g.qa_mapping_kind.eq('POINT_RECORD').all(),'Point classification mismatch')
        require(g.history.map(history_kind).eq(g.qa_history_format).all(),'History classification mismatch')
    else:
        require(g.data_type.map(mapping_kind).eq(g.qa_mapping_kind).all(),'Mapping method mismatch')
        require(g.qa_history_format.eq('NO_DEDICATED_HISTORY_FIELD').all(),'Unexpected polygon history classification')
    ids=pd.DataFrame({'qa_record_id':g.qa_record_id,'source_layer':name,'kind':'point' if point else 'polygon','gid_key':g.gid.map(text_key),'slide_no_key':g.slide_no.map(text_key) if point else '', 'geometry_key':[geometry_key(x) for x in g.geometry]})
    dates=pd.DataFrame()
    if point:
        lat=pd.to_numeric(g.latitude,errors='coerce'); lon=pd.to_numeric(g.longitude,errors='coerce')
        ok=lat.between(-90,90)&lon.between(-180,180)
        diff=np.full(len(g),np.nan)
        if ok.any():
            _,_,dist=GEOD.inv(lon[ok].to_numpy(),lat[ok].to_numpy(),g.geometry.x[ok].to_numpy(),g.geometry.y[ok].to_numpy()); diff[ok.to_numpy()]=dist
        require(np.allclose(diff,pd.to_numeric(g.qa_coord_difference_m,errors='coerce'),atol=0.01,rtol=0,equal_nan=True),'Coordinate difference report mismatch')
        parsed=[day_candidate(v) for v in g.history]
        dates=pd.DataFrame({'qa_record_id':g.qa_record_id,'source_layer':name,'history_raw':g.history,'history_format':g.qa_history_format,'date_status':[x[0] for x in parsed],'candidate_date':[x[1] for x in parsed],'trigger_raw':g.triggering,'coord_difference_m':diff,'coord_status':np.select([~ok,diff>100,diff>30],['MISSING_OR_INVALID_ATTRIBUTE','GT_100M','GT_30M_TO_100M'],default='LE_30M')})
    return ids,dates

def duplicate_candidates(ids):
    frames=[]
    for col, label in [('gid_key','SAME_GID_WITHIN_KIND'),('slide_no_key','SAME_SLIDE_NO'),('geometry_key','EXACT_NORMALIZED_GEOMETRY')]:
        x=ids[ids[col]!=''].copy()
        x['group_size']=x.groupby(['kind',col])[col].transform('size')
        x=x[x.group_size>1].copy()
        x['match_type']=label; x['match_key']=x[col]
        frames.append(x[['qa_record_id','source_layer','kind','match_type','match_key','group_size']])
    return pd.concat(frames,ignore_index=True)

def main(root):
    base=root/'data/processed/landslides/gsi_regional_v1'
    out=root/'data/processed/landslides/gsi_quality_v1'; out.mkdir(parents=True,exist_ok=True)
    report=pd.read_csv(base/'gsi_regional_layer_audit.csv')
    require(len(report)==32 and report.layer.nunique()==32 and report.status.eq('PASS').all(),'Expected 32 PASS layers')
    require(report.source_records.sum()==111962 and report.regional_candidates.sum()==111958 and report.invalid_records.sum()==0,'Unexpected source accounting')
    require(set(p.stem for p in base.glob('*.gpkg'))==set(report.layer),'Missing or unexpected GeoPackages')
    links=pd.read_csv(base/'gsi_district_links.csv',dtype={'qa_record_id':str,'ihr_state':str,'ihr_district':str,'qa_source_layer':str})
    require(links.qa_boundary_touch_only.isin([True,False]).all(),'Invalid touch flags')
    require(not links.duplicated(['qa_record_id','ihr_state','ihr_district']).any(),'Duplicate district links')
    require(set(links.qa_source_layer)==set(report.layer),'Link layer inventory mismatch')
    districts=gpd.read_file(root/'data/processed/boundary/ihr_districts.gpkg')[['state','district','geometry']].to_crs(4326).rename(columns={'state':'ihr_state','district':'ihr_district'}).reset_index(drop=True)
    districts.geometry=districts.geometry.make_valid()
    require(len(districts)==137 and not districts.duplicated(['ihr_state','ihr_district']).any() and districts.geometry.geom_type.isin(['Polygon','MultiPolygon']).all() and (~districts.geometry.is_empty).all(),'Invalid district inventory')
    ids=[]; dates=[]; audits=[]; maps=[]; hist=[]; types=[]
    for i,row in enumerate(report.to_dict('records'),1):
        name=row['layer']; print(f'[{i}/32] {name}',flush=True)
        g=gpd.read_file(base/f'{name}.gpkg',layer='candidates',engine='pyogrio')
        a,b=audit_layer(g,row,districts,links[links.qa_source_layer==name]); ids.append(a); dates.append(b)
        for col,target in [('qa_mapping_kind',maps),('qa_history_format',hist),('qa_original_geom_type',types)]:
            for k,n in g[col].value_counts().items(): target.append({'layer':name,'category':k,'records':n})
        audits.append({'layer':name,'records':len(g),'status':'PASS'}); print(f'  PASS | {len(g):,} records',flush=True)
    ids=pd.concat(ids,ignore_index=True); dates=pd.concat(dates,ignore_index=True)
    require(ids.qa_record_id.is_unique and set(ids.qa_record_id)==set(links.qa_record_id),'Global record/link identity mismatch')
    for rows,file in [(maps,'gsi_mapping_method_counts.csv'),(hist,'gsi_history_format_counts.csv')]:
        cols=['layer','category','records']; a=pd.DataFrame(rows).sort_values(cols).reset_index(drop=True); b=pd.read_csv(base/file).sort_values(cols).reset_index(drop=True)
        pd.testing.assert_frame_equal(a,b,check_dtype=False)
    dup=duplicate_candidates(ids)
    coverage=districts[['ihr_state','ihr_district']].merge(links.merge(ids[['qa_record_id','kind']],on='qa_record_id',validate='many_to_one').groupby(['ihr_state','ihr_district','kind']).qa_record_id.nunique().unstack(fill_value=0).reset_index(),how='left',on=['ihr_state','ihr_district']).fillna(0)
    coverage[['point','polygon']]=coverage[['point','polygon']].astype(int)
    tables={'gsi_readback_audit.csv':pd.DataFrame(audits),'gsi_duplicate_candidates.csv':dup,'gsi_point_history_audit.csv':dates,'gsi_district_geometry_counts.csv':coverage,'gsi_actual_type_summary.csv':pd.DataFrame(types),'gsi_date_status_counts.csv':dates.date_status.value_counts().rename_axis('status').reset_index(name='records'),'gsi_coordinate_status_counts.csv':dates.coord_status.value_counts().rename_axis('status').reset_index(name='records')}
    for name,df in tables.items(): df.to_csv(out/name,index=False)
    print('\nFINAL GSI CANDIDATE QUALITY AUDIT\nLayers: 32/32 PASS | Records:',len(ids))
    print('Districts with point candidates:',int((coverage.point>0).sum()),'/137')
    print('Districts with polygon candidates:',int((coverage.polygon>0).sum()),'/137')
    print('\nDuplicate candidate groups and records (categories can overlap):')
    if dup.empty: print('None')
    else: print(dup.groupby('match_type').agg(groups=('match_key',lambda x: len(x)),records=('qa_record_id','nunique')).drop(columns='groups').join(dup.drop_duplicates(['match_type','kind','match_key']).groupby('match_type').size().rename('groups')).to_string())
    print('\nDate status:\n'+dates.date_status.value_counts().to_string())
    print('\nCoordinate status:\n'+dates.coord_status.value_counts().to_string())
    print('\nReports:',out,'| No records deleted; source files unchanged.')

def self_test():
    assert day_candidate('28 April 2021')==('PARSED_DAY_UNVERIFIED','2021-04-28')
    assert day_candidate('03/04/2021')[0]=='AMBIGUOUS_NUMERIC_DAY'
    assert day_candidate('31/02/2021')[0]=='INVALID_DAY'
    assert day_candidate('2017, 11 July 2018')[0]=='OTHER_OR_MULTIPLE'
    assert day_candidate('2015')[0]=='YEAR_TEXT'
    p=shapely.box(0,0,1,1); assert geometry_key(p)==geometry_key(shapely.MultiPolygon([p]))
    d=pd.DataFrame({'qa_record_id':['a','b','c'],'source_layer':['x','y','z'],'kind':['point','point','polygon'],'gid_key':['1','1','1'],'slide_no_key':['s','s',''],'geometry_key':['x','x','y']})
    assert len(duplicate_candidates(d))==6
    g=gpd.GeoDataFrame({'qa_record_id':['a'],'qa_district_matches':[1]},geometry=[shapely.Point(0.5,0.5)],crs=4326)
    districts=gpd.GeoDataFrame({'ihr_state':['X'],'ihr_district':['Y']},geometry=[p],crs=4326)
    links=pd.DataFrame({'qa_record_id':['a'],'ihr_state':['X'],'ihr_district':['Y'],'qa_boundary_touch_only':[False]})
    with tempfile.TemporaryDirectory() as t:
        path=Path(t)/'test.gpkg'; g.to_file(path,layer='candidates',engine='pyogrio',index=False); compare_links(gpd.read_file(path,engine='pyogrio'),districts,links)
    print('Synthetic date, duplicate grouping, GeoPackage readback and district link tests: PASS')

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,default=ROOT); parser.add_argument('--self-test',action='store_true'); args=parser.parse_args()
    self_test()
    if not args.self_test: main(args.root)
