"""Finish GLC regional export and profile temporal fields before integration.
Read-only inputs, no deduplication, event-date inference or source modifications.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
from collections import Counter
import argparse, json, re
import pandas as pd
import numpy as np
import geopandas as gpd
import pyogrio
import shapely
from shapely.geometry import box
from extract_uglc_regional import load_districts

ROOT = Path(__file__).resolve().parent.parent
MISSING = {'','na','n/a','nan','none','null','nat','unknown','nil','-','--','<na>'}

def require(condition, message):
    if not condition:
        raise ValueError(message)

def read_csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig')

def glc_export(path, dest, districts, expected=11033, expected_regional=1179):
    a = read_csv(path)
    a.columns = a.columns.str.strip().str.lower().str.replace(' ', '_', regex=False)
    require(a.columns.is_unique and not any(c.startswith('qa_') for c in a), 'GLC field collision')
    require(len(a)==expected and {'longitude','latitude','event_id','event_date','location_accuracy','landslide_trigger'}.issubset(a.columns), 'GLC count/schema mismatch')
    a['qa_csv_record_number'] = np.arange(1,len(a)+1)
    a['qa_record_id'] = 'GLC:' + a.qa_csv_record_number.astype(str)
    x = pd.to_numeric(a.longitude,errors='coerce'); y = pd.to_numeric(a.latitude,errors='coerce')
    ok = x.between(-180,180) & y.between(-90,90)
    g = gpd.GeoDataFrame(a.loc[ok].copy(),geometry=gpd.points_from_xy(x[ok],y[ok]),crs=4326).reset_index(drop=True)
    tree = shapely.STRtree(districts.geometry.array)
    i,j = tree.query(g.geometry.array,predicate='intersects')
    links = pd.DataFrame({'qa_record_id':g.qa_record_id.iloc[i].to_numpy(),'ihr_state':districts.state.iloc[j].to_numpy(),'ihr_district':districts.district.iloc[j].to_numpy(),'qa_boundary_touch_only':shapely.touches(g.geometry.array[i],districts.geometry.array[j])})
    r = g[g.qa_record_id.isin(links.qa_record_id)].copy()
    require(len(r)==expected_regional, f'GLC regional count differs from initial inspection: {len(r)} vs {expected_regional}')
    r['qa_district_matches'] = r.qa_record_id.map(links.groupby('qa_record_id').size()).astype(int)
    r['qa_touch_only'] = r.qa_record_id.map(links.groupby('qa_record_id').qa_boundary_touch_only.all()).astype(bool)
    dest.mkdir(parents=True,exist_ok=False)
    target=dest/'candidates.gpkg'
    pyogrio.write_dataframe(r,target,layer='candidates',driver='GPKG')
    saved=pyogrio.read_dataframe(target,layer='candidates')
    require(len(saved)==len(r) and saved.crs.to_epsg()==4326 and saved.qa_record_id.is_unique and set(saved.qa_record_id)==set(r.qa_record_id), 'GLC readback ID/CRS mismatch')
    saved=saved.set_index('qa_record_id').loc[r.qa_record_id]
    require(shapely.equals(saved.geometry.array,r.geometry.array).all(), 'GLC readback geometry mismatch')
    require(saved.event_date.fillna('').astype(str).tolist()==r.event_date.tolist(), 'GLC date attribute readback mismatch')
    links.to_csv(dest/'glc_district_links.csv',index=False)
    r.drop(columns='geometry').to_csv(dest/'glc_candidate_attributes.csv',index=False)
    a.loc[~ok].to_csv(dest/'glc_invalid_coordinates.csv',index=False)
    summary=dict(source_records=len(a),invalid_coordinates=int((~ok).sum()),outside_valid_records=len(g)-len(r),regional_candidates=len(r),districts=len(links[['ihr_state','ihr_district']].drop_duplicates()),district_links=len(links),multiple_district_records=int((r.qa_district_matches>1).sum()),readback='PASS')
    (dest/'complete.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('GLC regional export:',json.dumps(summary,indent=2),flush=True)
    return r.drop(columns='geometry')

def signature(value):
    s=str(value).strip()
    if s.casefold() in MISSING: return 'MISSING_TOKEN'
    # Replace individual digit runs with their lengths; preserve separators and words.
    # Not a date parser: no guessing day/month order, epoch units, or occurrence meaning.
    return re.sub(r'\d+|[A-Za-z]+',lambda m: ('D'+str(len(m.group()))) if m.group().isdigit() else 'A',s)

def profile(frames, out):
    rows=[]; values=[]
    for source,layer,df,fields in frames:
        require('qa_record_id' in df and df.qa_record_id.ne('').all() and df.qa_record_id.is_unique, f'{source}/{layer}: record IDs invalid')
        for field in fields:
            require(field in df, f'{source}/{layer}: missing date field {field}')
            counts=Counter(df[field].fillna('').astype(str))
            groups={}
            for val,n in counts.items():
                sig=signature(val); groups.setdefault(sig,[]).append((val,n))
                values.append(dict(source=source,layer=layer,field=field,format_signature=sig,raw_value=val,records=n))
            for sig,items in groups.items():
                items.sort(key=lambda p:(-p[1],p[0]))
                rows.append(dict(source=source,layer=layer,field=field,format_signature=sig,records=sum(n for _,n in items),distinct_values=len(items),examples=json.dumps([v for v,_ in items[:5]],ensure_ascii=False)))
    result=pd.DataFrame(rows).sort_values(['source','layer','field','records'],ascending=[True,True,True,False])
    result.to_csv(out/'temporal_format_profile.csv',index=False)
    pd.DataFrame(values).to_csv(out/'temporal_raw_value_counts.csv',index=False)
    for (source,layer,field),g in result.groupby(['source','layer','field'],sort=False):
        print(f'\n{source} / {layer} / {field}: {g.records.sum():,} values')
        print(g[['format_signature','records','examples']].head(6).to_string(index=False))
    return result

def lesser_review(base,out,expected=3176,regional=3153):
    m=read_csv(base/'associated_point_fid_audit.csv')
    require(len(m)==expected and m.point_id.is_unique and m.orig_fid.ne('').all() and m.orig_fid.is_unique and m.polygon_id.ne('').all() and m.polygon_id.is_unique,'Lesser association inventory mismatch')
    require(m.status.isin(['POINT_COVERED','POINT_OUTSIDE_REFERENCED_POLYGON']).all(),'Unexpected Lesser association status')
    p=pyogrio.read_dataframe(base/'landslides_shimla_points.gpkg',layer='candidates',read_geometry=False,columns=['qa_record_id'])
    q=pyogrio.read_dataframe(base/'landslides_shimla.gpkg',layer='candidates',read_geometry=False,columns=['qa_record_id'])
    require(len(p)==len(q)==regional and p.qa_record_id.is_unique and q.qa_record_id.is_unique,'Lesser regional counts mismatch')
    require(set(p.qa_record_id)<=set(m.point_id) and set(q.qa_record_id)<=set(m.polygon_id),'Lesser regional association IDs mismatch')
    m['point_in_region']=m.point_id.isin(p.qa_record_id); m['polygon_in_region']=m.polygon_id.isin(q.qa_record_id)
    m['representation_status']='ASSOCIATED_POINT_NOT_INDEPENDENT'
    review=m[(m.status!='POINT_COVERED')|(m.point_in_region!=m.polygon_in_region)].copy()
    review.to_csv(out/'lesser_association_review.csv',index=False)
    print('\nLesser association review:')
    print(m.groupby(['status','point_in_region','polygon_in_region']).size().to_string())
    return m

def counted(path,expected):
    df=read_csv(path)
    require(len(df)==expected,f'{path.name}: {len(df)} rows; expected {expected}')
    return df

def run(root,out):
    base=root/'data/processed/landslides'
    out.mkdir(parents=True,exist_ok=False)
    d=load_districts(root)
    glc=glc_export(root/'data/raw/landslides/glc_global.csv',out/'glc_regional',d)
    frames=[('GLC','points',glc,['event_date','event_time'])]
    gsi=counted(base/'gsi_quality_v1/gsi_point_history_audit.csv',27484)
    frames.append(('GSI','points',gsi,['history_raw']))
    for kind,n in [('points',2110),('polygons',5430)]:
        f=counted(base/f'uglc_quality_v1/{kind}/date_accuracy_candidates.csv',n)
        frames.append(('UGLC',kind,f,['START DATE','END DATE']))
    c=counted(base/'coolr_regional_v1/coolr_candidate_attributes.csv',1504)
    require(c.qa_source_layer.value_counts().to_dict()=={'COOLR_Events_Points':177,'COOLR_Reports_Points':1323,'COOLR_Reports_Polygons':4},'COOLR layer counts mismatch')
    for layer,f in c.groupby('qa_source_layer'):
        frames.append(('COOLR',layer,f,['event_date','event_time','sat_date_before','sat_date_after']))
    gps=counted(base/'lesser_himalayas_regional_v1/gps_original_attributes.csv',359)
    frames.append(('LESSER_HIMALAYAS_2023','GPS',gps,['DateTimeS','DateTime']))
    lesser_review(base/'lesser_himalayas_regional_v1',out)
    profile(frames,out)
    (out/'complete.json').write_text(json.dumps({'status':'PASS','purpose':'GLC export and temporal format profile only','date_fields':[{'source':s,'layer':l,'records':len(f),'fields':fields} for s,l,f,fields in frames]},indent=2),encoding='utf-8')

def self_test():
    with TemporaryDirectory() as tmp:
        p=Path(tmp)
        d=gpd.GeoDataFrame({'state':['X','X'],'district':['A','B']},geometry=[box(70,20,71,21),box(71,20,72,21)],crs=4326)
        df=pd.DataFrame({'longitude':['70.5','71','80','200'],'latitude':['20.5']*4,'event_id':['1','2','3','4'],'event_date':['01/02/2023 12:00:00 PM','2023-01-02','','NA'],'event_time':['']*4,'location_accuracy':['exact']*4,'landslide_trigger':['rain']*4})
        df.to_csv(p/'glc.csv',index=False)
        a=glc_export(p/'glc.csv',p/'glc',d,4,2)
        r=profile([('GLC','points',a,['event_date','event_time'])],p)
        assert r.records.sum()==4 and signature('01/02/2023')=='D2/D2/D4' and signature('NA')=='MISSING_TOKEN'
        b=p/'lesser'; b.mkdir()
        pd.DataFrame({'point_id':['p1','p2'],'polygon_id':['q1','q2'],'orig_fid':['0','1'],'status':['POINT_COVERED','POINT_OUTSIDE_REFERENCED_POLYGON']}).to_csv(b/'associated_point_fid_audit.csv',index=False)
        for name,ids,geo in [('landslides_shimla_points',['p1'],gpd.points_from_xy([70.5],[20.5])),('landslides_shimla',['q1'],[box(70,20,71,21)])]:
            pyogrio.write_dataframe(gpd.GeoDataFrame({'qa_record_id':ids},geometry=geo,crs=4326),b/(name+'.gpkg'),layer='candidates')
        m=lesser_review(b,p,2,1)
        assert len(m)==2 and len(read_csv(p/'lesser_association_review.csv'))==1
    print('Synthetic GLC regional extraction, readback, date profile and association review: PASS')

def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--self-test',action='store_true'); args=parser.parse_args()
    if args.self_test: self_test(); return
    dest=ROOT/'data/processed/landslides/integration_preflight_v1'; stage=dest.with_name(dest.name+'.staging')
    require(not dest.exists() and not stage.exists(),'Output or staging directory already exists; inspect it before rerunning. No overwrite.')
    run(ROOT,stage); stage.rename(dest)
    print('\nOutput:',dest,'| Source and prior processed files unchanged.')
if __name__=='__main__': main()
