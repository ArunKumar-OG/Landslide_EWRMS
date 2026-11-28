"""Read-only attribute quality sidecar; no labels approved, no records removed."""
import argparse, csv, json, math, re
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
import pyogrio
import pandas as pd
from standardize_landslide_dates import read, write, require

ROOT=Path(__file__).resolve().parent.parent
EXPECTED={'GLC':1179,'GSI':111958,'UGLC':7540,'COOLR':1504,'LESSER_HIMALAYAS_2023':6665}
MISSING={'','na','n/a','nan','none','null','nat','unknown','nil','nd','-','--','<na>','.'}
def txt(v):
    return '' if v is None or pd.isna(v) else str(v).strip()
def norm(v): return re.sub(r'\s+',' ',txt(v).casefold().replace('_',' ')).strip()
def missing(v): return norm(v) in MISSING

def accuracy(source,raw):
    s=norm(raw)
    if missing(s) or s=='no field': return '', 'UNKNOWN_OR_UNREPORTED'
    if source=='UGLC':
        try: n=float(s)
        except ValueError: return '', 'UNRECOGNIZED_ACCURACY'
        if not math.isfinite(n) or n<0: return '', 'UNKNOWN_OR_INVALID'
        if n==0: return 0,'ZERO_REPORTED_UNVERIFIED'
    elif source in {'GLC','COOLR'}:
        if s=='exact': return '', 'EXACT_CATEGORY_UNVERIFIED'
        m=re.fullmatch(r'(\d+(?:\.\d+)?)\s*(km|m)',s)
        if not m: return '', 'UNRECOGNIZED_ACCURACY'
        n=float(m[1])*(1000 if m[2]=='km' else 1)
        if not math.isfinite(n) or n<=0: return '', 'UNKNOWN_OR_INVALID'
    else: return '', 'UNKNOWN_OR_UNREPORTED'
    return n, 'LE_30M_REPORTED' if n<=30 else 'GT_30_LE_100M' if n<=100 else 'GT_100_LE_1000M' if n<=1000 else 'GT_1000M'

# Deliberately conservative: exact labels only; free text and compound terms need review.
RAIN={'rain','rainfall','heavy rain','heavy rainfall','downpour','continuous rain','monsoon','tropical cyclone','incessant rainfall','rainfall (t)'}
SEISMIC={'earthquake','seismic','seismic (t)'}
HUMAN={'construction','mining','road cutting','road cut','anthropic (t)'}
OTHER={'snowfall snowmelt','snowmelt','flooding','volcanic (t)','volcanic'}
def trigger(raw,source,origin):
    s=norm(raw); o=norm(origin)
    if source=='UGLC' and 'earthquake triggered ground failure' in o:
        return 'EARTHQUAKE_INVENTORY_REVIEW' # provenance is not individual trigger verification
    if source=='LESSER_HIMALAYAS_2023': return 'RAINFALL_CONTEXT_ONLY'
    if missing(s): return 'UNKNOWN'
    if s in RAIN: return 'RAINFALL_REPORTED'
    if s in SEISMIC: return 'SEISMIC_REPORTED'
    if s in HUMAN: return 'HUMAN_REPORTED'
    if s in OTHER: return 'OTHER_REPORTED'
    if source=='UGLC': return 'FACTOR_OR_COMPOUND_REVIEW'
    # Flag possible mixed causes, not assert rainfall as sole trigger.
    if re.search(r'\b(rain\w*|downpour|monsoon)\b',s): return 'RAINFALL_TEXT_REVIEW'
    return 'OTHER_OR_COMPOUND_REVIEW'

def failure(raw):
    s=norm(raw)
    if missing(s): return 'UNKNOWN'
    if s in {'slide','landslide','landslides','rock slide','debris slide','earth slide','translational slide','rotational slide','shallow translational','shallow translational failure','deep rotational failure','shallow rotational failure','shallow planar failure'}: return 'LANDSLIDE_OR_SLIDE'
    if s in {'debris flow','mudslide','mud flow','mudflow','earth flow','debris avalanche','flow','rock avalanche'}: return 'FLOW_OR_AVALANCHE_REVIEW'
    if s in {'rock fall','rockfall','fall','falls','topple','toppling','falls/topple'}: return 'FALL_OR_TOPPLE'
    if s in {'complex','complex landslide','multiple'}: return 'COMPLEX'
    if s in {'subsidence','sinkhole','snow avalanche','snowavalanche','erosion','liquefaction','lateral spread','riverbank collapse'}: return 'OTHER_FAILURE_REVIEW'
    return 'UNMAPPED_TYPE_REVIEW'

def classify(source,layer,r,t):
    point=layer=='points' or 'POINT_' in layer or layer.endswith('_Points') or layer=='Field_GPS_landslides' or layer=='landslides_shimla_points'
    rep=t['representation_status']; kind='POINT' if point else 'POLYGON'
    origin=source; idraw=''; oldid=''; acc=''; typ=''; trig=''; factors=''; source_trigger=''; reliability=''; record_type=''; mapping=''; source_name=''; source_link=''
    if source=='GSI':
        origin='GSI'; idraw=txt(r['gid']); oldid=txt(r.get('slide_no')); mapping=txt(r['qa_mapping_kind']); typ=txt(r['movement_r'] if point else r['landslide_']); trig=txt(r.get('triggering')); record_type='inventory'
    elif source=='UGLC':
        origin=txt(r['OLD DATASET']); idraw=txt(r['ID']); oldid=txt(r['OLD ID']); acc=txt(r['ACCURACY']); typ=txt(r['TYPE']); factors=txt(r['PHYSICAL FACTORS']); source_trigger=txt(r.get('TRIGGER')); trig=factors; reliability=txt(r['RELIABILITY']); record_type=txt(r['RECORD TYPE']); mapping='CATALOGUE_POINT' if point else 'CATALOGUE_POLYGON'; source_link=txt(r.get('LINK'))
    elif source in {'GLC','COOLR'}:
        idraw=txt(r['event_id']); oldid=txt(r.get('event_import_id')); acc=txt(r.get('location_accuracy')); typ=txt(r['landslide_category']); trig=txt(r['landslide_trigger']); source_name=txt(r.get('source_name')); source_link=txt(r.get('source_link')); mapping='REPORTED_POINT' if point else 'REPORTED_POLYGON'; record_type='event' if 'Events' in layer else 'report'
    else:
        idraw=txt(r.get('Id')); oldid=txt(r.get('ORIG_FID')); typ=txt(r.get('Category') if layer=='Field_GPS_landslides' else r.get('category')); mapping='FIELD_GPS_POINT' if layer=='Field_GPS_landslides' else 'ASSOCIATED_POINT' if point else 'MAPPED_POLYGON'; record_type='inventory'
    metres,tier=accuracy(source,acc); tc=trigger(trig,source,origin); fc=failure(typ)
    if rep=='ASSOCIATED_POINT_NOT_INDEPENDENT': spatial='ASSOCIATED_REPRESENTATION'
    elif mapping=='BUFFERED_POINT': spatial='BUFFER_NOT_FOOTPRINT'
    elif mapping=='MAPPED_POLYGON': spatial='MAPPED_FOOTPRINT_CANDIDATE'
    elif not point: spatial='POLYGON_METHOD_REVIEW'
    elif tier in {'ZERO_REPORTED_UNVERIFIED','EXACT_CATEGORY_UNVERIFIED','LE_30M_REPORTED'}: spatial='POINT_REPORTED_FINE_REVIEW'
    elif tier in {'GT_30_LE_100M','GT_100_LE_1000M','GT_1000M'}: spatial='POINT_COARSE_REPORTED'
    else: spatial='POINT_ACCURACY_UNCONFIRMED'
    day=t['date_precision']=='DAY' and bool(t['start_date']) and t['start_date']==t['end_date']
    # Only a screening queue, not model labels. No automatic approval even if all criteria match.
    if rep=='ASSOCIATED_POINT_NOT_INDEPENDENT': dynamic='ASSOCIATED_REPRESENTATION'
    elif not day: dynamic='NO_SINGLE_DAY_CANDIDATE'
    elif tc in {'EARTHQUAKE_INVENTORY_REVIEW','SEISMIC_REPORTED','HUMAN_REPORTED','OTHER_REPORTED'}: dynamic='NONRAIN_OR_SOURCE_REVIEW'
    elif tc!='RAINFALL_REPORTED': dynamic='TRIGGER_REVIEW'
    elif spatial=='POINT_COARSE_REPORTED': dynamic='COARSE_LOCATION_REVIEW'
    elif spatial=='BUFFER_NOT_FOOTPRINT': dynamic='BUFFER_LOCATION_REVIEW'
    else: dynamic='RAIN_DAY_SPATIAL_AND_EVENT_REVIEW'
    return dict(source=source,layer=layer,qa_record_id=t['qa_record_id'],geometry_kind=kind,original_dataset=origin,original_id=idraw,original_secondary_id=oldid,source_name=source_name,source_link=source_link,source_import_raw=txt(r.get('event_import_source')),record_type_raw=record_type,mapping_method=mapping,mapping_method_raw=txt(r.get('data_type') if source=='GSI' else r.get('method')),coordinate_agreement_raw=txt(r.get('qa_coord_difference_m')),accuracy_raw=acc,reported_accuracy_m=metres,accuracy_tier=tier,reliability_raw=reliability,spatial_status=spatial,type_raw=typ,type_group=fc,trigger_raw=trig,physical_factors_raw=factors,legacy_trigger_raw=source_trigger,trigger_group=tc,date_status=t['date_status'],date_precision=t['date_precision'],start_date=t['start_date'],end_date=t['end_date'],representation_status=rep,dynamic_screen=dynamic,training_label_approved=False)

# Use the audited existing GeoPackages; only attributes, not global source geometry.
def plan(base,temporal):
    groups=sorted({(r['source'],r['layer']) for r in temporal})
    for s,l in groups:
        if s=='GLC': p=base/'integration_preflight_v1/glc_regional/candidates.gpkg'
        elif s=='UGLC': p=base/f'uglc_regional_v1/{l}/candidates.gpkg'
        else: p=base/({'GSI':'gsi_regional_v1','COOLR':'coolr_regional_v1','LESSER_HIMALAYAS_2023':'lesser_himalayas_regional_v1'}[s])/(l+'.gpkg')
        yield s,l,p

def process(base,expected=EXPECTED):
    temporal=read(base/'temporal_standardization_v1/temporal_candidates.csv')
    marker=json.loads((base/'temporal_standardization_v1/complete.json').read_text(encoding='utf-8'))
    require(marker['status']=='PASS' and marker['records']==len(temporal),'Temporal marker mismatch')
    require(Counter(t['source'] for t in temporal)==expected,'Unexpected source counts')
    index={ (t['source'],t['qa_record_id']):t for t in temporal}; require(len(index)==len(temporal),'Duplicate temporal keys')
    out=[]; seen=set(); grouped=Counter((t['source'],t['layer']) for t in temporal)
    for s,l,p in plan(base,temporal):
        frame=pyogrio.read_dataframe(p,layer='candidates',read_geometry=False)
        require(len(frame)==grouped[s,l] and frame.qa_record_id.notna().all() and frame.qa_record_id.is_unique,f'Invalid layer counts or IDs: {p}')
        for r in frame.to_dict('records'):
            key=(s,txt(r['qa_record_id'])); require(key in index and key not in seen and index[key]['layer']==l,f'Unmatched/duplicate key {key}')
            seen.add(key); out.append(classify(s,l,r,index[key]))
        print(f'{s} / {l}: {len(frame):,} matched',flush=True)
    require(seen==set(index),'Incomplete temporal join'); return out

def export(rows,out):
    fields=list(rows[0]); write(out/'quality_candidates.csv',rows,fields)
    saved=read(out/'quality_candidates.csv'); require(len(saved)==len(rows) and all(all(y[k]==str(x[k]) for k in fields) for x,y in zip(rows,saved)),'CSV readback mismatch')
    for field in ('accuracy_tier','spatial_status','type_group','trigger_group','dynamic_screen'):
        c=Counter((r['source'],r['geometry_kind'],r[field]) for r in rows)
        write(out/(field+'_counts.csv'),[dict(source=s,geometry_kind=k,category=v,records=n) for (s,k,v),n in sorted(c.items())],['source','geometry_kind','category','records'])
        print(field,dict(Counter(r[field] for r in rows)))
    c=Counter((r['source'],r['trigger_raw'],r['trigger_group']) for r in rows)
    write(out/'raw_trigger_mapping.csv',[dict(source=s,raw=v,category=k,records=n) for (s,v,k),n in sorted(c.items())],['source','raw','category','records'])
    c=Counter((r['source'],r['type_raw'],r['type_group']) for r in rows)
    write(out/'raw_type_mapping.csv',[dict(source=s,raw=v,category=k,records=n) for (s,v,k),n in sorted(c.items())],['source','raw','category','records'])
    report=dict(status='PASS',records=len(rows),source_counts=dict(Counter(r['source'] for r in rows)),day_candidates=sum(r['date_precision']=='DAY' and bool(r['start_date']) and r['start_date']==r['end_date'] for r in rows),approved_training_labels=0,excluded_repaired_review_polygons=6,associated_points=sum(r['representation_status']=='ASSOCIATED_POINT_NOT_INDEPENDENT' for r in rows))
    (out/'complete.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report,indent=2))

def self_test():
    assert accuracy('UGLC','-99999')[1]=='UNKNOWN_OR_INVALID'
    assert accuracy('UGLC','0')[1]=='ZERO_REPORTED_UNVERIFIED'
    assert accuracy('GLC','1km')==(1000,'GT_100_LE_1000M')
    assert accuracy('COOLR','exact')[0]==''
    assert trigger('rainfall (T)','UGLC','NASA')=='RAINFALL_REPORTED'
    assert trigger('rainfall (T)','UGLC','ETGFI - Earthquake Triggered Ground Failure Inventories (USGS)')=='EARTHQUAKE_INVENTORY_REVIEW'
    assert trigger('road cut & rainfall','GSI','GSI')=='RAINFALL_TEXT_REVIEW'
    assert failure('subsidence')=='OTHER_FAILURE_REVIEW'
    t=dict(qa_record_id='x',representation_status='BASELINE_CANDIDATE',date_precision='DAY',date_status='DAY_CANDIDATE_UNVERIFIED',start_date='2023-01-01',end_date='2023-01-01')
    assert classify('GSI','LANDSLIDE_POLYGON_STATE_U',dict(gid=1,qa_mapping_kind='BUFFERED_POINT',landslide_='Slide'),t)['spatial_status']=='BUFFER_NOT_FOOTPRINT'
    assert classify('GSI','LANDSLIDE_POINT_STATE_U',dict(gid=1,qa_mapping_kind='POINT_RECORD',movement_r='Slide',triggering='Rainfall'),t)['dynamic_screen']=='RAIN_DAY_SPATIAL_AND_EVENT_REVIEW'
    u={'OLD DATASET':'ETGFI - Earthquake Triggered Ground Failure Inventories (USGS)','ID':'1','OLD ID':'2','ACCURACY':'0','TYPE':'rock slide','PHYSICAL FACTORS':'seismic (T)','RELIABILITY':'1','RECORD TYPE':'event'}
    assert classify('UGLC','polygons',u,t)['dynamic_screen']=='NONRAIN_OR_SOURCE_REVIEW'
    assert classify('COOLR','COOLR_Events_Points',dict(event_id=1,landslide_category='landslide',landslide_trigger='Heavy Rain'),t)['spatial_status']=='POINT_ACCURACY_UNCONFIRMED'
    assert classify('LESSER_HIMALAYAS_2023','landslides_shimla_points',dict(Id=1,ORIG_FID=0,category='slide'),t|{'representation_status':'ASSOCIATED_POINT_NOT_INDEPENDENT'})['dynamic_screen']=='ASSOCIATED_REPRESENTATION'
    with TemporaryDirectory() as tmp:
        import geopandas as gpd
        from shapely.geometry import Point
        b=Path(tmp); t=b/'temporal_standardization_v1'; t.mkdir(); g=b/'integration_preflight_v1/glc_regional'; g.mkdir(parents=True)
        row=dict(source='GLC',layer='points',qa_record_id='GLC:1',representation_status='BASELINE_CANDIDATE',date_precision='DAY',date_status='DAY_CANDIDATE_UNVERIFIED',start_date='2023-01-01',end_date='2023-01-01')
        write(t/'temporal_candidates.csv',[row],list(row)); (t/'complete.json').write_text(json.dumps(dict(status='PASS',records=1)))
        pyogrio.write_dataframe(gpd.GeoDataFrame([dict(qa_record_id='GLC:1',event_id='1',location_accuracy='5km',landslide_category='landslide',landslide_trigger='rain')],geometry=[Point(78,30)],crs=4326),g/'candidates.gpkg',layer='candidates')
        rows=process(b,{'GLC':1}); assert rows[0]['dynamic_screen']=='COARSE_LOCATION_REVIEW'; export(rows,b)
        row['qa_record_id']='wrong'; write(t/'temporal_candidates.csv',[row],list(row))
        try: process(b,{'GLC':1})
        except ValueError: pass
        else: raise AssertionError('Unmatched key accepted')
    print('Synthetic classification, GeoPackage join, mismatch rejection and CSV readback: PASS')

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--self-test',action='store_true'); a=p.parse_args()
    if a.self_test: self_test(); return
    base=ROOT/'data/processed/landslides'; out=base/'quality_standardization_v1'; stage=out.with_name(out.name+'.staging')
    require(not out.exists() and not stage.exists(),'Output or staging exists; inspect before retrying; no overwrite')
    rows=process(base); stage.mkdir(); export(rows,stage); stage.rename(out); print('Output:',out,'| Source and prior processed files unchanged.')
if __name__=='__main__': main()
