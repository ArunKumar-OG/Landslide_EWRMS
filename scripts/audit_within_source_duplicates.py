"""Within-source duplicate evidence, not event merging. Read-only inputs."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import geopandas as gpd
import pandas as pd
import pyogrio
import shapely
from standardize_landslide_quality import EXPECTED, plan, missing
from standardize_landslide_dates import require
from audit_gsi_candidates import geometry_key, text_key

ROOT = Path(__file__).resolve().parent.parent
KEY = ['source', 'qa_record_id']
GROUP_COLS = ['source','match_type','namespace','match_key','group_id','records',
              'geometry_kinds','distinct_geometry_count','date_evidence',
              'distinct_day_count','missing_interval_count','day_candidates','rain_day_review_records']
MEMBER_COLS = ['group_id','source','match_type','namespace','match_key','qa_record_id','layer',
               'geometry_kind','start_date','end_date','date_status','spatial_status','dynamic_screen']


def read_csv(p):
    return pd.read_csv(p, dtype=str, keep_default_na=False, encoding='utf-8-sig')


def verify_csv(p, df):
    saved=read_csv(p)
    require(list(saved.columns)==list(df.columns) and len(saved)==len(df),f'{p.name}: CSV readback mismatch')
    # CSV has no saved DataFrame index; compare ordered cell values only.
    require(saved.reset_index(drop=True).equals(df.fillna('').astype(str).reset_index(drop=True)),f'{p.name}: CSV value mismatch')


def load_quality(base, expected):
    q = read_csv(base/'quality_standardization_v1/quality_candidates.csv')
    t = read_csv(base/'temporal_standardization_v1/temporal_candidates.csv')
    for folder, df in [('quality_standardization_v1',q),('temporal_standardization_v1',t)]:
        marker = json.loads((base/folder/'complete.json').read_text(encoding='utf-8'))
        require(marker['status']=='PASS' and marker['records']==len(df),f'{folder}: marker mismatch')
        require(Counter(df.source)==expected and not df.duplicated(KEY).any(),f'{folder}: source count or ID mismatch')
        require(df.qa_record_id.ne('').all(),'Empty record ID')
    cols = ['layer','start_date','end_date','date_status','date_precision','representation_status']
    a=q.set_index(KEY).sort_index(); b=t.set_index(KEY).sort_index()
    require(a.index.equals(b.index) and a[cols].equals(b[cols]),'Quality and temporal tables disagree')
    require(q.training_label_approved.str.lower().eq('false').all(),'Unexpected approved labels')
    return q


def read_hashes(base, q):
    out=[]
    for source, layer, path in plan(base,q[['source','layer']].to_dict('records')):
        frame = pyogrio.read_dataframe(path,layer='candidates',columns=['qa_record_id'])
        selected=q.loc[(q.source==source)&(q.layer==layer),['qa_record_id','geometry_kind']]
        require(frame.crs is not None and frame.crs.to_epsg()==4326,f'{path}: CRS mismatch')
        require(frame.qa_record_id.notna().all() and frame.qa_record_id.is_unique and set(frame.qa_record_id.astype(str))==set(selected.qa_record_id),f'{path}: record mismatch')
        geom=frame.geometry
        require(geom.notna().all() and not geom.is_empty.any() and geom.is_valid.all(),f'{path}: invalid geometry')
        actual=geom.geom_type.map({'Point':'POINT','Polygon':'POLYGON','MultiPolygon':'POLYGON'})
        declared=frame.qa_record_id.astype(str).map(selected.set_index('qa_record_id').geometry_kind)
        require(actual.eq(declared).all(),f'{path}: unexpected geometry kind')
        out.extend((source,str(i),geometry_key(g)) for i,g in zip(frame.qa_record_id,geom))
        print(f'{source} / {layer}: {len(frame):,} geometry hashes',flush=True)
    hashes=pd.DataFrame(out,columns=KEY+['geometry_hash'])
    require(len(hashes)==len(q) and not hashes.duplicated(KEY).any(),'Hash accounting mismatch')
    return q.merge(hashes,on=KEY,validate='one_to_one')


def id_key(v):
    return '' if missing(v) else str(v).strip()


def date_evidence(df):
    # Date strings already standardized to ISO. Intervals are uncertainty, not exact dates.
    # Do not treat UTC/IST ambiguity as a verified single day.
    known=df.start_date.ne('') & df.end_date.ne('')
    day=known & df.date_precision.eq('DAY') & df.start_date.eq(df.end_date)
    days=set(df.loc[day,'start_date'])
    k=df.loc[known]
    if len(days)>1: status='DIFFERENT_DAY_CANDIDATES'
    elif len(k)>1 and k.start_date.max()>k.end_date.min(): status='DISJOINT_REPORTED_INTERVALS'
    elif not len(k): status='NO_COMPARABLE_DATE'
    elif len(k)<len(df): status='PARTIAL_DATE_INFORMATION'
    elif day.all(): status='SAME_DAY_CANDIDATE'
    else: status='OVERLAPPING_OR_SINGLE_INTERVAL'
    return status,len(days),int((~known).sum()),int(day.sum())


def collect(q):
    groups=[]; members=[]
    def add(frame,match,ns,key):
        if len(frame)<2: return
        source=frame.source.iloc[0]
        gid=hashlib.sha256(json.dumps([source,match,ns,key],ensure_ascii=False).encode()).hexdigest()
        status,days,unknown,n_day=date_evidence(frame)
        common=dict(source=source,match_type=match,namespace=ns,match_key=key,group_id=gid)
        groups.append(common|dict(records=len(frame),geometry_kinds=';'.join(sorted(set(frame.geometry_kind))),distinct_geometry_count=frame.geometry_hash.nunique(),date_evidence=status,distinct_day_count=days,missing_interval_count=unknown,day_candidates=n_day,rain_day_review_records=int(frame.dynamic_screen.eq('RAIN_DAY_SPATIAL_AND_EVENT_REVIEW').sum())))
        for r in frame.to_dict('records'):
            members.append(common|{c:r[c] for c in MEMBER_COLS if c not in common})
    # Exact normalized 2-D coordinates, no tolerance, no rounding. Keep associated points in audit.
    for (s,k,h),f in q.loc[q.duplicated(['source','geometry_kind','geometry_hash'],keep=False)].groupby(['source','geometry_kind','geometry_hash'],sort=True):
        add(f,'EXACT_NORMALIZED_GEOMETRY',k,h)
    g=q[q.source=='GSI'].copy()
    for col,match in [('original_id','SAME_GID_WITHIN_KIND'),('original_secondary_id','SAME_SLIDE_NO')]:
        g['_key']=g[col].map(text_key)
        f=g[g._key.ne('')]; f=f[f.duplicated(['geometry_kind','_key'],keep=False)]
        for (k,v),f1 in f.groupby(['geometry_kind','_key']): add(f1,match,k,v)
    for s in ['GLC','COOLR','UGLC']:
        f=q[q.source==s].copy(); f['_key']=f.original_id.map(id_key); f=f[f._key.ne('')]
        # COOLR event IDs may connect reports and events; not necessarily one physical failure.
        # UGLC ID namespace is kept separate for points and polygons.
        f['_ns']=f.geometry_kind if s=='UGLC' else 'EVENT_ID_ALL_LAYERS'
        for (ns,v),f1 in f[f.duplicated(['_ns','_key'],keep=False)].groupby(['_ns','_key']):
            add(f1,'SAME_CATALOGUE_ID',ns,v)
    u=q[q.source=='UGLC'].copy(); u['_key']=u.original_secondary_id.map(id_key); u['_origin']=u.original_dataset.map(id_key)
    u=u[u._key.ne('')&u._origin.ne('')]
    for (origin,v),f in u[u.duplicated(['_origin','_key'],keep=False)].groupby(['_origin','_key']):
        add(f,'SAME_ORIGINAL_DATASET_ID',origin,v)
    return pd.DataFrame(groups,columns=GROUP_COLS),pd.DataFrame(members,columns=MEMBER_COLS)


def check_gsi(base,members):
    old=read_csv(base/'gsi_quality_v1/gsi_duplicate_candidates.csv')
    new=members[members.source=='GSI']
    a=set(zip(old.match_type,old.kind.str.upper(),old.match_key,old.qa_record_id))
    b=set(zip(new.match_type,new.namespace,new.match_key,new.qa_record_id))
    require(len(a)==len(old) and len(b)==len(new) and a==b,'GSI duplicate evidence differs from earlier audit')
    print('GSI duplicate evidence cross-check: PASS')


def lesser_links(base,q,expected=3153):
    x=read_csv(base/'lesser_himalayas_regional_v1/associated_point_fid_audit.csv')
    l=q[q.source=='LESSER_HIMALAYAS_2023']; p=l[l.layer=='landslides_shimla_points']; poly=l[l.layer=='landslides_shimla']
    require(len(p)==expected and x.point_id.is_unique,'Lesser association inventory mismatch')
    x=x[x.point_id.isin(p.qa_record_id)].copy()
    require(len(x)==expected and set(x.point_id)==set(p.qa_record_id),'Lesser association missing IDs')
    require(x.polygon_id.is_unique and set(x.polygon_id)==set(poly.qa_record_id),'Lesser referenced polygon mismatch')
    require(x.status.isin(['POINT_COVERED','POINT_OUTSIDE_REFERENCED_POLYGON']).all(),'Unexpected Lesser association status')
    x['disposition']='ASSOCIATED_NOT_INDEPENDENT'
    x.loc[x.status!='POINT_COVERED','disposition']='ASSOCIATION_GEOMETRY_REVIEW_NOT_INDEPENDENT'
    return x


def run(base,out,expected=EXPECTED,production=True):
    q=read_hashes(base,load_quality(base,expected)); groups,members=collect(q)
    if production:
        check_gsi(base,members)
        linked=lesser_links(base,q)
        require(Counter(linked.status)=={'POINT_COVERED':3144,'POINT_OUTSIDE_REFERENCED_POLYGON':9},'Lesser link count mismatch')
    else: linked=pd.DataFrame(columns=['point_id','polygon_id','status','disposition'])
    require(groups.group_id.is_unique and not members.duplicated(['group_id']+KEY).any(),'Duplicate group membership')
    require(set(members.group_id)==set(groups.group_id),'Group ID mismatch')
    require(members.groupby('group_id').size().to_dict()==groups.set_index('group_id').records.to_dict(),'Group membership count mismatch')
    require(set(zip(members.source,members.qa_record_id)).issubset(set(zip(q.source,q.qa_record_id))),'Unknown member ID')
    counts=members.groupby(KEY).group_id.nunique().rename('duplicate_evidence_groups').reset_index()
    flags=q[KEY+['layer','geometry_kind','geometry_hash','representation_status','date_precision','dynamic_screen']].merge(counts,on=KEY,how='left',validate='one_to_one')
    flags['duplicate_evidence_groups']=flags.duplicate_evidence_groups.fillna(0).astype(int)
    flags['disposition']='RETAIN_NO_MATCH_IN_THIS_STAGE'
    flags.loc[flags.duplicate_evidence_groups.gt(0),'disposition']='RETAIN_DUPLICATE_REVIEW'
    flags.loc[flags.representation_status.eq('ASSOCIATED_POINT_NOT_INDEPENDENT'),'disposition']='ASSOCIATED_NOT_INDEPENDENT'
    flags['training_label_approved']=False
    screen=q[q.dynamic_screen.eq('RAIN_DAY_SPATIAL_AND_EVENT_REVIEW')].groupby(['source','spatial_status','type_group']).size().rename('records').reset_index()
    summary=groups.groupby(['source','match_type']).agg(groups=('group_id','size'),membership_rows=('records','sum')).reset_index()
    unique=members.groupby(['source','match_type']).qa_record_id.nunique().rename('records').reset_index()
    summary=summary.merge(unique,on=['source','match_type'],validate='one_to_one')
    for name,df in [('duplicate_groups.csv',groups),('duplicate_group_members.csv',members),('within_source_record_flags.csv',flags),('duplicate_summary.csv',summary),('lesser_associated_links.csv',linked),('rain_day_spatial_profile.csv',screen)]:
        p=out/name; df.to_csv(p,index=False)
        verify_csv(p,df)
    report=dict(status='PASS',records=len(q),groups=len(groups),membership_rows=len(members),records_with_duplicate_evidence=int(flags.duplicate_evidence_groups.gt(0).sum()),associated_points=len(linked),approved_training_labels=0,records_deleted=0)
    print('\nWITHIN-SOURCE DUPLICATE EVIDENCE (categories can overlap)\n'+summary.to_string(index=False))
    print('\nGroup date evidence:\n'+groups.groupby(['match_type','date_evidence']).size().to_string())
    print('\nRain/day review pool by source and spatial status:\n'+screen.groupby(['source','spatial_status']).records.sum().to_string())
    print(json.dumps(report,indent=2)); (out/'complete.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return groups,members,flags


def self_test():
    from shapely.geometry import Point, Polygon, MultiPolygon
    assert geometry_key(Polygon([(0,0),(1,0),(1,1),(0,0)]))==geometry_key(MultiPolygon([Polygon([(1,1),(1,0),(0,0),(1,1)])]))
    with TemporaryDirectory() as tmp:
        b=Path(tmp); g=b/'integration_preflight_v1/glc_regional'; g.mkdir(parents=True)
        rows=[]
        for i,d,oid in [(1,'2023-01-01','X'),(2,'2023-01-02','X'),(3,'',''),(4,'','')]:
            rows.append(dict(source='GLC',layer='points',qa_record_id=f'GLC:{i}',geometry_kind='POINT',original_id=oid,original_secondary_id='',original_dataset='GLC',start_date=d,end_date=d,date_status='DAY_CANDIDATE_UNVERIFIED' if d else 'MISSING',date_precision='DAY' if d else 'UNKNOWN',representation_status='BASELINE_CANDIDATE',dynamic_screen='RAIN_DAY_SPATIAL_AND_EVENT_REVIEW' if d else 'NO_SINGLE_DAY_CANDIDATE',spatial_status='POINT_ACCURACY_UNCONFIRMED',type_group='LANDSLIDE_OR_SLIDE',training_label_approved=False))
        q=pd.DataFrame(rows)
        for folder,name in [('quality_standardization_v1','quality_candidates.csv'),('temporal_standardization_v1','temporal_candidates.csv')]:
            p=b/folder; p.mkdir(); q.to_csv(p/name,index=False); (p/'complete.json').write_text(json.dumps(dict(status='PASS',records=4)))
        pyogrio.write_dataframe(gpd.GeoDataFrame({'qa_record_id':q.qa_record_id},geometry=[Point(78,30),Point(78,30),Point(79,31),Point(80,31)],crs=4326),g/'candidates.gpkg',layer='candidates')
        out=b/'out'; out.mkdir(); groups,members,flags=run(b,out,{'GLC':4},False)
        assert len(groups)==2 and len(members)==4 and flags.duplicate_evidence_groups.tolist()==[2,2,0,0]
        assert groups.date_evidence.eq('DIFFERENT_DAY_CANDIDATES').all()
        assert not flags.training_label_approved.any()
        # Regression: the real Lesser association filter removes outside records
        # and keeps nonconsecutive indices. CSV roundtrip must not compare indices.
        lp=b/'lesser_himalayas_regional_v1'; lp.mkdir()
        assoc=pd.DataFrame({'point_id':['P0','P1','P2'], 'polygon_id':['A0','A1','A2'],
                            'status':['POINT_COVERED','POINT_COVERED','POINT_OUTSIDE_REFERENCED_POLYGON'],
                            'orig_fid':['0','1','2'], 'point_Id':['','1','2']})
        assoc.to_csv(lp/'associated_point_fid_audit.csv',index=False)
        lq=pd.DataFrame({'source':['LESSER_HIMALAYAS_2023']*4,
                         'layer':['landslides_shimla_points']*2+['landslides_shimla']*2,
                         'qa_record_id':['P0','P2','A0','A2']})
        linked=lesser_links(b,lq,expected=2)
        assert linked.index.tolist()==[0,2]
        p=out/'lesser_associated_links_regression.csv'; linked.to_csv(p,index=False)
        assert not read_csv(p).equals(linked.fillna('').astype(str))
        verify_csv(p,linked)
        damaged=read_csv(p); damaged.loc[1,'polygon_id']='WRONG'; damaged.to_csv(p,index=False)
        try: verify_csv(p,linked)
        except ValueError: pass
        else: raise AssertionError('Corrupted association CSV accepted')
        print('Filtered Lesser association CSV and corruption rejection: PASS')
        q.loc[0,'start_date']='2023-02-01'; q.to_csv(b/'quality_standardization_v1/quality_candidates.csv',index=False)
        try: load_quality(b,{'GLC':4})
        except ValueError: pass
        else: raise AssertionError('Temporal mismatch accepted')
    print('Synthetic geometry normalization, ID grouping, conflicting dates, join and readback: PASS')


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--self-test',action='store_true'); a=p.parse_args()
    if a.self_test: self_test(); return
    base=ROOT/'data/processed/landslides'; out=base/'within_source_duplicates_v1'; stage=out.with_name(out.name+'.staging')
    require(not out.exists() and not stage.exists(),'Output or staging already exists; inspect before retrying; no overwrite')
    stage.mkdir(); run(base,stage); stage.rename(out); print('Output:',out,'| Source and prior outputs unchanged.')

if __name__=='__main__': main()
