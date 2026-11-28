"""Cross-source duplicate evidence. No automatic merging or label approval.
Requires the completed within_source_duplicates_v1 audit. Run from scripts/.
"""
import argparse, csv, json, itertools, math
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
import pyogrio
import shapely
from shapely import STRtree
from pyproj import Geod, Transformer
from shapely.ops import transform
from standardize_landslide_quality import EXPECTED, plan, missing
from standardize_landslide_dates import require
from audit_gsi_candidates import geometry_key

ROOT = Path(__file__).resolve().parent.parent
KEY = ['source','qa_record_id']
GEOD = Geod(ellps='WGS84')
AREA = Transformer.from_crs(4326,6933,always_xy=True)
# Search radius is a diagnostic threshold, NOT a claim of location accuracy.
RADIUS = 100.0
COLS = ['source_a','id_a','layer_a','kind_a','source_b','id_b','layer_b','kind_b',
        'match_type','match_key','distance_m','intersection_area_m2','iou','fraction_a','fraction_b',
        'date_evidence','start_a','end_a','start_b','end_b','spatial_a','spatial_b',
        'trigger_a','trigger_b','dynamic_a','dynamic_b','disposition']

def read(p): return pd.read_csv(p,dtype=str,keep_default_na=False,encoding='utf-8-sig')
def marker(p): return json.loads(p.read_text(encoding='utf-8'))
def idkey(v): return '' if missing(v) else str(v).strip()

def load(base):
    q=read(base/'quality_standardization_v1/quality_candidates.csv')
    t=read(base/'temporal_standardization_v1/temporal_candidates.csv')
    w=read(base/'within_source_duplicates_v1/within_source_record_flags.csv')
    for folder,f in [('quality_standardization_v1',q),('temporal_standardization_v1',t),('within_source_duplicates_v1',w)]:
        m=marker(base/folder/'complete.json')
        require(m['status']=='PASS' and m['records']==len(f),f'{folder}: incomplete audit')
        require(Counter(f.source)==EXPECTED and not f.duplicated(KEY).any(),f'{folder}: inventory mismatch')
    require(q.training_label_approved.str.lower().eq('false').all(),'Unexpected approved labels')
    require(w.training_label_approved.str.lower().eq('false').all(),'Unexpected approved flags')
    a=q.set_index(KEY).sort_index(); b=t.set_index(KEY).sort_index(); c=w.set_index(KEY).sort_index()
    cols=['layer','start_date','end_date','date_status','date_precision','representation_status']
    require(a.index.equals(b.index) and a[cols].equals(b[cols]) and a.index.equals(c.index),'Temporal/quality/flag mismatch')
    cols=['layer','geometry_kind','representation_status','date_precision','dynamic_screen']
    require(a[cols].equals(c[cols]),'Flag attributes mismatch')
    require(c.geometry_hash.str.fullmatch(r'[0-9a-f]{64}').all(),'Invalid geometry hashes')
    return q.merge(w[KEY+['geometry_hash']],on=KEY,validate='one_to_one')

def dates(a,b):
    x,y=a['start_date'],a['end_date']; u,v=b['start_date'],b['end_date']
    if not (x and y and u and v): return 'MISSING_OR_PARTIAL_DATE'
    if a['date_precision']=='UTC_IST_DAY_AMBIGUITY' or b['date_precision']=='UTC_IST_DAY_AMBIGUITY': return 'TIMEZONE_REVIEW'
    if y<u or v<x: return 'DISJOINT_REPORTED_INTERVALS'
    if a['date_precision']==b['date_precision']=='DAY' and x==y==u==v: return 'SAME_DAY_CANDIDATE'
    return 'OVERLAPPING_REPORTED_INTERVALS'

def pair(a,b,kind,key='',distance='',area='',iou='',fa='',fb=''):
    require(a['source']<b['source'],'Pair order/source mismatch')
    return [a['source'],a['qa_record_id'],a['layer'],a['geometry_kind'],b['source'],b['qa_record_id'],b['layer'],b['geometry_kind'],kind,key,distance,area,iou,fa,fb,dates(a,b),a['start_date'],a['end_date'],b['start_date'],b['end_date'],a['spatial_status'],b['spatial_status'],a['trigger_group'],b['trigger_group'],a['dynamic_screen'],b['dynamic_screen'],'REVIEW_NOT_MERGED']

def id_evidence(records):
    # Namespace matches are hypotheses: NASA catalogue IDs may be version-dependent.
    by=defaultdict(list)
    for r in records:
        if r['source'] in ('GLC','COOLR') and idkey(r['original_id']):
            by[r['source'],idkey(r['original_id'])].append(r)
    for (s,k),aa in by.items():
        if s=='COOLR':
            for a,b in itertools.product(aa,by.get(('GLC',k),[])):
                yield pair(a,b,'NASA_CATALOGUE_ID_REVIEW',k)
    for u in records:
        if u['source']!='UGLC': continue
        origin=u['original_dataset'].casefold(); k=idkey(u['original_secondary_id'])
        if not k or 'cooperative open online landslide repository' not in origin: continue
        for s in ('COOLR','GLC'):
            for a in by.get((s,k),[]):
                yield pair(a,u,'UGLC_NASA_ORIGINAL_ID_REVIEW',k)
    # Do not match Global Fatal Landslide Catalog IDs to NASA GLC merely by acronym.

def geometry_records(base,q):
    result={}
    for s,l,p in plan(base,q[['source','layer']].to_dict('records')):
        f=pyogrio.read_dataframe(p,layer='candidates',columns=['qa_record_id'])
        sub=q[(q.source==s)&(q.layer==l)].set_index('qa_record_id')
        require(f.crs is not None and f.crs.to_epsg()==4326 and f.qa_record_id.notna().all() and f.qa_record_id.is_unique and set(f.qa_record_id.astype(str))==set(sub.index),f'{p}: geometry identity mismatch')
        g=shapely.force_2d(f.geometry.array)
        require(shapely.is_valid(g).all() and not shapely.is_empty(g).any(),f'{p}: invalid geometry')
        ordered=sub.loc[f.qa_record_id.astype(str).tolist()].reset_index().to_dict('records')
        for r,geom in zip(ordered,g):
            rid=r['qa_record_id']
            require(geometry_key(geom)==r['geometry_hash'],f'{p}: geometry hash changed: {rid}')
            if r['representation_status']=='ASSOCIATED_POINT_NOT_INDEPENDENT': continue
            kind='POINT' if geom.geom_type=='Point' else 'POLYGON' if geom.geom_type in ('Polygon','MultiPolygon') else ''
            require(kind==r['geometry_kind'],f'{p}: geometry kind mismatch')
            result.setdefault(s,[]).append((r,geom))
        print(f'{s} / {l}: {len(f):,} geometries checked',flush=True)
    require(sum(map(len,result.values()))==len(q)-int(q.representation_status.eq('ASSOCIATED_POINT_NOT_INDEPENDENT').sum()),'Spatial input accounting mismatch')
    return result

def spatial_evidence(a,b):
    # Both lists belong to distinct sources in alphabetical order.
    # WGS84 envelope for point proximity, followed by exact ellipsoidal distance.
    # At the study-region latitudes (21–38 N), 0.002 degrees exceeds 100 m.
    require(all(20<=g.bounds[1]<=g.bounds[3]<=40 for r,g in itertools.chain(a,b) if r['geometry_kind']=='POINT'),'Point outside supported latitude range; review proximity envelope')
    for ka,kb in itertools.product(('POINT','POLYGON'),repeat=2):
        aa=[x for x in a if x[0]['geometry_kind']==ka]; bb=[x for x in b if x[0]['geometry_kind']==kb]
        if not aa or not bb: continue
        bg=np.array([g for _,g in bb],dtype=object); tree=STRtree(bg)
        for off in range(0,len(aa),256):
            chunk=aa[off:off+256]; ag=np.array([g for _,g in chunk],dtype=object)
            if ka==kb=='POINT':
                x=shapely.get_x(ag); y=shapely.get_y(ag)
                ij=tree.query(shapely.box(x-.002,y-.002,x+.002,y+.002))
            else: ij=tree.query(ag,predicate='intersects')
            for i,j in zip(*ij):
                ra,ga=chunk[i]; rb,gb=bb[j]
                exact=ra['geometry_hash']==rb['geometry_hash']
                if ka==kb=='POINT':
                    d=abs(GEOD.inv(ga.x,ga.y,gb.x,gb.y)[2])
                    if d>RADIUS: continue
                    yield pair(ra,rb,'EXACT_NORMALIZED_GEOMETRY' if exact else 'POINT_DISTANCE_LE_100M',distance=round(d,4))
                elif ka!=kb:
                    point,poly=(ga,gb) if ka=='POINT' else (gb,ga)
                    yield pair(ra,rb,'POINT_POLYGON_BOUNDARY' if poly.touches(point) else 'POINT_INSIDE_POLYGON')
                else:
                    # Equal-area projection; calculate overlap metrics, not a merge threshold.
                    if ga.touches(gb): continue
                    try:
                        pa,pb=transform(AREA.transform,ga),transform(AREA.transform,gb)
                        if not pa.is_valid or not pb.is_valid or pa.area<=0 or pb.area<=0:
                            yield pair(ra,rb,'POLYGON_OVERLAP_METRIC_REVIEW'); continue
                        inter=pa.intersection(pb).area
                    except shapely.errors.GEOSException:
                        yield pair(ra,rb,'POLYGON_OVERLAP_METRIC_REVIEW'); continue
                    if inter<=0: continue # touch-only is not positive-area overlap
                    fa=min(1.,inter/pa.area); fb=min(1.,inter/pb.area)
                    iou=min(1.,inter/(pa.area+pb.area-inter))
                    yield pair(ra,rb,'EXACT_NORMALIZED_GEOMETRY' if exact else 'POLYGON_POSITIVE_AREA_OVERLAP',area=round(inter,4),iou=round(iou,8),fa=round(fa,8),fb=round(fb,8))

def run(base,out):
    q=load(base); records=q.to_dict('records'); by=geometry_records(base,q)
    counts=Counter(); flags=Counter(); n=0
    path=out/'cross_source_evidence.csv'
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f); w.writerow(COLS)
        def consume(rows):
            nonlocal n
            for row in rows:
                w.writerow(row); n+=1; counts[(row[0],row[4],row[8],row[15])]+=1
                flags[(row[0],row[1])]+=1; flags[(row[4],row[5])]+=1
        consume(id_evidence(records))
        for a,b in itertools.combinations(sorted(by),2):
            before=n; consume(spatial_evidence(by[a],by[b])); print(f'{a} vs {b}: {n-before:,} spatial evidence rows',flush=True)
    summary=pd.DataFrame([(*k,v) for k,v in sorted(counts.items())],columns=['source_a','source_b','match_type','date_evidence','evidence_rows'])
    summary.to_csv(out/'cross_source_summary.csv',index=False)
    flag=q[KEY+['layer','representation_status','dynamic_screen','spatial_status']].copy()
    flag['cross_source_evidence_rows']=[flags.get(tuple(k),0) for k in flag[KEY].itertuples(index=False,name=None)]
    flag['disposition']='RETAIN_NO_MATCH_IN_THIS_STAGE'; flag.loc[flag.cross_source_evidence_rows.gt(0),'disposition']='RETAIN_CROSS_SOURCE_REVIEW'
    flag.loc[flag.representation_status.eq('ASSOCIATED_POINT_NOT_INDEPENDENT'),'disposition']='ASSOCIATED_NOT_INDEPENDENT'
    flag['training_label_approved']=False; flag.to_csv(out/'cross_source_record_flags.csv',index=False)
    # Independent streaming accounting and key validation, not memory-heavy pair storage.
    known=set(q[KEY].itertuples(index=False,name=None)); check=Counter(); reread=0; reread_flags=Counter()
    with path.open(newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f); require(reader.fieldnames==COLS,'Evidence schema mismatch')
        for r in reader:
            a=(r['source_a'],r['id_a']); b=(r['source_b'],r['id_b'])
            require(a in known and b in known and a[0]<b[0] and r['disposition']=='REVIEW_NOT_MERGED','Invalid readback pair')
            reread+=1; check[(a[0],b[0],r['match_type'],r['date_evidence'])]+=1; reread_flags[a]+=1; reread_flags[b]+=1
    require(reread==n and check==counts and reread_flags==flags,'Evidence readback accounting mismatch')
    for name,df in [('cross_source_summary.csv',summary),('cross_source_record_flags.csv',flag)]:
        require(read(out/name).equals(df.astype(str).reset_index(drop=True)),f'{name}: value mismatch')
    report=dict(status='PASS',records=len(q),evidence_rows=n,records_with_cross_source_evidence=len(flags),associated_points_excluded_from_spatial_search=int(q.representation_status.eq('ASSOCIATED_POINT_NOT_INDEPENDENT').sum()),point_distance_threshold_m=RADIUS,approved_training_labels=0,records_deleted=0)
    print(summary.to_string(index=False)); print(json.dumps(report,indent=2)); (out/'complete.json').write_text(json.dumps(report,indent=2),encoding='utf-8')

def self_test():
    from shapely.geometry import Point,box
    def r(s,i,g,day='2023-07-01',**kw):
        return dict(source=s,qa_record_id=i,layer='test',geometry_kind='POINT' if g.geom_type=='Point' else 'POLYGON',geometry_hash=geometry_key(g),start_date=day,end_date=day,date_precision='DAY',spatial_status='REVIEW',trigger_group='UNKNOWN',dynamic_screen='REVIEW',original_id='',original_secondary_id='',original_dataset=s)|kw,g
    a=[r('COOLR','a',Point(78,30),original_id='123'),r('COOLR','b',box(78,30,78.002,30.002))]
    b=[r('GLC','c',Point(78,30),'2023-07-02',original_id='123'),r('GLC','d',Point(78.0005,30)),r('GLC','e',Point(78.0015,30)),r('GLC','f',box(78.001,30.001,78.003,30.003))]
    e=list(spatial_evidence(a,b)); types=Counter(x[8] for x in e)
    assert types['EXACT_NORMALIZED_GEOMETRY']==1 and types['POINT_DISTANCE_LE_100M']==1
    assert types['POLYGON_POSITIVE_AREA_OVERLAP']==1 and types['POINT_POLYGON_BOUNDARY']==3
    assert next(x for x in e if x[8]=='EXACT_NORMALIZED_GEOMETRY')[15]=='DISJOINT_REPORTED_INTERVALS'
    u=r('UGLC','u',Point(78,30),original_secondary_id='123',original_dataset='Cooperative Open Online Landslide Repository - NASA')[0]
    v=r('UGLC','v',Point(78,30),original_secondary_id='123',original_dataset='Global fatal landslide Catalog')[0]
    ids=list(id_evidence([x[0] for x in a+b]+[u,v])); assert len(ids)==3 and all(x[5]!='v' for x in ids)
    print('Synthetic cross-source exact/near point, polygon overlap, date conflict and namespace tests: PASS')

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--self-test',action='store_true'); p.add_argument('--root',type=Path,default=ROOT); a=p.parse_args()
    if a.self_test: self_test(); return
    base=a.root/'data/processed/landslides'; out=base/'cross_source_duplicates_v1'; stage=out.with_name(out.name+'.staging')
    require(not out.exists() and not stage.exists(),'Output or staging already exists; inspect before retrying; no overwrite')
    stage.mkdir(); run(base,stage); stage.rename(out); print('Output:',out,'| Original files unchanged.')
if __name__=='__main__': main()
