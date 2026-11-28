"""Conservative duplicate-resolution proposals; never deletes or approves labels.
Requires completed quality, within-source and cross-source audits. CSV-only.
"""
import argparse, hashlib, itertools, json
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
import pandas as pd

ROOT=Path(__file__).resolve().parent.parent
KEY=['source','qa_record_id']
EXPECTED={'COOLR':1504,'GLC':1179,'GSI':111958,'LESSER_HIMALAYAS_2023':6665,'UGLC':7540}
ID_TYPES={'NASA_CATALOGUE_ID_REVIEW','UGLC_NASA_ORIGINAL_ID_REVIEW'}
SPATIAL_TYPES={'EXACT_NORMALIZED_GEOMETRY','POINT_DISTANCE_LE_100M'}
RAIN='RAIN_DAY_SPATIAL_AND_EVENT_REVIEW'
ASSOC='ASSOCIATED_POINT_NOT_INDEPENDENT'

def require(ok,msg):
    if not ok: raise ValueError(msg)
def read(p): return pd.read_csv(p,dtype=str,keep_default_na=False,encoding='utf-8-sig')
def marker(p): return json.loads(p.read_text(encoding='utf-8'))
def token(prefix,items):
    return prefix+hashlib.sha256(json.dumps(items,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()[:24]
def export(out,name,df):
    x=df.fillna('').astype(str).reset_index(drop=True); p=out/name
    x.to_csv(p,index=False); require(read(p).equals(x),f'{name}: readback mismatch')
def keys(f): return set(f[KEY].itertuples(index=False,name=None))
def load(base,expected):
    q=read(base/'quality_standardization_v1/quality_candidates.csv')
    t=read(base/'temporal_standardization_v1/temporal_candidates.csv')
    w=read(base/'within_source_duplicates_v1/within_source_record_flags.csv')
    c=read(base/'cross_source_duplicates_v1/cross_source_record_flags.csv')
    for folder,f in [('quality_standardization_v1',q),('temporal_standardization_v1',t),('within_source_duplicates_v1',w),('cross_source_duplicates_v1',c)]:
        m=marker(base/folder/'complete.json')
        require(m['status']=='PASS' and m['records']==len(f),f'{folder}: marker mismatch')
        require(Counter(f.source)==expected and not f.duplicated(KEY).any() and f.qa_record_id.ne('').all(),f'{folder}: IDs/counts')
        require(keys(f)==keys(q),f'{folder}: key mismatch')
        if 'training_label_approved' in f: require(f.training_label_approved.str.lower().eq('false').all(),'Unexpected approved label')
    a=q.set_index(KEY).sort_index()
    for f,cols in [(t,['layer','start_date','end_date','date_precision','date_status','representation_status']), (w,['layer','geometry_kind','representation_status','date_precision','dynamic_screen']), (c,['layer','representation_status','dynamic_screen','spatial_status'])]:
        require(a[cols].equals(f.set_index(KEY).sort_index()[cols]),'Prior audit attribute mismatch')
    require(w.geometry_hash.str.fullmatch('[0-9a-f]{64}').all(),'Invalid geometry hash')
    q=q.merge(w[KEY+['geometry_hash','duplicate_evidence_groups']],on=KEY,validate='one_to_one').merge(c[KEY+['cross_source_evidence_rows']],on=KEY,validate='one_to_one')
    e=read(base/'cross_source_duplicates_v1/cross_source_evidence.csv')
    g=read(base/'within_source_duplicates_v1/duplicate_groups.csv')
    m=read(base/'within_source_duplicates_v1/duplicate_group_members.csv')
    l=read(base/'within_source_duplicates_v1/lesser_associated_links.csv')
    wm=marker(base/'within_source_duplicates_v1/complete.json'); cm=marker(base/'cross_source_duplicates_v1/complete.json')
    require(len(e)==cm['evidence_rows'] and len(g)==wm['groups'] and len(m)==wm['membership_rows'],'Evidence count mismatch')
    require(not g.group_id.duplicated().any() and not m.duplicated(['group_id']+KEY).any(),'Duplicate membership')
    require(m.groupby('group_id').size().to_dict()==g.set_index('group_id').records.astype(int).to_dict(),'Group counts mismatch')
    require(keys(m).issubset(keys(q)),'Unknown within-source member')
    wc=Counter(m[KEY].itertuples(index=False,name=None)); cc=Counter()
    for r in e.itertuples(index=False):
        a=(r.source_a,r.id_a); b=(r.source_b,r.id_b)
        require(a[0]<b[0] and r.disposition=='REVIEW_NOT_MERGED','Unexpected cross evidence')
        cc[a]+=1; cc[b]+=1
    require(set(cc).issubset(keys(q)),'Unknown cross-source member')
    require(len(wc)==wm['records_with_duplicate_evidence'] and len(cc)==cm['records_with_cross_source_evidence'],'Flag count mismatch')
    for r in q.itertuples(index=False):
        k=(r.source,r.qa_record_id)
        require(wc[k]==int(r.duplicate_evidence_groups) and cc[k]==int(r.cross_source_evidence_rows),'Evidence flag mismatch')
    lp=q[(q.source=='LESSER_HIMALAYAS_2023') & (q.representation_status==ASSOC)]
    poly=q[(q.source=='LESSER_HIMALAYAS_2023') & (q.layer=='landslides_shimla')]
    require(l.point_id.is_unique and l.polygon_id.is_unique and set(l.point_id)==set(lp.qa_record_id) and set(l.polygon_id)==set(poly.qa_record_id),'Association mismatch')
    require(l.status.isin(['POINT_COVERED','POINT_OUTSIDE_REFERENCED_POLYGON']).all(),'Unknown association status')
    return q,e,l

def propose(q,e,l):
    rec={(r['source'],r['qa_record_id']):r for r in q.to_dict('records')}
    # A spatial group is an exact 2-D representation, never a physical event.
    static={}; static_rows=[]
    eligible=q[q.representation_status.ne(ASSOC)]
    # Hash each distinct key once; group only duplicate geometries to avoid 120k tiny frames.
    by_geometry={}
    for r in eligible[KEY+['geometry_kind','geometry_hash']].itertuples(index=False,name=None):
        source,rid,kind,h=r; k=(kind,h)
        if k not in by_geometry: by_geometry[k]=token('S_',[kind,h])
        static[source,rid]=by_geometry[k]
    repeated=eligible[eligible.duplicated(['geometry_kind','geometry_hash'],keep=False)]
    for (kind,h),f in repeated.groupby(['geometry_kind','geometry_hash'],sort=False):
        static_rows.append(dict(static_geometry_group=by_geometry[kind,h],geometry_kind=kind,records=len(f),sources=';'.join(sorted(set(f.source))),mapped_footprint_candidates=int(f.spatial_status.eq('MAPPED_FOOTPRINT_CANDIDATE').sum()),day_candidates=int(f.date_precision.eq('DAY').sum())))
    # Combine evidence for each ordered pair. Recheck dates from the authoritative table.
    pairs=defaultdict(list)
    for r in e.to_dict('records'):
        a=(r['source_a'],r['id_a']); b=(r['source_b'],r['id_b']); pairs[a,b].append(r)
    strong=set(); graph=defaultdict(set); pair_rows=[]
    for (a,b),rows in sorted(pairs.items()):
        x,y=rec[a],rec[b]; kinds={r['match_type'] for r in rows}; dates={r['date_evidence'] for r in rows}
        require(len(dates)==1,'Inconsistent pair date evidence'); d=next(iter(dates))
        same_day=x['date_precision']==y['date_precision']=='DAY' and bool(x['start_date']) and x['start_date']==x['end_date']==y['start_date']==y['end_date']
        require((d=='SAME_DAY_CANDIDATE')==same_day,'Pair date evidence mismatch')
        id_ok=bool(kinds&ID_TYPES); spatial_ok=False
        for r in rows:
            if r['match_type'] in SPATIAL_TYPES and x['geometry_kind']==y['geometry_kind']=='POINT':
                try: distance=float(r['distance_m'])
                except ValueError: continue
                if 0<=distance<=100 and (r['match_type']!='EXACT_NORMALIZED_GEOMETRY' or x['geometry_hash']==y['geometry_hash']): spatial_ok=True
        # Restrict proposals to the NASA point records, with no compound/nonrain trigger conflict.
        rain_ok=x['trigger_group']==y['trigger_group']=='RAINFALL_REPORTED'
        independent=x['representation_status']!=ASSOC and y['representation_status']!=ASSOC
        # Group non-rain/unknown candidates only through manual review.
        strong_ok=id_ok and spatial_ok and same_day and rain_ok and independent
        if strong_ok:
            strong.add((a,b)); graph[a].add(b); graph[b].add(a); reason='STRONG_REPORTED_EVENT_EVIDENCE'
        elif d=='DISJOINT_REPORTED_INTERVALS': reason='DATE_CONFLICT_RETAIN'
        elif d=='TIMEZONE_REVIEW': reason='TIMEZONE_REVIEW_RETAIN'
        else: reason='MANUAL_REVIEW_RETAIN'
        pair_rows.append(dict(source_a=a[0],id_a=a[1],source_b=b[0],id_b=b[1],methods=';'.join(sorted(kinds)),date_evidence=d,evidence_rows=len(rows),id_evidence=id_ok,spatial_point_evidence=spatial_ok,both_reported_rain=rain_ok,proposal=reason))
    # No transitive chaining: accept only fully connected components with one record/source.
    # Even those are provisional event groups, not verified identities or training labels.
    ev={}; ev_rows=[]; visited=set()
    for a in sorted(graph):
        if a in visited: continue
        todo=[a]; comp=set()
        while todo:
            k=todo.pop()
            if k in comp: continue
            comp.add(k); todo.extend(graph[k]-comp)
        visited.update(comp); members=sorted(comp)
        clique=all((x,y) in strong for x,y in itertools.combinations(members,2))
        unique_sources=len({k[0] for k in members})==len(members)
        # If there is other within-source evidence, don't automatically consolidate that record.
        within_clear=all(int(rec[k]['duplicate_evidence_groups'])==0 for k in members)
        status='PROPOSED_SAME_REPORTED_EVENT' if clique and unique_sources and within_clear else 'AMBIGUOUS_COMPONENT_RETAIN_SEPARATELY'
        gid=token('E_',members)
        for k in members: ev[k]=(gid,status)
        ev_rows.append(dict(event_review_group=gid,status=status,records=len(members),sources=';'.join(sorted({k[0] for k in members})),reported_day=rec[members[0]]['start_date'] if len({rec[k]['start_date'] for k in members})==1 else '',complete_pair_evidence=clique,one_per_source=unique_sources,no_within_source_evidence=within_clear))
    assoc=l.set_index('point_id').to_dict('index'); flags=[]
    for k,r in rec.items():
        associated=r['representation_status']==ASSOC
        link=assoc.get(k[1],{}) if k[0]=='LESSER_HIMALAYAS_2023' and associated else {}
        gid,status=ev.get(k,('','NO_STRONG_EVENT_GROUP'))
        flags.append({**r,'static_geometry_group':static.get(k,''),'event_review_group':gid,'event_review_status':status,'associated_polygon_id':link.get('polygon_id',''),'association_status':link.get('status',''),'training_label_approved':False,'record_deleted':False})
    return pd.DataFrame(flags),pd.DataFrame(pair_rows,columns=['source_a','id_a','source_b','id_b','methods','date_evidence','evidence_rows','id_evidence','spatial_point_evidence','both_reported_rain','proposal']),pd.DataFrame(static_rows,columns=['static_geometry_group','geometry_kind','records','sources','mapped_footprint_candidates','day_candidates']),pd.DataFrame(ev_rows,columns=['event_review_group','status','records','sources','reported_day','complete_pair_evidence','one_per_source','no_within_source_evidence'])

def run(base,out,expected=EXPECTED):
    q,e,l=load(base,expected); f,p,s,v=propose(q,e,l)
    require(len(f)==len(q) and keys(f)==keys(q) and not f.duplicated(KEY).any(),'Resolution accounting mismatch')
    require(p.evidence_rows.sum()==len(e),'Pair accounting mismatch')
    rain=f[f.dynamic_screen.eq(RAIN)].copy()
    # Distinguish footprint representations from all exact geometries; no sample approval.
    footprints=f[f.spatial_status.eq('MAPPED_FOOTPRINT_CANDIDATE')]
    require(footprints.static_geometry_group.ne('').all(),'Footprint group missing')
    for name,df in [('resolution_record_flags.csv',f),('cross_source_pair_review.csv',p),('static_duplicate_geometry_groups.csv',s),('event_review_groups.csv',v),('rain_day_review_candidates.csv',rain)]: export(out,name,df)
    summary=rain.groupby(['source','spatial_status','event_review_status']).size().rename('records').reset_index(); export(out,'rain_day_resolution_summary.csv',summary)
    approved=v[v.status.eq('PROPOSED_SAME_REPORTED_EVENT')]
    report=dict(status='PASS',records=len(f),unique_cross_source_pairs=len(p),strong_pair_proposals=int(p.proposal.eq('STRONG_REPORTED_EVENT_EVIDENCE').sum()),event_review_components=len(v),proposed_event_groups=len(approved),records_in_proposed_event_groups=int(approved.records.sum()),ambiguous_event_components=int(v.status.eq('AMBIGUOUS_COMPONENT_RETAIN_SEPARATELY').sum()),independent_representation_candidates=int(f.static_geometry_group.ne('').sum()),distinct_exact_geometry_groups=f.loc[f.static_geometry_group.ne(''),'static_geometry_group'].nunique(),mapped_footprint_candidates=len(footprints),distinct_mapped_footprint_geometries=footprints.static_geometry_group.nunique(),rain_day_review_candidates=len(rain),rain_day_records_in_proposed_groups=int(rain.event_review_status.eq('PROPOSED_SAME_REPORTED_EVENT').sum()),associated_points=int(f.representation_status.eq(ASSOC).sum()),records_deleted=0,approved_training_labels=0)
    print(summary.to_string(index=False)); print(json.dumps(report,indent=2)); (out/'complete.json').write_text(json.dumps(report,indent=2),encoding='utf-8')

def self_test():
    def r(s,i,h='a',day='2023-07-01',within='0'):
        return dict(source=s,qa_record_id=i,geometry_kind='POINT',geometry_hash=h*64,representation_status='BASELINE_CANDIDATE',date_precision='DAY',start_date=day,end_date=day,trigger_group='RAINFALL_REPORTED',duplicate_evidence_groups=within,spatial_status='POINT_REPORTED_FINE_REVIEW',dynamic_screen=RAIN)
    def pair(a,b,kind,date='SAME_DAY_CANDIDATE'):
        return dict(source_a=a[0],id_a=a[1],source_b=b[0],id_b=b[1],match_type=kind,date_evidence=date,distance_m='0' if kind in SPATIAL_TYPES else '')
    a=('COOLR','a'); b=('GLC','b'); c=('UGLC','c')
    q=pd.DataFrame([r(*a),r(*b),r(*c),r('GSI','d',day='2023-07-02')]); links=pd.DataFrame(columns=['point_id','polygon_id','status'])
    e=pd.DataFrame([pair(x,y,k) for x,y in itertools.combinations([a,b,c],2) for k in ['NASA_CATALOGUE_ID_REVIEW','EXACT_NORMALIZED_GEOMETRY']])
    # All three share the same reported day and full pair evidence.
    f,p,s,v=propose(q,e,links); assert len(v)==1 and v.status.iloc[0]=='PROPOSED_SAME_REPORTED_EVENT' and v.records.iloc[0]==3 and f.static_geometry_group.nunique()==1
    # A later event at the exact same location stays outside the event group.
    assert f.loc[f.qa_record_id=='d','event_review_group'].iloc[0]==''
    # Missing triangle edge must not create a transitive event merge.
    f,p,s,v=propose(q,e[~((e.source_a=='COOLR')&(e.source_b=='UGLC'))],links); assert v.status.iloc[0]=='AMBIGUOUS_COMPONENT_RETAIN_SEPARATELY'
    # Within-source duplicates block even a fully connected proposal.
    q.loc[q.qa_record_id=='a','duplicate_evidence_groups']='1'; f,p,s,v=propose(q,e,links); assert v.status.iloc[0]=='AMBIGUOUS_COMPONENT_RETAIN_SEPARATELY'
    with TemporaryDirectory() as tmp:
        export(Path(tmp),'test.csv',f); assert len(read(Path(tmp)/'test.csv'))==4
    print('Synthetic exact spatial grouping, recurrent dates, complete-clique guard, within-source guard and CSV readback: PASS')

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--self-test',action='store_true'); p.add_argument('--root',type=Path,default=ROOT); args=p.parse_args()
    if args.self_test: self_test(); return
    base=args.root/'data/processed/landslides'; out=base/'duplicate_resolution_v1'; stage=out.with_name(out.name+'.staging')
    require(not out.exists() and not stage.exists(),'Output/staging exists; inspect before retrying, no overwrite')
    stage.mkdir(); run(base,stage); stage.rename(out); print('Output:',out,'| Inputs unchanged; no records merged or deleted.')
if __name__=='__main__': main()
