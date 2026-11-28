"""Provisional static footprint and dynamic review queues; never approves labels.
Read-only CSV inputs. Preserve all provenance and group-level conflicts.
"""
import argparse, json
from pathlib import Path
from tempfile import TemporaryDirectory
import pandas as pd
from resolve_landslide_duplicates import read, marker, require, export, keys, KEY, EXPECTED, ASSOC, RAIN
from collections import Counter

ROOT=Path(__file__).resolve().parent.parent
PROPOSED='PROPOSED_SAME_REPORTED_EVENT'
FOOT='MAPPED_FOOTPRINT_CANDIDATE'
SLIDE={'LANDSLIDE_OR_SLIDE'}
REVIEW_TYPES={'UNKNOWN','UNMAPPED_TYPE_REVIEW','FLOW_OR_AVALANCHE_REVIEW','FALL_OR_TOPPLE','COMPLEX'}

def load(base):
    p=base/'duplicate_resolution_v1'; m=marker(p/'complete.json'); f=read(p/'resolution_record_flags.csv'); g=read(p/'event_review_groups.csv'); q=read(base/'quality_standardization_v1/quality_candidates.csv'); t=read(base/'temporal_standardization_v1/temporal_candidates.csv')
    require(m['status']=='PASS' and m['records']==len(f) and m['records_deleted']==m['approved_training_labels']==0,'Resolution marker mismatch')
    for x in [f,q,t]:
        require(Counter(x.source)==EXPECTED and not x.duplicated(KEY).any() and x.qa_record_id.ne('').all(),'Input IDs/counts')
        require(keys(x)==keys(f),'Input keys differ')
    require(f.training_label_approved.str.lower().eq('false').all() and f.record_deleted.str.lower().eq('false').all(),'Unexpected label approval or deletion')
    require(f.set_index(KEY).sort_index()[q.columns.difference(KEY)].equals(q.set_index(KEY).sort_index()[q.columns.difference(KEY)]),'Quality values differ')
    cols=['layer','date_precision','date_status','start_date','end_date','representation_status']
    require(f.set_index(KEY).sort_index()[cols].equals(t.set_index(KEY).sort_index()[cols]),'Temporal values differ')
    require(g.event_review_group.is_unique and len(g)==m['event_review_components'],'Event group marker mismatch')
    eg=f[f.event_review_group.ne('')]; require(eg.groupby('event_review_group').size().to_dict()==g.set_index('event_review_group').records.astype(int).to_dict(),'Event group count mismatch')
    require(eg.groupby('event_review_group').event_review_status.nunique().eq(1).all() and eg.groupby('event_review_group').event_review_status.first().to_dict()==g.set_index('event_review_group').status.to_dict(),'Event status mismatch')
    require(f.representation_status.eq(ASSOC).sum()==m['associated_points'] and f.static_geometry_group.ne('').sum()==m['independent_representation_candidates'],'Representation count mismatch')
    require(f.loc[f.static_geometry_group.ne(''),'static_geometry_group'].nunique()==m['distinct_exact_geometry_groups'],'Geometry group count mismatch')
    z=f[f.spatial_status.eq(FOOT)]; require(len(z)==m['mapped_footprint_candidates'] and z.static_geometry_group.nunique()==m['distinct_mapped_footprint_geometries'],'Footprint count mismatch')
    require(f.dynamic_screen.eq(RAIN).sum()==m['rain_day_review_candidates'],'Rain queue mismatch')
    require(f.event_review_status.eq(PROPOSED).sum()==m['records_in_proposed_event_groups'],'Proposed member mismatch')
    # Retain temporal role and satellite context without inferring occurrence dates.
    return f.merge(t[KEY+['date_role','sat_before','sat_after','sat_window_status','context_start','context_end','raw_start','raw_end','alternative_ist_day']],on=KEY,validate='one_to_one')

def choose(f):
    require(not f.duplicated(KEY).any(),'Duplicate record keys')
    f=f.copy(); f['static_queue']='NOT_MAPPED_FOOTPRINT_REVIEW'; f['static_group_representative']='False'
    f.loc[f.representation_status.eq(ASSOC),'static_queue']='ASSOCIATED_NOT_INDEPENDENT'
    f.loc[f.spatial_status.eq('BUFFER_NOT_FOOTPRINT'),'static_queue']='BUFFER_NOT_FOOTPRINT'
    fp=f[f.spatial_status.eq(FOOT)].copy()
    require(fp.static_geometry_group.ne('').all() and fp.geometry_kind.eq('POLYGON').all(),'Footprint geometry mismatch')
    # Group-level type conflicts are not masked by choosing a known slide representative.
    fp['has_slide']=fp.type_group.isin(SLIDE)
    fp['other_failure']=fp.type_group.eq('OTHER_FAILURE_REVIEW')
    fp['other_movement']=fp.type_group.isin({'FLOW_OR_AVALANCHE_REVIEW','FALL_OR_TOPPLE','COMPLEX'})
    fp['unknown_type']=~fp.type_group.isin(SLIDE|{'OTHER_FAILURE_REVIEW','FLOW_OR_AVALANCHE_REVIEW','FALL_OR_TOPPLE','COMPLEX'})
    agg=fp.groupby('static_geometry_group').agg(members=('qa_record_id','size'),has_slide=('has_slide','any'),other_failure=('other_failure','any'),other_movement=('other_movement','any'),unknown_type=('unknown_type','any')).reset_index()
    agg['static_queue']='FOOTPRINT_TYPE_UNCONFIRMED_REVIEW'
    agg.loc[agg.has_slide & ~agg.other_failure & ~agg.other_movement,'static_queue']='SLIDE_FOOTPRINT_PRIORITY_REVIEW'
    agg.loc[agg.other_movement & ~agg.other_failure,'static_queue']='MOVEMENT_TYPE_REVIEW'
    agg.loc[agg.other_failure,'static_queue']='OTHER_FAILURE_OR_TYPE_CONFLICT_REVIEW'
    # Stable representative for geometry access only, not event/date selection.
    rep=fp.sort_values(['static_geometry_group','has_slide','source','qa_record_id'],ascending=[True,False,True,True]).drop_duplicates('static_geometry_group')
    agg=agg.merge(rep[['static_geometry_group','source','layer','qa_record_id','type_raw','type_group']],on='static_geometry_group',validate='one_to_one')
    f.loc[fp.index,'static_queue']=fp.static_geometry_group.map(agg.set_index('static_geometry_group').static_queue)
    f.loc[rep.index,'static_group_representative']='True'
    rain=f[f.dynamic_screen.eq(RAIN)].copy()
    rain['review_unit']=rain.apply(lambda r: r.event_review_group if r.event_review_status==PROPOSED else json.dumps([r.source,r.qa_record_id],ensure_ascii=False,separators=(',',':')),axis=1)
    require(rain.review_unit.ne('').all(),'Missing review unit')
    # Only priority for verification. A fine reported accuracy is not independent evidence.
    rain['priority']='LOCATION_AND_DATE_REVIEW'
    rain.loc[rain.spatial_status.eq('POINT_REPORTED_FINE_REVIEW'),'priority']='REPORTED_FINE_LOCATION_REVIEW'
    rain.loc[rain.spatial_status.eq('POLYGON_METHOD_REVIEW'),'priority']='POLYGON_METHOD_AND_DATE_REVIEW'
    rain.loc[rain.type_group.eq('OTHER_FAILURE_REVIEW'),'priority']='OTHER_FAILURE_REVIEW'
    rain.loc[rain.event_review_status.eq('AMBIGUOUS_COMPONENT_RETAIN_SEPARATELY'),'priority']='AMBIGUOUS_EVENT_REVIEW'
    # Record-level queue plus full proposed group context (including coarse-location members).
    selected=set(rain.loc[rain.event_review_status.eq(PROPOSED),'event_review_group'])
    context=f[f.event_review_group.isin(selected)].copy()
    units=rain.groupby('review_unit').agg(rain_day_records=('qa_record_id','size'),sources=('source',lambda x:';'.join(sorted(set(x)))),start_date=('start_date','first'),distinct_days=('start_date','nunique'),priorities=('priority',lambda x:';'.join(sorted(set(x))))).reset_index()
    require(units.distinct_days.eq(1).all(),'Review unit contains conflicting days')
    counts=context.groupby('event_review_group').size().to_dict(); units['full_proposed_group_members']=units.review_unit.map(counts).fillna(0).astype(int)
    f['dynamic_review_unit']=''; f.loc[rain.index,'dynamic_review_unit']=rain.review_unit
    f['training_label_approved']='False'
    return f,agg,rain,units,context

def run(f,out):
    a,s,r,u,c=choose(f)
    require(len(a)==len(f) and keys(a)==keys(f),'Record accounting mismatch')
    require(len(s)==int(a.static_group_representative.eq('True').sum()) and int(s.members.sum())==int(a.spatial_status.eq(FOOT).sum()),'Static accounting mismatch')
    require(int(u.rain_day_records.sum())==len(r),'Dynamic accounting mismatch')
    for name,x in [('candidate_selection_flags.csv',a),('static_footprint_review_units.csv',s),('dynamic_rain_day_review_records.csv',r),('dynamic_review_units.csv',u),('dynamic_proposed_group_context.csv',c)]: export(out,name,x)
    for name,x,cols in [('static_review_summary.csv',s,['static_queue']),('dynamic_review_summary.csv',r,['source','priority'])]:
        summary=x.groupby(cols).size().rename('records').reset_index(); export(out,name,summary); print(summary.to_string(index=False))
    report=dict(status='PASS',records=len(a),mapped_footprint_records=int(s.members.sum()),static_footprint_review_units=len(s),static_priority_review_units=int(s.static_queue.eq('SLIDE_FOOTPRINT_PRIORITY_REVIEW').sum()),rain_day_review_records=len(r),dynamic_review_units=len(u),proposed_groups_in_dynamic_queue=int(u.full_proposed_group_members.gt(0).sum()),full_proposed_group_context_records=len(c),coarse_location_review_records=int(a.dynamic_screen.eq('COARSE_LOCATION_REVIEW').sum()),records_deleted=0,approved_training_labels=0)
    (out/'complete.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report,indent=2))

def self_test():
    def row(i,**kw):
        return dict(source='GSI',qa_record_id=str(i),layer='test',geometry_kind='POLYGON',static_geometry_group='s'+str(i),representation_status='BASELINE_CANDIDATE',spatial_status=FOOT,type_group='LANDSLIDE_OR_SLIDE',type_raw='slide',dynamic_screen='NO_SINGLE_DAY_CANDIDATE',event_review_group='',event_review_status='NO_STRONG_EVENT_GROUP',start_date='',training_label_approved='False')|kw
    f=pd.DataFrame([row(1),row(2,static_geometry_group='s1',type_group='OTHER_FAILURE_REVIEW'),row(3),row(4,representation_status=ASSOC,spatial_status='ASSOCIATED_REPRESENTATION',geometry_kind='POINT',static_geometry_group=''),row(5,source='GLC',geometry_kind='POINT',spatial_status='POINT_REPORTED_FINE_REVIEW',dynamic_screen=RAIN,event_review_group='E_a',event_review_status=PROPOSED,start_date='2023-01-01'),row(6,source='COOLR',geometry_kind='POINT',spatial_status='POINT_ACCURACY_UNCONFIRMED',dynamic_screen=RAIN,event_review_group='E_a',event_review_status=PROPOSED,start_date='2023-01-01'),row(7,source='UGLC',geometry_kind='POINT',spatial_status='POINT_COARSE_REPORTED',dynamic_screen='COARSE_LOCATION_REVIEW',event_review_group='E_a',event_review_status=PROPOSED,start_date='2023-01-01')])
    a,s,r,u,c=choose(f); assert len(s)==2 and len(r)==2 and len(u)==1 and len(c)==3
    assert s.set_index('static_geometry_group').loc['s1','static_queue']=='OTHER_FAILURE_OR_TYPE_CONFLICT_REVIEW'
    assert s.set_index('static_geometry_group').loc['s3','static_queue']=='SLIDE_FOOTPRINT_PRIORITY_REVIEW'
    assert a.loc[a.qa_record_id=='4','static_group_representative'].iloc[0]=='False'
    with TemporaryDirectory() as tmp: run(f,Path(tmp))
    print('Synthetic type conflict, associated exclusion, event grouping, coarse context retention and CSV readback: PASS')

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',type=Path,default=ROOT); p.add_argument('--self-test',action='store_true'); args=p.parse_args()
    if args.self_test: self_test(); return
    base=args.root/'data/processed/landslides'; out=base/'candidate_selection_v1'; stage=out.with_name(out.name+'.staging')
    require(not out.exists() and not stage.exists(),'Output/staging exists; inspect before retrying, no overwrite')
    f=load(base); stage.mkdir(); run(f,stage); stage.rename(out); print('Output:',out,'| Inputs unchanged; no training labels approved.')
if __name__=='__main__': main()
