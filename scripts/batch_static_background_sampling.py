"""Regional 1:1 background sampling using the validated pilot implementation.
No confirmed absences, no approved labels, no source changes. Requires pilot script.
"""
import argparse
import json
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pyogrio
import rasterio

from pilot_static_background_sampling import (
    PILOTS, BUFFER, SEED, DIST, COLS, ROOT, paths, load_districts, read,
    export, marker, require, local_observations, sample_grid,
    self_test as pilot_self_test,
)

KEY = ['source', 'qa_record_id']
GRID = DIST + ['row', 'col']
COORDS = ['x', 'y', 'longitude', 'latitude']
STATS = ['proposals_evaluated', 'outside_district', 'inventory_buffer_rejected', 'feature_or_land_rejected']
COUNTS = ['inventory_points', 'inventory_polygons', 'review_candidates', 'selected_positive_centres']


def compare_pilot(f, r, old_f, old_r):
    """Compare actual selections as well as the reported counts, not just totals."""
    for k, v in r.items():
        require(k in old_r and str(v) == str(old_r[k]), f'Pilot report mismatch: {r["state"]}/{r["district"]} {k}')
    a = f.sort_values('background_id').reset_index(drop=True)
    b = old_f.sort_values('background_id').reset_index(drop=True)
    exact = [k for k in COLS if k not in COORDS]
    require(len(a)==len(b) and a[exact].astype(str).equals(b[exact].astype(str)), 'Pilot cell or attribute mismatch')
    require(np.allclose(a[COORDS].to_numpy(dtype=float), b[COORDS].to_numpy(dtype=float), rtol=0, atol=1e-8), 'Pilot coordinate mismatch')


def grid_keys(f):
    return set(f.assign(row=f.row.astype(int), col=f.col.astype(int))[GRID].itertuples(index=False, name=None))


def metric_keys(f):
    return set(f.assign(epsg=f.epsg.astype(int), x=f.x.astype(float).round(6), y=f.y.astype(float).round(6))[["epsg","x","y"]].itertuples(index=False,name=None))


def check_samples(samples, positives):
    require(samples.background_id.is_unique and len(grid_keys(samples)) == len(samples), 'Repeated background grid cell or ID')
    require(len(metric_keys(samples)) == len(samples), 'Repeated background metric centre')
    require(not (grid_keys(samples) & grid_keys(positives)), 'Background-positive grid collision')
    require(not (metric_keys(samples) & metric_keys(positives)), 'Background-positive metric centre collision')
    require(np.isfinite(samples[COORDS].to_numpy(dtype=float)).all(), 'Invalid background coordinates')
    require(samples.sample_kind.eq('BACKGROUND_UNVERIFIED').all() and samples.approved_training_label.astype(str).str.lower().eq('false').all(), 'Unexpected label approval')
    require(samples.exclusion_buffer_m.astype(float).eq(BUFFER).all(), 'Exclusion buffer mismatch')


def account(report, samples, quotas):
    require(not report.duplicated(DIST).any() and set(report[DIST].itertuples(index=False,name=None)) == set(quotas), 'District audit coverage mismatch')
    got = samples.groupby(DIST).size().to_dict()
    require(set(got) <= set(quotas), 'Unknown sample district')
    for r in report.to_dict('records'):
        key = (r['state'], r['district'])
        require(r['target']==quotas[key] and r['selected']==got.get(key,0) and r['shortfall']==r['target']-r['selected'] >= 0, f'{key}: quota mismatch')
        require(r['proposals_evaluated']==r['selected']+sum(r[k] for k in STATS[1:]), f'{key}: proposal accounting mismatch')
        expected = 'SKIP_NO_BASELINE' if not r['target'] else ('PASS' if not r['shortfall'] else 'REVIEW_SHORTFALL')
        require(r['status']==expected, f'{key}: inconsistent status')
    require(int(report.selected.sum())==len(samples), 'Regional count mismatch')


def run(root, out):
    base = root/'data/processed/landslides'
    p = base/'static_positive_sampling_batch_v1'
    u = base/'unified_inventory_v1'
    prior = base/'static_background_sampling_pilot_v1'
    m=marker(p/'complete.json'); um=marker(u/'complete.json'); pm=marker(prior/'complete.json')
    require(m['status']=='PASS' and m['scope']=='ALL_DISTRICTS' and m['districts']==137 and m['center_baseline_candidates']==41884 and m['overlap25_sensitivity_candidates']==6133 and m['approved_training_labels']==0, 'Positive batch mismatch')
    require(um['status']=='PASS' and um['records']==128846 and um['approved_training_labels']==0, 'Unified inventory mismatch')
    require(pm['status']=='PASS' and pm['scope']=='PILOT_ONLY' and pm['districts']==4 and pm['target']==pm['selected']==2219 and pm['shortfall']==0 and pm['seed']==SEED and pm['exclusion_buffer_m']==BUFFER and pm['approved_training_labels']==pm['confirmed_absences']==0, 'Successful unchanged background pilot required')
    a=read(p/'center_baseline_candidates.csv'); b=read(p/'overlap25_sensitivity_candidates.csv'); allp=pd.concat([a,b],ignore_index=True)
    require(len(a)==41884 and len(b)==6133 and not allp.duplicated(KEY).any() and len(grid_keys(allp))==len(metric_keys(allp))==len(allp), 'Positive counts or uniqueness mismatch')
    require(a.method.eq('CENTER_BASELINE').all() and b.method.eq('OVERLAP_25_SENSITIVITY').all() and allp.approved_training_label.str.lower().eq('false').all(), 'Positive flag mismatch')
    for k in COORDS: allp[k]=pd.to_numeric(allp[k],errors='raise')
    require(np.isfinite(allp[COORDS]).all().all() and allp.longitude.between(-180,180).all() and allp.latitude.between(-90,90).all(), 'Invalid positive coordinates')
    gpkg=u/'unified_landslide_inventory.gpkg'; repaired=base/'uglc_quality_v1/polygons/repaired_review_candidates.gpkg'
    for path,layer,n in [(gpkg,'inventory_points',35785),(gpkg,'inventory_polygons',93061),(repaired,'review_candidates',6)]:
        require(pyogrio.read_info(path,layer=layer)['features']==n,f'{layer}: source count mismatch')
    ds=load_districts(root); districts=sorted(ds[DIST].itertuples(index=False,name=None))
    require(len(districts)==137 and set(allp[DIST].itertuples(index=False,name=None))<=set(districts),'District inventory mismatch')
    ac=a.groupby(DIST).size().to_dict(); bc=b.groupby(DIST).size().to_dict()
    quotas={k:int(ac.get(k,0)) for k in districts}
    require(sum(quotas.values())==41884 and sum(v>0 for v in quotas.values())==90,'Baseline quota mismatch')
    ps=read(p/'district_sampling_audit.csv')
    require(len(ps)==137 and not ps.duplicated(DIST).any() and set(ps[DIST].itertuples(index=False,name=None))==set(districts) and ps.audit_crosscheck.eq('PASS').all(),'Positive district audit mismatch')
    for r in ps.to_dict('records'):
        k=(r['state'],r['district']); require(int(r['baseline_selected'])==quotas[k] and int(r['overlap25_selected'])==bc.get(k,0),f'{k}: positive count disagreement')
    old=read(prior/'background_candidates.csv'); oldr=read(prior/'district_background_audit.csv')
    require(len(old)==2219 and len(oldr)==4 and not oldr.duplicated(DIST).any() and set(oldr[DIST].itertuples(index=False,name=None))==set(PILOTS),'Pilot CSV count mismatch')
    check_samples(old,allp)
    require(set(old[DIST].itertuples(index=False,name=None))<=set(PILOTS),'Unexpected pilot district')
    frames=[]; reports=[]
    for ix,(s,d) in enumerate(districts,1):
        target=quotas[s,d]
        # No candidates needed in zero-baseline districts. Still rerun the zero-target pilot.
        checked=target>0 or (s,d) in PILOTS
        if checked:
            district=ds[(ds.state==s)&(ds.district==d)]; require(len(district)==1,'Missing district')
            refpath,files=paths(root,s,d)
            with ExitStack() as stack:
                ref=stack.enter_context(rasterio.open(refpath)); srcs={k:stack.enter_context(rasterio.open(v)) for k,v in files.items()}
                projected=district.to_crs(ref.crs).geometry.iloc[0]; require(projected.is_valid and not projected.is_empty,'Invalid projected district')
                tree,counts=local_observations(gpkg,repaired,allp,projected,ref.crs,BUFFER)
                f,r=sample_grid(ref,srcs,projected,tree,target,s,d); r.update(counts)
        else:
            f=pd.DataFrame(columns=COLS)
            r=dict(state=s,district=d,target=0,selected=0,shortfall=0,status='SKIP_NO_BASELINE',**dict.fromkeys(STATS,0),**dict.fromkeys(COUNTS,''))
        if (s,d) in PILOTS:
            compare_pilot(f,r,old[(old.state==s)&(old.district==d)],oldr[(oldr.state==s)&(oldr.district==d)].iloc[0].to_dict())
            r['pilot_crosscheck']='PASS'
        else: r['pilot_crosscheck']='NOT_PILOT'
        r['inventory_checked']=checked
        if len(f): frames.append(f)
        reports.append(r)
        print(f'[{ix}/137] {s}/{d} | target {target} | selected {r["selected"]} | {r["status"]} | pilot {r["pilot_crosscheck"]}',flush=True)
    samples=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(columns=COLS)
    report=pd.DataFrame(reports)
    check_samples(samples,allp); account(report,samples,quotas)
    require(report.pilot_crosscheck.eq('PASS').sum()==4,'Incomplete pilot checks')
    state=report.groupby('state',as_index=False).agg(districts=('district','size'),districts_with_target=('target',lambda x:int(x.gt(0).sum())),target=('target','sum'),selected=('selected','sum'),shortfall=('shortfall','sum'),proposals_evaluated=('proposals_evaluated','sum'),outside_district=('outside_district','sum'),inventory_buffer_rejected=('inventory_buffer_rejected','sum'),feature_or_land_rejected=('feature_or_land_rejected','sum'))
    require(int(state.target.sum())==41884 and int(state.selected.sum())==len(samples),'State accounting mismatch')
    export(out,'background_candidates.csv',samples); export(out,'district_background_audit.csv',report); export(out,'state_background_summary.csv',state)
    meta=dict(status='PASS' if not report.shortfall.any() else 'REVIEW',scope='ALL_DISTRICTS',districts=137,districts_with_target=90,districts_skipped_no_baseline=47,districts_with_shortfall=int(report.shortfall.gt(0).sum()),target=41884,selected=len(samples),shortfall=int(report.shortfall.sum()),pilot_crosschecks=4,seed=SEED,exclusion_buffer_m=BUFFER,approved_training_labels=0,confirmed_absences=0)
    (out/'complete.json').write_text(json.dumps(meta,indent=2),encoding='utf-8'); print(json.dumps(meta,indent=2))


def self_test():
    pilot_self_test()
    f=pd.DataFrame([dict(background_id='BG_T_Test_1_1',state='T',district='Test',row=1,col=1,epsg=32643,x=500045.,y=3300045.,longitude=75.,latitude=30.,exclusion_buffer_m=BUFFER,sample_kind='BACKGROUND_UNVERIFIED',approved_training_label=False)],columns=COLS)
    r=dict(state='T',district='Test',target=1,selected=1,shortfall=0,status='PASS',proposals_evaluated=2,outside_district=1,inventory_buffer_rejected=0,feature_or_land_rejected=0)
    z=dict(state='T',district='Empty',target=0,selected=0,shortfall=0,status='SKIP_NO_BASELINE',**dict.fromkeys(STATS,0))
    p=f.copy(); p['row']=2; p['x']+=30
    check_samples(f,p); account(pd.DataFrame([r,z]),f,{('T','Test'):1,('T','Empty'):0})
    with TemporaryDirectory() as td:
        path=Path(td); export(path,'f.csv',f); old=read(path/'f.csv')
        compare_pilot(f,r,old,{k:str(v) for k,v in r.items()})
        def rejects(fn):
            try: fn()
            except ValueError: return
            raise AssertionError('Expected rejection')
        wrong=old.copy(); wrong.loc[0,'col']='2'
        rejects(lambda:compare_pilot(f,r,wrong,{k:str(v) for k,v in r.items()}))
        rejects(lambda:check_samples(f,f))
        rejects(lambda:account(pd.DataFrame([r,z]),f,{('T','Test'):2,('T','Empty'):0}))
    print('Regional accounting, exact pilot-cell comparison, corruption and positive collision rejection: PASS')


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=ROOT); ap.add_argument('--self-test',action='store_true'); args=ap.parse_args()
    if args.self_test: self_test(); return
    out=args.root/'data/processed/landslides/static_background_sampling_batch_v1'; stage=out.with_name(out.name+'.staging')
    require(not out.exists() and not stage.exists(),'Output or staging exists; preserve or rename it before rerunning')
    stage.mkdir(parents=True); run(args.root,stage); stage.rename(out); print('Output:',out,'| Inputs unchanged; no confirmed absences or approved labels.')

if __name__=='__main__': main()
