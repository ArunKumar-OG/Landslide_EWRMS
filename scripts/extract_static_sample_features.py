"""Extract raw static raster values at validated sample cells; no final labels.
Default: four-district pilot. --all-districts: regional extraction with pilot check.
Keep prior audit scripts alongside this script. All inputs are read-only.
"""
import argparse
from collections import defaultdict
from contextlib import ExitStack
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_origin
from pyproj import Transformer
import shapely
from shapely.geometry import box
from audit_footprint_grid_batch import ROOT, paths, FEATURES, LC_CODES, load_districts, read, export, marker, require
from pilot_static_background_sampling import PILOTS

DIST = ['state','district']
CELL = DIST+['row','col']
COORD = ['x','y','longitude','latitude']
BASE='CENTER_BASELINE'
EXTRA='OVERLAP_25_SENSITIVITY'
BG='BACKGROUND_UNVERIFIED'
META=['sample_id','state','district','row','col','epsg']+COORD+['sample_kind','source','qa_record_id','background_id','overlap_fraction','exclusion_buffer_m','provisional_target','approved_training_label']
CATEGORICAL={'lithology','landcover'}


def assemble(a,b,c):
    frames=[]
    for f,kind,label in [(a,BASE,1),(b,EXTRA,1),(c,BG,0)]:
        f=f.copy()
        require(f.approved_training_label.str.lower().eq('false').all(), 'Unexpected approved sample')
        require(f['method' if label else 'sample_kind'].eq(kind).all(),'Sample kind mismatch')
        for k in ['row','col','epsg']:
            v=pd.to_numeric(f[k],errors='raise'); require(np.isfinite(v).all() and (v==np.floor(v)).all(),f'Invalid {k}'); f[k]=v.astype('int64')
        for k in COORD: f[k]=pd.to_numeric(f[k],errors='raise')
        require(np.isfinite(f[COORD].to_numpy()).all() and f.longitude.between(-180,180).all() and f.latitude.between(-90,90).all(),'Invalid coordinates')
        require(f[DIST].ne('').all().all() and (f[['row','col']]>=0).all().all(),'Invalid district/cell')
        if label:
            require(f[['source','qa_record_id']].ne('').all().all(),'Missing provenance')
            v=pd.to_numeric(f.overlap_fraction,errors='raise'); require(v.gt(0).all() and v.le(1).all() and (kind!=EXTRA or v.ge(.25).all()),'Invalid footprint fraction')
            f['overlap_fraction']=v; f['sample_id']=[json.dumps(['POS',s,q],ensure_ascii=False,separators=(',',':')) for s,q in zip(f.source,f.qa_record_id)]
        else:
            require(f.background_id.ne('').all() and pd.to_numeric(f.exclusion_buffer_m,errors='raise').eq(300).all(),'Invalid background metadata')
            f['sample_id']='BG:'+f.background_id
        f['sample_kind']=kind; f['provisional_target']=label; f['approved_training_label']=False
        frames.append(f.reindex(columns=META,fill_value=''))
    out=pd.concat(frames,ignore_index=True)
    require(not out.sample_id.duplicated().any() and not out.duplicated(CELL).any(),'Duplicate sample ID or district-grid cell')
    require(not out.assign(x=out.x.round(6),y=out.y.round(6)).duplicated(['epsg','x','y']).any(),'Duplicate metric centre')
    return out


def load(root):
    base=root/'data/processed/landslides'; p=base/'static_positive_sampling_batch_v1'; b=base/'static_background_sampling_batch_v1'
    pm,bm=marker(p/'complete.json'),marker(b/'complete.json')
    for m in (pm,bm): require(m['status']=='PASS' and m['scope']=='ALL_DISTRICTS' and m['districts']==137 and m['approved_training_labels']==0,'Prior batch incomplete')
    require(pm['center_baseline_candidates']==41884 and pm['overlap25_sensitivity_candidates']==6133,'Positive marker mismatch')
    require(bm['target']==bm['selected']==41884 and bm['shortfall']==bm['confirmed_absences']==0 and bm['exclusion_buffer_m']==300,'Background marker mismatch')
    a=read(p/'center_baseline_candidates.csv'); e=read(p/'overlap25_sensitivity_candidates.csv'); c=read(b/'background_candidates.csv')
    require((len(a),len(e),len(c))==(41884,6133,41884),'Input row count mismatch')
    require(a.groupby(DIST).size().equals(c.groupby(DIST).size()) and a.groupby(DIST).ngroups==90,'District class counts mismatch')
    return assemble(a,e,c)


def extract(refpath,files,f,district,block=256):
    require(set(files)==set(FEATURES),'Expected 18 features')
    f=f.copy().reset_index(drop=True)
    with ExitStack() as stack:
        ref=stack.enter_context(rasterio.open(refpath)); srcs={k:stack.enter_context(rasterio.open(files[k])) for k in FEATURES}
        require(ref.crs is not None and ref.crs.is_projected and ref.crs.linear_units.lower() in ('metre','meter','metres','meters') and np.allclose(ref.res,(30,30)) and ref.transform.b==ref.transform.d==0 and ref.transform.a>0 and ref.transform.e<0,'Invalid 30 m grid')
        for k,src in srcs.items(): require(src.count==1 and src.shape==ref.shape and src.crs==ref.crs and src.transform.almost_equals(ref.transform,precision=1e-7),f'{k}: grid mismatch')
        rr=f.row.to_numpy(dtype=np.int64); cc=f.col.to_numpy(dtype=np.int64)
        require(f.epsg.eq(ref.crs.to_epsg()).all() and ((rr>=0)&(rr<ref.height)&(cc>=0)&(cc<ref.width)).all(),'Sample CRS or bounds mismatch')
        x=ref.transform.c+(cc+.5)*ref.transform.a; y=ref.transform.f+(rr+.5)*ref.transform.e
        require(np.allclose(x,f.x,rtol=0,atol=1e-5) and np.allclose(y,f.y,rtol=0,atol=1e-5),'Stored projected centre mismatch')
        lon,lat=Transformer.from_crs(ref.crs,4326,always_xy=True).transform(x,y)
        require(np.allclose(lon,f.longitude,rtol=0,atol=1e-7) and np.allclose(lat,f.latitude,rtol=0,atol=1e-7),'Stored geographic centre mismatch')
        require(shapely.covers(district,shapely.points(x,y)).all(),'Sample outside district')
        groups=defaultdict(list)
        for i,(r,c) in enumerate(zip(rr,cc)): groups[(r//block,c//block)].append(i)
        for k,src in srcs.items():
            values=np.full(len(f),np.nan,dtype=np.float64)
            for (br,bc),indices in groups.items():
                r0=int(br*block); c0=int(bc*block); w=Window(c0,r0,min(block,ref.width-c0),min(block,ref.height-r0)); ii=np.asarray(indices)
                arr=src.read(1,window=w,masked=True); z=arr[rr[ii]-r0,cc[ii]-c0]; v=np.asarray(z.data,dtype=np.float64)
                ok=~np.ma.getmaskarray(z)&np.isfinite(v)
                if src.nodata is not None: ok &= v!=src.nodata
                if k=='lithology': ok &= np.isin(v,np.arange(1,16))
                if k=='landcover': ok &= np.isin(v,[j for j in LC_CODES if j not in (70,80)])
                if not ok.all(): raise ValueError(f'{k}: invalid sampled value for {f.sample_id.iloc[ii[np.flatnonzero(~ok)[0]]]}')
                values[ii]=v
            require(np.isfinite(values).all(),f'{k}: missing extracted value')
            f[k]=values.astype(np.int64) if k in CATEGORICAL else values
    return f


def summarize(f):
    rows=[]; cats=[]
    for kind,g in f.groupby('sample_kind'):
        for k in FEATURES:
            if k in CATEGORICAL:
                for v,n in g[k].value_counts().sort_index().items(): cats.append(dict(sample_kind=kind,feature=k,category=int(v),count=int(n)))
            else:
                v=g[k]; rows.append(dict(sample_kind=kind,feature=k,count=len(v),minimum=v.min(),p01=v.quantile(.01),median=v.median(),p99=v.quantile(.99),maximum=v.max(),mean=v.mean(),std=v.std(ddof=0)))
    return pd.DataFrame(rows),pd.DataFrame(cats)


def run(root,out,all_districts):
    f=load(root); ds=load_districts(root); known=set(ds[DIST].itertuples(index=False,name=None)); require(set(f[DIST].itertuples(index=False,name=None))<=known,'Unknown sample district')
    chosen=known if all_districts else set(PILOTS); frames=[]; reports=[]
    for ix,(s,d) in enumerate(sorted(chosen),1):
        g=f[(f.state==s)&(f.district==d)]; ref,files=paths(root,s,d)
        if len(g):
            with rasterio.open(ref) as r: district=ds[(ds.state==s)&(ds.district==d)].to_crs(r.crs).geometry.iloc[0]
            h=extract(ref,files,g,district); frames.append(h)
        counts=g.sample_kind.value_counts(); report=dict(state=s,district=d,baseline=int(counts.get(BASE,0)),background=int(counts.get(BG,0)),overlap25=int(counts.get(EXTRA,0)),status='PASS' if len(g) else 'SKIP_NO_SAMPLES')
        reports.append(report); print(f'[{ix}/{len(chosen)}] {s}/{d} | {report}',flush=True)
    final=pd.concat(frames,ignore_index=True).sort_values('sample_id').reset_index(drop=True); require(len(final)==len(f) if all_districts else len(final)==sum(tuple(x) in chosen for x in f[DIST].itertuples(index=False,name=None)),'Extracted count mismatch')
    require(not final.sample_id.duplicated().any() and np.isfinite(final[list(FEATURES)].to_numpy(dtype=float)).all(),'Invalid feature output')
    baseline=final[final.sample_kind.ne(EXTRA)].reset_index(drop=True); extra=final[final.sample_kind.eq(EXTRA)].reset_index(drop=True)
    require(baseline[baseline.sample_kind.eq(BASE)].groupby(DIST).size().equals(baseline[baseline.sample_kind.eq(BG)].groupby(DIST).size()),'Unbalanced extraction')
    if all_districts:
        pilot=root/'data/processed/landslides/static_feature_extraction_pilot_v1'; m=marker(pilot/'complete.json'); require(m['status']=='PASS' and m['scope']=='PILOT_ONLY','Missing successful pilot')
        for name,table in [('baseline_features.csv',baseline),('overlap25_features.csv',extra)]:
            old=read(pilot/name); current=table[[tuple(v) in set(PILOTS) for v in table[DIST].itertuples(index=False,name=None)]].fillna('').astype(str).sort_values('sample_id').reset_index(drop=True)
            require(current.equals(old.sort_values('sample_id').reset_index(drop=True)),f'{name}: pilot values mismatch')
    export(out,'baseline_features.csv',baseline); export(out,'overlap25_features.csv',extra)
    stats,cats=summarize(final); export(out,'continuous_feature_summary.csv',stats); export(out,'categorical_feature_counts.csv',cats); export(out,'district_extraction_audit.csv',pd.DataFrame(reports))
    result=dict(status='PASS',scope='ALL_DISTRICTS' if all_districts else 'PILOT_ONLY',districts=len(chosen),baseline_positive=int(baseline.provisional_target.sum()),background=int(baseline.provisional_target.eq(0).sum()),baseline_rows=len(baseline),overlap25_rows=len(extra),raw_features=len(FEATURES),baseline_feature_values=len(baseline)*len(FEATURES),missing_feature_values=0,approved_training_labels=0,confirmed_absences=0,pilot_value_crosscheck='PASS' if all_districts else 'NOT_APPLICABLE')
    (out/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result,indent=2))


def self_test():
    with TemporaryDirectory() as tmp:
        p=Path(tmp); tr=from_origin(500000,3300000,30,30); files={k:p/(k+'.tif') for k in FEATURES}
        def write(k,arr,transform=tr):
            with rasterio.open(files[k],'w',driver='GTiff',height=3,width=3,count=1,dtype='float32',crs='EPSG:32643',transform=transform,nodata=-9999) as dst: dst.write(np.asarray(arr,dtype=np.float32),1)
        for k in FEATURES: write(k,np.full((3,3),10 if k=='landcover' else 2 if k=='lithology' else 5))
        f=pd.DataFrame({'sample_id':['a','b'],'row':[0,2],'col':[0,2],'epsg':[32643]*2,'x':[500015.,500075.],'y':[3299985.,3299925.]})
        f['longitude'],f['latitude']=Transformer.from_crs(32643,4326,always_xy=True).transform(f.x.to_numpy(),f.y.to_numpy()); district=box(500000,3299910,500090,3300000)
        write('elevation',np.arange(9).reshape(3,3)+100)
        h=extract(files['elevation'],files,f,district,block=2); require(h.elevation.tolist()==[100.,108.] and h.slope.eq(5).all() and h.lithology.eq(2).all() and h.landcover.eq(10).all(),'Synthetic extraction failed'); export(p,'test.csv',h)
        def rejects(fn):
            try: fn()
            except ValueError: return
            raise AssertionError('Expected rejection')
        wrong=f.copy(); wrong.loc[0,'x']+=30; rejects(lambda:extract(files['elevation'],files,wrong,district))
        for k,val in [('slope',-9999),('lithology',16),('landcover',80)]:
            arr=np.full((3,3),5 if k=='slope' else 2 if k=='lithology' else 10); arr[0,0]=val; write(k,arr); rejects(lambda:extract(files['elevation'],files,f,district)); arr[0,0]=arr[1,1]; write(k,arr)
        write('tpi',np.full((3,3),5),from_origin(500030,3300000,30,30)); rejects(lambda:extract(files['elevation'],files,f,district))
    print('Synthetic 18-feature extraction, block indexing, coordinate validation, NoData, geology, landcover, shifted-grid rejection and CSV readback: PASS')


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=ROOT); ap.add_argument('--self-test',action='store_true'); ap.add_argument('--all-districts',action='store_true'); args=ap.parse_args()
    if args.self_test: self_test(); return
    out=args.root/'data/processed/landslides'/('static_feature_extraction_batch_v1' if args.all_districts else 'static_feature_extraction_pilot_v1'); stage=out.with_name(out.name+'.staging'); require(not out.exists() and not stage.exists(),'Output/staging already exists; preserve or rename before rerun'); stage.mkdir(parents=True); run(args.root,stage,args.all_districts); stage.rename(out); print('Output:',out,'| Inputs unchanged; provisional targets only.')

if __name__=='__main__': main()
