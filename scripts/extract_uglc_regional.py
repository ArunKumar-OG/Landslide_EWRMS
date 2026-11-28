"""UGLC regional extraction, preserving source attributes and full geometries.
WKT is WGS84 per authors' documentation; no country filtering or date imputation.
Only bbox candidates receive full topology validation. Raw files unchanged.
"""
from pathlib import Path
from collections import Counter
from tempfile import TemporaryDirectory
import argparse
import csv
import json
import shutil
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import pyogrio
from shapely.geometry import Point, box

ROOT = Path(__file__).resolve().parent.parent
FILES = {'points': ('UGLC_point.csv', 1060503), 'polygons': ('UGLC_poly.csv', 984109)}
REASONS = ('missing_wkt','unreadable_wkt','empty_geometry','unexpected_type',
           'invalid_bounds','outside_bbox','invalid_candidate_geometry',
           'outside_region','regional_candidates')
csv.field_size_limit(128*1024*1024)


def load_districts(root):
    d = gpd.read_file(root/'data/processed/boundary/ihr_districts.gpkg')
    if d.crs is None or len(d)!=137 or d[['state','district']].isna().any().any() or d.duplicated(['state','district']).any():
        raise ValueError('Unexpected district inventory or CRS')
    d=d[['state','district','geometry']].to_crs(4326).reset_index(drop=True)
    d.geometry=d.geometry.make_valid()
    if not d.geometry.geom_type.isin(['Polygon','MultiPolygon']).all() or not d.geometry.is_valid.all() or d.geometry.is_empty.any():
        raise ValueError('Invalid district polygons')
    return d


def extract(path, dest, districts, kind, expected, batch_size=2000):
    dest.mkdir(parents=True,exist_ok=False)
    target=dest/'candidates.gpkg'
    stats=Counter({k:0 for k in REASONS}); stats.update({'source_records':0,'blank_rows':0,'multiple_district_records':0,'touch_only_records':0,'district_links':0})
    profiles={}; district_counts=Counter(); types=Counter()
    tree=shapely.STRtree(districts.geometry.array)
    west,south,east,north=districts.total_bounds
    allowed=[0] if kind=='points' else [3,6]  # Shapely type IDs
    with path.open('r',encoding='utf-8-sig',newline='') as inp, (dest/'uglc_district_links.csv').open('w',encoding='utf-8',newline='') as lf, (dest/'uglc_geometry_issues.csv').open('w',encoding='utf-8',newline='') as qf:
        # Same header-based delimiter detection as initial inspection.
        first=inp.readline(); inp.seek(0)
        delimiter=max(['|',',',';','\t'],key=lambda sep:len(next(csv.reader([first],delimiter=sep))))
        if delimiter!='|': raise ValueError(f'Expected pipe delimiter; detected {delimiter!r}')
        reader=csv.reader(inp,delimiter=delimiter,strict=True)
        header=next(reader)
        norm=[s.strip().upper() for s in header]
        required={'WKT_GEOM','ID','OLD DATASET','OLD ID','ACCURACY','START DATE','END DATE','RELIABILITY','RECORD TYPE'}
        if not required.issubset(norm) or len(set(norm))!=len(norm) or '' in norm: raise ValueError(f'Unexpected schema: {header}')
        if any(s.startswith('QA_') or s in {'GEOMETRY','FID'} for s in norm): raise ValueError('Reserved column collision')
        wkt_i=norm.index('WKT_GEOM'); id_i=norm.index('ID')
        attr=[i for i in range(len(header)) if i!=wkt_i]
        for k in ['OLD DATASET','COUNTRY','ACCURACY','RELIABILITY','RECORD TYPE','TYPE','PHYSICAL FACTORS','TRIGGER','AFFIDABILITY']:
            if k in norm: profiles[k]=Counter()
        link=csv.writer(lf); link.writerow(['qa_record_id','ihr_state','ihr_district','qa_boundary_touch_only'])
        issues=csv.writer(qf); issues.writerow(['qa_record_id','source_id','csv_record_number','reason'])
        print(f'{kind}: {path.name} | delimiter {delimiter!r} | expected {expected:,}',flush=True)
        def process(rows, numbers):
            if not rows: return
            n=len(rows)
            wkts=np.array([r[wkt_i] for r in rows],dtype=object)
            missing=np.array([str(x).strip().casefold() in {'','null','nan','none','na','n/a'} for x in wkts])
            geoms=shapely.from_wkt(np.where(missing,None,wkts),on_invalid='ignore')
            codes=shapely.get_type_id(geoms)
            for code,count in zip(*np.unique(codes,return_counts=True)): types[int(code)]+=int(count)
            reason=np.full(n,'',dtype=object)
            def flag(mask,label): reason[(reason=='') & mask]=label
            flag(missing,'missing_wkt'); flag(shapely.is_missing(geoms),'unreadable_wkt')
            flag(shapely.is_empty(geoms),'empty_geometry'); flag(~np.isin(codes,allowed),'unexpected_type')
            b=shapely.bounds(geoms)
            flag(~np.isfinite(b).all(axis=1)|(b[:,0]<-180)|(b[:,2]>180)|(b[:,1]<-90)|(b[:,3]>90),'invalid_bounds')
            flag((b[:,2]<west)|(b[:,0]>east)|(b[:,3]<south)|(b[:,1]>north),'outside_bbox')
            idx=np.flatnonzero(reason=='')
            # Keep originals in raw CSV; use horizontal geometry for intersections.
            if len(idx):
                candidate=shapely.force_2d(geoms[idx]); good=shapely.is_valid(candidate)
                reason[idx[~good]]='invalid_candidate_geometry'
                idx=idx[good]; candidate=candidate[good]
                if len(idx):
                    pairs=tree.query(candidate,predicate='intersects')
                    matched=np.unique(pairs[0]); reason[idx]='outside_region'; reason[idx[matched]]='regional_candidates'
                    if len(matched):
                        touch=shapely.touches(candidate[pairs[0]],districts.geometry.array[pairs[1]])
                        group_n=Counter(pairs[0].tolist()); group_touch=Counter(pairs[0][touch].tolist())
                        stats['multiple_district_records']+=sum(v>1 for v in group_n.values())
                        stats['touch_only_records']+=sum(group_touch[k]==v for k,v in group_n.items())
                        stats['district_links']+=len(pairs[0])
                        for a,j,t in zip(pairs[0],pairs[1],touch):
                            st,di=districts.iloc[int(j)][['state','district']]
                            link.writerow([f'UGLC_{kind}:{numbers[idx[a]]}',st,di,bool(t)])
                            district_counts[(st,di)]+=1
                        selected=idx[matched]
                        attrs=pd.DataFrame([[rows[i][j] for j in attr] for i in selected],columns=[header[j] for j in attr],dtype=object)
                        attrs['qa_record_id']=[f'UGLC_{kind}:{numbers[i]}' for i in selected]
                        attrs['qa_csv_record_number']=[numbers[i] for i in selected]
                        attrs['qa_source_file']=path.name
                        attrs['qa_had_z']=shapely.has_z(geoms[selected])
                        attrs['qa_district_matches']=[group_n[int(k)] for k in matched]
                        attrs['qa_touch_only']=[group_touch[int(k)]==group_n[int(k)] for k in matched]
                        frame=gpd.GeoDataFrame(attrs,geometry=candidate[matched],crs=4326)
                        pyogrio.write_dataframe(frame,target,layer='candidates',driver='GPKG',append=target.exists(),promote_to_multi=(kind=='polygons'))
                        for key,counter in profiles.items():
                            pos=norm.index(key)
                            counter.update(rows[i][pos].strip() or '<BLANK>' for i in selected)
            stats.update(reason.tolist())
            for i,r in enumerate(reason):
                if r not in {'outside_bbox','outside_region','regional_candidates'}:
                    issues.writerow([f'UGLC_{kind}:{numbers[i]}',rows[i][id_i],numbers[i],r])
        rows=[]; numbers=[]; weight=0
        for row_num,row in enumerate(reader,1):
            if not row or all(not x.strip() for x in row): stats['blank_rows']+=1; continue
            stats['source_records']+=1
            if len(row)!=len(header): raise ValueError(f'Malformed CSV record {row_num}: {len(row)} fields')
            rows.append(row); numbers.append(row_num); weight+=sum(map(len,row))
            if len(rows)>=batch_size or weight>=8*1024*1024:
                process(rows,numbers); rows=[]; numbers=[]; weight=0
            if stats['source_records']%100000==0:
                print(f"  Scanned {stats['source_records']:,} | written candidates {stats['regional_candidates']:,}",flush=True)
        process(rows,numbers)
    if stats['source_records']!=expected: raise ValueError(f'Record count mismatch: {stats["source_records"]} vs {expected}')
    if sum(stats[k] for k in REASONS)!=expected: raise ValueError('Record accounting mismatch')
    if target.exists():
        info=pyogrio.read_info(target,layer='candidates',force_feature_count=True)
        if info['features']!=stats['regional_candidates'] or info['crs']!='EPSG:4326': raise ValueError('Saved candidate count/CRS mismatch')
    elif stats['regional_candidates']: raise ValueError('Missing candidate file')
    if sum(district_counts.values())!=stats['district_links']: raise ValueError('District link accounting mismatch')
    bad=sum(stats[k] for k in REASONS if k not in {'outside_bbox','outside_region','regional_candidates'})
    stats['geometry_issues']=bad
    report=dict(kind=kind,source_file=path.name,**stats,status='PASS' if bad==0 else 'REVIEW')
    pd.DataFrame([report]).to_csv(dest/'uglc_regional_audit.csv',index=False)
    pd.DataFrame([{'ihr_state':r.state,'ihr_district':r.district,'candidate_records':district_counts[(r.state,r.district)]} for r in districts.itertuples()]).to_csv(dest/'uglc_district_counts.csv',index=False)
    pd.DataFrame([{'field':k,'value':v,'records':n} for k,c in profiles.items() for v,n in c.most_common()],columns=['field','value','records']).to_csv(dest/'uglc_attribute_counts.csv',index=False)
    pd.DataFrame([{'shapely_type_id':k,'records':v} for k,v in sorted(types.items())]).to_csv(dest/'uglc_geometry_types.csv',index=False)
    (dest/'complete.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('\nFINAL UGLC REGIONAL AUDIT:',kind)
    print(pd.Series(report).to_string())
    print('Districts with candidates:',len(district_counts),'/',len(districts))
    for key in ['OLD DATASET','ACCURACY','RELIABILITY','RECORD TYPE']:
        print('\n'+key+':'); print(pd.Series(dict(profiles.get(key,{}).most_common(20))).to_string())
    return report


def self_test():
    d=gpd.GeoDataFrame({'state':['A','A'],'district':['One','Two']},geometry=[box(70,20,71,21),box(71,20,72,21)],crs=4326)
    header=['WKT_GEOM','ID','OLD DATASET','OLD ID','ACCURACY','START DATE','END DATE','RELIABILITY','RECORD TYPE']
    with TemporaryDirectory() as tmp:
        root=Path(tmp)
        for kind,wkts in [('points',['POINT (70.5 20.5)','POINT (71 20.5)','POINT (80 20)','bad','POINT EMPTY','POINT (72.1 20.5)']),('polygons',[box(70.4,20.4,71.2,20.6).wkt,box(72,20.4,72.2,20.6).wkt,box(80,20,81,21).wkt,'POLYGON ((70.1 20.1,70.9 20.9,70.1 20.9,70.9 20.1,70.1 20.1))','POLYGON EMPTY',box(70.1,20.1,70.2,20.2).wkt])]:
            path=root/f'{kind}.csv'
            with path.open('w',newline='',encoding='utf-8') as f:
                w=csv.writer(f,delimiter='|'); w.writerow(header)
                for i,g in enumerate(wkts): w.writerow([g,str(i),'source|quoted',str(i),'10','2023-07-01','2023-07-01','high','event'])
            r=extract(path,root/kind,d,kind,len(wkts),batch_size=2)
            assert r['regional_candidates']==(2 if kind=='points' else 3)
            assert r['multiple_district_records']==1 and r['touch_only_records']==1
            assert r['geometry_issues']==2
            saved=gpd.read_file(root/kind/'candidates.gpkg')
            assert saved['OLD DATASET'].eq('source|quoted').all() and saved.qa_record_id.is_unique
            assert saved.geometry.is_valid.all() and len(saved)==r['regional_candidates']
    print('\nSynthetic CSV, geometry, boundary, append and readback tests: PASS')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind',choices=['points','polygons','all'],default='all')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test: self_test(); return
    d=load_districts(ROOT)
    raw=ROOT/'data/raw/landslides/UGLC'
    base=ROOT/'data/processed/landslides/uglc_regional_v1'; base.mkdir(parents=True,exist_ok=True)
    for kind in FILES if args.kind=='all' else [args.kind]:
        filename,expected=FILES[kind]; path=raw/kind/filename
        final=base/kind; stage=base/f'{kind}.staging'
        if final.exists():
            print(f'{kind}: completed output already exists at {final}; not overwritten. Inspect it before rerunning.',flush=True); continue
        if stage.exists(): raise RuntimeError(f'Previous incomplete output at {stage}; inspect or remove that staging directory before retrying')
        report=extract(path,stage,d,kind,expected)
        stage.rename(final)
        print(f'Output: {final} | Raw file unchanged.',flush=True)

if __name__=='__main__': main()
