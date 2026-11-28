"""Readback and targeted review of UGLC extraction. Does not modify prior outputs.
Run alongside extract_uglc_regional.py. Repairs are review candidates, not labels.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import pyogrio
import extract_uglc_regional as e


def links_for(frame, districts):
    tree = shapely.STRtree(districts.geometry.array)
    pairs = tree.query(frame.geometry.array, predicate='intersects')
    touch = shapely.touches(frame.geometry.array[pairs[0]], districts.geometry.array[pairs[1]])
    return pd.DataFrame({'qa_record_id': frame.qa_record_id.to_numpy()[pairs[0]],
        'ihr_state': districts.state.to_numpy()[pairs[1]],
        'ihr_district': districts.district.to_numpy()[pairs[1]],
        'qa_boundary_touch_only': touch})


def check_equal(a, b, cols):
    a = a[cols].sort_values(cols).reset_index(drop=True)
    b = b[cols].sort_values(cols).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_dtype=False)


def polygon_parts(g):
    if g.geom_type == 'Polygon':
        return [g]
    if g.geom_type in ('MultiPolygon', 'GeometryCollection'):
        return [p for child in g.geoms for p in polygon_parts(child)]
    return []


def repair(g):
    fixed = shapely.make_valid(shapely.force_2d(g))
    parts = polygon_parts(fixed)
    if not parts:
        return None, fixed.geom_type
    out = shapely.union_all(parts)
    if out.is_empty or not out.is_valid or out.geom_type not in ('Polygon', 'MultiPolygon') or out.area <= 0:
        return None, fixed.geom_type
    return out, fixed.geom_type


def read_issue_rows(path, issues, kind, expected):
    wanted = {int(r.csv_record_number): r for r in issues.itertuples(index=False)}
    if len(wanted) != len(issues):
        raise ValueError('Duplicate issue row numbers')
    found = []
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f, delimiter='|', strict=True)
        header = next(reader)
        norm = {v.strip().upper(): v for v in header}
        n = 0
        for i, row in enumerate(reader, 1):
            if not row or all(not v.strip() for v in row):
                continue
            n += 1
            if len(row) != len(header):
                raise ValueError(f'Malformed record {i}')
            if i in wanted:
                attrs = dict(zip(header, row)); issue = wanted[i]
                if attrs[norm['ID']] != issue.source_id or issue.qa_record_id != f'UGLC_{kind}:{i}':
                    raise ValueError(f'Issue provenance mismatch {i}')
                found.append((issue, attrs, norm))
            if n % 250000 == 0:
                print(f'  Checked source rows {n:,}; recovered issues {len(found):,}', flush=True)
    if n != expected or len(found) != len(wanted):
        raise ValueError('Source record/issue accounting mismatch')
    return found


def review_issues(path, issues, kind, expected, districts, out):
    result, candidates = [], []
    tree = shapely.STRtree(districts.geometry.array)
    for issue, attrs, norm in read_issue_rows(path, issues, kind, expected):
        geom = shapely.from_wkt(attrs[norm['WKT_GEOM']], on_invalid='ignore')
        b = shapely.bounds(geom)
        row = dict(qa_record_id=issue.qa_record_id, source_id=issue.source_id,
                   csv_record_number=int(issue.csv_record_number), reason=issue.reason,
                   old_dataset=attrs.get(norm.get('OLD DATASET'), ''),
                   old_id=attrs.get(norm.get('OLD ID'), ''),
                   original_type=geom.geom_type if geom is not None else '',
                   west=b[0], south=b[1], east=b[2], north=b[3],
                   validity_reason=shapely.is_valid_reason(geom), repaired_type='',
                   repair_result_type='', district_matches=0, touch_only=False, disposition='UNRESOLVED')
        if issue.reason == 'invalid_bounds':
            if np.isfinite(b).all() and -180 <= b[0] <= b[2] <= 180 and -90 <= b[1] <= b[3] <= 90:
                raise ValueError('Bounds issue no longer reproducible')
            row['disposition'] = 'QUARANTINE_INVALID_BOUNDS'
        elif issue.reason == 'invalid_candidate_geometry' and kind == 'polygons':
            if geom is None or shapely.is_valid(shapely.force_2d(geom)):
                raise ValueError('Invalid polygon issue no longer reproducible')
            try:
                fixed, result_type = repair(geom)
                row['repair_result_type'] = result_type
                if fixed is None:
                    row['disposition'] = 'NO_VALID_POLYGON_REMAINS'
                else:
                    row['repaired_type'] = fixed.geom_type
                    matches = tree.query(fixed, predicate='intersects')
                    row['district_matches'] = len(matches)
                    if len(matches):
                        row['touch_only'] = bool(shapely.touches(fixed, districts.geometry.array[matches]).all())
                        row['disposition'] = 'REPAIRED_TOUCH_ONLY' if row['touch_only'] else 'REPAIRED_REGIONAL_REVIEW'
                        candidates.append({**{k:v for k,v in attrs.items() if k != norm['WKT_GEOM']},
                                           'qa_record_id':issue.qa_record_id, 'qa_csv_record_number':int(issue.csv_record_number),
                                           'qa_repair_method':'make_valid_polygon_parts_union', 'geometry':fixed})
                    else:
                        row['disposition'] = 'REPAIRED_OUTSIDE_REGION'
            except shapely.GEOSException as ex:
                row['disposition'] = 'REPAIR_FAILED'; row['repair_error'] = str(ex)
        result.append(row)
    pd.DataFrame(result).to_csv(out/'issue_review.csv', index=False)
    if candidates:
        f = gpd.GeoDataFrame(candidates, geometry='geometry', crs=4326)
        target = out/'repaired_review_candidates.gpkg'
        pyogrio.write_dataframe(f,target,layer='review_candidates',promote_to_multi=True)
        saved = gpd.read_file(target, layer='review_candidates')
        if len(saved)!=len(f) or not saved.geometry.is_valid.all() or set(saved.qa_record_id)!=set(f.qa_record_id):
            raise ValueError('Repair layer readback mismatch')
        links_for(saved,districts).to_csv(out/'repaired_district_links.csv', index=False)
    counts = Counter(r['disposition'] for r in result)
    if sum(counts.values()) != len(issues):
        raise ValueError('Review accounting mismatch')
    print('Issue dispositions:', dict(counts), flush=True)
    return dict(counts)


def audit(kind, source, base, out, districts, expected):
    out.mkdir(parents=True,exist_ok=False)
    report=json.loads((base/'complete.json').read_text(encoding='utf-8'))
    original=pd.read_csv(base/'uglc_regional_audit.csv',keep_default_na=False).iloc[0].to_dict()
    for k,v in report.items():
        if k not in original or original[k]!=v:
            raise ValueError(f'Marker/report mismatch {k}')
    if report['kind']!=kind or report['source_file']!=source.name or report['source_records']!=expected or sum(report[k] for k in e.REASONS)!=expected:
        raise ValueError('Extraction accounting mismatch')
    f=gpd.read_file(base/'candidates.gpkg',layer='candidates')
    allowed=['Point'] if kind=='points' else ['Polygon','MultiPolygon']
    bounds=f.geometry.bounds.to_numpy()
    if (len(f)!=report['regional_candidates'] or f.crs.to_epsg()!=4326 or not f.qa_record_id.is_unique
        or not f.geometry.geom_type.isin(allowed).all() or not f.geometry.is_valid.all() or f.geometry.is_empty.any()
        or not np.isfinite(bounds).all() or (bounds[:,0]<-180).any() or (bounds[:,2]>180).any()
        or (bounds[:,1]<-90).any() or (bounds[:,3]>90).any()):
        raise ValueError('Saved candidate integrity failure')
    if not f.qa_record_id.eq('UGLC_'+kind+':'+f.qa_csv_record_number.astype(str)).all() or not f.qa_source_file.eq(source.name).all():
        raise ValueError('Candidate provenance mismatch')
    links=links_for(f,districts)
    old=pd.read_csv(base/'uglc_district_links.csv',keep_default_na=False)
    check_equal(links,old,list(links.columns))
    by=links.groupby('qa_record_id').qa_boundary_touch_only.agg(['size','all'])
    if len(by)!=len(f) or len(links)!=report['district_links'] or int((by['size']>1).sum())!=report['multiple_district_records'] or int(by['all'].sum())!=report['touch_only_records']:
        raise ValueError('District link summary mismatch')
    if not np.array_equal(f.qa_district_matches.to_numpy(),f.qa_record_id.map(by['size']).to_numpy()) or not np.array_equal(f.qa_touch_only.to_numpy(),f.qa_record_id.map(by['all']).to_numpy()):
        raise ValueError('Stored per-record link metadata mismatch')
    counts=links.groupby(['ihr_state','ihr_district']).size().to_dict()
    dcounts=pd.DataFrame([{'ihr_state':r.state,'ihr_district':r.district,'candidate_records':counts.get((r.state,r.district),0)} for r in districts.itertuples()])
    check_equal(dcounts,pd.read_csv(base/'uglc_district_counts.csv'),list(dcounts.columns))
    norm={c.strip().upper():c for c in f.columns}
    old_attrs=pd.read_csv(base/'uglc_attribute_counts.csv',dtype={'field':str,'value':str},keep_default_na=False)
    attrs=[]
    for key in old_attrs.field.unique():
        c=f[norm[key]].fillna('').astype(str).str.strip().replace('', '<BLANK>').value_counts()
        attrs.extend({'field':key,'value':v,'records':int(n)} for v,n in c.items())
    check_equal(pd.DataFrame(attrs),old_attrs,['field','value','records'])
    acc=pd.to_numeric(f[norm['ACCURACY']],errors='coerce')
    tier=pd.Series(np.select([acc.eq(0),acc.gt(0)&acc.le(30),acc.gt(30)&acc.le(100),acc.gt(100)&acc.le(1000),acc.gt(1000)],
                             ['ZERO_REPORTED','GT_0_LE_30','GT_30_LE_100','GT_100_LE_1000','GT_1000'],default='UNKNOWN_OR_INVALID'),index=f.index)
    prof=pd.DataFrame({'source':f[norm['OLD DATASET']],'accuracy_tier':tier,'reliability':f[norm['RELIABILITY']],
                       'record_type':f[norm['RECORD TYPE']]})
    prof.value_counts(dropna=False).rename('records').reset_index().to_csv(out/'quality_profile.csv',index=False)
    f[[norm[k] for k in ['ID','OLD DATASET','OLD ID','START DATE','END DATE','ACCURACY','RELIABILITY','RECORD TYPE']]+['qa_record_id']].to_csv(out/'date_accuracy_candidates.csv',index=False)
    print(f'\n{kind}: readback PASS | {len(f):,} records | {len(counts)} districts',flush=True)
    print('Reported accuracy tiers:\n'+tier.value_counts().to_string(),flush=True)
    issues=pd.read_csv(base/'uglc_geometry_issues.csv',dtype=str,keep_default_na=False)
    bad={k:report[k] for k in e.REASONS if k not in ['regional_candidates','outside_bbox','outside_region'] and report[k]}
    if len(issues)!=report['geometry_issues'] or dict(issues.reason.value_counts())!=bad or set(issues.qa_record_id)&set(f.qa_record_id):
        raise ValueError('Issue inventory mismatch')
    dispositions=review_issues(source,issues,kind,expected,districts,out)
    summary={'kind':kind,'readback':'PASS','regional_candidates':len(f),'districts':len(counts),'reviewed_issues':len(issues),'dispositions':dispositions}
    (out/'complete.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary


def self_test():
    from shapely.geometry import box
    d=gpd.GeoDataFrame({'state':['A','A'],'district':['One','Two']},geometry=[box(70,20,71,21),box(71,20,72,21)],crs=4326)
    header=['WKT_GEOM','ID','OLD DATASET','OLD ID','ACCURACY','START DATE','END DATE','RELIABILITY','RECORD TYPE']
    with TemporaryDirectory() as tmp:
        root=Path(tmp)
        for kind, wkts in [('points',['POINT (70.5 20.5)','POINT (71 20.5)','POINT (200 30)']),('polygons',[box(70.4,20.4,71.2,20.6).wkt,'POLYGON ((70.1 20.1,70.9 20.9,70.1 20.9,70.9 20.1,70.1 20.1))'])]:
            path=root/f'{kind}.csv'
            with path.open('w',encoding='utf-8',newline='') as fp:
                w=csv.writer(fp,delimiter='|');w.writerow(header)
                for i,g in enumerate(wkts):w.writerow([g,str(i),'source|quoted',str(i),'0','2023-07-01','2023-07-01','1','event'])
            e.extract(path,root/kind,d,kind,len(wkts),batch_size=2)
            r=audit(kind,path,root/kind,root/(kind+'_audit'),d,len(wkts))
            assert r['reviewed_issues']==1
            assert r['dispositions']==({'QUARANTINE_INVALID_BOUNDS':1} if kind=='points' else {'REPAIRED_REGIONAL_REVIEW':1})
    print('Synthetic readback, source row recovery, invalid bounds and polygon repair: PASS')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--self-test',action='store_true'); p.add_argument('--kind',choices=['points','polygons','all'],default='all'); a=p.parse_args()
    if a.self_test: self_test();return
    root=e.ROOT; d=e.load_districts(root)
    raw=root/'data/raw/landslides/UGLC'; base=root/'data/processed/landslides/uglc_regional_v1'; out=root/'data/processed/landslides/uglc_quality_v1';out.mkdir(parents=True,exist_ok=True)
    for kind in e.FILES if a.kind=='all' else [a.kind]:
        name,n=e.FILES[kind]; final=out/kind; stage=out/(kind+'.staging')
        if final.exists() or stage.exists(): raise FileExistsError(f'Existing audit output {final} or {stage}; preserve/rename before rerunning')
        summary=audit(kind,raw/kind/name,base/kind,stage,d,n);stage.rename(final)
        print(json.dumps(summary,indent=2),flush=True)
    print('Audit reports:',out,'| Original data and extraction unchanged.')

if __name__=='__main__':main()
