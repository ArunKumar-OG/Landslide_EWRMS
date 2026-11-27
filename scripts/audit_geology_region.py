"""Read-only source audit; writes a repaired derived GPKG and CSV diagnostics."""
from pathlib import Path
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
import geopandas as gpd
import pyogrio
from shapely import make_valid, union_all
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, box
from shapely.validation import explain_validity

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'data/processed/geology'
SOURCE = OUT / 'glim_regional_candidates.gpkg'
BOUNDARY = ROOT / 'data/processed/boundary/ihr_districts.gpkg'
CODES = ['su','ss','py','sm','sc','ev','va','vi','vb','pa','pi','pb','mt','wb','ig','nd']
CLASS_IDS = {code: i+1 for i,code in enumerate(CODES)}
MIN_COVERAGE = 95.0
MAX_CONFLICT = 0.1 # percent of district area, diagnostic only


def polygon_parts(g):
    if g is None or g.is_empty:
        return []
    if g.geom_type == 'Polygon':
        return [g]
    if g.geom_type in ('MultiPolygon', 'GeometryCollection'):
        return [p for child in g.geoms for p in polygon_parts(child)]
    return []


def repair(g):
    if g is None or g.is_empty:
        raise ValueError('Null or empty source geometry')
    if g.is_valid and g.geom_type in ('Polygon','MultiPolygon'):
        return g
    parts = polygon_parts(make_valid(g))
    if not parts:
        raise ValueError('No polygonal area remains after make_valid')
    result = union_all(parts)
    if result.is_empty or not result.is_valid or result.geom_type not in ('Polygon','MultiPolygon'):
        raise ValueError('Repair did not produce valid polygonal geometry')
    return result


def coverage(geom, candidates):
    """Area union by class; conflict = footprint covered by >=2 distinct known classes."""
    pieces = defaultdict(list)
    individual_area = defaultdict(float)
    for code, poly in candidates:
        clipped = poly.intersection(geom)
        if not clipped.is_empty and clipped.area > 0:
            pieces[code].append(clipped)
            individual_area[code] += clipped.area
    merged = {c: union_all(ps) for c,ps in pieces.items()}
    known = [merged[c] for c in CODES if c != 'nd' and c in merged]
    seen = GeometryCollection()
    conflicts = []
    for p in known:
        overlap = seen.intersection(p)
        if overlap.area > 0:
            conflicts.append(overlap)
        seen = seen.union(p)
    conflict = union_all(conflicts)
    nd = merged.get('nd', GeometryCollection())
    mapped = seen.union(nd)
    total = geom.area
    if total <= 0:
        raise ValueError('District has zero polygonal area')
    pct = lambda a: max(0., min(100., 100.*a/total))
    result = dict(district_area_km2=total/1e6,
                  known_coverage_pct=pct(seen.area),
                  mapped_coverage_pct=pct(mapped.area),
                  unmapped_gap_pct=pct(total-mapped.area),
                  nd_only_pct=pct(nd.difference(seen).area),
                  nd_known_overlap_pct=pct(nd.intersection(seen).area),
                  known_class_conflict_pct=pct(conflict.area),
                  same_class_overlap_excess_km2=sum(max(0.,individual_area[c]-merged[c].area) for c in merged)/1e6)
    result['status'] = ('PASS' if result['known_coverage_pct'] >= MIN_COVERAGE
                        and result['known_class_conflict_pct'] <= MAX_CONFLICT
                        and result['nd_known_overlap_pct'] <= MAX_CONFLICT else 'REVIEW')
    class_areas = {c: merged[c].area/1e6 if c in merged else 0. for c in CODES}
    return result, class_areas


def smoke_tests():
    bow = Polygon([(0,0),(2,2),(0,2),(2,0),(0,0)])
    assert not bow.is_valid and repair(bow).is_valid and abs(repair(bow).area-2)<1e-9
    d = box(0,0,10,10)
    result, classes = coverage(d,[('su',box(0,0,6,10)),('su',box(0,0,2,10)),
                                  ('mt',box(5,0,8,10)),('nd',box(7,0,9,10))])
    for key, value in [('known_coverage_pct',80),('mapped_coverage_pct',90),('unmapped_gap_pct',10),
                       ('nd_only_pct',10),('nd_known_overlap_pct',10),('known_class_conflict_pct',10)]:
        assert abs(result[key]-value)<1e-9,(key,result[key])
    assert abs(classes['su']-60/1e6)<1e-12
    assert abs(result['same_class_overlap_excess_km2']-20/1e6)<1e-12
    assert coverage(d,[('mt',d)])[0]['status']=='PASS'
    assert coverage(d,[])[0]['status']=='REVIEW'
    print('Synthetic geometry and coverage tests: PASS',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    smoke_tests()
    if args.self_test: return 0
    geo=gpd.read_file(SOURCE,layer='geology',engine='pyogrio')
    districts=gpd.read_file(BOUNDARY,engine='pyogrio').to_crs(geo.crs)
    if len(districts)!=137 or districts[['state','district']].duplicated().any():
        raise RuntimeError('Unexpected district inventory')
    if geo.crs is None or geo.crs.to_authority()!=('ESRI','54012'):
        raise RuntimeError(f'Expected original equal-area ESRI:54012 CRS; found {geo.crs}')
    if len(geo)!=10815 or geo.source_fid.isna().any() or geo.source_fid.duplicated().any():
        raise RuntimeError('Unexpected regional polygon inventory or nonunique source_fid')
    codes=geo.xx_normalized.astype('string')
    if not codes.isin(CODES).all() or not (geo.litho_normalized.astype('string').str[:2]==codes).fillna(False).all():
        raise RuntimeError('Lithology classification mismatch')
    geo['class_id']=codes.map(CLASS_IDS).astype('uint8')
    print(f'Regional polygons: {len(geo)} | districts: {len(districts)}')
    fixes=[]; repaired=[]; failed=0
    duplicate=geo.IDENTITY_.duplicated(keep=False)
    geo.loc[duplicate,['source_fid','IDENTITY_','xx_normalized','litho_normalized']].to_csv(OUT/'glim_duplicate_identifiers.csv',index=False)
    for row in geo.itertuples(index=False):
        g=row.geometry
        valid=g is not None and not g.is_empty and g.is_valid
        try:
            new=repair(g)
            if not valid:
                fixes.append(dict(source_fid=row.source_fid,source_id=row.IDENTITY_,code=row.xx_normalized,
                                  reason=explain_validity(g),before_area_km2=g.area/1e6 if g is not None else np.nan,
                                  after_area_km2=new.area/1e6,status='REPAIRED'))
            repaired.append(new)
        except Exception as exc:
            failed+=1; repaired.append(None)
            fixes.append(dict(source_fid=row.source_fid,source_id=row.IDENTITY_,code=row.xx_normalized,
                              reason=str(exc),before_area_km2=np.nan,after_area_km2=np.nan,status='FAIL'))
    pd.DataFrame(fixes,columns=['source_fid','source_id','code','reason','before_area_km2','after_area_km2','status']).to_csv(OUT/'glim_geometry_repair_audit.csv',index=False)
    print(f'Repaired: {len(fixes)-failed} | Unrepairable: {failed} | Rows sharing a source identifier: {int(duplicate.sum())}')
    if failed:
        raise RuntimeError('Unrepairable geometries found; inspect repair CSV before proceeding')
    geo=geo.set_geometry(gpd.GeoSeries(repaired,index=geo.index,crs=geo.crs))
    geo['geometry_repaired']=geo.source_fid.isin([f['source_fid'] for f in fixes])
    if not geo.geometry.is_valid.all(): raise RuntimeError('Invalid output geometry')
    repaired_path=OUT/'glim_regional_repaired.gpkg'
    # Separate derived file, never replace original or candidate data.
    pyogrio.write_dataframe(geo,repaired_path,layer='geology',driver='GPKG',promote_to_multi=True)
    pd.DataFrame({'class_id':list(CLASS_IDS.values()),'code':list(CLASS_IDS)}).to_csv(OUT/'glim_class_lookup.csv',index=False)
    idx=geo.sindex
    rows=[]; area_rows=[]
    for i,row in enumerate(districts.itertuples(index=False),1):
        d=repair(row.geometry)
        hits=idx.query(d,predicate='intersects')
        selected=geo.iloc[hits]
        result,areas=coverage(d,zip(selected.xx_normalized,selected.geometry))
        key={'state_code':row.state,'district':row.district}
        rows.append({**key,'candidate_polygons':len(hits),**result})
        area_rows.extend({**key,'code':c,'class_id':CLASS_IDS[c],'area_km2':a} for c,a in areas.items() if a>0)
        print(f'[{i}/137] {row.state}/{row.district} | known {result["known_coverage_pct"]:.4f}% | conflict {result["known_class_conflict_pct"]:.4f}% | {result["status"]}',flush=True)
    report=pd.DataFrame(rows)
    report.to_csv(OUT/'glim_district_coverage_audit.csv',index=False)
    pd.DataFrame(area_rows).to_csv(OUT/'glim_district_class_area.csv',index=False)
    print('\nFINAL GEOLOGY AUDIT')
    print('Repaired dataset:',repaired_path)
    print('Districts:',len(report))
    print(report.status.value_counts().to_string())
    cols=['known_coverage_pct','mapped_coverage_pct','unmapped_gap_pct','nd_only_pct','nd_known_overlap_pct','known_class_conflict_pct']
    print(report[cols].agg(['min','mean','max']).T.to_string())
    print('\nLowest known coverage:')
    print(report.nsmallest(5,'known_coverage_pct')[['state_code','district']+cols+['status']].to_string(index=False))
    print('\nLargest class conflicts:')
    print(report.nlargest(5,'known_class_conflict_pct')[['state_code','district']+cols+['status']].to_string(index=False))
    print('Area-based audit only. No source files modified; overlap resolution and rasterization pending.')
    return 0 if (report.status=='PASS').all() else 2

if __name__=='__main__': raise SystemExit(main())
