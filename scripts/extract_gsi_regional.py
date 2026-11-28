"""Read-only raw GSI extraction and geometry audit. Run from project scripts/.
Outputs are candidates, not final training labels. Invalid records are logged,
not repaired or silently dropped. Full source attributes are preserved.
"""
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from zipfile import ZipFile
from collections import Counter
import argparse
import shutil
import json
import re
import numpy as np
import pandas as pd
import geopandas as gpd
import pyogrio
import shapely
from pyproj import Geod

ROOT = Path(__file__).resolve().parent.parent
GEOD = Geod(ellps='WGS84')
MISSING = {'', 'na', 'n/a', 'nan', 'none', 'null', 'nil', '-', '--', '.'}

def safe_name(name):
    p = PurePosixPath(name.replace('\\', '/'))
    if p.is_absolute() or '..' in p.parts or not p.parts or any(':' in x for x in p.parts):
        raise ValueError(f'Unsafe archive path: {name}')
    return p

def load_layer(raw, row, temp):
    outer = (raw / row['file']).resolve()
    if not outer.is_relative_to(raw.resolve()):
        raise ValueError('Archive outside raw root')
    inner, member = row['member'].split('!/', 1)
    safe_name(inner); mp = safe_name(member)
    inner_path = temp / 'inner.zip'
    with ZipFile(outer) as z, z.open(inner) as src, inner_path.open('wb') as dst:
        shutil.copyfileobj(src, dst)
    with ZipFile(inner_path) as z:
        # Extract exact GeoJSON or shapefile and matching sidecars only.
        names = [n for n in z.namelist() if not n.endswith('/')]
        if member not in names:
            raise ValueError(f'Missing inspected member: {member}')
        selected = [member]
        if mp.suffix.lower() == '.shp':
            selected = [n for n in names if safe_name(n).parent == mp.parent
                        and safe_name(n).stem.lower() == mp.stem.lower()
                        and safe_name(n).suffix.lower() in {'.shp','.shx','.dbf','.prj','.cpg','.qpj'}]
            if not {'.shp','.shx','.dbf'}.issubset({safe_name(n).suffix.lower() for n in selected}):
                raise ValueError('Incomplete shapefile')
        for name in selected:
            dest = temp / 'data' / Path(*safe_name(name).parts)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(name) as src, dest.open('wb') as dst:
                shutil.copyfileobj(src, dst)
    return pyogrio.read_dataframe(temp / 'data' / Path(*mp.parts), fid_as_index=True)

def clean_text(value):
    return '' if pd.isna(value) else str(value).strip()

def mapping_kind(value):
    s = clean_text(value).casefold()
    if s in MISSING: return 'UNKNOWN'
    if 'point' in s and 'buffer' in s: return 'BUFFERED_POINT'
    if s == 'mapped as polygon': return 'MAPPED_POLYGON'
    return 'OTHER_UNVERIFIED'

def history_kind(value):
    # Format classification only; no assertion of actual occurrence date.
    s = clean_text(value).casefold()
    if s in MISSING: return 'MISSING'
    if re.fullmatch(r'(?:19|20)\d{2}', s): return 'YEAR_TEXT'
    if re.fullmatch(r'\d{1,2}\s+[a-z]+\s+(?:19|20)\d{2}', s) or re.fullmatch(r'\d{1,2}[-/]\d{1,2}[-/](?:19|20)\d{2}', s) or re.fullmatch(r'(?:19|20)\d{2}-\d{2}-\d{2}', s): return 'DAY_FORMAT_CANDIDATE'
    if re.fullmatch(r'[a-z]+\s+(?:19|20)\d{2}', s): return 'MONTH_FORMAT_CANDIDATE'
    return 'OTHER_OR_MULTIPLE'

def audit_layer(g, row, districts):
    name = row['layer']; point = name.startswith('LANDSLIDE_POINT_')
    if len(g) != int(row['records']): raise ValueError('Record count changed since inspection')
    if g.crs is None or g.crs.to_epsg() != 4326: raise ValueError(f'Unexpected CRS: {g.crs}')
    if not g.index.is_unique: raise ValueError('Nonunique feature IDs')
    g = g.copy(); g['qa_source_fid'] = g.index.astype(str); g = g.reset_index(drop=True)
    g['qa_record_id'] = name + ':' + g['qa_source_fid']
    g['qa_source_layer'] = name
    g['qa_original_geom_type'] = g.geometry.geom_type.fillna('NULL')
    g['qa_had_z'] = shapely.has_z(g.geometry.array)
    allowed = ['Point'] if point else ['Polygon','MultiPolygon']
    reason = np.full(len(g), '', dtype=object)
    def flag(mask, label):
        reason[(reason == '') & np.asarray(mask, dtype=bool)] = label
    flag(g.geometry.isna(), 'NULL')
    flag(g.geometry.is_empty, 'EMPTY')
    flag(~g['qa_original_geom_type'].isin(allowed), 'UNEXPECTED_TYPE')
    bounds = g.geometry.bounds.to_numpy()
    flag(~np.isfinite(bounds).all(axis=1) | (bounds[:,0] < -180) | (bounds[:,2] > 180) | (bounds[:,1] < -90) | (bounds[:,3] > 90), 'INVALID_BOUNDS')
    flag(~g.geometry.is_valid, 'INVALID_GEOMETRY')
    g['qa_geometry_issue'] = reason
    bad = g.loc[reason != '', ['qa_record_id','qa_source_fid','qa_original_geom_type','qa_geometry_issue']].copy()
    bad['qa_source_layer'] = name
    if 'gid' in g: bad['gid'] = g.loc[bad.index, 'gid']
    good = g.loc[reason == ''].copy()
    good.geometry = shapely.force_2d(good.geometry.array)
    # Keep full geometries. Polygon touching-only matches are explicitly flagged.
    joined = gpd.sjoin(good[['qa_record_id','geometry']], districts, how='inner', predicate='intersects')
    if not joined.empty:
        target = districts.geometry.loc[joined['index_right'].to_numpy()].array
        joined['qa_boundary_touch_only'] = shapely.touches(joined.geometry.array, target)
    else: joined['qa_boundary_touch_only'] = pd.Series(dtype=bool)
    links = joined[['qa_record_id','ihr_state','ihr_district','qa_boundary_touch_only']].reset_index(drop=True)
    links['qa_source_layer'] = name
    # Keep touching-only records as candidates, not eligible labels by default.
    regional = good[good.qa_record_id.isin(links.qa_record_id)].copy()
    match_n = links.groupby('qa_record_id').size()
    regional['qa_district_matches'] = regional.qa_record_id.map(match_n).astype(int)
    if point:
        regional['qa_mapping_kind'] = 'POINT_RECORD'
        regional['qa_history_format'] = regional['history'].map(history_kind) if 'history' in regional else 'MISSING'
        regional['qa_coord_difference_m'] = np.nan
        if {'latitude','longitude'}.issubset(regional.columns):
            lat = pd.to_numeric(regional.latitude, errors='coerce'); lon = pd.to_numeric(regional.longitude, errors='coerce')
            ok = lat.between(-90,90) & lon.between(-180,180)
            if ok.any():
                _,_,d = GEOD.inv(lon[ok].to_numpy(), lat[ok].to_numpy(), regional.geometry.x[ok].to_numpy(), regional.geometry.y[ok].to_numpy())
                regional.loc[ok,'qa_coord_difference_m'] = d
    else:
        regional['qa_mapping_kind'] = regional['data_type'].map(mapping_kind) if 'data_type' in regional else 'UNKNOWN'
        regional['qa_history_format'] = 'NO_DEDICATED_HISTORY_FIELD'
    stats = dict(layer=name, source_records=len(g), invalid_records=len(bad), valid_records=len(good), regional_candidates=len(regional), outside_valid_records=len(good)-len(regional), multiple_district_records=int((match_n>1).sum()), touch_only_records=int((links.groupby('qa_record_id').qa_boundary_touch_only.all()).sum()), status='PASS' if not len(bad) else 'REVIEW')
    assert stats['source_records'] == stats['invalid_records'] + stats['regional_candidates'] + stats['outside_valid_records']
    types = g.groupby(['qa_original_geom_type','qa_geometry_issue'],dropna=False).size().reset_index(name='records'); types['layer']=name
    return regional, links, bad, stats, types

def main(root=ROOT):
    raw = root/'data/raw/landslides'; base = root/'data/processed/landslides'
    out = base/'gsi_regional_v1'; out.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(base/'inspection_gsi_v1/gsi_layers.csv')
    if len(manifest)!=32 or manifest.layer.nunique()!=32 or not manifest.status.eq('PASS').all() or int(manifest.records.sum())!=111962:
        raise ValueError('Expected 32 PASS layers and 111962 records; check manifest')
    districts = gpd.read_file(root/'data/processed/boundary/ihr_districts.gpkg')[['state','district','geometry']].to_crs(4326).rename(columns={'state':'ihr_state','district':'ihr_district'}).reset_index(drop=True)
    if len(districts)!=137 or districts.duplicated(['ihr_state','ihr_district']).any(): raise ValueError('Unexpected district inventory')
    districts.geometry = districts.geometry.make_valid()
    if not districts.geometry.geom_type.isin(['Polygon','MultiPolygon']).all() or districts.geometry.is_empty.any(): raise ValueError('Invalid district polygon structure')
    stats=[]; links=[]; bad=[]; types=[]; maps=[]; histories=[]; errors=[]
    for i,row in enumerate(manifest.to_dict('records'),1):
        name=row['layer']; print(f'[{i}/32] {name}',flush=True)
        if not re.fullmatch(r'[A-Za-z0-9_]+',name): raise ValueError('Unsafe layer name')
        try:
            with TemporaryDirectory(prefix='gsi_') as t:
                g=load_layer(raw,row,Path(t))
                regional, l, b, s, ty=audit_layer(g,row,districts)
                dest=out/f'{name}.gpkg'; staging=out/f'{name}.tmp.gpkg'
                staging.unlink(missing_ok=True)
                if len(regional):
                    pyogrio.write_dataframe(regional,staging,layer='candidates',driver='GPKG',promote_to_multi=not name.startswith('LANDSLIDE_POINT_'))
                    info=pyogrio.read_info(staging,layer='candidates')
                    if info['features']!=len(regional): raise ValueError('Readback feature count mismatch')
                    staging.replace(dest)
                else: dest.unlink(missing_ok=True)
                stats.append(s); links.append(l); bad.append(b); types.append(ty)
                for col, rows in [('qa_mapping_kind',maps),('qa_history_format',histories)]:
                    for k,n in regional[col].value_counts().items(): rows.append(dict(layer=name,category=k,records=n))
                print(f"  {s['status']} | regional {len(regional):,} | invalid {len(b):,} | outside {s['outside_valid_records']:,}",flush=True)
        except Exception as exc:
            errors.append(dict(layer=name,error=str(exc))); print('  ERROR:',exc,flush=True)
    def save(items, filename, columns=None):
        data=pd.concat(items,ignore_index=True) if items and isinstance(items[0],pd.DataFrame) else pd.DataFrame(items,columns=columns)
        data.to_csv(out/filename,index=False); return data
    report=save(stats,'gsi_regional_layer_audit.csv')
    link=save(links,'gsi_district_links.csv'); save(bad,'gsi_invalid_geometry_records.csv'); save(types,'gsi_actual_geometry_types.csv')
    m=save(maps,'gsi_mapping_method_counts.csv',['layer','category','records']); h=save(histories,'gsi_history_format_counts.csv',['layer','category','records']); save(errors,'gsi_errors.csv',['layer','error'])
    if not link.empty:
        counts=link.groupby(['ihr_state','ihr_district']).qa_record_id.nunique().rename('candidate_records').reset_index()
        districts[['ihr_state','ihr_district']].merge(counts,how='left').fillna({'candidate_records':0}).to_csv(out/'gsi_district_counts.csv',index=False)
    print('\nFINAL GSI REGIONAL AUDIT')
    print(f'Layers completed: {len(stats)}/32 | errors: {len(errors)}')
    if len(report):
        print(report[['source_records','invalid_records','valid_records','regional_candidates','outside_valid_records','multiple_district_records','touch_only_records']].sum().to_string())
        print(report.status.value_counts().to_string())
    for label,df in [('Mapping method',m),('History format',h)]:
        print('\n'+label+':'); print(df.groupby('category').records.sum().to_string())
    print('Output:',out,'| Raw archives unchanged.')
    if errors: return 1
    return 0

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,default=ROOT); args=parser.parse_args(); raise SystemExit(main(args.root))
