"""Read-only WorldCover source audit; writes CSV reports, not raster data."""
from pathlib import Path
import re
import sys
import argparse
import tempfile

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import MaskFlags
from rasterio.features import geometry_mask
from rasterio.windows import Window, bounds as window_bounds
from rasterio.transform import from_origin
from shapely.geometry import box
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / 'data/raw/landcover'
OUT = ROOT / 'data/processed/landcover'
BOUNDARY = ROOT / 'data/processed/boundary/ihr_districts.gpkg'
CLASSES = (10,20,30,40,50,60,70,80,90,95,100)
NAMES = ('Tree cover','Shrubland','Grassland','Cropland','Built-up',
         'Bare or sparse vegetation','Snow and ice','Permanent water bodies',
         'Herbaceous wetland','Mangroves','Moss and lichen')
RES = 1/12000
LUT = np.full(256, 255, dtype=np.uint8)
LUT[0] = 0
LUT[list(CLASSES)] = CLASSES
INVALID_CODES = [x for x in range(256) if x not in (0, *CLASSES)]


def histogram(a):
    return np.bincount(a.ravel(), minlength=256).astype(np.int64, copy=False)


def scan_tile(src, districts, block=2048, progress=False):
    """Return effective class counts, raw counts and per-district counts.
    0 denotes source NoData/masked known classes; 255 denotes illegal raw codes.
    District inclusion is by native source pixel centre, not all_touched.
    """
    raw_hist = np.zeros(256, dtype=np.int64)
    hist = raw_hist.copy()
    per_district = np.zeros((len(districts), 256), dtype=np.int64)
    masked_nonzero = 0
    si = districts.sindex
    geoms = list(districts.geometry)
    extra_mask = set(src.mask_flag_enums[0]) not in ({MaskFlags.nodata}, {MaskFlags.all_valid})
    nblocks = ((src.width+block-1)//block)*((src.height+block-1)//block)
    done = 0
    for y in range(0, src.height, block):
        for x in range(0, src.width, block):
            win = Window(x,y,min(block,src.width-x),min(block,src.height-y))
            raw = src.read(1, window=win)
            raw_hist += histogram(raw)
            a = LUT[raw]
            if extra_mask:
                masked = src.read_masks(1, window=win) == 0
                masked_nonzero += int(np.count_nonzero(masked & (raw != 0)))
                a[masked & (a != 255)] = 0
            hist += histogram(a)
            candidates = si.query(box(*window_bounds(win, src.transform)), predicate='intersects')
            if len(candidates):
                t = src.window_transform(win)
                for idx in candidates:
                    inside = geometry_mask([geoms[idx]], out_shape=a.shape,
                                           transform=t, invert=True, all_touched=False)
                    if inside.any():
                        per_district[idx] += histogram(a[inside])
            done += 1
            if progress and (done % 100 == 0 or done == nblocks):
                print(f'  Blocks: {done}/{nblocks}', flush=True)
    if int(hist.sum()) != src.width*src.height or hist.sum() != raw_hist.sum():
        raise RuntimeError('Tile pixel accounting failed')
    if int(hist[255]) != int(raw_hist[INVALID_CODES].sum()):
        raise RuntimeError('Unexpected-code accounting failed')
    return hist, raw_hist, per_district, masked_nonzero


def stats(h):
    n = int(h.sum())
    valid = int(h[list(CLASSES)].sum())
    nd = int(h[0]); bad = int(h[255])
    if n != valid+nd+bad:
        raise RuntimeError('Class count accounting failed')
    return dict(pixels=n, valid_pixels=valid, nodata_pixels=nd, unexpected_pixels=bad,
                coverage_pct=100*valid/n if n else 0.0,
                nodata_pct=100*nd/n if n else 0.0)


def validate_tile(src, name):
    m = re.fullmatch(r'ESA_WorldCover_10m_2021_v200_N(\d{2})E(\d{3})_Map\.tif', name)
    if not m:
        raise RuntimeError(f'Unexpected product filename: {name}')
    lat, lon = map(int,m.groups())
    expected = from_origin(lon, lat+3, RES, RES)
    if (lat%3 or lon%3 or src.crs is None or src.crs.to_epsg()!=4326 or
        src.count!=1 or src.dtypes[0]!='uint8' or src.nodata!=0 or
        src.shape!=(36000,36000) or
        not np.allclose(tuple(src.transform),tuple(expected),rtol=0,atol=1e-10)):
        raise RuntimeError(f'Metadata/grid mismatch: {name}')
    return (lat,lon), box(*src.bounds)


def self_test():
    # Two adjacent tiles; one crosses a district boundary and contains NoData,
    # invalid raw code, and an explicitly masked otherwise-valid pixel.
    districts = gpd.GeoDataFrame(geometry=[box(1,0,5,4),box(0,0,1,4)],crs='EPSG:4326')
    total = np.zeros((2,256),dtype=np.int64)
    with tempfile.TemporaryDirectory() as td:
        for i in range(2):
            p = Path(td)/f'{i}.tif'
            a = np.full((4,3),10 if i==0 else 40,dtype=np.uint8)
            if i==0: a[0,1]=0; a[1,1]=123
            with rasterio.open(p,'w',driver='GTiff',height=4,width=3,count=1,
                               dtype='uint8',crs='EPSG:4326',nodata=0,
                               transform=from_origin(i*3,4,1,1)) as ds:
                ds.write(a,1)
                if i==1:
                    mask=np.full(a.shape,255,dtype=np.uint8); mask[0,0]=0; ds.write_mask(mask)
            with rasterio.open(p) as ds:
                h,r,d,m=scan_tile(ds,districts,block=2)
                h2,r2,d2,m2=scan_tile(ds,districts,block=4)
                assert np.array_equal(h,h2) and np.array_equal(d,d2) and m==m2
                assert h.sum()==12
                if i==0: assert h[0]==1 and h[255]==1 and r[123]==1
                else: assert h[0]==1 and m==1
                total+=d
        assert total[0].sum()==16 and total[1].sum()==4
        assert total[0,0]==2 and total[0,255]==1
        assert total[0,10]==6 and total[0,40]==7
        assert total[1,10]==4
    print('Synthetic pixel, tile-seam, mask and block-equivalence tests: PASS')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    self_test()
    if args.self_test: return 0
    districts=gpd.read_file(BOUNDARY).to_crs('EPSG:4326').reset_index(drop=True)
    if len(districts)!=137 or districts[['state','district']].duplicated().any():
        raise RuntimeError('District inventory mismatch')
    if districts.geometry.isna().any() or districts.geometry.is_empty.any() or not districts.geometry.is_valid.all():
        raise RuntimeError('Invalid district geometry; review before auditing')
    inv=pd.read_csv(OUT/'landcover_tile_inventory.csv')
    relevant=inv['relevant'].astype(str).str.lower().eq('true')
    if not relevant.any() or not inv.loc[relevant,'status'].eq('PASS').all():
        raise RuntimeError('Relevant tile metadata must pass first')
    paths=[RAW / name for name in inv.loc[relevant,'file']]
    if len(paths)!=20 or len(set(paths))!=len(paths):
        raise RuntimeError(f'Expected 20 distinct relevant tiles; found {len(paths)}')
    tile_keys=set(); footprints=[]
    for path in paths:
        with rasterio.open(path) as src:
            key,footprint=validate_tile(src,path.name)
        if key in tile_keys: raise RuntimeError(f'Duplicate tile grid: {path}')
        tile_keys.add(key); footprints.append(footprint)
    combined=unary_union(footprints)
    if not all(combined.covers(g) or g.difference(combined).area <= g.area*1e-10 for g in districts.geometry):
        raise RuntimeError('Relevant tile footprints no longer cover all districts')
    print(f'\nWORLD COVER PIXEL AUDIT | {len(paths)} tiles | {len(districts)} districts')
    tile_rows=[]; tile_class_rows=[]; all_district=np.zeros((len(districts),256),dtype=np.int64)
    for i,path in enumerate(paths,1):
        print(f'\n[{i}/{len(paths)}] {path.name}',flush=True)
        with rasterio.open(path) as src:
            h,raw,d,masked=scan_tile(src,districts,progress=True)
        all_district+=d
        s=stats(h)
        bad_codes=';'.join(f'{v}:{raw[v]}' for v in INVALID_CODES if raw[v])
        status='REVIEW' if s['unexpected_pixels'] else 'PASS'
        tile_rows.append(dict(file=str(path.relative_to(RAW)),**s,
                              masked_nonzero_pixels=masked,unexpected_codes=bad_codes,status=status))
        for code,name in zip(CLASSES,NAMES):
            tile_class_rows.append(dict(file=str(path.relative_to(RAW)),code=code,landcover=name,pixels=int(h[code])))
        print(f"  {status} | valid {s['coverage_pct']:.6f}% | NoData {s['nodata_pixels']:,} | unexpected {s['unexpected_pixels']:,}",flush=True)
        if bad_codes: print('  Unexpected raw codes:',bad_codes)
    district_rows=[]; district_class_rows=[]
    for i,row in districts.iterrows():
        h=all_district[i]; s=stats(h)
        status='PASS' if s['pixels'] and s['nodata_pixels']==0 and s['unexpected_pixels']==0 else 'REVIEW'
        district_rows.append(dict(state=row['state'],district=row['district'],**s,status=status))
        for code,name in zip(CLASSES,NAMES):
            district_class_rows.append(dict(state=row['state'],district=row['district'],code=code,landcover=name,pixels=int(h[code]),percentage=100*int(h[code])/s['pixels'] if s['pixels'] else 0))
    tiles=pd.DataFrame(tile_rows); dist=pd.DataFrame(district_rows)
    classes=pd.DataFrame(district_class_rows)
    for df,name in [(tiles,'landcover_pixel_audit.csv'),(pd.DataFrame(tile_class_rows),'landcover_tile_class_counts.csv'),(dist,'landcover_district_pixel_coverage.csv'),(classes,'landcover_district_class_counts.csv')]:
        df.to_csv(OUT/name,index=False)
    print('\nFINAL LAND COVER PIXEL AUDIT')
    print('Tiles scanned:',len(tiles),'| source pixels:',int(tiles.pixels.sum()))
    print('Tile status:\n',tiles.status.value_counts().to_string())
    print('Unexpected raw pixels:',int(tiles.unexpected_pixels.sum()))
    print('District status:\n',dist.status.value_counts().to_string())
    print('District coverage (%):\n',dist.coverage_pct.describe().to_string())
    print('\nLowest district coverage:')
    print(dist.nsmallest(5,'coverage_pct').to_string(index=False))
    print('\nDistrict native class counts:')
    print(classes.groupby(['code','landcover']).pixels.sum().to_string())
    print('\nReports:',OUT,'| Original rasters unchanged.')
    return int(not (tiles.status.eq('PASS').all() and dist.status.eq('PASS').all()))

if __name__=='__main__':
    sys.exit(main())
