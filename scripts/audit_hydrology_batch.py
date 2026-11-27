"""Read-only inventory, pixel and consistency audit of saved hydrology batch rasters."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import re
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window
from pyproj import Geod
from rasterio.warp import transform_bounds

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'data/processed/hydrology/batch_v1'
REFS = ROOT / 'data/processed/terrain_batch/terrain_v2'
BOUNDARY = ROOT / 'data/processed/boundary/ihr_districts.gpkg'
FEATURES = ('upa','log_upa','hand','hydro_slope','twi','spi','log_spi')
GRIDS = ('native','aligned')
RES = 1/1200
NODATA = -9999.0
FLOOR = np.tan(np.deg2rad(.1))
GEOD = Geod(ellps='WGS84')


def norm(s):
    return re.sub(r'[^a-z0-9]', '', str(s).lower())


def windows(w,h):
    for y in range(0,h,512):
        for x in range(0,w,512):
            yield Window(x,y,min(512,w-x),min(512,h-y))


def valid(a):
    return np.isfinite(a) & (a != NODATA)


def relation_errors(a, native_width=None):
    """Validate masks and derived values. Native slope is stored as float32 degrees."""
    v={k:valid(x) for k,x in a.items()}
    errors={}
    for original,derived in [('upa','log_upa'),('spi','log_spi')]:
        m=v[original]&v[derived]
        errors[derived+'_mask']=int(np.count_nonzero(v[original]!=v[derived]))
        errors[derived+'_value']=int(np.count_nonzero(~np.isclose(a[derived][m],np.log1p(a[original][m].astype('float64')),rtol=2e-6,atol=2e-6)))
    expected_spi=v['upa']&v['hydro_slope']
    expected_twi=expected_spi&(a['upa']>0)
    errors['spi_mask']=int(np.count_nonzero(v['spi']!=expected_spi))
    errors['twi_mask']=int(np.count_nonzero(v['twi']!=expected_twi))
    if native_width is not None:
        sca=a['upa'].astype('float64')*1e6/native_width
        slope=a['hydro_slope'].astype('float64')
        # Bound the possible original angle using adjacent representable float32 angles.
        lo=np.maximum(0,np.nextafter(a['hydro_slope'],np.float32(-np.inf)).astype('float64'))
        hi=np.minimum(89.999999999,np.nextafter(a['hydro_slope'],np.float32(np.inf)).astype('float64'))
        grad_lo=np.tan(np.deg2rad(lo)); grad_hi=np.tan(np.deg2rad(hi))
        for k in ('spi','twi'):
            m=v[k] & (expected_spi if k=='spi' else expected_twi)
            if k=='spi':
                lower=(sca*grad_lo)[m]; upper=(sca*grad_hi)[m]
            else:
                with np.errstate(divide='ignore',invalid='ignore'):
                    lower=np.log(sca/np.maximum(grad_hi,FLOOR))[m]
                    upper=np.log(sca/np.maximum(grad_lo,FLOOR))[m]
            # Allow float32 storage of UPA and the derived value as well.
            tol=2e-6+2e-6*np.maximum(np.abs(lower),np.abs(upper))
            actual=a[k][m].astype('float64')
            errors[k+'_formula']=int(np.count_nonzero((actual<lower-tol)|(actual>upper+tol)))
    return errors


def self_test():
    a={k:np.full((3,3),NODATA,dtype='float32') for k in FEATURES}
    a['upa'][1,1]=.81; a['hand'][1,1]=5; a['hydro_slope'][1,1]=45
    a['log_upa'][1,1]=np.log1p(.81); a['spi'][1,1]=9000
    a['twi'][1,1]=np.log(9000); a['log_spi'][1,1]=np.log1p(9000)
    assert not any(relation_errors(a,90).values())
    a['twi'][1,1]+=1
    assert relation_errors(a,90)['twi_formula']==1
    a['log_upa'][1,1]=NODATA
    assert relation_errors(a,90)['log_upa_mask']==1
    print('Synthetic consistency checks: PASS')


def same_grid(a,b):
    return a.crs==b.crs and a.shape==b.shape and a.transform==b.transform


def audit(refpath, geom, marker, anchor):
    state,district=refpath.parent.parent.name,refpath.parent.name
    directory=OUT/state/district
    rows=[]
    summary={'state':state,'district':district}
    errors=[]
    if marker.get('status')!='PASS' or marker.get('version')!='merit_batch_v1_floor_0.1':
        errors.append('marker status/version')
    if marker.get('state')!=state or marker.get('district')!=district:
        errors.append('marker identity')
    with rasterio.open(refpath) as ref:
        for grid in GRIDS:
            with ExitStack() as stack:
                src={k:stack.enter_context(rasterio.open(directory/grid/f'{k}.tif')) for k in FEATURES}
                base=src['upa']; tr=base.transform
                if grid=='aligned':
                    if not same_grid(base,ref) or not np.allclose(base.res,(30,30),rtol=0,atol=1e-8):
                        raise ValueError('Aligned reference grid mismatch')
                else:
                    if base.crs.to_epsg()!=4326 or not np.allclose(base.res,(RES,RES),rtol=0,atol=1e-12) or tr.b!=0 or tr.d!=0 or tr.a<=0 or tr.e>=0:
                        raise ValueError('Invalid native grid')
                    offsets=((tr.c-anchor.c)/RES,(anchor.f-tr.f)/RES)
                    if any(abs(x-round(x))>1e-5 for x in offsets):
                        raise ValueError('Native source grid misalignment')
                    west,south,east,north=transform_bounds(ref.crs,'EPSG:4326',*ref.bounds,densify_pts=21)
                    if not (base.bounds.left<=west and base.bounds.bottom<=south and base.bounds.right>=east and base.bounds.top>=north):
                        raise ValueError('Native grid does not cover reference bounds')
                for k,s in src.items():
                    if not same_grid(s,base) or s.count!=1 or s.dtypes[0]!='float32' or s.nodata!=NODATA:
                        raise ValueError(f'Metadata mismatch {grid}/{k}')
                g=gpd.GeoSeries([geom],crs='EPSG:4326').to_crs(base.crs).iloc[0]
                if not g.is_valid: g=g.buffer(0)
                counts={k:0 for k in FEATURES}; bad=counts.copy(); outside=counts.copy()
                mins={k:np.inf for k in FEATURES}; maxs={k:-np.inf for k in FEATURES}
                relation={}; total=common=terrain=joint=floor=zeros=0
                for i,w in enumerate(windows(base.width,base.height),1):
                    inside=geometry_mask([g],out_shape=(int(w.height),int(w.width)),transform=base.window_transform(w),invert=True)
                    a={k:s.read(1,window=w) for k,s in src.items()}
                    both=inside.copy(); total+=int(inside.sum())
                    for k,x in a.items():
                        m=valid(x); both &= m
                        invalid=~np.isfinite(x)
                        if k!='twi': invalid |= m&(x<0)
                        if k=='hydro_slope': invalid |= m&(x>=90)
                        bad[k]+=int(invalid.sum()); outside[k]+=int((m&~inside).sum())
                        y=x[m&inside]; counts[k]+=y.size
                        if y.size: mins[k]=min(mins[k],float(y.min())); maxs[k]=max(maxs[k],float(y.max()))
                    common+=int(both.sum())
                    width=None
                    if grid=='native':
                        lat=tr.f+(np.arange(int(w.row_off),int(w.row_off+w.height))+.5)*tr.e
                        z=np.zeros_like(lat)
                        _,_,dx=GEOD.inv(z,lat,z+RES,lat)
                        _,_,dy=GEOD.inv(z,lat-RES/2,z,lat+RES/2)
                        width=np.sqrt(dx*dy)[:,None]
                        floor+=int((inside&valid(a['twi'])&(a['hydro_slope']<.1)).sum())
                        zeros+=int((inside&valid(a['upa'])&(a['upa']==0)).sum())
                    else:
                        e=ref.read(1,window=w,masked=True)
                        ev=inside&~np.ma.getmaskarray(e)&np.isfinite(e.data)&(e.data!=NODATA)
                        terrain+=int(ev.sum()); joint+=int((both&ev).sum())
                    for k,n in relation_errors(a,width).items(): relation[k]=relation.get(k,0)+n
                    if i%1000==0: print(f'  {grid}: audited {i} blocks',flush=True)
                if not total: raise ValueError(f'No district pixels {grid}')
                if total!=marker.get(grid+'_pixels'): errors.append(grid+' pixel count')
                for k in FEATURES:
                    cov=100*counts[k]/total
                    key=('native_' if grid=='native' else '')+k+'_coverage'
                    if abs(cov-float(marker[key]))>1e-7: errors.append(key+' marker mismatch')
                    if bad[k] or (grid=='aligned' and outside[k]) or cov<95: errors.append(grid+'/'+k+' invalid/range/coverage')
                    rows.append(dict(state=state,district=district,grid=grid,feature=k,district_pixels=total,valid_pixels=counts[k],coverage=cov,invalid_pixels=bad[k],valid_outside_district=outside[k],minimum=mins[k] if counts[k] else None,maximum=maxs[k] if counts[k] else None))
                for k,n in relation.items():
                    if n: errors.append(f'{grid}/{k}: {n}')
                summary[grid+'_common_coverage']=100*common/total
                if grid=='native':
                    summary['twi_floor_percentage']=100*floor/total
                    summary['zero_upa_pixels']=zeros
                    if zeros!=marker.get('zero_upa_pixels'): errors.append('zero UPA marker mismatch')
                    # Stored slope rounding can change the floor count at exactly 0.1 degrees.
                    summary['twi_floor_pixels_marker']=marker.get('twi_floor_pixels')
                else:
                    summary['common_with_terrain_coverage']=100*joint/total
                    for key,n in [('hydro_common_coverage',common),('terrain_elevation_coverage',terrain),('common_with_terrain_coverage',joint)]:
                        if abs(100*n/total-float(marker[key]))>1e-7: errors.append(key+' marker mismatch')
                    if common/total<.95: errors.append('aligned common coverage below 95%')
    summary.update(status='PASS' if not errors else 'REVIEW',issues='; '.join(errors))
    return rows,summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--district',help='Optional STATE/District pilot')
    args=p.parse_args(); self_test()
    refs={(x.parent.parent.name,x.parent.name):x for x in REFS.glob('*/*/elevation.tif')}
    if len(refs)!=137: raise RuntimeError('Expected 137 terrain reference rasters')
    expected={OUT/s/d/g/f'{k}.tif' for s,d in refs for g in GRIDS for k in FEATURES}
    actual={x for x in OUT.glob('*/*/*/*.tif') if '_work' not in x.relative_to(OUT).parts}
    markers={OUT/s/d/'complete.json' for s,d in refs}
    actual_markers={x for x in OUT.glob('*/*/complete.json') if '_work' not in x.relative_to(OUT).parts}
    print(f'Rasters: {len(actual)}/{len(expected)} | missing {len(expected-actual)} | unexpected {len(actual-expected)}')
    print(f'Completion markers: {len(actual_markers)}/{len(markers)}')
    if expected!=actual or markers!=actual_markers: raise RuntimeError('Output inventory mismatch')
    report=pd.read_csv(OUT/'hydrology_batch_report.csv',dtype={'state':str,'district':str})
    if len(report)!=137 or set(zip(report.state,report.district))!=set(refs) or not report.status.eq('PASS').all(): raise RuntimeError('Batch report inventory/status mismatch')
    gdf=gpd.read_file(BOUNDARY).to_crs('EPSG:4326')
    geoms={(norm(r.state),norm(r.district)):r.geometry for r in gdf.itertuples()}
    if len(gdf)!=137 or len(geoms)!=137 or set(geoms)!={(norm(s),norm(d)) for s,d in refs}: raise RuntimeError('Boundary inventory mismatch')
    anchor_path=next(iter(sorted((ROOT/'data/raw/hydrology/merit_hydro/elv').rglob('*.tif'))))
    with rasterio.open(anchor_path) as a: anchor=a.transform
    chosen=sorted(refs)
    if args.district:
        k=tuple(map(norm,args.district.replace('\\','/').split('/')))
        chosen=[(s,d) for s,d in chosen if (norm(s),norm(d))==k]
        if len(chosen)!=1: p.error('Unknown district')
    rows=[]; summaries=[]
    with rasterio.Env(GDAL_CACHEMAX=128*1024**2):
        for i,(s,d) in enumerate(chosen,1):
            print(f'[{i}/{len(chosen)}] {s}/{d}',flush=True)
            try:
                marker=json.loads((OUT/s/d/'complete.json').read_text(encoding='utf-8'))
                geom=geoms[norm(s),norm(d)]
                if not geom.is_valid: geom=geom.buffer(0)
                r,m=audit(refs[s,d],geom,marker,anchor)
                rows.extend(r); summaries.append(m)
                print(f"  {m['status']} | native common {m['native_common_coverage']:.4f}% | aligned common {m['aligned_common_coverage']:.4f}% | joint {m['common_with_terrain_coverage']:.4f}%",flush=True)
                if m['issues']: print('  '+m['issues'],flush=True)
            except Exception as e:
                summaries.append(dict(state=s,district=d,status='ERROR',issues=str(e)))
                print(f'  ERROR: {e}',flush=True)
    suffix='_pilot' if args.district else ''
    df=pd.DataFrame(rows); sd=pd.DataFrame(summaries)
    df.to_csv(OUT/f'hydrology_quality_features{suffix}.csv',index=False)
    sd.to_csv(OUT/f'hydrology_quality_districts{suffix}.csv',index=False)
    print('\nFINAL HYDROLOGY AUDIT')
    print(f'Districts: {len(sd)} | rasters read: {len(df)} / {14*len(chosen)}')
    print(sd.status.value_counts().to_string())
    if not df.empty: print(df.groupby(['grid','feature']).coverage.agg(['min','mean','max']).to_string())
    cols=['native_common_coverage','aligned_common_coverage','common_with_terrain_coverage','twi_floor_percentage']
    cols=[x for x in cols if x in sd]
    if cols: print('\nDistrict-level statistics:\n'+sd[cols].agg(['min','mean','max']).T.to_string())
    print(f'Reports: {OUT} / hydrology_quality_*{suffix}.csv')
    return int(not sd.status.eq('PASS').all() or len(df)!=14*len(chosen))


if __name__=='__main__': sys.exit(main())
