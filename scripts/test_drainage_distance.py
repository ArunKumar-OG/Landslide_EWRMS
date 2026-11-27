"""Haridwar pilot: capped Euclidean distance to MERIT-derived stream cell centres.
Uses original UPA cells, projected to district UTM, not clipped district UPA.
Keeps all existing outputs untouched. Requires test_hydrology_features.py.
"""
from pathlib import Path
import math
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from affine import Affine
from pyproj import Transformer
from rasterio.features import geometry_mask
from rasterio.warp import transform_bounds
from scipy.spatial import cKDTree
from test_hydrology_features import read_mosaic

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw/hydrology/merit_hydro/upa"
REF = ROOT / "data/processed/terrain_batch/terrain_v2/UK/Haridwar/elevation.tif"
BOUNDARY = ROOT / "data/processed/boundary/ihr_districts.gpkg"
OUT = ROOT / "data/processed/hydrology/drainage_pilot/UK/Haridwar"
THRESHOLDS = (1., 5., 10.)
CAP = 10_000.
BUFFER = 12_000.
UNKNOWN_MARGIN = 150.
NODATA = -9999.
RES = 1/1200
BLOCK = 512

def distances(tree, points):
    if tree is None:
        return np.full(len(points), np.inf)
    return tree.query(points, k=1, distance_upper_bound=CAP+UNKNOWN_MARGIN)[0]

def capped_distance(tree, unknown, points):
    d = np.minimum(distances(tree, points), CAP)
    u = distances(unknown, points)
    # Conservatively reject results if a missing source cell might hide
    # a closer channel. The buffer also keeps the mosaic edge beyond CAP.
    d[u <= d + UNKNOWN_MARGIN] = np.nan
    return d

def self_test():
    pts = np.array([[0., 0.], [90., 0.], [20000., 0.]])
    tree = cKDTree([[0., 0.]])
    np.testing.assert_allclose(capped_distance(tree, None, pts), [0.,90.,CAP])
    np.testing.assert_allclose(capped_distance(None, None, pts), CAP)
    u = cKDTree([[100., 0.]])
    assert np.isnan(capped_distance(tree, u, pts)[1])
    assert capped_distance(tree, None, pts)[0] == 0.
    print("Synthetic drainage distance tests: PASS", flush=True)

def tree_for(mask, tr, project):
    r,c = np.nonzero(mask)
    if not len(r):
        return None
    lon = tr.c + (c+0.5)*tr.a
    lat = tr.f + (r+0.5)*tr.e
    x,y = project.transform(lon,lat)
    coords = np.column_stack((x,y))
    if not np.isfinite(coords).all():
        raise ValueError("Nonfinite projected source centres")
    return cKDTree(coords)

def main():
    self_test()
    paths = sorted(RAW.rglob("*.tif"))
    if len(paths) != 12:
        raise RuntimeError(f"Expected 12 UPA tiles, got {len(paths)}")
    with rasterio.open(paths[0]) as src:
        anchor = src.transform
    OUT.mkdir(parents=True,exist_ok=True)
    with rasterio.open(REF) as ref:
        if ref.crs is None or not ref.crs.is_projected or ref.res != (30.,30.):
            raise ValueError("Expected projected 30 m terrain grid")
        if ref.transform.b != 0 or ref.transform.d != 0:
            raise ValueError("Expected north-up grid")
        if ref.crs.linear_units.lower() not in ("metre","meter","m"):
            raise ValueError("Expected metric projected coordinates")
        b = ref.bounds
        west,south,east,north = transform_bounds(
            ref.crs, "EPSG:4326", b.left-BUFFER,b.bottom-BUFFER,
            b.right+BUFFER,b.top+BUFFER,densify_pts=101)
        c0 = math.floor((west-anchor.c)/RES)-2
        c1 = math.ceil((east-anchor.c)/RES)+2
        r0 = math.floor((anchor.f-north)/RES)-2
        r1 = math.ceil((anchor.f-south)/RES)+2
        tr = anchor * Affine.translation(c0,r0)
        w,h = c1-c0,r1-r0
        if w*h > 10_000_000:
            raise RuntimeError("Pilot memory limit exceeded")
        print(f"Buffered native grid: {w} x {h}",flush=True)
        upa = read_mosaic(paths,tr,w,h)
        valid = np.isfinite(upa) & (upa >= 0)
        project = Transformer.from_crs("EPSG:4326",ref.crs,always_xy=True)
        unknown = tree_for(~valid,tr,project)
        print(f"Missing buffered UPA cells: {(~valid).sum():,}",flush=True)
        gdf = gpd.read_file(BOUNDARY).to_crs(ref.crs)
        sel = gdf[(gdf.state=="UK") & (gdf.district=="Haridwar")]
        if len(sel)!=1: raise RuntimeError("Haridwar boundary match failed")
        geom = sel.geometry.iloc[0]
        if not geom.is_valid: geom = geom.buffer(0)
        if geom.is_empty: raise RuntimeError("Empty district geometry")
        inside = geometry_mask([geom],out_shape=ref.shape,
                               transform=ref.transform,invert=True)
        total = int(inside.sum())
        if total==0: raise RuntimeError("No district pixels")
        profile = ref.profile.copy()
        profile.update(dtype="float32",count=1,nodata=NODATA,
                       compress="deflate",predictor=3,tiled=True,
                       blockxsize=256,blockysize=256,BIGTIFF="IF_SAFER")
        previous = None
        report = []
        for threshold in THRESHOLDS:
            stream = valid & (upa>=threshold)
            tree = tree_for(stream,tr,project)
            print(f"\nThreshold {threshold:g} km2 | buffered stream cells: {stream.sum():,}",flush=True)
            out = np.full(ref.shape,NODATA,dtype=np.float32)
            for r in range(0,ref.height,BLOCK):
                for c in range(0,ref.width,BLOCK):
                    rr,cc = np.nonzero(inside[r:r+BLOCK,c:c+BLOCK])
                    if not len(rr): continue
                    rr,cc = rr+r,cc+c
                    x = ref.transform.c+(cc+0.5)*ref.transform.a
                    y = ref.transform.f+(rr+0.5)*ref.transform.e
                    d = capped_distance(tree,unknown,np.column_stack((x,y)))
                    good = np.isfinite(d)
                    out[rr[good],cc[good]] = d[good].astype(np.float32)
            ok = inside & (out!=NODATA)
            vals = out[ok]
            if not len(vals): raise RuntimeError("No valid distance cells")
            if previous is not None:
                common = ok & (previous!=NODATA)
                if np.any(out[common]+0.001 < previous[common]):
                    raise RuntimeError("Threshold monotonicity failed")
            path = OUT / f"distance_upa_{threshold:g}km2.tif"
            with rasterio.open(path,"w",**profile) as dst:
                dst.write(out,1)
                dst.set_band_description(1,"Distance to modeled drainage cell centre (m)")
                dst.update_tags(source="MERIT Hydro v1.0.1 UPA",threshold_km2=str(threshold),
                                cap_m=str(CAP),buffer_m=str(BUFFER),units="m",
                                native_resolution="3 arc-seconds",method="UTM cell-centre KDTree")
            row = dict(threshold_km2=threshold,stream_cells_buffer=int(stream.sum()),
                       district_pixels=total,valid_pixels=int(ok.sum()),
                       coverage_percent=100*ok.sum()/total,
                       capped_percent=100*np.count_nonzero(vals>=CAP)/len(vals),
                       min_m=float(vals.min()),median_m=float(np.median(vals)),
                       p95_m=float(np.percentile(vals,95)),max_m=float(vals.max()))
            report.append(row)
            print(f"Coverage: {row['coverage_percent']:.4f}% | median: {row['median_m']:.2f} m | "
                  f"P95: {row['p95_m']:.2f} m | max: {row['max_m']:.2f} m | "
                  f"capped: {row['capped_percent']:.4f}%",flush=True)
            previous = out
        pd.DataFrame(report).to_csv(OUT/"drainage_sensitivity.csv",index=False)
        print("\nThreshold monotonicity: PASS (common valid cells)")
        print("Output:",OUT)
        print("Pilot complete; review coverage and distributions before batch.")
if __name__=="__main__":
    main()
