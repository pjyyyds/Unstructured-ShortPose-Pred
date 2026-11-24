"""Cross-check every source Car center against the same-row metric coordinates, without targets or images."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(b):
    return hashlib.sha256(b).hexdigest()


def axis_rule(value):
    rules = {"x":(0,1), "+x":(0,1), "-x":(0,-1), "y":(1,1), "+y":(1,1), "-y":(1,-1)}
    require(isinstance(value,str) and value in rules, "invalid axis")
    return rules[value]


def geometry(c):
    lo = np.array([c["x_min"], c["y_min"]], np.float64)
    hi = np.array([c["x_max"], c["y_max"]], np.float64)
    dims = np.array([c["bev_width"], c["bev_height"]], np.float64)
    require(np.isfinite([lo,hi,dims]).all() and (hi>lo).all(), "invalid/zero metric span")
    require((dims>1).all() and (dims==np.floor(dims)).all(), "invalid BEV dimensions")
    axes = [axis_rule(c[key]) for key in ("u_axis","v_axis")]
    require(axes[0][0]!=axes[1][0],"axes must be different")
    return lo,hi,dims,axes


def declared_metric(uv, c, subtract_one=False):
    lo,hi,dims,axes = geometry(c)
    ratio = np.asarray(uv,dtype=np.float64)/(dims-int(subtract_one))
    result = np.empty_like(ratio)
    for i,(j,sgn) in enumerate(axes):
        result[:,j] = lo[j]+(ratio[:,i] if sgn>0 else 1-ratio[:,i])*(hi[j]-lo[j])
    return result


def reflect_y(xy, c):
    a = xy.copy()
    a[:,1] = (c["y_min"]+c["y_max"]) - a[:,1]
    return a


def camera_orientation(c):
    """Exactly constrained four-point solve; compute determinant by analytic quotient derivative."""
    xx = np.array(c["pts_2d"],np.float64)
    yy = np.array(c["pts_3d"],np.float64)
    require(xx.shape==yy.shape==(4,2) and np.isfinite([xx,yy]).all(),"invalid four-point references")
    src = np.column_stack([xx,np.ones(4)])
    m = np.zeros((8,8)); m[::2,:3]=src; m[1::2,3:6]=src
    m[::2,6:] = -xx*yy[:,0:1];m[1::2,6:]=-xx*yy[:,1:2]
    h = np.append(np.linalg.solve(m,yy.ravel()),1.).reshape(3,3)
    qc = h@np.append(xx.mean(0),1.)
    require(abs(qc[2])>1e-12,"singular reference projection")
    jac = (h[:2,:2]*qc[2]-qc[:2,None]*h[2,:2])/qc[2]**2
    det = float(np.linalg.det(jac))
    require(np.isfinite(det) and abs(det)>1e-12,"degenerate Jacobian")
    return det


def metric(differences):
    a=np.concatenate(differences)
    r=np.linalg.norm(a,axis=1)
    return {"n":len(a),"median_m":float(np.median(r)),"p95_m":float(np.percentile(r,95)),
            "max_m":float(np.max(r)),"mean_m":float(r.mean()),"rows_within_1um":int(np.sum(r<=1e-6))}


def read_cars(path):
    blob=path.read_bytes()
    rows=[]
    for line in blob.splitlines():
        cells=line.split()
        if not cells:continue
        require(len(cells)==12,"unexpected annotation schema")
        if cells[1]==b"Car":
            rows.append([float(cells[j]) for j in (3,4,8,9)])
    a=np.array(rows,dtype=float).reshape(-1,4)
    require(np.isfinite(a).all(),"nonfinite Car coordinates")
    return a,digest(blob)


def run(root, repo=None, bev_root=None):
    conditions={}
    site_info=defaultdict(list)
    hashes={}
    clip_data={}
    all_data=defaultdict(list)
    by_site=defaultdict(lambda:defaultdict(list))
    if repo is not None:
        sys.path.insert(0,str(repo.resolve()))
        from src.data import pinns_extractor as frozen
        require(frozen.cv2 is not None,"OpenCV required to independently compare the frozen project flip")
        frozen_src=Path(frozen.__file__).resolve()
        require(frozen_src==repo.resolve()/"src/data/pinns_extractor.py","source imported outside supplied checkout")
        project_check={"source_sha256":digest(frozen_src.read_bytes()),"opencv":frozen.cv2.__version__}
    else:
        frozen=None;project_check=None
    if bev_root is not None:
        import cv2
    for cp in sorted((root/"videos").glob("*/calib.json")):
        raw=cp.read_bytes()
        hashes[cp.relative_to(root).as_posix()]=digest(raw)
        c=json.loads(raw)
        lo,hi,dim,axes=geometry(c)
        # Continuous outer corners at (0,0), (W,0), (W,H), (0,H). Last pixel center is (W-1,H-1).
        corners=np.array([[0,0],[1,0],[1,1],[0,1.]])*dim
        corner_diff=float(np.max(np.abs(declared_metric(corners,c)-np.asarray(c["pts_3d"]))))
        det=camera_orientation(c)
        flip=det>0
        if frozen is not None:require(flip==bool(frozen._check_right_handed_homography(c)),"camera parity differs from frozen loader")
        site=c["location"]+"_"+c["scene"]
        site_info[site].append(cp.parent.name)
        conditions[cp.parent.name]=(c,site,{"dimensions_wh":dim.astype(int).tolist(),"minimum_xy_m":lo.tolist(),
            "maximum_xy_m":hi.tolist(),"meters_per_bev_pixel_xy_for_actual_x_and_minus_y_axes":((hi-lo)/dim).tolist(),
            "u_axis":c["u_axis"],"v_axis":c["v_axis"],"max_abs_outer_corner_difference_m":corner_diff,
            "image_to_reference_metric_jacobian_det_at_mean":det,"loader_flip_y":bool(flip)})
    require(len(conditions)==22,"expected the 22 released condition calibrations")
    paths=sorted((root/"labels").glob("*.txt"))
    require(len(paths)==47,"expected the 47 label files")
    used=set()
    for path in paths:
        condition,index=path.stem.rsplit("_",1)
        require(index.isdecimal(),"invalid video index")
        c,site,info=conditions[condition]
        used.add(condition)
        lo,hi,dim,axes=geometry(c)
        # Reject unseen axis conventions from this release: fitting a reflection to outcomes is disallowed.
        require(axes==[(0,1),(1,-1)],"release-specific raw-Y reflection must be revalidated for new axis declarations")
        data,hs=read_cars(path);hashes[path.relative_to(root).as_posix()]=hs
        uv,world=data[:,:2],data[:,2:]
        source_report={"site":site,"n_car_rows":len(uv),"outside_center_bounds_count":int(np.any((uv<0)|(uv>dim-1),axis=1).sum())}
        if bev_root is not None:
            vp=bev_root/"videos"/condition/"videos"/f"bev_{index}.mp4"
            if vp.is_file():
                capture=cv2.VideoCapture(str(vp))
                try:
                    require(capture.isOpened(),"failed BEV header")
                    source_report["local_bev_header_wh"]=[int(capture.get(prop)) for prop in (cv2.CAP_PROP_FRAME_WIDTH,cv2.CAP_PROP_FRAME_HEIGHT)]
                finally:capture.release()
                require(source_report["local_bev_header_wh"]==dim.astype(int).tolist(),"BEV source raster differs from calibration")
        clip_data[path.stem]=source_report
        if not len(uv):continue
        decl=declared_metric(uv,c)
        minus1=declared_metric(uv,c,True)
        raw=reflect_y(decl,c)
        direct=lo+uv/dim*(hi-lo)
        require(np.allclose(raw,direct,rtol=0,atol=1e-12),"published-axis raw reflection vs direct scale mismatch")
        to_project=lambda xy:reflect_y(xy,c) if info["loader_flip_y"] else xy
        deltas={
            "raw_width_height": raw-world,
            "raw_width_height_minus_one": reflect_y(minus1,c)-world,
            "declared_axes_direct_vs_raw": decl-world,
            "old_contour_formula_minus_one_and_y_opposite_vs_parsed_trajectory":to_project(minus1)-to_project(world),
            "raw_scaled_then_frozen_project_transform_vs_parsed_trajectory":to_project(raw)-to_project(world)}
        source_report["metrics"]={name:metric([d]) for name,d in deltas.items()}
        for name,d in deltas.items():
            all_data[name].append(d);by_site[site][name].append(d)
    require(set(conditions)==used and sum(c["n_car_rows"] for c in clip_data.values())==47555,"incomplete source population")
    return {"purpose":"Release-wide coordinate correspondence: all original Car rows, not a forecast evaluation.",
            "official_fields":"https://github.com/Songan-Lab/PINNS#annotation-format",
            "program_sha256":digest(Path(__file__).read_bytes()),"inputs_sha256":hashes,"numpy":np.__version__,
            "inputs":{"calibrations":len(conditions),"clips":len(paths),"clips_with_car_rows":sum(c["n_car_rows"]>0 for c in clip_data.values()),
                      "source_rows_car":sum(c["n_car_rows"] for c in clip_data.values()),"clips_with_local_bev_header_verified":sum("local_bev_header_wh" in c for c in clip_data.values())},
            "project_check":project_check,"conditions":{k:v[2] for k,v in conditions.items()},
            "per_site":{site:{"calibrations":len(site_info[site]),
                      "clips":sum(c["site"]==site for c in clip_data.values()),
                      "metrics":{n:metric(ds) for n,ds in methods.items()}} for site,methods in by_site.items()},
            "pooled":{n:metric(ds) for n,ds in all_data.items()},"by_clip":clip_data,
            "limits":["Rows are all original 30-Hz label rows of type Car. A tracked object contributes repeatedly; no disjoint-people claim.",
                "The reference is the same released row's world columns, not an independent physical survey. No trajectory model or future window is accessed.",
                "The legacy contour expression is evaluated at the car centers as coordinate test points. These errors are not measured errors in actual road-edge positions.",
                "Frozen camera-Jacobian flip controls a later operation applied to raw ground labels. The calibration's BEV-axis sign and that loader operation are distinct.",
                "Use native raster coordinates; unknown resize, crop, padding, rotation, new axis conventions or degenerate bounds require explicit handling."]}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("metadata",type=Path);p.add_argument("output",type=Path)
    p.add_argument("--repo",type=Path,help="cross-check all camera flip decisions with the actual loader (requires OpenCV)")
    p.add_argument("--bev-root",type=Path,help="optionally check dimensions from locally present source MP4 headers")
    a=p.parse_args();r=run(a.metadata,a.repo,a.bev_root);a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n")
    for site,rec in r["per_site"].items():
        print(site,[(k,v["n"],v["median_m"],v["p95_m"],v["max_m"]) for k,v in rec["metrics"].items() if k in
                    ("raw_width_height","raw_width_height_minus_one","old_contour_formula_minus_one_and_y_opposite_vs_parsed_trajectory")])
