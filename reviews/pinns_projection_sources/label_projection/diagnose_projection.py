"""Coordinate-label consistency only. No prediction, video, or model is loaded."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


CLIPS = ["america_crossroad_summer_daytime_sunny_0", "america_crossroad_summer_nightfall_sunny_0",
         "america_crossroad_summer_nightfall_sunny_1"]
GEOM = ["pts_2d", "pts_3d", "x_min", "x_max", "y_min", "y_max", "u_size", "v_size",
        "bev_width", "bev_height", "u_axis", "v_axis", "fps"]


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def project(x, h):
    q = np.column_stack([x, np.ones(len(x))]) @ h.T
    assert np.isfinite(q).all() and np.all(np.abs(q[:, 2]) > 1e-12)
    return q[:, :2] / q[:, 2:]


def norm_transform(x):
    center = x.mean(0)
    scale = np.sqrt(2) / np.mean(np.linalg.norm(x-center, axis=1))
    return np.array([[scale, 0, -scale*center[0]], [0, scale, -scale*center[1]], [0, 0, 1.]])


def fit_h(x, y):
    tx, ty = norm_transform(x), norm_transform(y)
    xx, yy = project(x, tx), project(y, ty)
    m = np.zeros((2*len(x), 9))
    q = np.column_stack((xx, np.ones(len(x))))
    m[::2, :3] = q
    m[1::2, 3:6] = q
    m[::2, 6:] = -q * yy[:, 0:1]
    m[1::2, 6:] = -q * yy[:, 1:2]
    _, s, v = np.linalg.svd(m, full_matrices=len(x) == 4)
    assert s[7] > 1e-10 * s[0]
    h = np.linalg.solve(ty, v[-1].reshape(3, 3) @ tx)
    return h / h[2, 2]


def score(d):
    d = np.asarray(d)
    lengths = np.linalg.norm(d, axis=1)
    return {"n": len(d), "median_m": float(np.median(lengths)), "p95_m": float(np.percentile(lengths, 95)),
            "mean_m": float(lengths.mean()), "max_m": float(lengths.max()),
            "fraction_le_1cm": float(np.mean(lengths <= .01)), "fraction_le_10cm": float(np.mean(lengths <= .1)),
            "median_signed_xy_m": np.median(d, axis=0).tolist()}


def estimate(check, train, original):
    x, y = train[:, 2:4], train[:, 6:8]
    nominal = project(x, original)
    to_check = project(check[:, 2:4], original)
    xy1 = lambda z: np.column_stack((z, np.ones(len(z))))
    A = np.linalg.lstsq(xy1(nominal), y, rcond=None)[0]
    reestimated = fit_h(x, y)
    return {"original": to_check,
            "train_translation": to_check + (y - nominal).mean(0),
            "train_affine_on_original": xy1(to_check) @ A,
            "train_fixed_homography": project(check[:, 2:4], reestimated)}


def run(root):
    extracted, measured, input_sha = {}, {}, {}
    models = {}
    calib_geo = {}
    for cfile in sorted((root/"videos").glob("america_crossroad_*/calib.json")):
        c = json.loads(cfile.read_bytes())
        calib_geo[cfile.parent.name] = digest(json.dumps({k:c[k] for k in GEOM},sort_keys=True).encode())
    for clip in CLIPS:
        label = root/"labels"/(clip+".txt")
        config = root/"videos"/clip.rsplit("_",1)[0]/"calib.json"
        raw, cr = label.read_bytes(), config.read_bytes()
        input_sha[label.relative_to(root).as_posix()] = digest(raw)
        input_sha[config.relative_to(root).as_posix()] = digest(cr)
        c = json.loads(cr)
        assert c["fps"] == 30 and (c["u_axis"], c["v_axis"]) == ("x","-y")
        vals = defaultdict(list)
        for line in raw.splitlines():
            tick, typ, track, ox, oy, _, w, h, gx, gy, *_ = line.split()
            vals[typ].append([int(tick),int(track),float(ox),float(oy),float(w),float(h),float(gx),float(gy)])
        people, cars = np.array(vals[b"Person"]), np.array(vals[b"Car"])
        assert np.isfinite(people).all() and people[:,0].min() == 0 and len(set(map(tuple,people[:,:2])))==len(people)
        extracted[clip] = (people,cars,c); measured[clip] = people[people[:,0] % 3 == 1]
        pts = np.asarray(c["pts_3d"])
        pts[:,1] = c["y_min"] + c["y_max"] - pts[:,1]
        original = fit_h(np.asarray(c["pts_2d"]),pts)
        assert np.max(np.abs(project(np.asarray(c["pts_2d"]),original)-pts))<1e-7
        models[clip]=original
    assert len(set(calib_geo.values())) == 1
    pooled, by_clip = defaultdict(list), {}
    for clip in CLIPS:
        people, cars, c = extracted[clip]; check=measured[clip]
        train=np.concatenate([measured[k] for k in CLIPS if k!=clip])
        original=models[clip]
        local_results={}
        def add(name, differences):
            pooled[name].append(differences); local_results[name]=score(differences)
        projected = estimate(check,train,original)
        for name, xy in projected.items():
            add("leave_clip_out/"+name, xy-check[:,6:8])
        split = check[:,1].astype(int)%2
        held_ids, within = [], np.empty_like(check[:,6:8])
        for j in (0,1):
            tr, te = check[split!=j],check[split==j]
            assert not (set(tr[:,1]) & set(te[:,1]))
            held_ids.append({"held_persons":len(set(te[:,1])),"held_rows":len(te),
                             "train_persons":len(set(tr[:,1])), "train_rows":len(tr)})
            within[split==j]=estimate(te,tr,original)["train_fixed_homography"]
        add("same_clip_disjoint_person_fixed_homography",within-check[:,6:8])
        for phase in (0,1,2):
            group=people[people[:,0]%3==phase]
            add(f"nominal_source_phase_{phase}",project(group[:,2:4],original)-group[:,6:8])
        lookup={(int(r[0]),int(r[1])):r[6:8] for r in people}
        keep=np.array([all((int(r[0])+off,int(r[1])) in lookup for off in (-1,1)) for r in check])
        xy=project(check[keep,2:4],original)
        for off in (-1,0,1):
            truth=np.stack([lookup[int(r[0])+off,int(r[1])] for r in check[keep]])
            add(f"common_person_time_offset_{off:+d}",xy-truth)
        cars=cars[cars[:,0]%3==1]
        world_range=np.array([c["x_max"]-c["x_min"],c["y_max"]-c["y_min"]])
        dims=np.array([c["bev_width"],c["bev_height"]])
        assert (c["x_min"],c["y_min"])==(0,0)
        for m in (0,1):
            add("car_bev_wh"+("_minus_one" if m else ""), cars[:,2:4]/(dims-m)*world_range-cars[:,6:8])
        # Check image/BEV pixel-centre rounding effects of the two relevant published raster scales.
        nominal=project(check[:,2:4],original)
        add("original_after_rounding_to_nearest_BEV_pixel",np.rint(nominal*dims/world_range)*world_range/dims-check[:,6:8])
        add("original_after_rounding_to_nearest_camera_pixel",project(np.rint(check[:,2:4]),original)-check[:,6:8])
        by_clip[clip]={"all_person_rows":len(people),"sampled_persons":len(set(check[:,1])),
                       "fit_other_clip_names":[k for k in CLIPS if k!=clip], "fit_other_clip_rows":len(train),
                       "distinct_other_clip_track_instances":sum(len(set(measured[k][:,1])) for k in CLIPS if k!=clip),
                       "within_clip_alternating_track_folds":held_ids,"scores":local_results}
    return {"purpose":"Same-label coordinate correspondence diagnostic. Only same-row world annotations were used; no forecasting windows, model outputs, video or evaluation archive were read. Ground columns used for diagnostic mapping only; mappings are not deployment recommendations.",
            "official_source":"https://github.com/Songan-Lab/PINNS#annotation-format",
            "input_sha256":input_sha,"numpy":str(np.__version__),"program_sha256":digest(Path(__file__).read_bytes()),
            "seven_US_calibration_geometry_hashes_including_axes_raster_and_reference_points":calib_geo,
            "notes":["All project-grid person rows at raw frame modulo 3 equal 1 receive equal weight in the pooled primary summaries.",
                     "DLT uses centered isotropically normalized coordinates, unweighted linear SVD; no filtering, tunable parameter, robust search or test-based correction.",
                     "For phase matching, current stored person point is fixed; reference ground fields at t-1, t or t+1 come from the same person in the same clip, restricted to a common complete intersection.",
                     "BEV car centre versus world check uses positive raw ground axes resulting from the reflected-y label convention; this tests BEV-to-ground separately from person camera-to-BEV.",
                     "Independent people are distinct within a video; repeated IDs across recordings are treated as recording-local and do not certify physical identity across recordings."],
            "per_clip":by_clip,"pooled":{k:score(np.concatenate(v,axis=0)) for k,v in pooled.items()}}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("metadata",type=Path);p.add_argument("out",type=Path)
    p.add_argument("--reference",type=Path,help="optional original aggregate JSON, opened after independent calculations")
    a=p.parse_args(); report=run(a.metadata)
    if a.reference:
        ref=json.loads(a.reference.read_bytes())
        differences={}
        for name in [*CLIPS,"pooled"]:
            original = (ref["clips"][name]["project_10hz"] if name != "pooled"
                        else ref["project_10hz"]["by_min_annotated_frame_category"]["zero"])["distances"]["stored_xy"]
            calc = (report["per_clip"][name]["scores"] if name != "pooled" else report["pooled"])["leave_clip_out/original"]
            assert calc["n"]==original["n"]
            differences[name] = max(abs(calc[key]-original[key]) for key in ("median_m","p95_m","mean_m"))
        assert max(differences.values())<1e-10
        report["registered_four_point_crosscheck"]={"source_sha256":digest(a.reference.read_bytes()),
            "equal_row_counts":True, "maximum_absolute_mean_median_p95_differences_m":differences}
    a.out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    for k,v in report["pooled"].items(): print(k,v["n"],"median",round(v["median_m"],7),"p95",round(v["p95_m"],7))
