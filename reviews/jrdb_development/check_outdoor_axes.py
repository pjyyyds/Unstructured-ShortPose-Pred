"""Check eight horizontal axis conventions; optionally compare a legitimately obtained raw JRDB training JSON.

python3 reviews/jrdb_development/check_outdoor_axes.py --output axes.json
Add --raw-labels /path/to/labels_3d/<training-scene>.json (repeatable) to compare actual raw centres in metres.
"""
import argparse
import hashlib
from itertools import product
import json
from pathlib import Path

import numpy as np

from check_outdoor_projection import WIDTH, delta_pixel
from check_training_coordinates import TrainingArchive
from evaluate_outdoor_motion import ARCHIVES, DIGESTS, MEMBERS, SCENES, read_poses


def quantiles(values):
    return dict(zip(("p50","p95","max"), np.quantile(values,[.5,.95,1.]).tolist()))


def variants(xy):
    for swapped in (False, True):
        ax = ("y","x") if swapped else ("x","y")
        for sx,sy in product((1,-1),repeat=2):
            label = ",".join(("-" if sign<0 else "")+a for sign,a in zip((sx,sy),ax))
            yield label,(xy[:,::-1] if swapped else xy)*[sx,sy]


def check(tracks_blob, pose_blob, raw_path=None):
    records = {}
    for line in tracks_blob.splitlines():
        t = json.loads(line).get("track")
        if t is None:
            continue
        key=(int(t["f"]),int(t["p"]))
        if key in records:
            raise ValueError("Duplicate ST person/frame")
        if np.isfinite([t[k] for k in ("x","y","bb_left","bb_width","bb_height")]).all() and (
                0 < t["bb_width"] < WIDTH and t["bb_height"] > 0 and (t["x"] or t["y"])):
            records[key] = t
    keys=sorted(records)
    xy=np.array([[records[k]["x"],records[k]["y"]] for k in keys])
    centres=np.array([(records[k]["bb_left"]+records[k]["bb_width"]/2) % WIDTH for k in keys])
    poses=read_poses(pose_blob)
    def bearing(points):
        return ((np.arctan2(-points[:,1],points[:,0])+np.pi) % (2*np.pi))*WIDTH/(2*np.pi)
    errors={name:quantiles(delta_pixel(bearing(test_xy),centres)) for name,test_xy in variants(xy)}
    correct=bearing(xy)
    circular=delta_pixel
    frames=np.array([f for f,_ in keys]); matched=[]; permuted=[]
    for t in np.unique(frames):
        sel=frames==t
        if sel.sum()>1:
            matched.extend(circular(correct[sel],centres[sel]))
            permuted.extend(circular(correct[sel],np.roll(centres[sel],1)))
    common=set(keys)&poses
    out={"st_frame_person_pairs":len(keys),"st_people":len({p for _,p in keys}),
         "hst_finite_33_joint_same_frame_same_id_pairs":len(common),
         "people_with_matched_hst_pose":len({p for _,p in common}),
         "axis_conventions_vs_2d_box_center_px":errors,
         "same_frame_id_control":{"rows":len(matched),"correct_id_px":quantiles(matched),
                                  "cyclically_shifted_other_id_px":quantiles(permuted)},
         "independent_raw_center":None}
    if raw_path is None:
        return out
    raw_bytes=raw_path.read_bytes()
    raw={}
    for frame, objects in json.loads(raw_bytes)["labels"].items():
        for ob in objects:
            kind,p=ob["label_id"].split(":")
            if kind!="pedestrian": continue
            k=(int(Path(frame).stem),int(p))
            point=np.array([ob["box"][field] for field in ("cx","cy","cz")])
            if k in raw or not np.isfinite(point).all():
                raise ValueError("Duplicate or nonfinite raw JRDB centre")
            raw[k]=point
    pairs=sorted(set(keys)&raw.keys())
    if not pairs: raise ValueError("No raw/ST matched frames and people")
    source=np.array([raw[k][:2] for k in pairs])
    estimate=np.array([[records[k]["x"],records[k]["y"]] for k in pairs])
    denominator=float(np.sum(source**2))
    if not denominator: raise ValueError("No metric extent")
    coef=float(np.sum(estimate*source)/denominator)
    out["independent_raw_center"]={
        "raw_member_sha256":hashlib.sha256(raw_bytes).hexdigest(),"pairs":len(pairs),
        "unique_people":len({p for _,p in pairs}),"triple_raw_st_hst_ids":len(set(pairs)&poses),
        "similarity_scale_raw_to_st_fixed_origin_axes":coef,
        "unscaled_planar_distance_m":quantiles(np.linalg.norm(estimate-source,axis=1)),
        "distance_after_scale_in_st_units":quantiles(np.linalg.norm(estimate-coef*source,axis=1)),
        "signed_axis_unscaled_median_errors_m":{
            label:float(np.median(np.linalg.norm(alt-source,axis=1))) for label,alt in variants(estimate)}}
    return out


def run(members_root, raw_paths):
    if len({p.stem for p in raw_paths}) != len(raw_paths) or any(p.stem not in SCENES for p in raw_paths):
        raise ValueError("Accept one file per named outdoor training scene only")
    raw={p.stem:p for p in raw_paths}
    groups=("tracks","poses")
    archives={} if members_root else {k:TrainingArchive(*ARCHIVES[k]) for k in groups}
    scenes={}
    for i,scene in enumerate(SCENES):
        blobs={}
        for g in groups:
            path=MEMBERS[g].format(scene)
            data=(members_root/path).read_bytes() if members_root else archives[g].member(path)
            if hashlib.sha256(data).hexdigest() != DIGESTS[g][i]:raise ValueError(f"Unexpected {g} archive member")
            blobs[g]=data
        scenes[scene]=check(blobs["tracks"],blobs["poses"],raw.get(scene))
    return {"scope":"Three named outdoor training clips. Compare horizontal cyclic panorama bearing error to the matched 2D box "
                    "centre for all signed axis permutations. Same frame and person ID joins HST body-relative 33-joint estimates. "
                    "Neither bearings nor HST identity matches supply an independent raw metric pedestrian centre. When separately "
                    "provided, the optional raw train labels are compared without translation, rescaling or rotation first.",
            "axis_convention":"JRDB robot x forward, y left, z up. x,y in the ST export used directly. Horizontal panorama is "
                              "u=W*((atan2(-y,x)+pi) mod 2pi)/(2pi), W=3760 pixels.",
            "sources":{"sensor_format":"https://drive.google.com/file/d/1DyfU_A0x5OzdDGUWPPPWuW-epupIyvT7/view",
                       "projection":"https://github.com/JRDB-dataset/jrdb_toolkit/blob/4fbf7d6eba3255746000eb8c15f707af69561c5d/visualisation/visualise.py#L556-L591",
                       "release_checksums":"outdoor_motion_baselines.json"},
            "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"recordings":scenes}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--members-root",type=Path)
    p.add_argument("--raw-labels",type=Path,action="append",default=[])
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    report=run(args.members_root,args.raw_labels)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
