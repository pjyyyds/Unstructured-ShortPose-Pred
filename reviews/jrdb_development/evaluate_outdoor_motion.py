"""Unfitted 2.8-second motion baselines on three fixed, outdoor JRDB development recordings.

Run: python3 reviews/jrdb_development/evaluate_outdoor_motion.py --output report.json
Optionally pass --members-root ROOT for the nine extracted + checksum-verified training members.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import platform

import numpy as np

from check_outdoor_projection import SOURCES, WIDTH, delta_pixel
from check_training_coordinates import ARTIFACTS, TrainingArchive, rotation


STEP = 6  # raw JRDB frame indices between Social-Transmotion observations
HZ = 2.5
FUTURE = 7
SCENES = ("hewlett-packard-intersection-2019-01-24_0", "huang-lane-2019-02-12_0", "tressider-2019-03-16_1")
ARCHIVES = {"tracks": ARTIFACTS["tracks"][:2], "poses": SOURCES["keypoints"][:2], "odom": SOURCES["moving_odometry"][:2]}
MEMBERS = {"tracks": "jrdb/data/train/{}_shift_0.ndjson", "poses": "labels_3d_keypoints/train/{}.json",
           "odom": "odometry/train/{}.json"}
DIGESTS = {
    "tracks": ("29adfca6b4bfcf886d2832217d65b2eb7530c9cd4b9a15edfa94b476f1b1e818",
               "682752cfae6a08771d3547f233023b0d8d444346303baed2d103ce57b027affd",
               "d0da8c09a5593a8491ca825449d77547a64d5f3ecd35f185906b5ef5a64b76bc"),
    "poses": ("5db2360f7e0507dab0d7eaf734ce8f2d942253028b555d90c19d282d4b245e45",
              "51eb289d6f2ca68d83a5da7d5845e5eb8df7ff89d698b682f8261ba458b682cb",
              "b728921be8fb6b6a4de6ac410558c219ff344a13a2f40205ae446757c2a124b9"),
    "odom": ("25ba8073598b743de7e47f04ab09ed4b8ccc254eecc996d411fb31465e92b80d",
             "9f028bcf79854e581148ae8cfeef62ec5df19930839856586d4d6a754c014c8d",
             "b66ba960bb8be864aa1dc9e04c9b05c8e15e785cf5f400600ccc9151cb0e4c40"),
}
REFERENCES = {
    "outdoor_classification_all_three_table_5_1": "https://publications.rwth-aachen.de/record/1027011/files/1027011.pdf",
    "original_dataset_and_annotation_frequency": "https://arxiv.org/html/1910.11792#S4.SS2",
    "official_metric_ground_coordinates": "https://jrdb.erc.monash.edu/benchmark/",
    "official_trajectory_export": "https://github.com/vita-epfl/JRDB-Traj/blob/main/train_traj_extractor.py",
    "official_rig_calibration": "https://github.com/JRDB-dataset/jrdb_toolkit/tree/4fbf7d6eba3255746000eb8c15f707af69561c5d/calibration",
    "hst_data_and_odometry": "https://github.com/google-research/human-scene-transformer/tree/f60bc7fd63487f7135f70b7ed4a1f25a1f32c5bd/human_scene_transformer/data",
    "official_pose_format": "https://jrdb.erc.monash.edu/dataset/pose",
    "st_data_release": "https://github.com/vita-epfl/social-transmotion/releases/tag/ckpt_data",
}


def read_tracks(blob):
    tracks, angular, rates = defaultdict(dict), [], set()
    read_keys, missing, fields = 0, 0, set()
    for line in blob.splitlines():
        item = json.loads(line)
        if "scene" in item:
            rates.add(item["scene"]["fps"])
            continue
        t = item["track"]
        fields.update(t)
        f, p = int(t["f"]), int(t["p"])
        xy = (float(t["x"]), float(t["y"]))
        read_keys += 1
        if f % STEP or f in tracks[p]:
            raise ValueError("Off-grid or duplicate training track key")
        if not np.isfinite(xy).all() or np.hypot(*xy) == 0:
            missing += 1
            continue
        tracks[p][f] = xy
        w, h, left = t["bb_width"], t["bb_height"], t["bb_left"]
        if np.isfinite((w,h,left)).all() and 0 < w < WIDTH and h > 0:
            u = ((np.arctan2(-xy[1],xy[0]) + np.pi) % (2*np.pi)) * WIDTH / (2*np.pi)
            angular.append(delta_pixel(u, (left + w/2) % WIDTH))
    if rates != {HZ}:
        raise ValueError(f"Unexpected track sampling metadata: {rates}")
    return tracks, {"source_rows":read_keys, "excluded_invalid_or_zero_xy":missing, "track_fields":sorted(fields),
                    "bearing_vs_box_center_pairs":len(angular),
                    "bearing_vs_box_center_px_p50_p95":np.quantile(angular,[.5,.95]).tolist()}


def read_poses(blob):
    usable = set()
    for name, people in json.loads(blob)["labels"].items():
        for p in people:
            kind, ident = p["label_id"].split(":")
            if kind == "pedestrian":
                key = (int(Path(name).stem),int(ident))
                q = np.asarray(p["keypoints"])
                if q.shape != (99,) or key in usable:
                    raise ValueError("Unexpected HST joint shape or duplicate key")
                if np.isfinite(q).all():
                    usable.add(key)
    return usable


def odometry(blob, ticks):
    records = {int(Path(k).stem):v for k,v in json.loads(blob)["odometry"].items()}
    times = sorted(records)
    if times != list(range(len(records))) or not ticks.issubset(records):
        raise ValueError("Incomplete native odometry")
    pos = np.array([[records[t]["position"][a] for a in "xyz"] for t in times])
    rot = np.stack([rotation(np.array([records[t]["orientation"][a] for a in "xyzw"])) for t in times])
    relative = rot[0].T @ rot
    angle = np.rad2deg(np.arccos(np.clip((relative.trace(axis1=1,axis2=2)-1)/2,-1,1)))
    displacement = np.linalg.norm(pos[:,:2]-pos[0,:2],axis=1)
    if displacement.max() >= .001 or angle.max() >= .01:
        raise ValueError("Recording exceeds the specified stationary-rig bounds")
    return {"native_frames":len(times), "max_horizontal_displacement_m":float(displacement.max()),
            "max_3d_rotation_deg":float(angle.max())}


def evaluate_scene(blobs):
    tracks, support = read_tracks(blobs["tracks"])
    poses = read_poses(blobs["poses"])
    keys = {(f,p) for p,rows in tracks.items() for f in rows}
    ticks = {f for f,_ in keys}
    rig = odometry(blobs["odom"],ticks)
    if set(np.diff(sorted(ticks))) != {STEP}:
        raise ValueError("Scene-wide track sampling interval differs from the frozen grid")
    out = []
    owners, anchors, pose_counts = [], [], []
    possible, complete, runs_total = 0, 0, 0
    for person, rows in sorted(tracks.items()):
        times = np.asarray(sorted(rows))
        runs = np.split(times, np.flatnonzero(np.diff(times)!=STEP)+1)
        for run in runs:
            runs_total += 1
            possible += max(len(run)-1,0)
            complete += max(len(run)-FUTURE-1,0)
            for offset in range(0,len(run)-FUTURE-1,2+FUTURE):
                chunk = run[offset:offset+2+FUTURE]
                xy = np.stack([rows[t] for t in chunk])
                truth = xy[2:] - xy[1]
                delta = xy[1] - xy[0]
                cv = np.arange(1,FUTURE+1)[:,None] * delta
                dist = np.column_stack([np.linalg.norm(truth,axis=1),np.linalg.norm(truth-cv,axis=1)])
                out.append(np.stack([dist.mean(0),dist[-1]]).T)
                owners.append(person)
                anchors.append(int(chunk[1]))
                pose_counts.append(sum((int(t),person) in poses for t in chunk[:2]))
    scores, owners, anchors = np.stack(out), np.asarray(owners), np.asarray(anchors)
    counts = np.bincount(pose_counts,minlength=3)
    for p in np.unique(owners):
        if np.any(np.diff(anchors[owners==p]) < STEP*(2+FUTURE)):
            raise ValueError("Overlapping input/target positions on one track")
    person_means = np.stack([scores[owners==p].mean(0) for p in np.unique(owners)])
    def means(vals):
        return {m:dict(zip(("ade_m","fde_m"),v.tolist())) for m,v in zip(("stationary","last_two_cv"),vals.mean(0))}
    return {"sampling":{"nominal_rgb_hz":15,"track_hz":HZ,"native_step":STEP,"first_and_last_native_track_frame":[min(ticks),max(ticks)],
                        "distinct_track_frames":len(ticks)}, "odometry":rig,
            "support":{**support,"valid_xy_frame_person_pairs":len(keys),"people_with_xy":len({p for _,p in keys}),
                       "continuous_person_runs":runs_total,"all_consecutive_two_point_history_anchors":possible,
                       "history_anchors_with_all_seven_future_steps_before_thinning":complete,
                       "disjoint_complete_windows":len(scores),"people_in_scored_windows":len(np.unique(owners)),
                       "distinct_scored_anchor_times":len(np.unique(anchors)),"target_positions_scored":len(scores)*FUTURE,
                       "scored_windows_per_person_min_median_max":np.quantile(np.unique(owners,return_counts=True)[1],[0,.5,1]).tolist()},
            "pose_coverage":{"finite_hst_33x3_frame_person_entries_all_native_frames":len(poses),
                             "matched_st_frame_person_pairs":len(poses & keys),
                             "scored_windows_with_0_1_2_matched_history_frames":counts.tolist(),
                             "scored_people_with_both_historical_poses":len(set(owners[np.asarray(pose_counts)==2].tolist()))},
            "ordered_scored_keys_sha256":hashlib.sha256(json.dumps(list(zip(owners.tolist(),anchors.tolist())),
                                                            separators=(",",":")).encode()).hexdigest(),
            "window_weighted":means(scores),"person_equal_weighted":means(person_means)}


def run(members_root):
    archives = {} if members_root else {name: TrainingArchive(url,size) for name,(url,size) in ARCHIVES.items()}
    scenes = {}
    for i,scene in enumerate(SCENES):
        blobs, manifest = {}, {}
        for group,pattern in MEMBERS.items():
            member = pattern.format(scene)
            data = (members_root/member).read_bytes() if members_root else archives[group].member(member)
            digest = hashlib.sha256(data).hexdigest()
            if digest != DIGESTS[group][i]:
                raise ValueError(f"Unexpected {group} member for {scene}")
            blobs[group] = data
            manifest[group] = {"member":member,"uncompressed_bytes":len(data),"sha256":digest}
        scenes[scene] = {**evaluate_scene(blobs),"sources":manifest}
    macro = {m:{metric:float(np.mean([s["window_weighted"][m][metric] for s in scenes.values()]))
                for metric in ("ade_m","fde_m")} for m in ("stationary","last_two_cv")}
    return {
        "scope":"Unfitted development baselines on the three outdoor members of Social-Transmotion's ten shift_0 training recordings, "
                "classified using Table 5.1 of the cited study. All three are JRDB training recordings with verified nearly stationary odometry. "
                "No predictor was fitted; evaluations use release training records. Identities are recording-local and are never pooled across recordings.",
        "window_protocol":"Two consecutive real rows of the released track, 0.4 s apart at 2.5 Hz. Demand seven consecutive subsequent "
                          "rows through 2.8 s. Independently tile each uninterrupted recording/person run from its earliest record into "
                          "blocks of nine, dropping incomplete tails. All scored futures are complete and blocks of the same person share "
                          "no frames. Different people in a recording can appear at the same time. Score both predictions on the same windows.",
        "coordinates":"Predict offsets within each track's native exported horizontal frame. Static holds the last xy; constant velocity "
                      "uses the last two points. The benchmark specifies LiDAR horizontal metres and the authors' JRDB-Traj training "
                      "export reads 3D box cx/cy. Angular agreement with panorama boxes checks the direction; it cannot validate metric "
                      "scale or the export transform independently. Both models and all targets use the identical ST frame. "
                      "Wheel odometry establishes stationarity. Scoring native planar positions requires no camera-to-LiDAR transform. Original per-ID raw 3D centres were not available for an independent scale check.",
        "sampling_limit":"The source RGB is 15 Hz. JRDB documents manual 3D box annotations at 7.5 Hz and interpolation to 15 Hz, "
                         "with an interpolated flag in raw records. ST releases positions at 2.5 Hz and does not retain that flag. "
                         "No resampling/interpolation was performed here. HST provides estimated 33×3 landmarks on selected frames, "
                         "without image-plane confidence. Two native observations and a 2.8 s forecast cannot reproduce PINNS's five "
                         "10 Hz observations and 3.0 s endpoint. The records do not supply five genuine synchronized 10 Hz pose/ground measurements.",
        "interpretation":"Metrics describe three recording-specific external development checks in the source coordinate scale. "
                         "Person-equal and window-weighted estimates summarize different weightings. A recording-local track ID is not "
                         "guaranteed unique across recordings. No unseen-site model evaluation, pose effect, significance test or generalization "
                         "claim follows from these deterministic baselines.",
        "source_urls":REFERENCES,"archives":{k:v[0] for k,v in ARCHIVES.items()},
        "versions":{"python":platform.python_version(),"numpy":np.__version__},"script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "recordings": scenes, "three_recording_equal_window_means":macro}


if __name__ == "__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--members-root",type=Path)
    cli.add_argument("--output",type=Path,required=True)
    args=cli.parse_args()
    report=run(args.members_root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(report["three_recording_equal_window_means"])
