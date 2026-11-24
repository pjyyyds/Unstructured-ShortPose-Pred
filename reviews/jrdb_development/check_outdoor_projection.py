"""Join one named outdoor training recording in the public HST and Social-Transmotion releases."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from check_training_coordinates import ARTIFACTS, TrainingArchive, rotation


SCENE = "huang-lane-2019-02-12_0"  # Fig. 2(c), JRDB paper; JRDB, ST and HST development/training.
MOVING_SCENE = "memorial-court-2019-03-16_0"  # Fig. 2(d), JRDB paper; JRDB and HST training.
WIDTH = 3760  # JRDB benchmark paper, section 7.1.2, cylindrical panorama.
SOURCES = {
    "tracks_and_boxes": (*ARTIFACTS["tracks"][:2], f"jrdb/data/train/{SCENE}_shift_0.ndjson"),
    "keypoints": ("https://storage.googleapis.com/gresearch/human_scene_transformer/keypoints.zip?generation=1699186254967082",
                  378743220, f"labels_3d_keypoints/train/{SCENE}.json"),
    "moving_odometry": (ARTIFACTS["odometry"][0] + "?generation=1699184889005564",
                        ARTIFACTS["odometry"][1], f"odometry/train/{MOVING_SCENE}.json"),
}
MEMBER_SHA256 = {
    "tracks_and_boxes": "682752cfae6a08771d3547f233023b0d8d444346303baed2d103ce57b027affd",
    "keypoints": "51eb289d6f2ca68d83a5da7d5845e5eb8df7ff89d698b682f8261ba458b682cb",
    "moving_odometry": "ee9de392716120c169fc56dedc8abd72cf83f56ee9cb7f02b7b38e67a95ba151",
}


def delta_pixel(a, b):
    return np.abs((np.asarray(a) - np.asarray(b) + WIDTH / 2) % WIDTH - WIDTH / 2)


def distribution(values):
    return dict(zip(("p50", "p90", "p95", "max"), np.quantile(values, [.5, .9, .95, 1]).tolist()))


def moving_check(blob):
    odom = {int(Path(name).stem): v for name, v in json.loads(blob)["odometry"].items()}
    ticks = sorted(odom)
    if ticks != list(range(len(ticks))):
        raise ValueError("Moving training recording has a gap in the native frame index")
    pos = np.asarray([[odom[t]["position"][d] for d in "xyz"] for t in ticks])
    rot = np.stack([rotation(np.array([odom[t]["orientation"][d] for d in "xyzw"])) for t in ticks])
    local = (pos - pos[0]) @ rot[0]
    angles = np.rad2deg(np.arccos(np.clip(((rot[0].T @ rot).trace(axis1=1, axis2=2) - 1) / 2, -1, 1)))
    return {"recording": MOVING_SCENE, "native_frames": len(ticks),
            "max_translation_from_first_xy_m": float(np.linalg.norm(local[:, :2], axis=1).max()),
            "max_rotation_from_first_deg": float(angles.max()),
            "displacement_over_42_native_frames_xy_m": distribution(
                np.linalg.norm(local[42:, :2] - local[:-42, :2], axis=1)),
            "scope": "Robot wheel odometry only. Moving clip absent from Social-Transmotion training release. "
                     "No pedestrian coordinates or forecast targets used in this second recording."}


def main():
    blobs, provenance = {}, {}
    for name, (url, size, member) in SOURCES.items():
        archive = TrainingArchive(url, size)
        data = archive.member(member)
        digest = hashlib.sha256(data).hexdigest()
        if digest != MEMBER_SHA256[name]:
            raise ValueError(f"Unexpected content for frozen training member {member}")
        blobs[name] = data
        provenance[name] = {"url": url, "archive_member": member, "member_bytes": len(data),
                            "member_sha256": digest}
    tracks, sampled_hz = {}, set()
    for line in blobs["tracks_and_boxes"].splitlines():
        item = json.loads(line)
        if "scene" in item:
            sampled_hz.add(item["scene"]["fps"])
            continue
        t = item["track"]
        key = (t["f"], t["p"])
        v = tuple(float(t[n]) for n in ("x", "y", "bb_left", "bb_top", "bb_width", "bb_height"))
        if key in tracks and tracks[key] != v:
            raise ValueError(f"Conflicting box or centre for {key}")
        tracks[key] = v
    poses = {}
    for image, people in json.loads(blobs["keypoints"])["labels"].items():
        frame = int(Path(image).stem)
        for p in people:
            prefix, ident = p["label_id"].split(":")
            if prefix != "pedestrian":
                continue
            key = (frame, int(ident))
            points = np.asarray(p["keypoints"], dtype=float).reshape(33, 3)
            if key in poses:
                raise ValueError(f"Duplicate HST ID: {key}")
            poses[key] = points
    usable = {key: v for key, v in tracks.items()
              if np.isfinite(v).all() and 0 < v[4] < WIDTH and v[5] > 0 and np.hypot(*v[:2]) > 0}
    keys = sorted(usable)
    x, y, left, _, width, height = np.asarray([usable[k] for k in keys]).T
    center = (left + width / 2) % WIDTH
    # JRDB toolkit: (x, y, z) maps to reference (-y, -z, x); cylindrical u depends on atan2(-y,x).
    projected = (np.arctan2(-y, x) + np.pi) % (2 * np.pi) * WIDTH / (2 * np.pi)
    error = delta_pixel(projected, center)
    # Deterministic identity mismatch check: next ascending pedestrian ID in the same frame.
    by_frame = defaultdict(list)
    for i, (f, _) in enumerate(keys):
        by_frame[f].append(i)
    shuffled, paired = [], []
    for idx in by_frame.values():
        if len(idx) >= 2:
            shuffled.extend(delta_pixel(projected[idx], np.roll(center[idx], -1)))
            paired.extend(error[idx])
    matched = sorted(set(keys) & poses.keys())
    fully_finite = [k for k in matched if np.isfinite(poses[k]).all()]
    hips = [(poses[k][23] + poses[k][24]) / 2 for k in fully_finite]
    sample_steps = sorted({int(d) for d in np.diff(sorted(by_frame))})
    if sample_steps != [6] or sampled_hz != {2.5}:
        raise ValueError("Unexpected native-to-ST sampling schedule")
    report = {
        "recording": SCENE, "panorama_width_px": WIDTH, "source_members": provenance,
        "source_revisions": {"jrdb_toolkit": "4fbf7d6eba3255746000eb8c15f707af69561c5d",
                             "hst_code": "f60bc7fd63487f7135f70b7ed4a1f25a1f32c5bd",
                             "st_github_release_id": 135144126, "st_github_asset_id": 157559343},
        "jrdb_nominal_native_rgb_hz": 15, "native_frames_per_st_sample": 6,
        "st_track_samples_hz": sorted(sampled_hz), "st_unique_frame_person_pairs": len(tracks),
        "st_valid_boxes_and_nonzero_xy": len(keys), "st_people": len({k[1] for k in keys}),
        "st_distinct_frames": len(by_frame), "st_native_frame_range": [min(by_frame), max(by_frame)],
        "st_frame_steps": sample_steps,
        "st_boxes_at_or_above_jrdb_pose_general_6500px2_guideline": int(np.count_nonzero(width * height >= 6500)),
        "horiz_center_vs_jrdb_bearing_circular_error_px": distribution(error),
        "horiz_center_vs_jrdb_bearing_circular_error_deg": distribution(error * 360 / WIDTH),
        "projected_bearing_inside_box_horizontal_extent": int(np.count_nonzero(error <= width / 2)),
        "next_person_same_frame_control": {"eligible_pairs": len(paired), "correct_id_px": distribution(paired),
                                            "wrong_id_px": distribution(shuffled)},
        "hst_frame_person_pairs_all_native_frames": len(poses),
        "hst_matched_st_valid_pairs": len(matched), "hst_matched_st_people": len({k[1] for k in matched}),
        "hst_matched_all_33_xyz_finite": len(fully_finite),
        "hst_matched_finite_mid_hip_norm_native": distribution(np.linalg.norm(hips, axis=1)) if hips else None,
        "moving_outdoor_reference": moving_check(blobs["moving_odometry"]),
        "semantics": "Angular agreement tests horizontal axes/origin direction. It is invariant to positive radial scale. "
                     "Matched HST records store 33x3 body landmarks without pixel boxes or confidence. "
                     "An ID match and an angular check do not independently verify raw 3D box centres, metric scale "
                     "or a moving-sequence compensation convention.",
    }
    target = Path(__file__).with_name("outdoor_projection.json")
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
