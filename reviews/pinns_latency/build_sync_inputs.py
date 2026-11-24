"""Materialize same-camera, same-tick inputs for timing the frozen full PINNS predictor."""
import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from reviews.pinns_graph_control.run_graph_control import graph, typed_frame
from reviews.pinns_matched_multiseed.replay_helpers import CACHE_SHA
from scripts.build_pinns_causal_maps import edges_from_frames
from src.data.pinns_bev import LEGACY_RASTER
from src.data.features import build_motion_features, build_pose_and_quality_features
from src.data.pinns_extractor import _check_right_handed_homography, _parse_pinns_label_file
from src.data.scene_graph import build_local_polylines

LENGTHS = (5, 10, 20)
CHECKPOINT = "checkpoints/PINNS_T5_Ours_Full_seed42.pt"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def source_groups(clip, ticks):
    counts = Counter(zip(clip, ticks))
    return {str(n): sum(size >= n for size in counts.values()) for n in (1, 4, 8, 12)}


def write_inputs(cache_path, original_pose, metadata, bev_root, destination):
    cache_bytes = cache_path.read_bytes()
    if digest(cache_bytes) != CACHE_SHA:
        raise ValueError("Expected the frozen 623-window matched pose cache")
    cache_inputs = ("clip", "person_id", "t0", "hist_kpts", "hist_bbox", "img_h", "ego_type", "pose_inferred")
    with np.load(io.BytesIO(cache_bytes), allow_pickle=False) as archive:
        data = {k: archive[k] for k in cache_inputs}
        observed_xy = archive["xy"][:, :20].copy()
    with ZipFile(original_pose) as archive:
        pose_bytes = archive.read("pose.npz")
    with np.load(io.BytesIO(pose_bytes), allow_pickle=False) as pose:
        large_support = source_groups(pose["clip"].tolist(), pose["t0"].tolist())
        orig_keys = list(zip(pose["clip"].tolist(), pose["t0"].tolist(), pose["person_id"].tolist()))
        if len(set(orig_keys)) != 1482 or len(orig_keys) != 1482:
            raise ValueError("Unexpected original five-frame roster")
        old_pose = {k: pose[k] for k in ("hist_kpts", "hist_bbox", "hist_pos_relative", "ego_type", "img_h")}
    orig_rows = {key: i for i, key in enumerate(orig_keys)}
    groups = defaultdict(list)
    for i, (clip, tick, person) in enumerate(zip(data["clip"].tolist(), data["t0"].tolist(), data["person_id"].tolist())):
        groups[clip, tick].append((person, i))
    scenes = sorted(key for key, members in groups.items() if len(members) >= 12)
    if not scenes:
        raise ValueError("No camera tick contains twelve distinct timed targets")
    rows = {t: defaultdict(list) for t in LENGTHS}
    sources, public_support = {}, []
    cv2.setNumThreads(1)
    for clip, tick in scenes:
        condition, number = clip.rsplit("_", 1)
        calib_file = metadata / "videos" / condition / "calib.json"
        label_file = metadata / "labels" / f"{clip}.txt"
        bev_file = bev_root / "videos" / condition / "videos" / f"bev_{number}.mp4"
        calibration = json.loads(calib_file.read_bytes())
        site = calibration["location"] + "_" + calibration["scene"]
        flip = _check_right_handed_homography(calibration)
        labels = _parse_pinns_label_file(label_file, flip, calibration["y_min"], calibration["y_max"])
        history = [typed_frame(labels[t]) for t in range(tick - 4, tick + 1)]
        earlier = [typed_frame(labels[t]) for t in range(tick - 19, tick - 4)]
        # Frame zero of the official bird's-eye-view video precedes all 5/10/20 histories here.
        if tick - 19 <= 0:
            raise ValueError("Map frame must precede the earliest history")
        polylines = edges_from_frames(bev_file, [0], calibration, flip, bev_convention=LEGACY_RASTER)
        counts = Counter()
        members = sorted(groups[clip, tick])
        if len(members) != len(set(person for person, _ in members)):
            raise ValueError("Duplicate person at the same physical tick")
        for person, i in members:
            old = orig_rows[clip, tick, person]
            for k in ("hist_kpts", "hist_bbox"):
                np.testing.assert_array_equal(data[k][i, -5:], old_pose[k][old])
            for k in ("ego_type", "img_h"):
                np.testing.assert_array_equal(data[k][i], old_pose[k][old])
            if not data["pose_inferred"][i].all():
                raise ValueError("Incomplete cached pose extraction")
            xy = np.stack([frame["Person", person] for frame in earlier + history])
            np.testing.assert_allclose(xy - xy[-1], observed_xy[i], atol=1e-8, rtol=0)
            np.testing.assert_allclose(xy[-5:] - xy[-1], old_pose["hist_pos_relative"][old], atol=1e-6, rtol=0)
            edges, types, mask, *_ = graph(history, person, xy[-5:] - xy[-1])
            env, avail = build_local_polylines(polylines, xy[-1], map_available=bool(polylines))
            counts["neighbor_slots"] += int(mask.sum())
            counts["car_slots"] += int(types[:, 6].sum())
            counts["map_rows_with_points"] += int(bool(env[:, :, 10].any()))
            for t, store in rows.items():
                valid = np.ones(t, dtype=np.float32)
                motion = build_motion_features(xy[-t:] - xy[-1], valid, dt=.1)
                pose, q, qp, _ = build_pose_and_quality_features(
                    data["hist_kpts"][i, -t:], data["hist_bbox"][i, -t:], valid, float(data["img_h"][i]))
                values = dict(clip=clip, t0=tick, person_id=person, site=site, x_m=motion, x_p=pose,
                              q_vec=q, q_part=qp, ego_type_oh=np.eye(8, dtype=np.float32)[data["ego_type"][i]],
                              p_env=env, m_map=np.array([avail], dtype=np.float32),
                              nbr_edges=edges, nbr_type_oh=types, nbr_mask=mask)
                for name, value in values.items():
                    store[name].append(value)
        for path in (calib_file, label_file, bev_file):
            sources[str(path.relative_to(metadata) if path in (calib_file, label_file) else path.relative_to(bev_root))] = digest(path.read_bytes())
        public_support.append(dict(site=site, targets=len(members), scene_contours=len(polylines), **counts))
    destination.mkdir(parents=True, exist_ok=True)
    file_hashes = {}
    for t, store in rows.items():
        out = destination / f"sync_T{t}.npz"
        np.savez_compressed(out, **{name: np.asarray(vals) for name, vals in store.items()})
        file_hashes[out.name] = digest(out.read_bytes())
    result = {
        "files_sha256": file_hashes,
        "checkpoint": CHECKPOINT, "checkpoint_sha256": digest((ROOT / CHECKPOINT).read_bytes()),
        "matched_inputs_sha256": CACHE_SHA, "original_pose_member_sha256": digest(pose_bytes),
        "source_files_sha256": sources, "scene_support": public_support,
        "original_roster_qualifying_camera_ticks_by_minimum_targets": large_support,
        "matched_roster_qualifying_camera_ticks_by_minimum_targets": source_groups(data["clip"].tolist(), data["t0"].tolist()),
        "scope": "Use every matched-roster camera tick with at least 12 distinct targets. The source roster required complete future labels for the prior forecasting study. Timed inputs contain only measurements up to each decision, with the same people across histories. Person ID orders a nested subset at each target count. No sample of future trajectories is timed.",
        "clock": "Nominal 10 Hz source label grid subsampled from 30 fps. The original extractor's possible one-video-frame offset for some US clips remains. The source clock is common to all targets in a scene.",
        "pose": "PINNS annotated boxes and original frozen RTMPose. Joint confidence and missingness are retained. Last-five-frame keypoints and boxes match the separate original cache exactly.",
        "map": "Canny contours from video frame zero of the official same-clip BEV, no occupancy hull. Calibration and local 20 m radius use the registered code. These image-based proxies carry no verified curb semantics.",
        "neighbors": "Up to 12 nearest original typed Person/Car records within 10 m per target. Relative displacement, velocity and observation fractions use the latest five measured ground ticks for all history-length comparisons. Includes observed background actors outside the timed batch.",
        "timing_scope": "The real full-model input tensors can be loaded before timing. This fixture does not measure image decoding, tracking, pose or map extraction, feature transfer, or a new device.",
    }
    out = destination / "protocol.json"
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("matched_cache", "original_pose_zip", "pinns_metadata", "bev_root", "destination"):
        p.add_argument(name, type=Path)
    args = p.parse_args()
    result = write_inputs(args.matched_cache, args.original_pose_zip, args.pinns_metadata, args.bev_root, args.destination)
    print(result["scene_support"])
