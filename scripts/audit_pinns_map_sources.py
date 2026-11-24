"""Replay the archived PINNS map inputs and audit their semantic and temporal sources.

Run at the repository root:
    python -m scripts.audit_pinns_map_sources DATA_ROOT --output REPORT.json

DATA_ROOT contains labels/ and videos/<condition>/calib.json. BEV movies must
exist at videos/<condition>/videos/bev_<id>.mp4 under DATA_ROOT or --bev-root.
Only the original 47 development clips are in scope. No pose model is needed.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from src.data.pinns_bev import LEGACY_RASTER
from src.data.pinns_extractor import (
    _assemble_sequence_windows,
    _check_right_handed_homography,
    _parse_pinns_label_file,
    extract_static_bev_and_train_polylines,
)
from src.data.splits import assign_site_clip_splits, temporal_buffer_split_indices


ROOT = Path(__file__).resolve().parents[1]
CHANNELS = (
    "bev_contour", "wall", "planter_barrier", "temporary_barrier",
    "parked_rectangle", "pedestrian_hull", "closed", "tangent_valid",
)
IMPLEMENTATION = (
    "src/data/pinns_extractor.py", "src/data/scene_graph.py",
    "src/data/dataset_builder.py", "src/data/splits.py", "src/data/pinns_bev.py",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_for_clip(condition, index, frames, condition_split):
    """Mirror the archived per-condition choice of training rows and BEV donor."""
    name = f"{condition}_{index}"
    if condition_split:
        split, guard = condition_split[index]
        triggers = {frame: split for frame in frames[name] if frame >= min(frames[name]) + guard}
        donors = [f"{condition}_{i}" for i in sorted(condition_split) if condition_split[i][0] == "train"]
        limits = {donor: max(frames[donor]) for donor in donors}
        return triggers, condition, donors[0], limits, {}

    split_frames = temporal_buffer_split_indices(sorted(frames[name]))
    triggers = {int(frame): split for split, times in split_frames.items() for frame in times}
    train = split_frames["train"]
    limits = {name: int(train[-1]) if len(train) else -1}
    spans = {
        split: [3 * int(times[0]), 3 * int(times[-1])]
        for split, times in split_frames.items() if split != "train" and len(times)
    }
    return triggers, name, name, limits, spans


def source_frames(path):
    cap = cv2.VideoCapture(str(path))
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if count <= 0 or not abs(fps - 30) < 0.05:
            raise ValueError(f"Expected a decodable 30 Hz BEV movie: {path}")
        raw_frames = np.linspace(0, max(0, int(0.6 * count) - 1), 7, dtype=int).tolist()
        for frame in raw_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = cap.read()
            if not ok or image is None:
                raise ValueError(f"Missing BEV frame {frame} in {path}")
    finally:
        cap.release()
    return count, fps, raw_frames


def build_map(frames, limits, bev_clip, condition, calibration, flip_y, bev_root, spans, temporal):
    """Reproduce the archived inputs; retain source identities for the audit only."""
    filename = "bev_" + bev_clip.rsplit("_", 1)[1] + ".mp4"
    video = Path("videos") / condition / "videos" / filename
    count, fps, raw_frames = source_frames(bev_root / video)
    pedestrians = []
    vehicles = []
    for donor, cutoff in limits.items():
        for frame, objects in frames[donor].items():
            if frame > cutoff:
                continue
            for obj in objects:
                xy = (obj["wx"], obj["wy"])
                if obj["type"].lower().startswith("per"):
                    pedestrians.append((donor, obj["id"], frame, xy))
                elif obj["fixed"] == 1:
                    vehicles.append(xy)
    ped_xy = np.asarray([row[3] for row in pedestrians], dtype=np.float32).reshape(-1, 2)
    vehicle_xy = np.unique(np.round(vehicles, 1), axis=0) if vehicles else np.empty((0, 2))
    polylines = extract_static_bev_and_train_polylines(
        bev_root / video, calibration, ped_xy, vehicle_xy, flip_y, bev_convention=LEGACY_RASTER
    )
    vertices = {tuple(xy) for points, attrs in polylines if attrs[5] for xy in points}
    pool_frames = defaultdict(set)
    vertex_frames = defaultdict(set)
    for (clip, pid, frame, _), xy in zip(pedestrians, ped_xy):
        pool_frames[clip, pid].add(frame)
        if tuple(xy) in vertices:
            vertex_frames[clip, pid].add(frame)

    counts = sum((attrs.astype(int) for _, attrs in polylines), start=np.zeros(8, dtype=int))
    info = {
        "source_video": video.as_posix(), "source_video_sha256": digest(bev_root / video),
        "video_frames": count, "video_fps": fps, "sampled_raw_frames": raw_frames,
        "same_clip_val_test_trigger_spans_raw": spans,
        "sampled_inside_split_span": {
            split: any(low <= f <= high for f in raw_frames) for split, (low, high) in spans.items()
        },
        "sampled_after_temporal_training_cut": temporal and max(raw_frames) > 3 * limits[bev_clip],
        "source_track_max_10hz_by_clip": limits, "pedestrian_training_rows": len(pedestrians),
        "observed_fixed_vehicle_rows": len(vehicles), "rounded_unique_vehicle_centres": len(vehicle_xy),
        "polylines_by_channel_before_local_crop": dict(zip(CHANNELS, counts.tolist())),
        "served_clips": [],
    }
    return polylines, pool_frames, vertex_frames, info


def target_id(sample, tracks):
    """The archived sample stores only geometry. Match the current position, then observed history if needed."""
    t0 = sample["t0_frame_10hz"]
    matches = [pid for pid, track in tracks.items()
               if t0 in track and np.array_equal(track[t0], sample["p_t0"])]
    if len(matches) > 1:
        observed = np.flatnonzero(sample["hist_mask_20"])
        matches = [
            pid for pid in matches
            if all(np.array_equal(tracks[pid].get(t0 - 19 + j), sample["hist_pos_20"][j]) for j in observed)
        ]
    if len(matches) != 1:
        raise ValueError(f"Ambiguous target in {sample['seq_id']} at {t0}: {matches}")
    return matches[0]


def window_counts(sample, pool, vertices, map_info, same_bev_clip):
    t0 = sample["t0_frame_10hz"]
    scored = {t0 + 1 + int(i) for i in np.flatnonzero(sample["fut_mask"])}
    environment = sample["p_env"]
    valid = environment[:, :, 10].any(axis=1)
    active = environment[valid, 0, 2:10].astype(int).sum(axis=0)
    edge = bool(active[0])
    hull = bool(active[5])
    raw_frames = map_info["sampled_raw_frames"]
    overlaps = map_info["sampled_inside_split_span"]
    after_t0 = same_bev_clip and max(raw_frames) > 3 * t0
    in_next_3s = same_bev_clip and any(3 * t0 < f <= 3 * (t0 + 30) for f in raw_frames)
    past = any(frame <= t0 for frame in pool)
    counts = Counter({
        "windows": 1, "map_flag_one": int(sample["m_map"][0] == 1),
        "nonempty": int(valid.any()), "empty": int(not valid.any()),
        "source_pool_includes_same_person_clip": int(bool(pool)),
        "source_pool_includes_same_person_past_or_present": int(past),
        "source_pool_includes_same_person_scored_future_position": int(bool(scored.intersection(pool))),
        "usable_hull_source_pool_includes_same_person_past_or_present": int(hull and past),
        "usable_hull_same_person_past_or_present_at_global_hull_vertex": int(hull and any(f <= t0 for f in vertices)),
        "usable_hull_same_person_scored_future_at_global_hull_vertex": int(hull and bool(scored.intersection(vertices))),
        "usable_edge_from_temporal_map_with_source_after_training_cut":
            int(edge and map_info["sampled_after_temporal_training_cut"]),
        "usable_edge_with_same_clip_source_inside_val_trigger_span": int(edge and overlaps.get("val", False)),
        "usable_edge_with_same_clip_source_inside_test_trigger_span": int(edge and overlaps.get("test", False)),
        "same_clip_bev_after_t0": int(after_t0),
        "usable_edge_same_clip_bev_after_t0": int(edge and after_t0),
        "usable_edge_same_clip_bev_within_next_3s": int(edge and in_next_3s),
    })
    for name, count in zip(CHANNELS, active.tolist()):
        counts["windows_" + name] = int(count > 0)
        counts["instances_" + name] = count
    return counts


def summarize(counts):
    result = dict(counts)
    for key in ("map_flag_one", "nonempty", "empty"):
        result[key + "_rate"] = counts[key] / counts["windows"]
    result["windows_by_channel"] = {name: result.pop("windows_" + name) for name in CHANNELS}
    result["polyline_instances_by_channel"] = {name: result.pop("instances_" + name) for name in CHANNELS}
    return result


def audit(data_root, bev_root):
    cv2.setNumThreads(1)
    sources = defaultdict(dict)
    label_hashes = {}
    calibration_hashes = {}
    for path in sorted((data_root / "labels").glob("*.txt")):
        condition, index = path.stem.rsplit("_", 1)
        sources[condition][int(index)] = path
        label_hashes[path.stem] = digest(path)
    totals = defaultdict(Counter)
    metadata = {}
    clip_site = {}
    for condition, paths in sources.items():
        calibration_path = data_root / "videos" / condition / "calib.json"
        calibration_hashes[condition] = digest(calibration_path)
        calibration = json.loads(calibration_path.read_text())
        site = f"{calibration['location']}_{calibration['scene']}"
        flip_y = _check_right_handed_homography(calibration)
        condition_split = assign_site_clip_splits(paths)
        frames = {path.stem: _parse_pinns_label_file(path, flip_y, calibration["y_min"], calibration["y_max"])
                  for _, path in sorted(paths.items())}
        maps = {}
        for index, path in sorted(paths.items()):
            name = path.stem
            clip_site[name] = site
            if len(frames[name]) < 40:
                raise ValueError(f"Unexpectedly short original clip: {name}")
            triggers, map_id, bev_clip, limits, spans = split_for_clip(condition, index, frames, condition_split)
            if map_id not in maps:
                maps[map_id] = build_map(frames, limits, bev_clip, condition, calibration, flip_y,
                                         bev_root, spans, temporal=not condition_split)
            polylines, pool, vertices, map_info = maps[map_id]
            map_info["served_clips"].append(name)
            metadata[map_id] = map_info
            tracks = defaultdict(dict)
            for frame, objects in frames[name].items():
                for obj in objects:
                    if obj["type"].lower().startswith("per"):
                        tracks[obj["id"]][frame] = np.asarray([obj["wx"], obj["wy"]], dtype=np.float32)
            samples = _assemble_sequence_windows(name, frames[name], {}, polylines, triggers, calibration["v_size"], 4)
            for sample in samples:
                key = name, target_id(sample, tracks)
                counts = window_counts(sample, pool.get(key, ()), vertices.get(key, ()), map_info, name == bev_clip)
                for group in ("all", site, name):
                    totals[group, "all"].update(counts)
                    totals[group, sample["split"]].update(counts)

    reference = json.loads((ROOT / "results/dataset_diagnostics.json").read_text())["pinns"]
    expected = reference["split_counts"]
    if len(sources) != 22 or len(clip_site) != reference["num_sequences"] or len(set(clip_site.values())) != 4:
        raise ValueError("Expected the 47 original clips in 22 acquisition conditions at four physical sites")
    if {s: totals["all", s]["windows"] for s in expected} != expected:
        raise ValueError("Reconstructed windows differ from the archived split counts")
    ordered_splits = ("all", *expected)

    def groups(name):
        return {s: summarize(totals[name, s]) for s in ordered_splits if totals[name, s]["windows"]}

    return {
        "scope": ("Original 7,283 PINNS windows at four previously examined sites. Pose inference is omitted. "
                  "Map generation, cropping and window eligibility do not require pose outputs. V2 does not archive "
                  "per-window map tensors, so cross-version decoder/OpenCV bitwise identity is unverified. "
                  "The later 1,482-window M0/M1 held-out-site study does not use this map."),
        "source": {"opencv": cv2.__version__, "numpy": np.__version__,
                   "dataset": "Official PINNS release, labels, scene calibrations and BEV video; no road vector annotations used",
                   "source_sha256": {p: digest(ROOT / p) for p in IMPLEMENTATION},
                   "label_sha256": label_hashes, "calibration_sha256": calibration_hashes},
        "channel_order": CHANNELS,
        "sources_by_channel": {
            "bev_contour": "Generic Canny image contour from a seven-frame median; no manually labeled curb/sidewalk mask",
            "wall": "No extractor", "planter_barrier": "No extractor", "temporary_barrier": "No extractor",
            "parked_rectangle": "Synthetic axis-aligned 4.2 x 1.8 m rectangle at rounded fixed non-pedestrian centers; no instances",
            "pedestrian_hull": "Closed convex hull of training pedestrian positions; no annotated portal or walkable boundary",
            "closed": "Geometric flag on generated hulls/rectangles", "tangent_valid": "Geometric flag on generic image contours",
        },
        "definition": ("Per-channel counts are windows with one or more active lines, and instances of active lines. "
                       "Local extraction uses a 20 m nearest-original-vertex radius, 16 lines maximum, 10 points per line. "
                       "PINNS always sets m_map=1. A usable hull vertex match compares original float32 global vertices, "
                       "before resampling; identical coordinates can have multiple contributing rows. Track matches "
                       "require the same clip and person ID. Split-trigger spans run from the first to the last eligible "
                       "10 Hz anchor, expressed as raw zero-based BEV frames (3*t under the frozen convention). "
                       "Timing exposure does not establish that a moving object survives the median or affects a prediction."),
        "all": groups("all"), "sites": {site: groups(site) for site in sorted(set(clip_site.values()))},
        "clips": {clip: groups(clip) for clip in sorted(clip_site)}, "clip_to_site": clip_site,
        "map_construction": metadata,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--bev-root", type=Path, help="Separate BEV root with the same videos/ hierarchy")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.data_root, args.bev_root or args.data_root)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
