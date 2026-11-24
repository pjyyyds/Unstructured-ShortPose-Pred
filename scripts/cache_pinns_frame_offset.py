"""Replay RTMPose one raw frame later on the three zero-origin PINNS clips.

Run python -m scripts.cache_pinns_frame_offset DATA_ROOT FROZEN_POSE.npz MODEL.onnx OUTPUT.npz.
The original folds/report.json beside the frozen cache supplies its expected SHA-256.
For a standalone copy, pass --parent-sha256 from the verified cloud report.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np

from scripts.benchmark_heldout_sites import HISTORY, load_site_windows
from scripts.cache_heldout_pose import file_sha256
from src.data import pinns_extractor
from src.data.pinns_extractor import RTMPoseONNXExtractor


CLIPS = {
    "america_crossroad_summer_daytime_sunny_0": 103,
    "america_crossroad_summer_nightfall_sunny_0": 104,
    "america_crossroad_summer_nightfall_sunny_1": 390,
}
DATA_SHA = "0b276b34ab1b01dae91f67c56bfaea5cc931511b05b4fc1c92686a891e6f6801"
MODEL_SHA = "5c0a4bf67953e6d2ac43ce15e77dc9d5d354ae18430a47d2c5963a7bc5683e3c"
XY_ATOL_PX = 0.001
CONF_ATOL = 0.0001
CHANGED_FIELDS = {"hist_kpts", "pose_inferred"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def frozen_digest(path, expected):
    report_path = path.parent / "folds/report.json"
    if report_path.is_file():
        identity = json.loads(report_path.read_text())["identity"]
        require(identity["data_sha256"] == DATA_SHA, "Adjacent fold report refers to different source data")
        if expected is not None:
            require(identity["pose_sha256"] == expected, "Explicit and adjacent frozen cache digests disagree")
        expected = identity["pose_sha256"]
    require(expected is not None, "Provide the original folds/report.json or --parent-sha256 from its identity.pose_sha256")
    actual = file_sha256(path)
    require(actual == expected, "Frozen pose cache does not match the verified parent digest")
    return actual


def preserved_hashes(arrays, affected):
    # All source arrays are immutable except the two pose fields on affected rows and metadata.
    return {
        k: hashlib.sha256((v[~affected] if k in CHANGED_FIELDS else v).tobytes()).hexdigest()
        for k, v in arrays.items() if k != "metadata_json"
    }


def replay_clip(video, rows, arrays, extractor, expected_frames):
    """Replace pose/mask on these rows only. Rows must follow original within-clip batch order."""
    requests = defaultdict(list)
    for row in rows:
        for time, frame in enumerate(range(int(arrays["t0"][row]) - HISTORY + 1, int(arrays["t0"][row]) + 1)):
            box = arrays["hist_bbox"][row, time]
            if np.isfinite(box).all() and min(box[2:]) > 0:
                requests[frame].append((row, time))
    require(bool(requests), f"No valid pose crops in {video}")
    frames = sorted(requests)
    # Controls span both ends of the clip and the largest original batch (first on ties).
    controls = {frames[0], frames[-1], max(frames, key=lambda t: len(requests[t]))}
    control_rows = {}
    offset_rows = {}
    for frame in frames:
        selection = tuple(np.array(requests[frame], dtype=np.int64).T)
        offset_rows[3 * frame + 1] = selection
        if frame in controls:
            control_rows[3 * frame] = (selection, arrays["hist_kpts"][selection].copy(),
                                      arrays["pose_inferred"][selection].copy())
    require(max(offset_rows) < expected_frames, "Shifted request is beyond the video")
    old_pose = arrays["hist_kpts"][rows].copy()
    old_mask = arrays["pose_inferred"][rows].copy()
    arrays["hist_kpts"][rows] = 0
    arrays["pose_inferred"][rows] = False
    control_stats = []
    cap = cv2.VideoCapture(str(video))
    try:
        require(cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == expected_frames
                and np.isclose(cap.get(cv2.CAP_PROP_FPS), 30, atol=0.05)
                and np.all(arrays["img_h"][rows] == cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                f"Video geometry/rate/length mismatch: {video}")
        for raw in range(max(offset_rows) + 1):
            require(cap.grab(), f"Unable to grab raw frame {raw} in {video}")
            if raw not in offset_rows and raw not in control_rows:
                continue
            ok, image = cap.retrieve()
            require(ok, f"Unable to decode raw frame {raw} in {video}")
            is_control = raw in control_rows
            selection = control_rows[raw][0] if is_control else offset_rows[raw]
            observed = extractor.extract_crops(image, arrays["hist_bbox"][selection])
            require(observed.shape == (len(selection[0]), 17, 3) and np.isfinite(observed).all(),
                    f"Invalid RTMPose output at raw frame {raw} of {video}")
            if is_control:
                _, original, inferred = control_rows[raw]
                require(inferred.all(), f"The original cache lacks one of the zero-offset control crops at {video}/{raw}")
                diff = np.abs(observed - original)
                xy_max, conf_max = float(diff[..., :2].max()), float(diff[..., 2].max())
                cross = int(np.count_nonzero((observed[..., 2] >= .2) != (original[..., 2] >= .2)))
                require(xy_max <= XY_ATOL_PX and conf_max <= CONF_ATOL and cross == 0,
                        f"Zero-offset replay differs at {video}/{raw}: max {xy_max:g} px, confidence {conf_max:g}, "
                        f"{cross} threshold crossings. Use the original runtime or a full, matched zero-offset control.")
                control_stats.append({"raw_frame": raw, "crops": len(original), "xy_max_abs_px": xy_max,
                                      "confidence_max_abs": conf_max, "xy_mean_abs_px": float(diff[..., :2].mean()),
                                      "confidence_mean_abs": float(diff[..., 2].mean()), "confidence_0_2_crossings": cross})
            else:
                arrays["hist_kpts"][selection] = observed
                arrays["pose_inferred"][selection] = True
    finally:
        cap.release()
    return {
        "windows": len(rows), "requested_crops": sum(map(len, requests.values())),
        "original_inferred": int(old_mask.sum()), "offset_inferred": int(arrays["pose_inferred"][rows].sum()),
        "changed_pose_frames": int(np.any(old_pose != arrays["hist_kpts"][rows], axis=(-2, -1)).sum()),
        "changed_inference_flags": int(np.count_nonzero(old_mask != arrays["pose_inferred"][rows])),
        "zero_offset_controls": control_stats,
    }


def run(root, parent, model, output, parent_sha256=None):
    import onnxruntime as ort

    require(output.resolve() != parent.resolve() and not output.exists(), "Use a new output path for the offset cache")
    parent_sha = frozen_digest(parent, parent_sha256)
    require(file_sha256(model) == MODEL_SHA, "Unexpected RTMPose ONNX digest")
    groups, _, digest = load_site_windows(root)
    require(digest == DATA_SHA, "Unexpected source label/calibration digest")
    by_clip = defaultdict(list)
    for windows in groups.values():
        for window in windows:
            by_clip[window.clip].append(window)
    with np.load(parent, allow_pickle=False) as cached:
        arrays = {k: cached[k] for k in cached.files}
    metadata = json.loads(arrays["metadata_json"].item())
    require("frame_offset_sensitivity" not in metadata and metadata.get("frame_offset_raw", 0) == 0
            and metadata["model_sha256"] == MODEL_SHA and metadata["label_and_calibration_digest_sha256"] == DATA_SHA
            and metadata["site_counts"] == {site: len(w) for site, w in groups.items()}
            and set(metadata["video_sha256"]) == set(by_clip), "Unexpected original pose metadata or source identities")
    keys = list(zip(arrays["clip"], arrays["person_id"], arrays["t0"]))
    expected = [(w.clip, w.person_id, w.t0) for clip in sorted(by_clip) for w in by_clip[clip]]
    require(keys == expected and len(set(keys)) == 1482, "Window IDs/order differ from the frozen pose extraction batch order")
    require(arrays["hist_kpts"].shape == (1482, HISTORY, 17, 3) and arrays["pose_inferred"].shape == (1482, HISTORY),
            "Unexpected frozen pose dimensions")
    expected_history = np.stack([w.xy[:HISTORY] for clip in sorted(by_clip) for w in by_clip[clip]])
    require(np.all(arrays["hist_mask"] == 1) and np.array_equal(arrays["hist_pos_relative"], expected_history),
            "Frozen motion observations differ from the registered ground positions")
    affected = np.isin(arrays["clip"], list(CLIPS))
    preserved = preserved_hashes(arrays, affected)
    sources = {}
    for clip, count in CLIPS.items():
        condition, number = clip.rsplit("_", 1)
        video = root / "videos" / condition / "videos" / f"ori_{number}.mp4"
        require(file_sha256(video) == metadata["video_sha256"][clip], f"Original video digest mismatch for {clip}")
        label = root / "labels" / f"{clip}.txt"
        with label.open() as stream:
            # Vehicle records can use a different final frame in the same clip.
            frames = {int(float(fields[0])) for line in stream
                      if len(fields := line.split()) >= 11 and fields[1].lower().startswith("per")}
        require(frames == set(range(len(frames))) and len(by_clip[clip]) == count, f"Unexpected label origin/count in {clip}")
        sources[clip] = (video, len(frames))

    cv2.setNumThreads(1)
    extractor = RTMPoseONNXExtractor(onnx_path=model, batch_size=32)
    require(extractor.sess is not None and extractor.active_providers == metadata["providers"],
            f"Frozen providers {metadata['providers']} differ from current {extractor.active_providers}. "
            "Use original providers or a full same-backend zero-offset control before interpreting changes.")
    details = {}
    for clip, (video, frame_count) in sources.items():
        rows = np.flatnonzero(arrays["clip"] == clip)
        details[clip] = replay_clip(video, rows, arrays, extractor, frame_count)
        print(f"{clip}: replayed {details[clip]['requested_crops']} crops, zero-offset controls passed", flush=True)
    require(preserved == preserved_hashes(arrays, affected), "Unexpected mutation outside the affected pose/mask values")
    metadata.update(frame_offset_raw=1, offset_clips=sorted(CLIPS), frame_mapping=(
        "Three offset_clips use zero-based video frame 3 * observation index + 1; all others use 3 * observation index."
    ))
    metadata["frame_offset_sensitivity"] = {
        "parent_pose_sha256": parent_sha, "providers": extractor.active_providers,
        "onnxruntime_version": ort.__version__, "opencv_version": cv2.__version__, "numpy_version": np.__version__,
        "script_sha256": file_sha256(Path(__file__)), "extractor_source_sha256": file_sha256(Path(pinns_extractor.__file__)),
        "clips": details, "unchanged_windows": int((~affected).sum()),
        "preserved_arrays_sha256": preserved,
        "zero_offset_control": {"method": "First, last and largest original within-frame crop batches per changed clip. "
                                 "The original batch order and RTMPose batch_size=32 are preserved.",
                                "xy_atol_px": XY_ATOL_PX, "confidence_atol": CONF_ATOL,
                                "allowed_confidence_0_2_crossings": 0, "sampled_controls_passed": True},
    }
    arrays["metadata_json"] = np.array(json.dumps(metadata, sort_keys=True))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.link(temporary, output)  # Publish the complete cache without replacing an existing one.
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Saved {output}, sha256={file_sha256(output)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("frozen_pose", type=Path)
    parser.add_argument("onnx_model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--parent-sha256", help="Verified cloud identity.pose_sha256 when the fold report is elsewhere")
    args = parser.parse_args()
    run(args.data_root, args.frozen_pose, args.onnx_model, args.output, args.parent_sha256)
