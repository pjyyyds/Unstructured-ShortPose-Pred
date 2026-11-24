"""Fit and replay six-mode PINNS motion and pose models at matched 5/10/20-frame histories."""
import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
import torch

from scripts.audit_pinns_matched_histories import load_matched_histories
from scripts.benchmark_heldout_sites import TrackWindow
from scripts.cache_heldout_pose import build_clip, save_npz
from scripts.cache_pinns_frame_offset import CONF_ATOL, XY_ATOL_PX
from scripts.run_heldout_cloud import MODEL_SHA
from scripts.run_heldout_seed_sensitivity import configure_training, json_digest, split_indices
from scripts.train_pinns_heldout_sites import MODELS, Protocol, atomic_write, file_hash, fit_predict, mean_metrics, save_json
from scripts.verify_heldout_bundle import ATOL_M, RTOL, run_cpu_forward
from src.data.constants import DT, T_FUT
from src.data.features import build_motion_features, build_pose_and_quality_features
from src.data.pinns_extractor import RTMPoseONNXExtractor, _check_right_handed_homography, _parse_pinns_label_file
from src.models.predictor import UnstructuredShortPosePredictor

DURATIONS = (5, 10, 20)
PARENT_SHA = "b493a47536664ddb4fd66603c1bbf6b2c2e33f553520382027b3ae96b53447d0"
# First extraction run passed all per-clip controls before its first GPU checkpoint failed to reload.
FIRST_COLLECTOR_SHA = "9fb1a8bdb240ef2d3fd5f3a3a91370a2585961dfc4c6007a9c7bafa74d5c1788"


@dataclass
class Reference:
    keys: list
    index: dict
    xy: np.ndarray
    boxes: np.ndarray
    parent: dict
    parent_indices: np.ndarray
    parent_meta: dict
    label_digest: str


POSE_FIELDS = ("hist_kpts", "hist_bbox", "pose_inferred")
IDENTITY_FIELDS = ("clip", "person_id", "t0", "ego_type", "img_h")
CLOCK = "Frozen pilot clock, raw image index = 3 * sampled observation index."


def read_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def keys_of(arrays):
    return list(zip(*(arrays[name].tolist() for name in ("clip", "person_id", "t0"))))


def same(actual, expected, name):
    if actual.dtype != expected.dtype or actual.shape != expected.shape or not np.array_equal(actual, expected):
        raise ValueError(f"Cache/annotation mismatch in {name}")


def load_reference(data, pilot_pose):
    if file_hash(pilot_pose) != PARENT_SHA:
        raise ValueError("Unexpected frozen five-frame pilot pose SHA-256")
    cohort = load_matched_histories(data)
    old = read_arrays(pilot_pose)
    parent_meta = json.loads(old["metadata_json"].item())
    if (parent_meta["label_and_calibration_digest_sha256"] != cohort.source_digest
            or parent_meta["model_sha256"] != MODEL_SHA or not parent_meta["providers"]):
        raise ValueError("Frozen parent pose metadata differs from archived sources")
    original = keys_of(old)
    # The parent extractor ordered clips globally while retaining within-clip window order.
    expected = sorted(cohort.keys, key=lambda key: key[0])
    if original != expected or len(original) != len(set(original)):
        raise ValueError("Parent window ID order differs from the verified original crop batches")
    original_index = {key: i for i, key in enumerate(original)}
    selected = sorted(np.flatnonzero(cohort.strict), key=lambda i: cohort.keys[i])
    keys = [cohort.keys[i] for i in selected]
    parent_indices = np.array([original_index[k] for k in keys], dtype=int)
    xy = np.concatenate([cohort.history[selected], cohort.future[selected]], axis=1)
    if not cohort.mask[selected].all() or not np.all(old["hist_mask"][parent_indices] == 1):
        raise ValueError("A matched window lacks an observed position")
    same(old["hist_pos_relative"][parent_indices], xy[:, 15:20], "parent five observed positions")
    selected_index = {key: i for i, key in enumerate(keys)}
    index = {site: np.array([selected_index[k] for k, site_id in zip(cohort.keys, cohort.sites)
                             if site_id == site and k in selected_index], dtype=int)
             for site in sorted(set(cohort.sites.tolist()))}
    rows_by_clip = defaultdict(list)
    for i, (clip, _, _) in enumerate(keys):
        rows_by_clip[clip].append(i)
    boxes = np.zeros((len(keys), 20, 4), dtype=np.float32)
    types = np.zeros(len(keys), dtype=np.int64)
    heights = np.empty(len(keys), dtype=old["img_h"].dtype)
    for clip, rows in rows_by_clip.items():
        condition = clip.rsplit("_", 1)[0]
        calibration = json.loads((data / "videos" / condition / "calib.json").read_text())
        frame_objects = _parse_pinns_label_file(data / "labels" / (clip + ".txt"),
            _check_right_handed_homography(calibration), calibration["y_min"], calibration["y_max"])
        tracks = {(t, o["id"]): o for t, objects in frame_objects.items()
                  for o in objects if o["type"].lower().startswith("per")}
        for i in rows:
            _, person, anchor = keys[i]
            observed = [tracks[t, person] for t in range(anchor - 19, anchor + 1)]
            for j, o in enumerate(observed):
                box = [o["ox"], o["oy"] - 0.5 * o["h"], o["w"], o["h"]]
                if np.isfinite(box).all() and min(box[2:]) > 0:
                    boxes[i, j] = box
            previous, last = observed[-2:]
            speed = np.linalg.norm([last["wx"] - previous["wx"], last["wy"] - previous["wy"]]) / DT
            types[i] = 2 if last["fixed"] == 1 or speed < 0.20 else 0
            heights[i] = calibration["v_size"]
    same(old["hist_bbox"][parent_indices], boxes[:, -5:], "parent five boxes from official annotations")
    same(old["ego_type"][parent_indices], types, "parent ego type from official observations")
    same(old["img_h"][parent_indices], heights, "parent camera height")
    return Reference(keys, index, xy, boxes, old, parent_indices, parent_meta, cohort.source_digest)


def control_batch(parent, clip):
    """Largest fully recorded original within-frame crop batch, including non-matched windows."""
    requests = defaultdict(list)
    for row in np.flatnonzero(parent["clip"] == clip):
        for j in range(5):
            box = parent["hist_bbox"][row, j]
            if np.isfinite(box).all() and min(box[2:]) > 0:
                requests[int(parent["t0"][row]) - 4 + j].append((int(row), j))
    for frame in sorted(requests, key=lambda t: (-len(requests[t]), t)):
        rows, times = np.asarray(requests[frame], dtype=int).T
        if parent["pose_inferred"][rows, times].all():
            return 3 * frame, (rows, times)
    raise ValueError(f"No complete original crop batch available for {clip}")


def check_stats(stats, raw, crops, clip):
    if (stats["raw_frame"] != raw or stats["crops"] != crops or stats["confidence_0_2_crossings"] != 0
            or not 0 <= stats["xy_max_abs_px"] <= XY_ATOL_PX
            or not 0 <= stats["confidence_max_abs"] <= CONF_ATOL):
        raise ValueError(f"Failed original-frame control for {clip}: {stats}")


def zero_control(video, clip, parent, reader):
    raw, selection = control_batch(parent, clip)
    cap = cv2.VideoCapture(str(video))
    try:
        if (not cap.isOpened() or not np.isclose(cap.get(cv2.CAP_PROP_FPS), 30, atol=.05)
                or not np.all(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) == parent["img_h"][selection[0]])):
            raise ValueError(f"Unexpected zero-control video geometry for {clip}")
        for _ in range(raw + 1):
            if not cap.grab():
                raise ValueError(f"Cannot grab original control frame in {clip} at {raw}")
        ok, image = cap.retrieve()
        if not ok or image is None:
            raise ValueError(f"Cannot decode original control frame in {clip} at {raw}")
    finally:
        cap.release()
    pose = reader.extract_crops(image, parent["hist_bbox"][selection])
    original = parent["hist_kpts"][selection]
    if pose.shape != original.shape or not np.isfinite(pose).all():
        raise ValueError(f"Invalid original-frame pose output in {clip}")
    difference = np.abs(pose - original)
    stats = {"raw_frame": raw, "crops": len(pose), "xy_max_abs_px": float(difference[..., :2].max()),
             "confidence_max_abs": float(difference[..., 2].max()),
             "confidence_0_2_crossings": int(np.count_nonzero((pose[..., 2] >= .2) != (original[..., 2] >= .2)))}
    check_stats(stats, raw, len(pose), clip)
    return stats


def validate_cache(arrays, ref):
    meta = json.loads(arrays["metadata_json"].item())
    if (meta.get("schema") != 2 or meta.get("data_sha256") != ref.label_digest or meta.get("pose_parent_sha256") != PARENT_SHA
            or meta.get("onnx_sha256") != MODEL_SHA or meta.get("keys_sha256") != json_digest(ref.keys)
            or meta.get("providers") != ref.parent_meta["providers"] or meta.get("frame_mapping") != CLOCK
            or meta.get("collector_sha256") not in {FIRST_COLLECTOR_SHA, file_hash(Path(__file__))}):
        raise ValueError("Stale or incompatible matched cache provenance")
    if keys_of(arrays) != ref.keys:
        raise ValueError("Matched cache window IDs or order differ")
    n = len(ref.keys)
    for field in IDENTITY_FIELDS:
        same(arrays[field], ref.parent[field][ref.parent_indices], field)
    same(arrays["xy"], ref.xy, "all 20 historical and 30 future positions")
    same(arrays["hist_bbox"], ref.boxes, "all 20 boxes")
    for field in POSE_FIELDS:
        expected = ref.parent[field][ref.parent_indices]
        if arrays[field].shape != (n, 20, *expected.shape[2:]):
            raise ValueError(f"Wrong history shape in {field}")
        same(arrays[field][:, -5:], expected, "parent suffix " + field)
    pose, inferred = arrays["hist_kpts"], arrays["pose_inferred"]
    valid_box = (arrays["hist_bbox"][:, :, 2:] > 0).all(axis=2)
    if (not np.isfinite(pose).all() or np.any((pose[..., 2] < 0) | (pose[..., 2] > 1))
            or (inferred & ~valid_box).any() or pose[~inferred].any()
            or not np.array_equal(inferred[:, :15], valid_box[:, :15])):
        raise ValueError("Inconsistent keypoints, confidence or inference mask")
    if meta.get("prefix_sha256") != prefix_digest(arrays):
        raise ValueError("Prefix pose or inference mask differs from the extracted cache fingerprint")
    clips = set(arrays["clip"].tolist())
    control = meta.get("zero_offset_controls", {})
    if (control.get("xy_atol_px") != XY_ATOL_PX or control.get("confidence_atol") != CONF_ATOL
            or control.get("allowed_confidence_0_2_crossings") != 0 or control.get("passed") is not True
            or set(control.get("clips", {})) != clips):
        raise ValueError("Missing matched original-frame controls for the retained clips")
    for clip in clips:
        raw, selection = control_batch(ref.parent, clip)
        check_stats(control["clips"][clip], raw, len(selection[0]), clip)
    return meta


def prefix_digest(arrays):
    digest = hashlib.sha256()
    for field in ("hist_kpts", "pose_inferred"):
        digest.update(np.ascontiguousarray(arrays[field][:, :15]).tobytes())
    return digest.hexdigest()


def collect(data, model_path, pilot_pose, cache_path):
    ref = load_reference(data, pilot_pose)
    if cache_path.exists():
        validate_cache(read_arrays(cache_path), ref)
        return
    if file_hash(model_path) != MODEL_SHA:
        raise ValueError("Unexpected RTMPose ONNX digest")
    reader = RTMPoseONNXExtractor(onnx_path=model_path, batch_size=32)
    if reader.sess is None or reader.active_providers != ref.parent_meta["providers"]:
        raise RuntimeError(f"Frozen providers {ref.parent_meta['providers']} differ from current {reader.active_providers}")
    cv2.setNumThreads(1)
    import onnxruntime as ort
    old = ref.parent
    arrays = {field: old[field][ref.parent_indices] for field in IDENTITY_FIELDS}
    arrays["xy"] = ref.xy
    for field in POSE_FIELDS:
        suffix = old[field][ref.parent_indices]
        arrays[field] = np.zeros((len(ref.keys), 20, *suffix.shape[2:]), dtype=suffix.dtype)
        arrays[field][:, 15:] = suffix
    controls = {}
    clip_names = sorted(set(arrays["clip"].tolist()))
    for i, clip in enumerate(clip_names, 1):
        condition, suffix = clip.rsplit("_", 1)
        labels = data / "labels" / (clip + ".txt")
        video = data / "videos" / condition / "videos" / ("ori_" + suffix + ".mp4")
        if file_hash(video) != ref.parent_meta["video_sha256"][clip]:
            raise ValueError(f"Video identity mismatch: {clip}")
        calib = json.loads((data / "videos" / condition / "calib.json").read_text())
        controls[clip] = zero_control(video, clip, old, reader)
        indices = np.flatnonzero(arrays["clip"] == clip)
        requests = []
        for j in indices:
            _, person, anchor = ref.keys[j]
            for start in (0, 5, 10):
                end = start + 4
                # build_clip uses fixed only for its transient ego type; no prefix ego type enters the result.
                requests.append(TrackWindow(clip, person, anchor - 19 + end,
                                            ref.xy[j, start:end+1] - ref.xy[j, end], 0))
        prefix = build_clip(labels, calib, video, requests, reader)
        for field in POSE_FIELDS:
            arrays[field][indices, :15] = prefix[field].reshape((len(indices), 15, *prefix[field].shape[2:]))
        print(f"{i}/{len(clip_names)} {clip}: original control passed; {len(indices)} matched windows", flush=True)
    arrays["metadata_json"] = np.array(json.dumps({
        "schema": 2, "pose_parent_sha256": PARENT_SHA, "onnx_sha256": MODEL_SHA, "data_sha256": ref.label_digest,
        "keys_sha256": json_digest(ref.keys), "providers": list(reader.active_providers), "frame_mapping": CLOCK,
        "collector_sha256": file_hash(Path(__file__)), "prefix_sha256": prefix_digest(arrays),
        "onnxruntime_version": ort.__version__, "opencv_version": cv2.__version__,
        "zero_offset_controls": {"xy_atol_px": XY_ATOL_PX, "confidence_atol": CONF_ATOL,
                                 "allowed_confidence_0_2_crossings": 0, "passed": True, "clips": controls},
    }, sort_keys=True))
    validate_cache(arrays, ref)
    save_npz(cache_path, arrays)


def tensors_for(arrays, duration):
    if duration not in DURATIONS:
        raise ValueError(f"Unexpected observation duration: {duration}")
    history = arrays["xy"][:, 20 - duration:20]
    observed = np.ones(duration, dtype=np.float32)
    motion = np.stack([build_motion_features(pos, observed, dt=DT) for pos in history])
    poses = [build_pose_and_quality_features(k[-duration:], box[-duration:], observed, float(height))
             for k, box, height in zip(arrays["hist_kpts"], arrays["hist_bbox"], arrays["img_h"])]
    pose, quality = (np.stack([v[i] for v in poses]) for i in (0, 1))
    one_hot = np.eye(8, dtype=np.float32)[arrays["ego_type"]]
    future = arrays["xy"][:, 20:].astype(np.float32)
    return {k: torch.from_numpy(v) for k, v in {
        "x_m": motion, "x_p": pose, "q_vec": quality, "ego_type_oh": one_hot, "fut_xy": future,
        "fut_mask": np.ones((len(history), T_FUT), dtype=np.float32),
    }.items()}


def fit_and_score(data, cache, out, replay_only, pilot_pose):
    ref = load_reference(data, pilot_pose)
    arrays = read_arrays(cache)
    validate_cache(arrays, ref)
    keys, index = ref.keys, ref.index
    if not replay_only and not Path("/kaggle/working").is_dir():
        raise RuntimeError("Matched-duration training is enabled only on Kaggle; use --check-only or --replay-only locally")
    torch.set_num_threads(2)
    protocol = Protocol()
    device = torch.device("cuda:0")
    if not replay_only:
        configure_training(device, "2.11.0+cu130")
    base = {"protocol": asdict(protocol), "cache_sha256": file_hash(cache), "keys_sha256": json_digest(keys),
            "runner_sha256": file_hash(Path(__file__))}
    metrics, max_diff, per_window = {}, 0.0, {}
    for duration in DURATIONS:
        tensors = tensors_for(arrays, duration)
        metrics[duration] = {}
        for site, ids in index.items():
            training_sites, train, test = split_indices(index, keys, site)
            metrics[duration][site] = {}
            for variant in MODELS:
                identity = {**base, "duration": duration, "site": site, "model": variant, "train_sites": training_sites,
                            "train_ids_sha256": json_digest([keys[i] for i in train]), "test_ids_sha256": json_digest([keys[i] for i in test])}
                path = out / f"T{duration}/{site}/{variant}.pt"
                if not path.exists() and not replay_only:
                    print(f"Training T={duration}, hold out {site}, model {variant}, train {len(train)} test {len(ids)}", flush=True)
                    state, errors = fit_predict(variant, tensors, train, test, device, protocol)
                    atomic_write(path, lambda tmp: torch.save({"identity": identity, "model_state": state,
                                                               "errors": torch.from_numpy(errors)}, tmp))
                ck = torch.load(path, map_location="cpu", weights_only=True)
                if ck["identity"] != identity:
                    raise ValueError(f"Stale checkpoint: {path}")
                replay = run_cpu_forward(UnstructuredShortPosePredictor, variant, ck["model_state"], tensors, test, protocol.batch_size)
                cloud = ck["errors"].numpy()
                np.testing.assert_allclose(replay, cloud, atol=ATOL_M, rtol=RTOL)
                max_diff = max(max_diff, float(np.max(np.abs(replay - cloud))))
                metrics[duration][site][variant] = mean_metrics(cloud.astype(np.float64))
                per_window[duration, site, variant] = cloud.astype(np.float64)
    macro = {duration: {variant: {metric: float(np.mean([row[variant][metric] for row in sites.values()]))
                                 for metric in ("ade_m", "fde_m")} for variant in MODELS} for duration, sites in metrics.items()}
    differences = {duration: {site: mean_metrics(per_window[duration, site, "M1"] - per_window[duration, site, "M0"])
                              for site in index} for duration in DURATIONS}
    return {**base, "cohort": "Same track ordered anchors with complete 20+30 history/future, at least 50 ticks apart. Four previously studied sites.",
            "windows": len(keys), "site_counts": {s: len(v) for s,v in index.items()},
            "pose_inferred_by_duration": {t: int(arrays["pose_inferred"][:, -t:].sum()) for t in DURATIONS},
            "models_replayed": len(per_window), "max_cpu_vs_cloud_abs_m": max_diff, "site_metrics": metrics,
            "four_site_macro": macro, "paired_m1_minus_m0": differences}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data", type=Path)
    p.add_argument("pilot_pose", type=Path)
    p.add_argument("rtmpose_onnx", type=Path)
    p.add_argument("out", type=Path)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--replay-only", action="store_true")
    mode.add_argument("--check-only", action="store_true", help="Verify the frozen source and existing cache without pose extraction or training.")
    args = p.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "matched_observations.npz"
    if args.check_only:
        ref = load_reference(args.data, args.pilot_pose)
        if cache.exists():
            validate_cache(read_arrays(cache), ref)
        print(f"Verified {len(ref.keys)} matched 20+30 windows across {len(ref.index)} sites; existing cache checked: {cache.exists()}")
        return
    if not args.replay_only and not Path("/kaggle/working").is_dir():
        raise RuntimeError("Use --check-only or --replay-only locally; extraction and training run on Kaggle")
    if not cache.exists() and not args.replay_only:
        collect(args.data, args.rtmpose_onnx, args.pilot_pose, cache)
    report = fit_and_score(args.data, cache, out, args.replay_only, args.pilot_pose)
    if args.replay_only:
        print(json.dumps(report, indent=2))
        return
    save_json(out / "report.json", report)
    files = [out / "report.json", cache, *sorted(out.glob("T*/*/*.pt"))]
    if len(files) != 26:
        raise ValueError("Expected 24 checkpoints, observation cache and report")
    save_json(out / "sha256.json", {f.relative_to(out).as_posix(): file_hash(f) for f in files})
    def write_bundle(tmp):
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
            for f in [*files, out / "sha256.json"]:
                z.write(f, f.relative_to(out).as_posix())
    atomic_write(out / "matched_durations.zip", write_bundle)
    print(json.dumps(report["four_site_macro"]), flush=True)


if __name__ == "__main__":
    main()
