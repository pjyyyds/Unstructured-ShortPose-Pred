"""Evaluate three held-out American clips with shifted image frames and fixed M1 weights."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from scripts.analyze_heldout_pairs import SITE_COUNTS, check_keys, metric_dict
from scripts.benchmark_heldout_sites import load_site_windows
from scripts.cache_pinns_frame_offset import CLIPS, DATA_SHA, MODEL_SHA
from scripts.train_pinns_heldout_sites import Protocol, file_hash, save_json, source_hash, tensorize
from src.metrics.metrics_and_stats import compute_sample_trajectory_metrics
from src.models.predictor import UnstructuredShortPosePredictor


AMERICA = "america_crossroad"
IDS_SHA = "f5ee511a54ad818499d3dde6213f7f3b49332943ff2fc93d93aed87d316dbb91"
SOURCE_SHA = "e61a2310793e900c6188588c750e47c44e23fbf41f429fb94ccf8b2ea88fd094"
ATOL, RTOL = 2e-4, 2e-5
M1_INPUTS = ("x_m", "ego_type_oh", "x_p", "q_vec")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def cache_keys(cache):
    return list(zip(cache["clip"].tolist(), cache["person_id"].tolist(), cache["t0"].tolist()))


def check_cache_changes(original, shifted, keys):
    """Compare every cache column by window ID. Only keypoints and inference flags may change."""
    with np.load(original, allow_pickle=False) as base, np.load(shifted, allow_pickle=False) as alt:
        require(set(base.files) == set(alt.files), "Pose NPZ fields differ")
        base_order = check_keys(cache_keys(base), keys)
        alt_order = check_keys(cache_keys(alt), keys)
        clips = np.array([key[0] for key in keys])
        selected = np.isin(clips, list(CLIPS))
        changed_frames = np.zeros((len(keys), 5), dtype=bool)
        for name in sorted(set(base.files) - {"metadata_json"}):
            a, b = base[name][base_order], alt[name][alt_order]
            require(a.dtype == b.dtype and a.shape == b.shape, f"{name}: cache shape or dtype changed")
            if name in ("hist_kpts", "pose_inferred"):
                require(np.array_equal(a[~selected], b[~selected]), f"{name}: change outside the three clips")
                difference = a != b
                if name == "hist_kpts":
                    difference = difference.any(axis=(-2, -1))
                changed_frames |= difference
            else:
                require(np.array_equal(a, b), f"{name}: non-pose input changed")
    return {clip: int(changed_frames[clips == clip].sum()) for clip in CLIPS}


def expected_control_batches(pose_path):
    """Return the original first, last and largest crop batches per clip, keyed by raw frame."""
    result = {}
    with np.load(pose_path, allow_pickle=False) as cache:
        clips, times, boxes = (cache[k] for k in ("clip", "t0", "hist_bbox"))
        for clip in CLIPS:
            mask = clips == clip
            subset = boxes[mask]
            usable = np.isfinite(subset).all(axis=-1) & (subset[..., 2:] > 0).all(axis=-1)
            raw = 3 * (times[mask, None] + np.arange(-4, 1))
            frames, counts = np.unique(raw[usable], return_counts=True)
            require(len(frames) > 0, f"{clip}: missing frozen pose crops")
            indices = {0, len(frames) - 1, int(counts.argmax())}
            result[clip] = {int(frames[i]): int(counts[i]) for i in indices}
    return result


def verify_pose_controls(provenance, frozen_providers, expected):
    control = provenance.get("zero_offset_control", {})
    details = provenance.get("clips", {})
    require(bool(frozen_providers) and provenance.get("providers") == frozen_providers,
            "Shifted extraction providers differ from the frozen providers")
    require(control.get("sampled_controls_passed") is True and control.get("xy_atol_px") == .001
            and control.get("confidence_atol") == .0001 and control.get("allowed_confidence_0_2_crossings") == 0,
            "Required zero-offset replay controls or tolerances did not pass")
    require(set(details) == set(expected) == set(CLIPS), "Zero-offset controls must cover exactly the three clips")
    for clip, batches in expected.items():
        actual = {}
        for record in details[clip].get("zero_offset_controls", []):
            frame, count = record["raw_frame"], record["crops"]
            require(type(frame) is int and type(count) is int and count > 0 and frame not in actual,
                    f"{clip}: invalid or duplicate control frame")
            actual[frame] = count
            require(0 <= record["xy_max_abs_px"] <= .001 and 0 <= record["confidence_max_abs"] <= .0001
                    and record["confidence_0_2_crossings"] == 0, f"{clip}: zero-offset control outside tolerance")
        require(actual == batches, f"{clip}: missing or incorrect control frames/crop counts")


def read_frozen_fold(directory, site, site_keys, identity):
    fold = json.loads((directory / "fold.json").read_text())
    train_sites = sorted(set(SITE_COUNTS) - {site})
    require(fold["identity"] == identity and fold["train_sites"] == train_sites
            and fold["error_columns"] == ["ADE1_m", "FDE1_m"], f"{site}: incorrect fold identity or metric order")
    fold_order = check_keys(fold["window_keys"], site_keys)
    path = directory / "M1.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    require(checkpoint["identity"] == {**identity, "model": "M1", "test_site": site, "training_sites": train_sites},
            f"{site}: checkpoint identity mismatch")
    ck_order = check_keys(checkpoint["window_keys"], site_keys)
    saved = np.asarray(fold["paired_window_errors"]["M1"], dtype=np.float64)
    check_errors = checkpoint["errors"].numpy()
    require(saved.shape == check_errors.shape == (len(site_keys), 2), f"{site}: incorrect saved error shape")
    saved = saved[fold_order]
    require(np.isfinite(saved).all() and np.array_equal(saved, check_errors[ck_order]),
            f"{site}: checkpoint and fold errors disagree")
    model = UnstructuredShortPosePredictor("M1").eval()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model, saved, file_hash(path)


@torch.inference_mode()
def evaluate(model, tensors, test, device):
    rows = []
    for idx in torch.as_tensor(test).split(Protocol().batch_size):
        pred = model({k: tensors[k][idx].to(device) for k in M1_INPUTS})
        trajs, probs = pred["trajs"].cpu().numpy(), pred["mode_probs"].cpu().numpy()
        require(np.isfinite(trajs).all() and np.isfinite(probs).all(), "Non-finite trajectory or mode score")
        metrics = compute_sample_trajectory_metrics(trajs, probs, tensors["fut_xy"][idx].numpy(),
                                                    tensors["fut_mask"][idx].numpy())
        rows.append(np.column_stack((metrics["ADE_1"], metrics["FDE_1"])))
    return np.concatenate(rows)


def verify_replay(replayed, saved, site):
    require(replayed.shape == saved.shape and np.isfinite(replayed).all(), f"{site}: invalid frozen replay")
    matches = np.isclose(replayed, saved, atol=ATOL, rtol=RTOL)
    if not matches.all():
        row, col = np.argwhere(~matches)[0]
        raise ValueError(f"{site}: frozen replay differs from cloud at window {row}, metric {col}; "
                         f"replayed={replayed[row, col]}, saved={saved[row, col]}")
    return metric_dict(np.max(np.abs(replayed - saved), axis=0))


def summarize(base, shifted):
    return {"windows": len(base), "frozen_replay_m": metric_dict(base.mean(axis=0)),
            "shifted_m": metric_dict(shifted.mean(axis=0)),
            "shifted_minus_frozen_m": metric_dict((shifted - base).mean(axis=0))}


def run(data_root, result_dir, shifted_path, device):
    folds = result_dir / "folds"
    original = result_dir / "pose.npz"
    report = json.loads((folds / "report.json").read_text())
    require(report["total_windows"] == sum(SITE_COUNTS.values()) and set(report["sites"]) == set(SITE_COUNTS),
            "Expected the complete four-site result")
    groups, _, digest = load_site_windows(data_root)
    require(digest == DATA_SHA and {s: len(v) for s, v in groups.items()} == SITE_COUNTS,
            "Official label/calibration or site count mismatch")
    tensors, index, keys, _, _, meta = tensorize(groups, original)
    key_hash = hashlib.sha256(json.dumps(keys, separators=(",", ":")).encode()).hexdigest()
    expected = {"source_sha256": SOURCE_SHA, "data_sha256": digest, "ordered_ids_sha256": key_hash,
                "protocol": asdict(Protocol()), "pose_sha256": file_hash(original)}
    identity = report["identity"]
    require(source_hash() == SOURCE_SHA and key_hash == IDS_SHA and all(identity[k] == v for k, v in expected.items()),
            "Frozen training code, pose cache, protocol, or ordered IDs differ")
    require(meta == report["pose_metadata"] and meta["model_sha256"] == MODEL_SHA
            and meta["label_and_calibration_digest_sha256"] == digest, "Frozen pose provenance differs")
    us_clips = np.array([keys[i][0] for i in index[AMERICA]])
    require({c: int((us_clips == c).sum()) for c in CLIPS} == CLIPS, "Unexpected affected test IDs")

    poses_changed = check_cache_changes(original, shifted_path, keys)
    altered, alt_index, alt_keys, _, _, alt_meta = tensorize(groups, shifted_path)
    require(keys == alt_keys and all(np.array_equal(index[s], alt_index[s]) for s in index),
            "Shifted tensor order or site membership differs")
    provenance = alt_meta["frame_offset_sensitivity"]
    require(provenance["parent_pose_sha256"] == expected["pose_sha256"] and alt_meta["frame_offset_raw"] == 1
            and sorted(alt_meta["offset_clips"]) == sorted(CLIPS), "Shifted cache lineage or offset differs")
    for field in ("model_sha256", "label_and_calibration_digest_sha256", "video_sha256", "boxes", "site_counts"):
        require(meta[field] == alt_meta[field], f"Shifted cache source differs: {field}")

    verify_pose_controls(provenance, meta["providers"], expected_control_batches(original))

    base_errors, replay_checks, weights = {}, {}, {}
    for site, test in index.items():
        model, saved, weights[site] = read_frozen_fold(folds / site, site, [keys[i] for i in test], identity)
        base_errors[site] = evaluate(model.to(device), tensors, test, device)
        replay_checks[site] = verify_replay(base_errors[site], saved, site)
        if site == AMERICA:
            america_model = model
    # All four baseline replays must pass before the shifted forward evaluation starts.
    us = evaluate(america_model, altered, index[AMERICA], device)
    affected = np.isin(us_clips, list(CLIPS))
    require(np.array_equal(base_errors[AMERICA][~affected], us[~affected]),
            "Predictions changed for untouched American windows")
    sites = {s: summarize(base, us if s == AMERICA else base) for s, base in base_errors.items()}
    macro = {col: {metric: float(np.mean([v[col][metric] for v in sites.values()]))
                   for metric in ("ade_m", "fde_m")}
             for col in ("frozen_replay_m", "shifted_m", "shifted_minus_frozen_m")}
    return {
        "scope": "Post-hoc test-input sensitivity of four fixed seed-42 M1 checkpoints. Only the American "
                 "test fold has shifted images. The three other folds reuse their verified baseline replay because "
                 "all their test cache inputs and weights are identical. Training inputs and weights are fixed.",
        "interpretation": "Positive shifted-minus-frozen error means worse prediction. A +1 decoded 30 Hz frame "
                          "uses video index equal to the raw annotation number on the three clips. This comparison "
                          "does not establish the correct indexing or measure retraining effects. Pose extraction "
                          "can also vary across backends. The zero-offset control samples only selected frames.",
        "source_sha256": SOURCE_SHA, "label_and_calibration_sha256": digest, "ordered_ids_sha256": key_hash,
        "original_pose_sha256": expected["pose_sha256"], "shifted_pose_sha256": file_hash(shifted_path),
        "m1_checkpoint_sha256": weights, "cloud_torch": identity["torch_version"],
        "replay_device": str(device), "replay_torch": str(torch.__version__),
        "frozen_replay_tolerance": {"atol_m": ATOL, "rtol": RTOL},
        "max_abs_frozen_replay_minus_saved_m": replay_checks, "original_pose_providers": meta["providers"],
        "shifted_extraction_metadata": provenance, "changed_keypoint_or_inference_frames": poses_changed,
        "clips": {c: summarize(base_errors[AMERICA][us_clips == c], us[us_clips == c]) for c in CLIPS},
        "three_clips_combined": summarize(base_errors[AMERICA][affected], us[affected]),
        "other_american_windows": summarize(base_errors[AMERICA][~affected], us[~affected]),
        "sites": sites, "equal_site_mean": macro,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path, help="Original PINNS labels and calibrations")
    parser.add_argument("result_dir", type=Path, help="Extracted cloud result containing pose.npz and folds/")
    parser.add_argument("shifted_cache", type=Path, help="Complete 1,482-window +1-frame pose NPZ")
    parser.add_argument("--output", type=Path, help="JSON result file (default stdout)")
    parser.add_argument("--threads", type=int, default=2, help="CPU inference threads")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Use one device for both replays")
    args = parser.parse_args()
    if args.threads < 1 or (args.device == "cuda" and not torch.cuda.is_available()):
        parser.error("A positive thread count and an available evaluation device are required")
    if args.output:
        target = args.output.resolve()
        frozen = args.result_dir.resolve()
        if (target.suffix.lower() != ".json" or target.is_relative_to(frozen / "folds")
                or target.is_relative_to(frozen / "logs") or target.is_relative_to(args.data_root.resolve())
                or target in (args.shifted_cache.resolve(), (frozen / "pose.npz").resolve())):
            parser.error("Choose a JSON output separate from the source data, caches, frozen folds and logs")
    if args.device == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(args.threads)
    result = run(args.data_root, args.result_dir, args.shifted_cache, torch.device(args.device))
    if args.output:
        save_json(args.output, result)
    else:
        print(json.dumps(result, indent=2, allow_nan=False))
