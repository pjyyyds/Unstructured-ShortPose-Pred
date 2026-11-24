"""Reconstruct the archived PINNS five-frame test and evaluate first-order motion baselines.

Run from the repository root with python -m scripts.audit_pinns_short_window DATASET_ROOT.
For a metadata-only copy, supply --members with original ZIP member names, one per line.
The archived video release has labels/ and videos/ at its root.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2  # Required by the release's ground-coordinate handedness check.
import numpy as np
import torch

from src.data.constants import DT, FIXED_SEEDS, T_FUT
from src.data.features import build_motion_features
from src.data.pinns_extractor import (
    _assemble_sequence_windows,
    _check_right_handed_homography,
    _parse_pinns_label_file,
)
from src.data.splits import assign_site_clip_splits, temporal_buffer_split_indices
from src.metrics.metrics_and_stats import compute_sample_trajectory_metrics
from src.models.predictor import UnstructuredShortPosePredictor


METRICS = {"ade_m": "ADE_1", "fde_m": "FDE_1"}
MODELS = ("M0", "M1", "Ours_Full")
ROOT = Path(__file__).resolve().parents[1]


def reconstruct(dataset, members):
    folders = defaultdict(list)
    for label in sorted((dataset / "labels").glob("*.txt")):
        if label.name.lower().startswith("readme") or "_" not in label.stem:
            continue
        folder, index = label.stem.rsplit("_", 1)
        video = Path("videos") / folder / "videos" / f"ori_{index}.mp4"
        if members is not None and video.as_posix() not in members:
            continue
        if members is None and not (dataset / video).is_file():
            continue
        folders[folder].append((int(index), label))

    samples, sites, eligible_windows = [], {}, {}
    for folder, entries in sorted(folders.items()):
        calib = json.loads((dataset / "videos" / folder / "calib.json").read_text())
        site = f"{calib['location']}_{calib['scene']}"
        entries.sort()
        by_clip = assign_site_clip_splits([index for index, _ in entries])
        flip_y = _check_right_handed_homography(calib)
        for index, label in entries:
            sites[label.stem] = site
            if by_clip and by_clip[index][0] != "test":
                continue
            observations = _parse_pinns_label_file(label, flip_y, calib["y_min"], calib["y_max"])
            frames = np.array(sorted(observations))
            if len(frames) < 40:
                continue
            if by_clip:
                cutoff = frames[0] + by_clip[index][1]
                splits = {int(frame): "test" for frame in frames if frame >= cutoff}
            else:
                # The original code strides across the combined split anchors before filtering.
                splits = {int(frame): split for split, selected in temporal_buffer_split_indices(frames).items()
                          for frame in selected}
            if "test" not in splits.values():
                continue
            # Motion-only inference needs neither the video skeletons nor the scene map.
            windows = _assemble_sequence_windows(
                label.stem, observations, {}, [], splits, float(calib["v_size"]), window_stride=4
            )
            selected = [w for w in windows if w["split"] == "test"]
            samples.extend(selected)
            eligible_windows[label.stem] = len(selected)
    return samples, sites, eligible_windows


def errors(predictions, probabilities, truth, mask):
    measured = compute_sample_trajectory_metrics(predictions, probabilities, truth, mask)
    return {key: measured[metric] for key, metric in METRICS.items()}


def motion_baselines(samples, checkpoint):
    xm = np.stack([build_motion_features(s["hist_pos_20"][-5:], s["hist_mask_20"][-5:], dt=DT)
                   for s in samples])
    truth = np.stack([s["fut_xy"] for s in samples])
    mask = np.stack([s["fut_mask"] for s in samples])
    # build_motion_features computes final velocity from the last two valid frames and
    # their elapsed time, with zero velocity if no valid pair is available inside T=5.
    velocity = xm[:, -1, 2:4]
    offsets = np.arange(1, T_FUT + 1, dtype=np.float32) * DT
    cv = velocity[:, None, None, :] * offsets[None, None, :, None]
    probability = np.ones((len(samples), 1), dtype=np.float32)
    baselines = {
        "constant_velocity": errors(cv, probability, truth, mask),
        "stationary": errors(np.zeros_like(cv), probability, truth, mask),
    }

    model = UnstructuredShortPosePredictor("M0")
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    with torch.inference_mode():
        prediction = model({"x_m": torch.from_numpy(xm), "ego_type_oh": torch.from_numpy(
            np.stack([s["ego_type_oh"] for s in samples]))})
    seed42 = errors(prediction["trajs"].numpy(), prediction["mode_probs"].numpy(), truth, mask)
    return baselines, seed42, int(np.count_nonzero(xm[:, -1, 6] == 0))


def summarize(per_window, selected):
    count = int(np.count_nonzero(selected))
    if count == 0:
        return {"windows": 0, "models": {}}
    models = {
        name: {key: round(float(np.mean(values[selected], dtype=np.float64)), 6)
               for key, values in metrics.items()}
        for name, metrics in per_window.items()
    }
    for metrics in models.values():
        metrics["ade_minus_cv_m"] = round(metrics["ade_m"] - models["constant_velocity"]["ade_m"], 6)
    return {"windows": count, "models": models}


def equal_weight_ade(per_window, groups):
    """Average within each represented group first, giving every group the same weight."""
    observed = [groups == value for value in np.unique(groups)]
    return {name: round(float(np.mean([np.mean(values["ade_m"][sel], dtype=np.float64)
                                       for sel in observed])), 6)
            for name, values in per_window.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--members", type=Path, help="newline-separated original ZIP member names")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/PINNS_T5_M0_seed42.pt")
    parser.add_argument("--out", type=Path, default=ROOT / "results/pinns_motion_audit.json")
    args = parser.parse_args()
    members = set(args.members.read_text().splitlines()) if args.members else None
    samples, sites, eligible = reconstruct(args.dataset, members)
    master = json.loads((args.results / "master_experiment_report.json").read_text())
    expected_n = master["diagnostics"]["pinns"]["split_counts"]["test"]
    if len(samples) != expected_n:
        raise ValueError(f"Reconstructed {len(samples)} test windows, expected {expected_n}")
    # Use bounded CPU threading for the small, deterministic checkpoint comparison.
    torch.set_num_threads(1)
    per_window, seed42, fallback = motion_baselines(samples, args.checkpoint)
    clip = np.array([s["seq_id"] for s in samples])
    site = np.array([sites[c] for c in clip])
    transition = np.array([s["m_act"] > 0.5 and s["y_act"][:4].sum() > 0.5 for s in samples])
    complete_history = sum(int(np.all(s["hist_mask_20"][-5:] > 0.5)) for s in samples)
    valid_final_future = sum(int(s["fut_mask"][-1] > 0.5) for s in samples)
    selected_all = np.ones(len(samples), dtype=bool)
    checks = {}
    summaries = master["summaries"]
    for key, metric in METRICS.items():
        expected = summaries["PINNS_T5_M0"]["slices"]["Overall"][metric + "_seeds"][FIXED_SEEDS.index(42)]
        replay = float(np.mean(seed42[key], dtype=np.float64))
        np.testing.assert_allclose(replay, expected, rtol=0, atol=1e-4)
        checks[key] = {"checkpoint_replay": round(replay, 8), "archive_seed42": expected}

    with np.load(args.results / "paired_sample_errors.npz", allow_pickle=False) as archive:
        for model in MODELS:
            tag = f"PINNS_T5_{model}"
            per_window[model] = {}
            for key, metric in METRICS.items():
                values = archive[f"{tag}__{metric.lower().replace('_', '')}"]
                if values.shape != (expected_n,) or not np.isfinite(values).all():
                    raise ValueError(f"Invalid archived array: {tag} {metric}")
                # Transition membership depends on the reconstructed order. Also check
                # its stored moments, not just count and all-window means.
                for group, subset in [("Overall", selected_all), ("Transition", transition)]:
                    ref = summaries[tag]["slices"][group]
                    if int(subset.sum()) != int(ref["count"]):
                        raise ValueError(f"Subset membership count mismatch for {group}")
                    np.testing.assert_allclose(values[subset].mean(dtype=np.float64),
                                               ref[metric + "_mean"], rtol=0, atol=1e-5)
                per_window[model][key] = values

    out = {
        "protocol": {
            "experiment": "Post-hoc motion baselines on the existing V2 PINNS test split.",
            "history_frames": 5, "dt_s": DT, "future_steps": T_FUT, "anchor_stride_at_10hz": 4,
            "metric": "ADE across valid future positions. FDE at the last valid future position.",
            "cv": "Last two valid ground positions in the five-frame input, divided by their elapsed time. Zero if unavailable.",
            "learned_models": "Archived per-window top-1 errors averaged over five seeds. M0 seed 42 is separately replayed with motion features and its original static/moving ego type.",
            "sites": "Location and scene combined, with seasonal/weather folders pooled. Empty sites have no test estimate.",
            "comparison": "ade_minus_cv_m is model minus constant-velocity ADE, in metres. Lower is better.",
        },
        "verification": {"eligible_original_clips": len(sites),
                         "source_condition_folders": len({name.rsplit("_", 1)[0] for name in sites}),
                         "clips_with_evaluated_windows": len(np.unique(clip)), "seed42": checks,
                         "windows_with_all_five_history_frames": complete_history,
                         "windows_with_valid_future_at_3s": valid_final_future,
                         "windows_scored_at_earlier_last_valid_future": expected_n - valid_final_future,
                         "transition_windows_checked": int(transition.sum()),
                         "m0_ade_seed42_vs_archived_five_seed_per_window_pearson": round(float(
                             np.corrcoef(seed42["ade_m"], per_window["M0"]["ade_m"])[0, 1]), 6),
                         "static_ego_type_windows": sum(int(s["ego_type_oh"][2] > 0.5) for s in samples),
                         "velocity_unavailable_windows": fallback},
        "overall": summarize(per_window, selected_all),
        "equal_site_weight_ade_m": equal_weight_ade(per_window, site),
        "equal_clip_weight_ade_m": equal_weight_ade(per_window, clip),
        "by_site": {s: summarize(per_window, site == s) for s in sorted(set(sites.values()))},
        "by_clip_with_test_anchors": {c: summarize(per_window, clip == c) for c in sorted(eligible)},
    }
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(f"Verified {len(samples)} windows. Results written to {args.out}")


if __name__ == "__main__":
    main()
