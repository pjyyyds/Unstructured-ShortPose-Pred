"""Evaluate events implied by frozen PINNS forecasts, using the verified four-site artifacts."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.benchmark_heldout_sites import HISTORY, load_site_windows
from scripts.compare_pinns_velocity_estimators import predictions
from scripts.train_pinns_heldout_sites import Protocol, file_hash, save_json, source_hash, tensorize
from scripts.verify_heldout_bundle import ATOL_M, RTOL, check, errors_by_window, same_array
from src.data.actions import compute_action_labels
from src.data.constants import T_FUT
from src.models.predictor import UnstructuredShortPosePredictor


ROOT = Path(__file__).resolve().parents[1]
ACTIONS = ("start", "stop", "left", "right", "maintain")
METHODS = ("M0", "M1", "last_two", "ols_five")
OBSERVED = np.ones(HISTORY)
COMPLETE = np.ones(T_FUT)


def events(history, paths):
    """Derive five binary labels for each [window, trajectory mode] using identical event rules."""
    n, modes = paths.shape[:2]
    check(history.shape == (n, HISTORY, 2) and paths.shape == (n, modes, T_FUT, 2)
          and np.isfinite(paths).all(), "Invalid forecast shape or nonfinite trajectory")
    labels = np.zeros((n, modes, len(ACTIONS)), dtype=bool)
    for i, obs in enumerate(history):
        for j, path in enumerate(paths[i]):
            event, valid = compute_action_labels(obs, OBSERVED, path, COMPLETE)
            check(valid == 1., "A complete benchmark trajectory has an invalid event label")
            labels[i, j] = event
    return labels


def weighted_events(history, paths, mode_prob):
    labels = events(history, paths)
    probs = mode_prob.astype(np.float64)
    check(probs.shape == paths.shape[:2] and np.isfinite(probs).all() and (probs >= 0).all()
          and np.allclose(probs.sum(axis=1), 1., rtol=0, atol=1e-6), "Invalid trajectory probabilities")
    probs /= probs.sum(axis=1, keepdims=True)
    scores = (probs[..., None] * labels).sum(axis=1)
    # Every mode (or none) predicting an event is an exact score tie across windows.
    scores[labels.all(axis=1)] = 1.
    return scores


def class_scores(truth, score):
    """Non-interpolated AP; a threshold includes the entire group of tied scores."""
    positive = int(truth.sum())
    binary = score >= 0.5
    tp = int(np.count_nonzero(truth & binary))
    fp = int(np.count_nonzero(~truth & binary))
    fn = positive - tp
    f1, ap = None, None
    if positive:
        f1 = (2 * tp) / (2 * tp + fp + fn)
        order = np.argsort(-score, kind="stable")
        # Threshold boundaries are the exclusive ends of the equal-score blocks.
        ends = np.r_[np.flatnonzero(np.diff(score[order]) != 0) + 1, len(score)]
        cumulative_tp = np.cumsum(truth[order])[ends - 1]
        recall_incs = np.diff(np.r_[0, cumulative_tp]) / positive
        ap = float(np.sum(recall_incs * (cumulative_tp / ends)))
    return {"predicted_positive": int(binary.sum()), "tp": tp, "fp": fp, "fn": fn, "f1": f1, "ap": ap}


def summarize(truth, scores, clips):
    per_model = {}
    for method, score in scores.items():
        by_class = {name: class_scores(truth[:, j], score[:, j]) for j, name in enumerate(ACTIONS)}
        present = [m for m in by_class.values() if m["f1"] is not None]
        per_model[method] = {
            "per_class": by_class, "macro_f1": float(np.mean([m["f1"] for m in present])),
            "mAP": float(np.mean([m["ap"] for m in present])),
        }
    return {
        "windows": len(truth), "valid_windows": len(truth), "clips": len(set(clips)),
        "multilabel_windows": int((truth.sum(axis=1) > 1).sum()),
        "classes": {name: {"positive_windows": int(truth[:, j].sum()),
                           "clips_with_positives": len(set(clips[truth[:, j]]))}
                    for j, name in enumerate(ACTIONS)}, "methods": per_model,
    }


@torch.inference_mode()
def predict_neural(variant, state, tensors, indices):
    model = UnstructuredShortPosePredictor(variant).cpu().eval()
    model.load_state_dict(state, strict=True)
    inputs = ("x_m", "ego_type_oh") if variant == "M0" else ("x_m", "ego_type_oh", "x_p", "q_vec")
    paths, probs = [], []
    for rows in torch.as_tensor(indices).split(Protocol().batch_size):
        out = model({key: tensors[key][rows] for key in inputs})
        paths.append(out["trajs"].numpy())
        probs.append(out["mode_probs"].numpy())
    return np.concatenate(paths), np.concatenate(probs)


def evaluate(root, artifacts):
    replay = json.loads((ROOT / "results/pinns_heldout_replay.json").read_text())
    cloud = json.loads((artifacts / "folds/report.json").read_text())
    groups, _, digest = load_site_windows(root)
    pose = artifacts / "pose.npz"
    check(replay["label_and_calibration_sha256"] == digest == cloud["identity"]["data_sha256"], "Source data mismatch")
    check(source_hash() == cloud["identity"]["source_sha256"] == replay["frozen_source_sha256"], "Model source mismatch")
    check(file_hash(pose) == replay["pose_sha256"] == cloud["identity"]["pose_sha256"], "Pose cache mismatch")
    tensors, ids, keys, _, _, _ = tensorize(groups, pose)
    key_hash = hashlib.sha256(json.dumps(keys, separators=(",", ":")).encode()).hexdigest()
    check(replay["ordered_window_ids_sha256"] == key_hash == cloud["identity"]["ordered_ids_sha256"], "Window IDs differ")
    sites, all_truth, all_scores, all_clips, errors = {}, [], {m: [] for m in METHODS}, [], {}
    for site, windows in sorted(groups.items()):
        xy = np.stack([w.xy for w in windows])
        history, future = xy[:, :HISTORY], xy[:, HISTORY:]
        clips = np.array([w.clip for w in windows])
        truth = events(history, future[:, None])[:, 0]
        scores, errors[site] = {}, {}
        for name in ("M0", "M1"):
            path = artifacts / "folds" / site / (name + ".pt")
            check(file_hash(path) == replay["sites"][site]["checkpoint_sha256"][name], f"Wrong {site}/{name} weights")
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            check(ckpt["window_keys"] == [list(keys[i]) for i in ids[site]]
                  and ckpt["identity"]["training_sites"] == sorted(set(groups) - {site}), f"Incorrect {site} fold")
            paths, probs = predict_neural(name, ckpt["model_state"], tensors, ids[site])
            check(paths.shape[1] == 6, "Expected six forecast modes")
            scores[name] = weighted_events(history, paths, probs)
            top1 = paths[np.arange(len(paths)), probs.argmax(axis=1)]
            errors[site][name] = same_array(errors_by_window(top1, future), ckpt["errors"].numpy(),
                                             f"{site}/{name} archived top-mode ADE/FDE", atol=ATOL_M, rtol=RTOL)
        extrapolated = predictions(history)
        for name in ("last_two", "ols_five"):
            scores[name] = events(history, extrapolated[name][:, None])[:, 0].astype(float)
        sites[site] = summarize(truth, scores, clips)
        all_truth.append(truth)
        all_clips.append(clips)
        for method in METHODS:
            all_scores[method].append(scores[method])
        print(f"Evaluated {site}: {len(windows)} paired windows", flush=True)
    pooled = summarize(np.concatenate(all_truth), {m: np.concatenate(v) for m,v in all_scores.items()},
                       np.concatenate(all_clips))
    check(pooled["windows"] == 1482 and len(sites) == 4 and pooled["clips"] == 47, "Incomplete benchmark")
    return {
        "protocol": {
            "scope": "Post-hoc on four previously examined PINNS sites and eight frozen seed-42 networks. "
                     "Descriptive trajectory-implied events; M0/M1 have no trained action head.",
            "targets": "Use five historical ground points and 30 true future positions in the official compute_action_labels "
                       "rule (default speed/angle scales, 10 Hz). All windows complete. Start/stop can coexist with a turn.",
            "predictions": "Apply exactly the same rule to the observed history plus each predicted future. "
                           "For M0/M1 sum the probabilities of the six modes that imply each event. The two "
                           "constant-velocity forecasts have binary scores, giving their precision-recall curves fewer thresholds than the neural mixtures. No true future "
                           "is passed into forecasting or scoring the modes. No tuning, fitting, or threshold selection.",
            "metrics": "Use a fixed >=0.5 threshold per event. AP is non-interpolated, grouping all tied scores at "
                       "each unique threshold. Absent true positives make a class F1/AP null and exclude it from the "
                       "class macro, while its confusion counts remain. Equal-site averages weight the four within-site "
                       "class-macro metrics equally; pooled metrics instead concatenate all windows.",
        },
        "verification": {"frozen_bundle_sha256": replay["bundle_sha256"], "ordered_ids_sha256": key_hash,
                         "pose_sha256": replay["pose_sha256"], "source_data_sha256": digest,
                         "cpu_torch": str(torch.__version__), "numpy": np.__version__,
                         "replayed_top1_max_abs_error_m": errors},
        "sites": sites, "pooled": pooled,
        "equal_site_macro": {m: {k: float(np.mean([s["methods"][m][k] for s in sites.values()]))
                                 for k in ("macro_f1", "mAP")} for m in METHODS},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path, help="Registered PINNS metadata")
    parser.add_argument("artifacts", type=Path, help="Verified extracted bundle containing pose.npz and folds/")
    parser.add_argument("--output", type=Path, default=ROOT / "results/pinns_heldout_implied_actions.json")
    args = parser.parse_args()
    torch.set_num_threads(2)
    save_json(args.output, evaluate(args.data_root, args.artifacts))
