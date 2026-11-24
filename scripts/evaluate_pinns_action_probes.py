"""Cross-validate history-only linear action probes on the 47 previously examined PINNS clips.

From the repository root:
    python -m scripts.evaluate_pinns_action_probes PINNS_ROOT pose.npz --output action_probes.json
Add --design-only to verify input identity, folds, features and event counts without fitting.
Only labels/, calibrations (and videos or archive_members.txt) and the archived pose.npz are needed.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform

import cv2
import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import expit
import torch

from scripts.benchmark_heldout_sites import HISTORY, load_site_windows
from scripts.evaluate_heldout_implied_actions import ACTIONS, class_scores, events, summarize
from scripts.train_pinns_heldout_sites import file_hash, save_json, tensorize
from src.data.actions import _estimate_causal_velocities

ROOT = Path(__file__).resolve().parents[1]
N_FOLDS = 5
C = 0.1
NAMES = ("motion", "motion_last_pose", "motion_last_pose_ordered_changes")
PRECONDITIONS = ("start", "stop")


def assign_folds(sites, clips):
    """Balance windows per physical site without seeing outcomes; entire clips are indivisible."""
    assignment = np.full(len(clips), -1, dtype=int)
    if any(len(np.unique(sites[clips == clip])) != 1 for clip in set(clips)):
        raise ValueError("Clip names must be globally unique across sites")
    total_load = [0] * N_FOLDS
    for site in sorted(set(sites)):
        ids = np.flatnonzero(sites == site)
        counts = Counter(clips[ids].tolist())
        if len(counts) < N_FOLDS:
            raise ValueError(f"{site} has fewer than {N_FOLDS} clips")
        loads, clip_counts = [0] * N_FOLDS, [0] * N_FOLDS
        for clip, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            fold = min(range(N_FOLDS), key=lambda f: (loads[f], total_load[f], clip_counts[f], f))
            assignment[ids[clips[ids] == clip]] = fold
            loads[fold] += count
            total_load[fold] += count
            clip_counts[fold] += 1
    if np.any(assignment < 0):
        raise ValueError("A sample was never assigned to a fold")
    return assignment


def history_features(x_m, x_p, history):
    """Use only five observed frames; keep the two pose representations nested."""
    observed = np.ones(HISTORY, dtype=bool)
    v_pre = np.array([_estimate_causal_velocities(x, observed)[2][-3:].mean() for x in history])
    eligible = np.column_stack([v_pre <= 0.20, v_pre >= 0.60])
    motion = np.column_stack([x_m[:, :, :5].reshape(len(x_m), -1), v_pre, eligible])
    # Normalized root-relative (u,v); masks distinguish invalid joints from true zeros.
    xy, valid, hip = x_p[..., :2], x_p[..., 3] > 0.5, x_p[..., 4] > 0.5
    last = np.concatenate([xy[:, -1], valid[:, -1, :, None]], axis=-1)
    static = np.column_stack([motion, last.reshape(len(last), -1)])
    # Both frames require the same origin convention (hip center vs. box center).
    pairs = valid[:, :-1] & valid[:, 1:] & (hip[:, :-1] == hip[:, 1:])
    delta = np.where(pairs[..., None], np.diff(xy, axis=1), 0.)
    changes = np.concatenate([delta, pairs[..., None]], axis=-1)
    dynamic = np.column_stack([static, changes.reshape(len(changes), -1)])
    return dict(zip(NAMES, (motion, static, dynamic))), eligible


def standardize(x_train, x_test):
    mean, sd = x_train.mean(axis=0), x_train.std(axis=0)
    sd = np.where(sd > 0, sd, 1.)
    return (x_train - mean) / sd, (x_test - mean) / sd


def fit_scores(train, labels, test):
    """Fit five independent unweighted logistic models, sharing the training-only scaling."""
    x, xt = standardize(train, test)
    probabilities = np.empty((len(xt), len(ACTIONS)))
    fits = {}
    for j, name in enumerate(ACTIONS):
        y = labels[:, j].astype(np.float64)
        n_pos = int(y.sum())
        if n_pos == 0 or n_pos == len(y):
            probabilities[:, j] = n_pos / len(y)
            fits[name] = {"fit": "single-class training frequency", "iterations": 0}
            continue
        # Mean of the logistic loss plus ||w||²/(2*C*n); intercept is unpenalized.
        def objective(theta):
            w, intercept = theta[:-1], theta[-1]
            logits = x @ w + intercept
            residual = expit(logits) - y
            value = np.logaddexp(0., (1.-2.*y) * logits).mean() + (w @ w)/(2*C*len(y))
            gradient = np.r_[(x.T @ residual + w/C)/len(y), residual.mean()]
            return value, gradient
        initial = np.zeros(x.shape[1] + 1)
        initial[-1] = np.log(n_pos / (len(y) - n_pos))
        opt = minimize(objective, initial, method="L-BFGS-B", jac=True,
                       options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8})
        if not opt.success or not np.isfinite(opt.x).all():
            raise RuntimeError(f"{name}: logistic fit failed: {opt.message}")
        probabilities[:, j] = expit(xt @ opt.x[:-1] + opt.x[-1])
        fits[name] = {"fit": "L2 logistic", "iterations": int(opt.nit)}
    return probabilities, fits


def supports(truth, clips):
    return {"windows": len(clips), "clips": len(set(clips)), "events": {
        name: {"positive_windows": int(truth[:, j].sum()),
               "positive_clips": len(set(clips[truth[:, j]]))}
        for j, name in enumerate(ACTIONS)}}


def increments(methods):
    contrasts = ((NAMES[1], NAMES[0]), (NAMES[2], NAMES[1]))
    result = {}
    for larger, smaller in contrasts:
        per_event = {}
        for name in ACTIONS:
            a = methods[larger]["per_class"][name]
            b = methods[smaller]["per_class"][name]
            per_event[name] = {m: None if a[m] is None else a[m] - b[m] for m in ("ap", "f1")}
        result[f"{larger}_minus_{smaller}"] = per_event
    return result


def load_inputs(data_root, pose_path):
    replay = json.loads((ROOT / "results/pinns_heldout_replay.json").read_text())
    groups, _, source_digest = load_site_windows(data_root)
    pose_digest = file_hash(pose_path)
    if (source_digest, pose_digest) != (replay["label_and_calibration_sha256"], replay["pose_sha256"]):
        raise ValueError("Input hashes differ from the accepted PINNS metadata or pose cache")
    tensors, _, keys, _, _, _ = tensorize(groups, pose_path)
    key_digest = hashlib.sha256(json.dumps(keys, separators=(",", ":")).encode()).hexdigest()
    if key_digest != replay["ordered_window_ids_sha256"]:
        raise ValueError("Window keys differ from the accepted cohort")
    sites = np.array([s for s in sorted(groups) for _ in groups[s]])
    clips = np.array([k[0] for k in keys])
    xy = np.stack([w.xy for s in sorted(groups) for w in groups[s]])
    if (len(keys), len(set(clips)), len(groups)) != (1482, 47, 4):
        raise ValueError("Expected 1,482 windows, 47 clips and four sites")
    # Freeze the grouped design before extracting truth; no test event counts enter assignment.
    folds = assign_folds(sites, clips)
    features, eligible = history_features(tensors["x_m"].numpy(), tensors["x_p"].numpy(), xy[:, :HISTORY])
    truth = events(xy[:, :HISTORY], xy[:, None, HISTORY:])[:, 0]
    if np.any(truth[:, :2] & ~eligible) or any(not np.isfinite(x).all() for x in features.values()):
        raise ValueError("Unexpected start/stop eligibility or nonfinite historical features")
    ident = {"labels_and_calibration_sha256": source_digest, "pose_sha256": pose_digest,
             "ordered_window_ids_sha256": key_digest, "probe_script_sha256": file_hash(Path(__file__))}
    return keys, sites, clips, folds, features, eligible, truth, ident


PROTOCOL = {
    "scope": "Exploratory clip-grouped development cross-validation at four previously examined sites. "
             "Observable events in the next 30 frames (3 s), derived from historical and future ground tracks. "
             "Five one-vs-rest supervised probes per representation. No existing model weights are read or changed.",
    "folds": "Five fixed folds. Process sites in ascending name order. Within each site sort whole clips by descending window count then ascending name. "
             "Place each clip in the fold with fewest windows from that site, then fewest windows overall, then fewest clips from that site, "
             "then lowest fold ID. No event labels, scores, threshold optimization or class balancing in the split.",
    "features": "All models use the same 5 historical ground points: per-frame relative xy, vx, vy, speed, "
                "the event rule's mean pre-event speed and its start/stop eligibility flags. Last pose adds "
                "17 relative joint xy and valid flags at t0. Ordered changes add four successive 17-joint signed "
                "xy differences and pair-validity flags. A pair needs a valid joint with the same root convention "
                "at both endpoints. Invalid differences are zero. Pose coordinates are normalized by the respective "
                "box heights. The pose increments include availability information and detector noise.",
    "fit": "Same folds, unweighted samples, independent binary logistic models. Per-column mean and population "
           "SD estimated only on training windows; divisor 1 for zero SD. Fixed C=0.1, summed log loss plus "
           "||w||²/(2C), unpenalized intercept. SciPy L-BFGS-B, maxiter=2000, ftol=1e-12, gtol=1e-8. A "
           "single-class training event predicts its training frequency (0 or 1). No fit selects hyperparameters.",
    "metrics": "Each row receives one out-of-fold probability per event and representation. Window-pooled AP uses "
               "one threshold per unique score (ties grouped); F1 uses fixed >=0.5. Zero-positive evaluation "
               "events have null AP/F1, retain confusion counts and are excluded from that group's class macro. "
               "Fold and site metrics are descriptive. Each fold contains clips from every site.",
    "history_eligible": "Within each event's eligible windows, score the same trained models and a constant "
                        "baseline estimated only from eligible training rows in the respective fold. If none are "
                        "eligible in training, use its overall training event frequency. Eligibility uses only "
                        "the last three causal smoothed speeds: mean <=0.20 m/s for start, >=0.60 m/s for stop.",
}


def evaluate(data_root, pose_path, design_only=False):
    keys, sites, clips, fold_ids, features, eligible, truth, ident = load_inputs(data_root, pose_path)
    folds = {}
    report = {"protocol": PROTOCOL, "design_only": design_only, "verification": ident,
              "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                           "torch": str(torch.__version__), "opencv": cv2.__version__},
              "feature_dimensions": {name: x.shape[1] for name, x in features.items()},
              "cohort": supports(truth, clips), "folds": folds}
    out = {k: np.full(truth.shape, np.nan) for k in NAMES}
    rates = np.full((len(truth), len(PRECONDITIONS)), np.nan)
    for fold in range(N_FOLDS):
        test = fold_ids == fold
        train = ~test
        strata = {s: sorted(set(clips[(sites == s) & test].tolist())) for s in sorted(set(sites))}
        info = {"evaluation_clips_by_site": strata, "training": supports(truth[train], clips[train]),
                "evaluation": supports(truth[test], clips[test])}
        folds[str(fold)] = info
        if design_only:
            continue
        info["fits"] = {}
        for name, x in features.items():
            out[name][test], info["fits"][name] = fit_scores(x[train], truth[train], x[test])
        for j in range(len(PRECONDITIONS)):
            train_eligible = train & eligible[:, j]
            rates[test, j] = truth[train_eligible if train_eligible.any() else train, j].mean()
        info["scores"] = summarize(truth[test], {name: v[test] for name,v in out.items()}, clips[test])["methods"]
    if design_only:
        return report
    if any(not np.isfinite(v).all() for v in [*out.values(), rates]):
        raise ValueError("Missing out-of-fold predictions")
    report["pooled"] = summarize(truth, out, clips)
    report["pooled"]["increments"] = increments(report["pooled"]["methods"])
    report["sites"] = {s: summarize(truth[sites == s], {n:v[sites == s] for n,v in out.items()}, clips[sites == s])
                       for s in sorted(set(sites))}
    report["history_eligible"] = {}
    for j, name in enumerate(PRECONDITIONS):
        subset = eligible[:, j]
        models = {n:v[subset, j] for n,v in out.items()}
        models["training_eligible_prevalence"] = rates[subset, j]
        report["history_eligible"][name] = {
            "windows": int(subset.sum()), "clips": len(set(clips[subset])),
            "positive_windows": int(truth[subset, j].sum()),
            "methods": {n:class_scores(truth[subset, j], s) for n,s in models.items()}}
    report["out_of_fold"] = {"key_fields": ["clip", "person_id", "t0"], "actions": list(ACTIONS),
                             "window_keys": keys, "fold": fold_ids.tolist(), "truth": truth.astype(int).tolist(),
                             "probabilities": {n:v.tolist() for n,v in out.items()},
                             "start_stop_eligible": eligible.tolist(), "training_eligible_prevalences": rates.tolist()}
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data_root", type=Path)
    p.add_argument("pose_npz", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--design-only", action="store_true")
    args = p.parse_args()
    save_json(args.output, evaluate(args.data_root, args.pose_npz, args.design_only))
