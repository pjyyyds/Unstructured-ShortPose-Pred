"""Nested site-held-out single-trajectory residual ridge on five historical PINNS observations.

Use --design-only with --public to check identities, features and splits without fitting or scoring.
For the experiment also supply --private for the coefficients, transforms and keyed forecasts.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np

from scripts.analyze_heldout_pairs import BOOTSTRAP_SEED, SITE_COUNTS, intervals, metric_dict, paired_clip_draws
from scripts.benchmark_heldout_sites import HISTORY, load_site_windows
from scripts.compare_pinns_velocity_estimators import predictions
from scripts.evaluate_pinns_action_probes import NAMES, history_features
from scripts.train_pinns_heldout_sites import atomic_write, file_hash, save_json, source_hash
from src.data.constants import DT, T_FUT
from src.data.features import build_motion_features, build_pose_and_quality_features

ROOT = Path(__file__).resolve().parents[1]
ANCHOR = "last_two"
GRID = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0)
RIDGE = 0.1
LOW_SPEED = 0.20
PROTOCOL = {
    "scope": "Single-trajectory linear development prototype at four previously examined sites, separate from the multimodal neural models. "
    "The 5 consecutive observations span 0.4 s; all 30 future points through 3 s are scored.",
    "anchor_and_frame": "Last-two-point constant velocity (0.1 s gap). Rotate ground motion and target residuals into the "
    "last velocity heading. Below 0.20 m/s use the fixed global +x heading. Pose stays in image axes.",
    "features": "Fixed nested arms: motion; motion + 17 final box-height-normalized relative joint xy/valid flags; "
    "both plus four ordered same-root, pair-valid, signed joint changes with flags. All windows retained. "
    "Pose gains include differences in joint availability and detector noise.",
    "fit": "One training-window-weighted ridge per arm, predicting all 30 local residual xy pairs. Mean squared loss + "
    "0.1 times squared coefficient norm, unpenalized intercept. Training-only means and population SD, "
    "divide by 1 when SD <= 1e-10. No outer-held data enters fitting or selection.",
    "shrinkage": "Within three outer-training sites, leave each site out and fit on the other two. "
    "For each grid value, average ADE equally over three arms and three inner sites, windows equal within each. "
    "Take the smallest grid value attaining the minimum, refit on three training sites and use the same "
    "multiplier in all arms on outer test. Zero is exactly the original constant-velocity forecast.",
    "metrics": "Primary: ordered pose minus last-two constant velocity, equal-site mean ADE. Also report FDE and paired "
    "increments (ordered versus motion, ordered versus last pose, last pose versus motion). Negative favors "
    "the first method. Primary interval: 20,000 paired within-site whole-clip draws, PCG64 seed 2026, "
    "observed windows weighted within site, sites weighted equally. Conditional on four development sites "
    "and the selected fits. The zero candidate does not guarantee lower held-site error.",
}


def frames_from_history(history):
    step = history[:, -1] - history[:, -2]
    speed = np.linalg.norm(step, axis=1) / DT
    moving = speed >= LOW_SPEED
    forward = np.tile([1.0, 0.0], (len(step), 1))
    forward[moving] = step[moving] / np.linalg.norm(step[moving], axis=1)[:, None]
    left = np.column_stack((-forward[:, 1], forward[:, 0]))
    return np.stack([forward, left], axis=2), moving


def make_features(history, pose):
    frames, moving = frames_from_history(history)
    local = history @ frames
    observed = np.ones(HISTORY, dtype=np.float32)
    motion = np.stack([build_motion_features(h, observed) for h in local])
    features, _ = history_features(motion, pose, local)
    return {k: v.astype(np.float64) for k, v in features.items()}, frames, moving


def load_inputs(data_root, pose_path):
    replay = json.loads((ROOT / "results/pinns_heldout_replay.json").read_text())
    groups, _, raw_sha = load_site_windows(data_root)
    keys = [(w.clip, w.person_id, w.t0) for s in sorted(groups) for w in groups[s]]
    key_sha = hashlib.sha256(json.dumps(keys, separators=(",", ":")).encode()).hexdigest()
    pose_sha = file_hash(pose_path)
    if (raw_sha, key_sha, pose_sha) != (
        replay["label_and_calibration_sha256"],
        replay["ordered_window_ids_sha256"],
        replay["pose_sha256"],
    ):
        raise ValueError("Expected frozen labels, ordered keys and pilot pose cache")
    if {s: len(rows) for s, rows in groups.items()} != SITE_COUNTS or len(set(keys)) != len(keys):
        raise ValueError("Unexpected site counts or duplicate keys")
    sites = np.array([s for s in sorted(groups) for _ in groups[s]])
    xy = np.stack([w.xy for s in sorted(groups) for w in groups[s]])
    if xy.shape != (1482, HISTORY + T_FUT, 2) or not np.isfinite(xy).all() or np.any(xy[:, HISTORY - 1]):
        raise ValueError("Expected five complete origin-centered observations and 30 targets for each window")
    history = xy[:, :HISTORY]
    with np.load(pose_path, allow_pickle=False) as cache:
        cache_keys = list(zip(cache["clip"].tolist(), cache["person_id"].tolist(), cache["t0"].tolist()))
        lookup = {k: i for i, k in enumerate(cache_keys)}
        if len(lookup) != len(cache_keys) or set(keys) != set(lookup):
            raise ValueError("Source/cache window keys disagree")
        order = [lookup[k] for k in keys]
        np.testing.assert_allclose(cache["hist_pos_relative"][order], history, rtol=0, atol=1e-8)
        if not np.all(cache["hist_mask"] == 1):
            raise ValueError("Incomplete motion observation")
        kpts, boxes, masks, heights = [
            cache[n][order] for n in ("hist_kpts", "hist_bbox", "hist_mask", "img_h")
        ]
        pose = np.stack(
            [build_pose_and_quality_features(k, b, m, h)[0] for k, b, m, h in zip(kpts, boxes, masks, heights)]
        )
    features, frames, moving = make_features(history, pose)
    if not all(np.isfinite(v).all() for v in features.values()):
        raise ValueError("Nonfinite history feature")
    inputs = {
        "features": features,
        "frames": frames,
        "moving": moving,
        "keys": keys,
        "sites": sites,
        "clips": np.array([k[0] for k in keys]),
        "truth": xy[:, HISTORY:],
        "anchor": predictions(history)[ANCHOR],
    }
    dependencies = (
        "scripts/evaluate_pinns_action_probes.py",
        "scripts/analyze_heldout_pairs.py",
        "scripts/compare_pinns_velocity_estimators.py",
        "scripts/evaluate_pinns_pose_residuals.py",
    )
    identity = {
        "labels_and_calibration_sha256": raw_sha,
        "ordered_key_sha256": key_sha,
        "pose_sha256": pose_sha,
        "frozen_neural_source_and_reader_sha256": source_hash(),
        "script_sha256": {n: file_hash(ROOT / n) for n in dependencies},
    }
    if identity["frozen_neural_source_and_reader_sha256"] != replay["frozen_source_sha256"]:
        raise ValueError("Frozen feature builders or window reader changed")
    return inputs, identity


def ridge_fit(x, y):
    mean, scale = x.mean(0), x.std(0)
    scale[scale <= 1e-10] = 1.0
    z = (x - mean) / scale
    target = y.reshape(len(y), -1)
    intercept = target.mean(0)
    weights = np.linalg.solve(z.T @ z + len(z) * RIDGE * np.eye(x.shape[1]), z.T @ (target - intercept))
    return dict(mean=mean, scale=scale, intercept=intercept, weights=weights)


def predict_world_residual(fit, x, frames):
    local = ((x - fit["mean"]) / fit["scale"]) @ fit["weights"] + fit["intercept"]
    return local.reshape(len(x), T_FUT, 2) @ frames.transpose(0, 2, 1)


def errors(forecast, truth):
    distances = np.linalg.norm(forecast - truth, axis=2)
    return np.column_stack([distances.mean(1), distances[:, -1]])


def selected_index(inner_cost):
    if (
        inner_cost.ndim != 3
        or inner_cost.shape != (3, len(NAMES), len(GRID))
        or not np.isfinite(inner_cost).all()
    ):
        raise ValueError("Expected complete validation costs from three training sites and three arms")
    return int(np.argmin(inner_cost.mean(axis=(0, 1))))


def outer_fold(inputs, held):
    sites, anchor, frames, truth = (inputs[k] for k in ("sites", "anchor", "frames", "truth"))
    training = tuple(s for s in sorted(set(sites)) if s != held)
    train, test = np.flatnonzero(sites != held), np.flatnonzero(sites == held)
    if len(training) != 3 or not len(test):
        raise ValueError("Expected one held site and three training sites")
    # All fits and the inner scores use only rows inside these three sites.
    local_residual = (truth[train] - anchor[train]) @ frames[train]
    inner_cost = np.empty((len(training), len(NAMES), len(GRID)))
    stored = {"train_rows": train, "test_rows": test, "inner_validation_sites": np.array(training)}
    for j, val_site in enumerate(training):
        val = sites[train] == val_site
        val_rows, fit_rows = train[val], train[~val]
        stored[f"inner_{j}_training_rows"] = fit_rows
        stored[f"inner_{j}_validation_rows"] = val_rows
        for i, name in enumerate(NAMES):
            x = inputs["features"][name]
            fit = ridge_fit(x[fit_rows], local_residual[~val])
            for field, v in fit.items():
                stored[f"inner_{j}_{name}_{field}"] = v
            delta = predict_world_residual(fit, x[val_rows], frames[val_rows])
            for g, multiplier in enumerate(GRID):
                inner_cost[j, i, g] = errors(anchor[val_rows] + multiplier * delta, truth[val_rows])[
                    :, 0
                ].mean()
    chosen = selected_index(inner_cost)
    stored.update(inner_ade_by_site_arm_grid=inner_cost, selected_multiplier=np.array(GRID[chosen]))
    forecasts = {}
    for name, x in inputs["features"].items():
        fit = ridge_fit(x[train], local_residual)
        for field, v in fit.items():
            stored[f"final_{name}_{field}"] = v
        delta = predict_world_residual(fit, x[test], frames[test])
        forecasts[name] = anchor[test].copy() if GRID[chosen] == 0 else anchor[test] + GRID[chosen] * delta
    return test, forecasts, stored


def evaluate(inputs, identity, design_only=False):
    sites, clips, keys = [inputs[k] for k in ("sites", "clips", "keys")]
    distinct = sorted(set(sites))
    if len(distinct) != 4 or any(len(np.unique(sites[clips == c])) != 1 for c in set(clips)):
        raise ValueError("Expected four sites and no clip crossing sites")
    report = {
        "protocol": PROTOCOL,
        "identity": identity,
        "design_only": design_only,
        "grid": list(GRID),
        "ridge": RIDGE,
        "grid_arm_order": list(NAMES),
        "versions": {"numpy": np.__version__, "python": platform.python_version()},
        "feature_dimensions": {n: x.shape[1] for n, x in inputs["features"].items()},
        "windows": len(keys),
        "clips": len(set(clips)),
        "sites": {},
    }
    for s in distinct:
        test = sites == s
        report["sites"][s] = {
            "test_windows": int(test.sum()),
            "test_clips": len(set(clips[test])),
            "fixed_heading_windows": int((test & ~inputs["moving"]).sum()),
            "training_sites": [u for u in distinct if u != s],
        }
    if design_only:
        return report, None
    anchor, truth = inputs["anchor"], inputs["truth"]
    out = {n: np.full_like(anchor, np.nan) for n in NAMES}
    out[ANCHOR] = anchor
    private = {
        "identity_json": np.array(json.dumps(identity, sort_keys=True)),
        "site": sites,
        "clip": clips,
        "person_id": np.array([k[1] for k in keys]),
        "t0": np.array([k[2] for k in keys]),
        "truth": truth,
        "world_to_local_basis_columns": inputs["frames"],
        "moving_heading": inputs["moving"],
        "grid": np.array(GRID),
        "arm_names": np.array(NAMES),
        "outer_sites": np.array(distinct),
    }
    for name, x in inputs["features"].items():
        private["features_" + name] = x
    for j, s in enumerate(distinct):
        rows, forecasts, state = outer_fold(inputs, s)
        for name, f in forecasts.items():
            out[name][rows] = f
        private.update({f"outer_{j}_{k}": v for k, v in state.items()})
        report["sites"][s]["selected_multiplier"] = float(state["selected_multiplier"])
        report["sites"][s]["inner_ade_by_grid_m"] = (
            state["inner_ade_by_site_arm_grid"].mean(axis=(0, 1)).tolist()
        )
    if not all(np.isfinite(v).all() for v in out.values()):
        raise ValueError("Missing or nonfinite outer trajectory")
    scores = {n: errors(p, truth) for n, p in out.items()}
    contrasts = (
        (NAMES[2], ANCHOR),
        (NAMES[2], NAMES[0]),
        (NAMES[2], NAMES[1]),
        (NAMES[1], NAMES[0]),
        (NAMES[1], ANCHOR),
        (NAMES[0], ANCHOR),
    )
    for s in distinct:
        site = report["sites"][s]
        mask = sites == s
        site["mean_m"] = {n: metric_dict(d[mask].mean(0)) for n, d in scores.items()}
        site["paired_m"] = {
            a + "_minus_" + b: metric_dict((scores[a][mask] - scores[b][mask]).mean(0)) for a, b in contrasts
        }
    report["equal_site_macro_m"] = {
        n: metric_dict(np.mean([d[sites == s].mean(0) for s in distinct], axis=0)) for n, d in scores.items()
    }
    report["equal_site_paired_m"] = {
        a + "_minus_" + b: metric_dict(
            np.mean([(scores[a][sites == s] - scores[b][sites == s]).mean(0) for s in distinct], axis=0)
        )
        for a, b in contrasts
    }
    diff = scores[NAMES[2]] - scores[ANCHOR]
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    draws = [paired_clip_draws(diff[sites == s], clips[sites == s], rng) for s in distinct]
    report["primary_ordered_minus_anchor_percentile_95_m"] = intervals(np.mean(draws, axis=0))
    for name in out:
        private["prediction_" + name], private["errors_" + name] = out[name], scores[name]
    return report, private


def write_arrays(path, arrays):
    with path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("pose_npz", type=Path)
    parser.add_argument("--public", required=True, type=Path)
    parser.add_argument("--private", type=Path)
    parser.add_argument("--design-only", action="store_true")
    args = parser.parse_args()
    if args.design_only != (args.private is None):
        parser.error("Use --private only for an actual experiment, and provide it for every fit")
    targets = [p.resolve() for p in (args.public, args.private) if p is not None]
    if (
        len(set(targets)) != len(targets)
        or args.pose_npz.resolve() in targets
        or any(t == args.data_root.resolve() or args.data_root.resolve() in t.parents for t in targets)
    ):
        parser.error("Output paths must be distinct and outside the source inputs")
    inputs, identity = load_inputs(args.data_root, args.pose_npz)
    report, state = evaluate(inputs, identity, args.design_only)
    if state is not None:
        atomic_write(args.private, lambda temp: write_arrays(temp, state))
    save_json(args.public, report)
    print("Validated", report["windows"], "windows across", len(report["sites"]), "sites.", flush=True)
