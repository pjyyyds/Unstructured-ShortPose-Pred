"""Five-frame pedestrian motion and annotated neighbors as predictors of the next 3 s.

CPU, run at the repository root:
    python -m scripts.evaluate_pinns_neighbor_residuals DATA_ROOT --output REPORT.json
    python -m scripts.evaluate_pinns_neighbor_residuals --pilot /mounted/shortpose_heldout.zip --output REPORT.json

Without DATA_ROOT, stage only the 47 official labels, 22 calibrations and archive
member list. --design-only checks inputs, clip folds and support without fitting.
"""
import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import zipfile

import cv2
import numpy as np

from scripts.analyze_heldout_pairs import (
    BOOTSTRAP_SEED, REPLICATES, SITE_COUNTS, intervals, metric_dict, paired_clip_draws,
)
from scripts.benchmark_heldout_sites import HISTORY, constant_velocity, load_site_windows
from scripts.evaluate_pinns_action_probes import N_FOLDS, assign_folds
from scripts.train_pinns_heldout_sites import save_json
from src.data.constants import DT, T_FUT
from src.data.pinns_extractor import _check_right_handed_homography, _parse_pinns_label_file


ROOT = Path(__file__).resolve().parents[1]
FOLD_DIGEST = "913707e55ffa19f0458d1540fe3e8c5044a6c4c02f847f75c2f0f1eded9b1c4b"
PILOT_REPORT_SHA = "a0dc58b81ea8c1ee7a067c29e5ba6eeba22d58718f919368717c439b1f7ceca7"
ALPHA = 0.1
RADIUS_M = 10.0
METHODS = ("motion", "motion_availability", "motion_neighbors")
CLASSES = ("Person", "Car")
AVAILABILITY = ("count_log1p", "present", "velocity_valid")
GEOMETRY = ("dx_m", "dy_m", "distance_m", "dvx_mps", "dvy_mps", "closing_mps")


def json_hash(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def check_pilot(path, data_hash, window_hash):
    if path.is_dir():
        raw = (path / "folds/report.json").read_bytes()
    else:
        with zipfile.ZipFile(path) as archive:
            raw = archive.read("folds/report.json")
    if hashlib.sha256(raw).hexdigest() != PILOT_REPORT_SHA:
        raise ValueError("Unexpected seed-42 report")
    identity = json.loads(raw)["identity"]
    if (identity["data_sha256"], identity["ordered_ids_sha256"]) != (data_hash, window_hash):
        raise ValueError("Seed-42 and neighbor experiment cohorts differ")


def track_frames(parsed):
    """Keep only typed identities and observed ground coordinates; fixed/type-derived status is not a feature."""
    frames = {}
    for time, objects in parsed.items():
        frame = {}
        for obj in objects:
            if obj["type"] not in CLASSES:
                raise ValueError(f"Unexpected source object type: {obj['type']}")
            key = (obj["type"], obj["id"])
            if key in frame:
                raise ValueError(f"Duplicate typed ID at frame {time}: {key}")
            xy = np.array([obj["wx"], obj["wy"]], dtype=float)
            if np.isfinite(xy).all():
                frame[key] = xy
        frames[time] = frame
    return frames


def slope(times, xy):
    t = (np.asarray(times, dtype=float) - times[-1]) * DT
    t -= t.mean()
    points = np.asarray(xy, dtype=float)
    return (t @ (points - points[-1])) / (t @ t)


def neighbor_features(history_frames, person, expected_history):
    """Take exactly five snapshots ending at t0. Selection/ranking never sees another frame."""
    if len(history_frames) != HISTORY:
        raise ValueError("Neighbor history requires exactly five frames")
    ego_key = ("Person", person)
    observed = np.stack([frame[ego_key] for frame in history_frames])
    np.testing.assert_allclose(observed - observed[-1], expected_history, rtol=0, atol=1e-10)
    ego_velocity = slope(range(HISTORY), observed)
    anchor = history_frames[-1]
    masks, geometry, counts = [], [], []
    for kind in CLASSES:
        nearby = []
        for key, xy in anchor.items():
            if key == ego_key or key[0] != kind:
                continue
            distance = float(np.linalg.norm(xy - observed[-1]))
            if distance <= RADIUS_M:
                nearby.append((distance, key[1]))
        n = len(nearby)
        counts.append(n)
        features = np.zeros(len(GEOMETRY))
        valid = False
        if nearby:
            distance, ident = min(nearby)  # Tie-break on the numeric, same-class ID.
            key = kind, ident
            delta = anchor[key] - observed[-1]
            features[:3] = [*delta, distance]
            times = [j for j, frame in enumerate(history_frames) if key in frame]
            if len(times) >= 2:
                valid = True
                velocity = slope(times, [history_frames[j][key] for j in times]) - ego_velocity
                features[3:5] = velocity
                if distance > 0:
                    features[5] = -(delta @ velocity) / distance
        masks.extend((np.log1p(n), int(bool(n)), int(valid)))
        geometry.extend(features.tolist())
    return masks, geometry, counts


@dataclass
class Cohort:
    keys: list
    sites: np.ndarray
    clips: np.ndarray
    folds: np.ndarray
    feature_sets: dict
    counts: np.ndarray
    velocity_valid: np.ndarray
    histories: np.ndarray
    futures: np.ndarray
    identity: dict


def load_cohort(data_root, pilot=None):
    groups, _, data_hash = load_site_windows(data_root)
    rows = [window for site in sorted(groups) for window in groups[site]]
    keys = [(w.clip, w.person_id, w.t0) for w in rows]
    sites = np.asarray([site for site in sorted(groups) for _ in groups[site]])
    clips = np.asarray([w.clip for w in rows])
    folds = assign_folds(sites, clips)
    mapping = {c: int(f) for c, f in zip(clips.tolist(), folds.tolist())}
    id_hash, fold_hash = json_hash(keys), json_hash(mapping)
    reference = json.loads((ROOT / "results/pinns_heldout_replay.json").read_text())
    if (dict(Counter(sites.tolist())) != SITE_COUNTS or len(set(keys)) != 1482 or len(mapping) != 47
            or fold_hash != FOLD_DIGEST or data_hash != reference["label_and_calibration_sha256"]
            or id_hash != reference["ordered_window_ids_sha256"]):
        raise ValueError("Source labels, ordered IDs or fixed clip allocation differ from the accepted development cohort")
    if pilot is not None:
        check_pilot(pilot, data_hash, id_hash)

    xy = np.stack([w.xy for w in rows])
    availability = np.empty((len(rows), len(CLASSES) * len(AVAILABILITY)))
    neighbors = np.empty((len(rows), len(CLASSES) * len(GEOMETRY)))
    counts = np.empty((len(rows), len(CLASSES)), dtype=int)
    for clip in sorted(mapping):
        condition = clip.rsplit("_", 1)[0]
        calib = json.loads((data_root / "videos" / condition / "calib.json").read_text())
        parsed = _parse_pinns_label_file(data_root / "labels" / (clip + ".txt"),
                                         _check_right_handed_homography(calib), calib["y_min"], calib["y_max"])
        snapshots = track_frames(parsed)
        for i in np.flatnonzero(clips == clip):
            window = rows[i]
            history = [snapshots[t] for t in range(window.t0 - HISTORY + 1, window.t0 + 1)]
            availability[i], neighbors[i], counts[i] = neighbor_features(history, window.person_id, xy[i, :HISTORY])

    motion = xy[:, :HISTORY - 1].reshape(len(rows), -1)
    features = dict(zip(METHODS, (motion, np.column_stack([motion, availability]),
                                 np.column_stack([motion, availability, neighbors]))))
    if not all(np.isfinite(x).all() for x in features.values()):
        raise ValueError("Nonfinite historical features")
    identity = {"label_and_calibration_sha256": data_hash, "ordered_window_ids_sha256": id_hash,
                "clip_to_fold_sha256": fold_hash, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "pilot_report_sha256_verified": PILOT_REPORT_SHA if pilot is not None else None}
    identity["implementation_sha256"] = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
            "scripts/benchmark_heldout_sites.py", "scripts/evaluate_pinns_action_probes.py",
            "scripts/analyze_heldout_pairs.py", "src/data/pinns_extractor.py", "src/data/constants.py",
        )
    }
    valid = availability[:, [2, len(AVAILABILITY) + 2]].astype(bool)
    if np.any(valid & (counts == 0)):
        raise ValueError("A missing neighbor cannot have a measured velocity")
    return Cohort(keys, sites, clips, folds, features, counts, valid, xy[:, :HISTORY], xy[:, HISTORY:], identity)


def ridge_residual(x, train_residuals, x_held):
    """Solve mean squared ridge per target; the intercept and all scalers are training-only."""
    center, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale > 1e-10, scale, 1.)
    train = (x - center) / scale
    y = train_residuals.reshape(len(x), -1)
    intercept = y.mean(axis=0)
    weights = np.linalg.solve(train.T @ train + len(x) * ALPHA * np.eye(x.shape[1]), train.T @ (y - intercept))
    residual = intercept + ((x_held - center) / scale) @ weights
    return residual.reshape(len(x_held), T_FUT, 2)


def summarize(cohort, mask, scores=None):
    subset = cohort.counts[mask] > 0
    out = {"windows": int(mask.sum()), "clips": len(set(cohort.clips[mask])),
           "with_person": int(subset[:, 0].sum()), "with_car": int(subset[:, 1].sum()),
           "with_both": int(subset.all(axis=1).sum()), "without_neighbor": int((~subset.any(axis=1)).sum())}
    for i, kind in enumerate(CLASSES):
        present = int(subset[:, i].sum())
        valid = int(cohort.velocity_valid[mask, i].sum())
        out[kind.lower() + "_nearest_velocity"] = {
            "valid_windows": valid, "share_of_present": valid / present if present else None,
        }
    if scores and mask.any():
        out["means_m"] = {name: metric_dict(values[mask].mean(axis=0)) for name, values in scores.items()}
        out["paired_m"] = {
            name + "_minus_" + baseline: metric_dict((scores[name][mask] - scores[baseline][mask]).mean(axis=0))
            for name, baseline in ((METHODS[2], METHODS[0]), (METHODS[2], METHODS[1]), (METHODS[1], METHODS[0]))
        }
    return out


PROTOCOL = {
    "scope": "Five-fold whole-clip out-of-fold development at four previously examined PINNS sites. Official annotated "
             "positions/typed IDs for ego and neighbors; this is an oracle-tracking feature experiment on previously seen "
             "locations. No neural model, maps, pose, fixed/moving label, scene ID or image input. No test-set tuning.",
    "windows": "Disjoint complete 35-frame blocks per clip/person at 10 Hz. Five past points span 0.4 s. Score 30 future "
               "steps through +3 s. Neighbor absence never removes a window. Synchronized clips or the same person across "
               "different clip IDs cannot be identified from these public annotations.",
    "folds": "Same five previously frozen, site-stratified whole-clip folds as the action probes. "
             "Assignments depend only on site, clip name and window count. No event or residual enters allocation.",
    "features": "Motion: eight ground coordinates of the first four observations relative to the fifth. For each of "
                "Person and Car within 10 m inclusive of ego at t0, availability adds log(1+count), presence and usable "
                "nearest-neighbor velocity. Nearest ranks use t0 distance then lowest numeric ID. Geometry adds nearest "
                "dx,dy,distance,dvx,dvy,closing speed. Relative velocity subtracts ego five-point OLS from the neighbor's "
                "OLS over its available points within those same five frames (minimum two points); positive closing means "
                "approaching. Missing entries and coincident-position radial speed are zero, with channel masks as above. "
                "All features in calibrated world axes. Actor classes and IDs are annotated.",
    "fit": "CV uses the OLS velocity from five observed ego points. All three ridge models directly predict 30 xy "
           "residuals to that same CV. Warm-start, fitting on held-out clips, tuning, reweighting and clipping are absent. "
           "Training-only population standardization (SD <=1e-10 uses divisor 1), training mean residual as unpenalized "
           "intercept. Per coordinate, mean squared error plus 0.1 times squared coefficient norm; deterministic direct solve.",
    "primary": "Motion + availability + geometry minus motion-only ridge, equal-weighting four fixed sites after "
               "window-weighted means within each site. ADE is primary; FDE and geometry increment over availability "
               "are secondary. Negative is less error. Report all OOF windows and prespecified neighbor-presence strata.",
    "interpretation": "Neighbor counts, directions and distances may proxy site or traffic patterns. Differences cannot "
                      "establish a causal social response. Each ridge is fitted separately, so its predictions can differ "
                      "even on windows with no measured neighbor.",
    "interval": "95% percentile intervals on paired primary ADE/FDE. With PCG64 seed 2026, jointly resample whole clips "
                "with replacement 20,000 times within each of the four fixed sites, preserving all clip windows; average "
                "the four site means equally. Conditional on realized OOF fits; shared training folds and possible "
                "cross-clip dependence are not modeled. Descriptive development uncertainty, not out-of-site generalization.",
}


def evaluate(cohort, design_only=False):
    all_rows = np.ones(len(cohort.keys), dtype=bool)
    by_site = {site: cohort.sites == site for site in sorted(set(cohort.sites))}
    by_fold = {f: cohort.folds == f for f in range(N_FOLDS)}
    report = {"protocol": PROTOCOL, "design_only": design_only, "identity": cohort.identity,
              "versions": {"python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__},
              "feature_dimensions": {name: int(x.shape[1]) for name, x in cohort.feature_sets.items()},
              "clip_to_fold": {clip: int(fold) for clip, fold in zip(cohort.clips.tolist(), cohort.folds.tolist())}}
    scores = {}
    if not design_only:
        cv = constant_velocity(cohort.histories)
        forecast = {name: np.full_like(cv, np.nan) for name in METHODS}
        for test in by_fold.values():
            train = ~test
            if set(cohort.clips[test]).intersection(cohort.clips[train]) or not test.any() or not train.any():
                raise ValueError("Empty or overlapping held-out clip fold")
            residual = cohort.futures[train] - cv[train]
            for name, x in cohort.feature_sets.items():
                forecast[name][test] = cv[test] + ridge_residual(x[train], residual, x[test])
        for name, prediction in {"ols_five": cv, **forecast}.items():
            distances = np.linalg.norm(prediction - cohort.futures, axis=2)
            if not np.isfinite(distances).all():
                raise ValueError(f"Incomplete OOF coverage in {name}")
            scores[name] = np.column_stack([distances.mean(axis=1), distances[:, -1]])
    report["all"] = summarize(cohort, all_rows, scores)
    report["sites"] = {site: summarize(cohort, mask, scores) for site, mask in by_site.items()}
    report["folds"] = {str(f): summarize(cohort, mask, scores) for f, mask in by_fold.items()}
    person, car = (cohort.counts > 0).T
    report["strata"] = {name: summarize(cohort, mask, scores) for name, mask in
                        (("none", ~(person | car)), ("person_only", person & ~car),
                         ("car_only", ~person & car), ("both", person & car))}
    if scores:
        report["four_site_equal_means_m"] = {
            name: metric_dict(np.mean([values[mask].mean(axis=0) for mask in by_site.values()], axis=0))
            for name, values in scores.items()}
        report["four_site_equal_paired_m"] = {
            contrast: {m: float(np.mean([s["paired_m"][contrast][m] for s in report["sites"].values()]))
                       for m in ("ade_m", "fde_m")} for contrast in report["all"]["paired_m"]}
        primary = scores[METHODS[2]] - scores[METHODS[0]]
        rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
        draws = [paired_clip_draws(primary[mask], cohort.clips[mask], rng) for mask in by_site.values()]
        report["primary_percentile_95_m"] = intervals(np.mean(draws, axis=0))
        report["bootstrap_replicates"] = REPLICATES
        report["out_of_fold"] = {"window_key_fields": ["clip", "person_id", "t0"], "keys": cohort.keys,
                                 "fold": cohort.folds.tolist(), "neighbor_counts_person_car": cohort.counts.tolist(),
                                 "metric_order": ["ade_m", "fde_m"], "errors": {n: v.tolist() for n, v in scores.items()}}
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path, nargs="?", help="Optional official metadata root including archive_members.txt")
    parser.add_argument("--metadata-cache", type=Path, default=Path("/tmp/shortpose_neighbor_metadata"))
    parser.add_argument("--pilot", type=Path, help="Optional mounted accepted seed-42 directory or its shortpose_heldout.zip")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--design-only", action="store_true")
    args = parser.parse_args()
    root = args.data_root
    if root is None:
        from scripts.stage_pinns_metadata import stage_metadata
        root = stage_metadata(args.metadata_cache)
    cohort = load_cohort(root, args.pilot)
    output = evaluate(cohort, args.design_only)
    save_json(args.output, output)
    print("Verified", len(cohort.keys), "window IDs and", len(output["clip_to_fold"]), "clip folds.", flush=True)
    print("Design only:" if args.design_only else "Paired four-site differences (m):",
          output["all"] if args.design_only else output["four_site_equal_paired_m"], flush=True)
    print("Output:", args.output, flush=True)
