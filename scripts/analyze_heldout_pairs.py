"""Compare paired PINNS held-site predictions against motion and fixed velocity baselines."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.benchmark_heldout_sites import HISTORY, load_site_windows
from scripts.compare_pinns_velocity_estimators import predictions


ROOT = Path(__file__).resolve().parents[1]
SITE_COUNTS = {"america_crossroad": 816, "italy_park": 84, "japan_crossroad": 432, "thailand_street": 150}
METRICS = ("ade_m", "fde_m")
REFERENCES = ("M0", "ols_five", "last_two")
REPLICATES = 20_000
BOOTSTRAP_SEED = 2026


def metric_dict(values):
    return {key: float(value) for key, value in zip(METRICS, values)}


def check_keys(keys, expected):
    if any(len(k) != 3 or type(k[0]) is not str or type(k[1]) is not int or type(k[2]) is not int for k in keys):
        raise ValueError("Fold keys must contain a clip name, integer person ID and integer anchor")
    index = {tuple(k): i for i, k in enumerate(keys)}
    if len(index) != len(keys) or len(expected) != len(set(expected)) or set(index) != set(expected):
        raise ValueError("Duplicate, extra or missing clip/person/anchor key in the fold or source labels")
    return [index[k] for k in expected]


def velocity_errors(windows):
    xy = np.stack([w.xy for w in windows])
    forecasts = predictions(xy[:, :HISTORY])
    result = {}
    for name in ("ols_five", "last_two"):
        distances = np.linalg.norm(forecasts[name] - xy[:, HISTORY:], axis=-1)
        result[name] = np.column_stack((distances.mean(axis=1), distances[:, -1]))
    return result


def assert_metric_match(values, reference, atol=1e-10):
    np.testing.assert_allclose(values, [reference[k] for k in METRICS], rtol=0, atol=atol)


def paired_clip_draws(deltas, clip_names, rng):
    """Sample the original number of clips uniformly within this site; retain all windows."""
    clips, group = np.unique(clip_names, return_inverse=True)
    sizes = np.bincount(group)
    clip_sums = np.stack([deltas[group == i].sum(axis=0) for i in range(len(clips))])
    selected = rng.integers(0, len(clips), size=(REPLICATES, len(clips)))
    return clip_sums[selected].sum(axis=1) / sizes[selected].sum(axis=1)[:, None]


def intervals(draws):
    lower, upper = np.quantile(draws, (0.025, 0.975), axis=0)
    return {k: [float(lo), float(hi)] for k, lo, hi in zip(METRICS, lower, upper)}


def analyze(folds_dir, labels_dir):
    groups, _, digest = load_site_windows(labels_dir)
    velocity_ref = json.loads((ROOT / "results/pinns_velocity_estimators.json").read_text())
    cv_ref = json.loads((ROOT / "results/pinns_heldout_sites.json").read_text())
    run = json.loads((folds_dir / "report.json").read_text())
    identity = run["identity"]
    if set(groups) != set(SITE_COUNTS) or set(run["sites"]) != set(SITE_COUNTS) or run["total_windows"] != 1482:
        raise ValueError("Expected the four registered PINNS sites and 1,482 windows")
    if (digest != identity["data_sha256"]
            or digest != velocity_ref["verification"]["label_and_calibration_digest_sha256"]
            or digest != cv_ref["protocol"]["label_and_calibration_digest_sha256"]):
        raise ValueError("The fold, labels and previously published velocity baselines must use the same source")
    if identity["protocol"]["seed"] != 42 or identity["protocol"]["epochs"] != 28:
        raise ValueError("Expected the locked single-seed, 28-epoch fitting protocol")
    if run["window_key_fields"] != ["clip", "person_id", "t0_10hz"]:
        raise ValueError("Unexpected window-key definition")
    all_keys = [(w.clip, w.person_id, w.t0) for site in sorted(groups) for w in groups[site]]
    ids_digest = hashlib.sha256(json.dumps(all_keys, separators=(",", ":")).encode()).hexdigest()
    if len(set(all_keys)) != 1482 or identity["ordered_ids_sha256"] != ids_digest:
        raise ValueError("The folds do not match the complete registered list of window identities")

    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    site_stats, site_draws = {}, []
    for site, expected_n in sorted(SITE_COUNTS.items()):
        windows = groups[site]
        fold = json.loads((folds_dir / site / "fold.json").read_text())
        if (len(windows) != expected_n or fold["test_windows"] != expected_n or fold["identity"] != identity
                or set(fold["train_sites"]) != set(SITE_COUNTS) - {site} or fold["train_windows"] != 1482 - expected_n
                or fold["error_columns"] != ["ADE1_m", "FDE1_m"]):
            raise ValueError(f"Fold specification or training/held-out records mismatch for {site}")
        order = check_keys(fold["window_keys"], [(w.clip, w.person_id, w.t0) for w in windows])
        errors = velocity_errors(windows)
        for name in ("M0", "M1", "constant_velocity"):
            values = np.asarray(fold["paired_window_errors"][name], dtype=float)
            if values.shape != (expected_n, 2) or not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"Invalid paired ADE/FDE values for {name} at {site}")
            errors[name] = values[order]
            assert_metric_match(errors[name].mean(axis=0), fold["metrics"][name], atol=1e-6)
            assert_metric_match(errors[name].mean(axis=0), run["sites"][site]["metrics"][name], atol=1e-6)
        np.testing.assert_allclose(errors["constant_velocity"], errors["ols_five"], rtol=0, atol=1e-10)
        assert_metric_match(errors["ols_five"].mean(axis=0), cv_ref["sites"][site]["constant_velocity"])
        clips = np.asarray([w.clip for w in windows])
        by_clip_ref = velocity_ref["sites"][site]["by_clip"]
        if set(clips) != set(by_clip_ref):
            raise ValueError(f"Archived and reconstructed source clips differ at {site}")
        for model in ("ols_five", "last_two"):
            assert_metric_match(errors[model].mean(axis=0), velocity_ref["sites"][site]["metrics"][model])
            for clip, record in by_clip_ref.items():
                mask = clips == clip
                if int(mask.sum()) != record["windows"]:
                    raise ValueError(f"Inconsistent historical clip count for {clip}")
                assert_metric_match(errors[model][mask].mean(axis=0), record["metrics"][model])

        deltas = {name: errors["M1"] - errors[name] for name in REFERENCES}
        draws = paired_clip_draws(deltas["M0"], clips, rng)
        site_draws.append(draws)
        site_stats[site] = {
            "windows": expected_n, "clips": len(by_clip_ref),
            "m1_minus_reference_m": {name: metric_dict(delta.mean(axis=0)) for name, delta in deltas.items()},
            "m1_minus_m0_percentile_95_m": intervals(draws),
        }

    macro = {
        name: {metric: float(np.mean([site["m1_minus_reference_m"][name][metric] for site in site_stats.values()]))
               for metric in METRICS} for name in REFERENCES
    }
    return {
        "protocol": {
            "primary": "M1 minus M0. Negative differences favor M1.",
            "secondary": "M1 minus five-point least-squares velocity; M1 minus last-two-point velocity.",
            "metrics": "Top-mode 30-step ADE and final-step FDE, in metres. Equal window weight within each site, then equal weight across the four sites.",
            "bootstrap": "For each replicate, sample with replacement the original number of complete clips independently within each fixed site. Use the same sampled clips for paired models and both metrics. Include all windows in each selected clip, then average the four site means equally.",
            "seed": BOOTSTRAP_SEED, "replicates": REPLICATES, "rng": "NumPy PCG64",
            "interval": "Marginal 2.5th and 97.5th percentiles, linear interpolation; no multiplicity adjustment. Intervals are for the primary comparison only.",
            "scope": "Post-hoc analysis of four fixed, previously seen locations with one fitted seed (42). Assumes approximate independence between clips within a location. Conditional on these fits, the intervals exclude training-seed, retraining and new-site variation.",
        },
        "verification": {
            "label_and_calibration_sha256": digest, "ordered_ids_sha256": ids_digest,
            "pose_cache_sha256": identity["pose_sha256"], "windows": len(all_keys),
            "velocity_check": "Five-point errors matched the cloud fold for every key. Five- and two-point ADE and FDE matched the prior clip and site summaries to 1e-10 m.",
        },
        "sites": site_stats, "equal_site_m1_minus_reference_m": macro,
        "equal_site_m1_minus_m0_percentile_95_m": intervals(np.mean(site_draws, axis=0)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folds_dir", type=Path, help="Cloud result directory containing report.json and each site's fold.json")
    parser.add_argument("labels_dir", type=Path, help="Official PINNS labels and camera metadata")
    parser.add_argument("--output", type=Path, default=ROOT / "results/pinns_heldout_paired.json")
    args = parser.parse_args()
    result = analyze(args.folds_dir, args.labels_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
