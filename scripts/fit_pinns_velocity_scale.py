"""Fit one constant-velocity scale on three PINNS sites and evaluate on the fourth.

Run python -m scripts.fit_pinns_velocity_scale DATA_ROOT from the project root.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.benchmark_heldout_sites import HISTORY, load_sites, score
from src.data.constants import T_FUT


ROOT = Path(__file__).resolve().parents[1]
METRICS = ("ade_m", "fde_m")


def unscaled(history):
    # All positions are relative to the last observation; every interval is 0.1 s.
    step = history[:, -1:] - history[:, -2:-1]
    return step * np.arange(1, T_FUT + 1)[None, :, None]


def fit_scale(training):
    numerator, denominator = 0.0, 0.0
    for windows in training.values():
        c = unscaled(windows[:, :HISTORY])
        y = windows[:, HISTORY:]
        # Reduce within site first, giving each physical site the same total weight.
        numerator += (c * y).sum(axis=-1).mean()
        denominator += (c * c).sum(axis=-1).mean()
    if denominator <= 0 or not np.isfinite([numerator, denominator]).all():
        raise ValueError("The constant-velocity scale is not identifiable on these training sites.")
    return float(np.clip(numerator / denominator, 0, 1))


def benchmark(root, reference_path):
    sites, _, digest = load_sites(root)
    reference = json.loads(reference_path.read_text())
    if (digest != reference["verification"]["label_and_calibration_digest_sha256"]
            or set(sites) != set(reference["sites"])):
        raise ValueError("Source data or site names differ from the registered last-two baseline.")
    folds = {}
    for held_out in sorted(sites):
        training = {site: sites[site] for site in sorted(sites) if site != held_out}
        scale = fit_scale(training)
        test = sites[held_out]
        c = unscaled(test[:, :HISTORY])
        y = test[:, HISTORY:]
        original = score(c, y)
        scaled = score(scale * c, y)
        archived = reference["sites"][held_out]
        if len(test) != archived["windows"]:
            raise ValueError(f"Test-window count differs for {held_out}")
        for key in METRICS:
            np.testing.assert_allclose(original[key], archived["metrics"]["last_two"][key], rtol=0, atol=1e-10)
        folds[held_out] = {
            "train_site_windows": {site: len(rows) for site, rows in training.items()},
            "train_windows": sum(map(len, training.values())), "test_windows": len(test),
            "fitted_scale": scale, "last_two": original, "scaled_last_two": scaled,
            "scaled_minus_last_two": {key: scaled[key] - original[key] for key in METRICS},
        }
    macro = {name: {key: float(np.mean([fold[name][key] for fold in folds.values()])) for key in METRICS}
             for name in ("last_two", "scaled_last_two", "scaled_minus_last_two")}
    for key in METRICS:
        np.testing.assert_allclose(macro["last_two"][key], reference["macro"]["last_two"][key], rtol=0, atol=1e-10)
    return {
        "protocol": {
            "scope": "One exploratory alternative on four PINNS sites examined in earlier work. One fixed coefficient fit per held-out site, no validation or tuning.",
            "windows": reference["protocol"]["windows"],
            "coordinates": reference["protocol"]["coordinates"],
            "forecast": "C[t] = t * (last position - penultimate position), t=1..30, relative to the last observation. Compare a*C to C.",
            "fit": "Minimize squared 2D position errors with no intercept. a = clip(sum_s mean_windows,steps(C dot Y) / sum_s mean_windows,steps(C dot C), 0, 1), using the three training sites only.",
            "metrics": "ADE is the mean Euclidean position error at steps 1..30; FDE is the Euclidean error at 3.0 s. Macro gives each held-out site equal weight. Negative deltas mean lower errors.",
        },
        "verification": {
            "baseline_reference_file": reference_path.name, "label_and_calibration_digest_sha256": digest,
            "coefficient_inputs": "fit_scale receives only the three arrays named in train_site_windows. The held-out array is read for evaluation after fitting.",
            "baseline_checked": "Counts and unscaled ADE/FDE from all four folds plus the macro ADE/FDE; tolerance 1e-10 m.",
        },
        "test_windows_total": sum(fold["test_windows"] for fold in folds.values()),
        "macro": macro, "sites": folds,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--reference", type=Path, default=ROOT / "results/pinns_velocity_estimators.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/pinns_velocity_scale.json")
    args = parser.parse_args()
    result = benchmark(args.data_root, args.reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
