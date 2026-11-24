"""Compare three fixed motion estimators on the complete PINNS site-holdout windows.

Run python -m scripts.compare_pinns_velocity_estimators DATA_ROOT from the project root.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from scripts.benchmark_heldout_sites import HISTORY, constant_velocity, load_site_windows, score
from src.data.constants import DT, T_FUT


ROOT = Path(__file__).resolve().parents[1]
METRICS = ("ade_m", "fde_m")
METHODS = ("last_two", "ols_five", "median_four")


def predictions(history):
    # The protocol retains complete consecutive observations and centers on the last one.
    velocities = np.diff(history, axis=1) / DT
    elapsed = DT * np.arange(1, T_FUT + 1)[None, :, None]
    return {
        "last_two": velocities[:, -1:, :] * elapsed,
        "ols_five": constant_velocity(history),
        "median_four": np.median(velocities, axis=1, keepdims=True) * elapsed,
    }


def scores(windows):
    future = windows[:, HISTORY:]
    return {method: score(predicted, future)
            for method, predicted in predictions(windows[:, :HISTORY]).items()}


def verify_ols(measured, expected):
    for key in METRICS:
        np.testing.assert_allclose(measured[key], expected[key], rtol=0, atol=1e-10)


def benchmark(root, reference_path):
    reference = json.loads(reference_path.read_text())
    sites, counts, digest = load_site_windows(root)
    if (set(sites) != set(reference["sites"])
            or digest != reference["protocol"]["label_and_calibration_digest_sha256"]):
        raise ValueError("Input site names or label/calibration digest differ from the reference.")

    output = {}
    for site, rows in sorted(sites.items()):
        archived = reference["sites"][site]
        by_clip = defaultdict(list)
        for row in rows:
            by_clip[row.clip].append(row)
        if (len(rows) != archived["test_windows"]
                or len(by_clip) != archived["clips_with_windows"]
                or any(archived[key] != value for key, value in counts[site].items())):
            raise ValueError(f"Reference clip, track, or window counts differ for {site}.")
        metrics = scores(np.stack([row.xy for row in rows]))
        verify_ols(metrics["ols_five"], archived["constant_velocity"])
        output[site] = {
            "windows": len(rows), **counts[site], "metrics": metrics,
            "by_clip": {
                clip: {
                    "windows": len(clip_rows),
                    "person_tracks_with_windows": len({row.person_id for row in clip_rows}),
                    "metrics": scores(np.stack([row.xy for row in clip_rows])),
                }
                for clip, clip_rows in sorted(by_clip.items())
            },
        }
    macro = {
        method: {key: float(np.mean([site["metrics"][method][key] for site in output.values()]))
                 for key in METRICS}
        for method in METHODS
    }
    verify_ols(macro["ols_five"], reference["macro"]["constant_velocity"])
    total = sum(site["windows"] for site in output.values())
    if total != reference["test_windows_total"]:
        raise ValueError("Total number of windows differs from the reference.")
    return {
        "protocol": {
            "task": "Three fixed, training-free motion estimators evaluated on each held-out site of the existing PINNS motion benchmark. All four sites were previously examined.",
            "windows": reference["protocol"]["windows"],
            "coordinates": reference["protocol"]["coordinates"],
            "estimators": {
                "last_two": "Displacement between the last two observed positions divided by 0.1 s.",
                "ols_five": "Ordinary least-squares slope per coordinate using all five observed positions.",
                "median_four": "Coordinatewise median of four consecutive interframe velocities, averaging the two middle values on each axis.",
            },
            "forecast": "All methods extrapolate at constant velocity from the last observed position. Forecasts use only the five observed positions and fixed 0.1 s spacing.",
            "metrics": "Mean Euclidean distance across 30 targets (ADE) and at 3.0 s (FDE), in metres. Each window has equal weight within a clip or a site. The macro mean weights the four site means equally. Descriptive comparison without significance testing.",
        },
        "verification": {
            "reference_file": reference_path.name,
            "label_and_calibration_digest_sha256": digest,
            "windows_from_reference_loader_with_clip_identity": total,
            "ols_reference_checked": "ADE and FDE for each site and their four-site mean, absolute tolerance 1e-10 m.",
        },
        "test_windows_total": total, "macro": macro, "sites": output,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--reference", type=Path, default=ROOT / "results/pinns_heldout_sites.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/pinns_velocity_estimators.json")
    args = parser.parse_args()
    result = benchmark(args.data_root, args.reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
