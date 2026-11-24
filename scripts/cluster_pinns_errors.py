"""Paired clip bootstrap for five-frame M1 minus constant-velocity ADE on PINNS."""

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SEED = 2026
REPLICATES = 20_000


def interval_summary(estimate, draws):
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {
        "estimate_m": round(float(estimate), 6),
        "percentile_95_interval_m": [round(float(lower), 6), round(float(upper), 6)],
    }


def compute(audit):
    clips = audit["by_clip_with_test_anchors"]
    rows = [clips[name] for name in sorted(clips) if clips[name]["windows"] > 0]
    sizes = np.asarray([record["windows"] for record in rows], dtype=np.int64)
    differences = np.asarray([record["models"]["M1"]["ade_minus_cv_m"] for record in rows])
    if len(rows) != audit["verification"]["clips_with_evaluated_windows"] or sizes.sum() != audit["overall"]["windows"]:
        raise ValueError("Clip counts do not reproduce the source audit")
    if not np.all(np.isfinite(differences)):
        raise ValueError("Non-finite paired clip ADE difference")

    # Draw whole clips with replacement. A repeated clip contributes all its windows again.
    rng = np.random.Generator(np.random.PCG64(SEED))
    sample = rng.integers(0, len(rows), size=(REPLICATES, len(rows)))
    sample_weights = sizes[sample]
    sample_differences = differences[sample]
    pooled = np.average(differences, weights=sizes)
    equal_clip = np.mean(differences)
    reference = audit["overall"]["models"]["M1"]["ade_minus_cv_m"]
    if abs(pooled - reference) > 2e-6:
        raise ValueError("Weighted clip means disagree with the archived window mean")

    return {
        "source": "results/pinns_motion_audit.json",
        "metric": "Five-frame PINNS top-mode ADE in metres, M1 minus constant velocity. Negative favors M1.",
        "windows": int(sizes.sum()),
        "nonempty_clips": len(rows),
        "observed_site_windows": {key: value["windows"] for key, value in audit["by_site"].items()},
        "method": {
            "sampling": "Unstratified paired resampling of nonempty clips with replacement, each clip equally likely. Each replicate draws the original number of clips and retains every paired window inside each selected clip.",
            "seed": SEED,
            "generator": "NumPy PCG64, source clip names sorted lexicographically",
            "replicates": REPLICATES,
            "interval": "2.5th and 97.5th percentiles of bootstrap estimates, linear interpolation.",
            "input_precision": "Source clip/model means are rounded to six decimal places in metres.",
        },
        "window_weighted_difference": interval_summary(
            pooled, np.sum(sample_weights * sample_differences, axis=1) / np.sum(sample_weights, axis=1)
        ),
        "equal_clip_difference": interval_summary(equal_clip, np.mean(sample_differences, axis=1)),
        "scope": [
            "Post-hoc comparison on the original V2 split reused during development. M1's five-seed average is held fixed throughout.",
            "The bootstrap approximates independent clips and retains dependence within clips. Correlation between clips from a shared site may remain.",
            "The observations cover four fixed locations. The intervals describe clip resampling on this observed pool and do not estimate transfer to new sites.",
        ],
    }


if __name__ == "__main__":
    results = ROOT / "results"
    audit = json.loads((results / "pinns_motion_audit.json").read_text(encoding="utf-8"))
    report = compute(audit)
    output = results / "pinns_clip_bootstrap.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(output)
