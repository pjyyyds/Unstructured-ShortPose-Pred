"""Recount the 497 original test windows and summarize two saved ridge forecasts."""
import argparse
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np


ARCHIVE = "093a4d1901fe392e9181897287b86cddf686bd46d1a063a706c733cf7ca58fda"
REPLAY = "af471b7be832e47b74eee0d4232b15cba177393262468cf06166eabaed90cc7e"
TEST_KEYS = "2ab8d0197625f7cfb1ddb63cc6acbb62d14d904bfed7fc9ab262396aa52faa6d"
ARMS = ("availability", "image_geometry")
METRICS = ("ade_m", "fde_last_valid_m", "fde_exact_3s_m")


def read_archive(path):
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != ARCHIVE:
        raise ValueError("Expected the registered private V2 result archive")
    with ZipFile(io.BytesIO(payload)) as z:
        if z.testzip() is not None:
            raise ValueError("Invalid ZIP CRC")
        for name, expected in json.loads(z.read("sha256.json")).items():
            if hashlib.sha256(z.read(name)).hexdigest() != expected:
                raise ValueError("Invalid archive member digest")
        raw = z.read("causal_geometry_replay.npz")
        if hashlib.sha256(raw).hexdigest() != REPLAY:
            raise ValueError("Unexpected trajectories and masks")
        summary = json.loads(z.read("causal_geometry_summary.json"))
    with np.load(io.BytesIO(raw), allow_pickle=False) as rows:
        test = rows["split"] == "test"
        keys = list(zip(rows["clip"][test].tolist(), rows["person_id"][test].tolist(), rows["t0"][test].tolist()))
        digest = hashlib.sha256(json.dumps(sorted(keys), separators=(",", ":")).encode()).hexdigest()
        if digest != TEST_KEYS or len(set(keys)) != 497:
            raise ValueError("Unexpected test cohort")
        fields = ("site", "clip", "future_mask", "truth_relative", *("prediction_" + name for name in ARMS))
        return {name: rows[name][test] for name in fields}, summary


def support(mask, clips):
    complete = mask.all(axis=1)
    # A gap is a missing tick strictly between two observed ticks. A trailing or leading loss is distinct.
    inside = (np.maximum.accumulate(mask, axis=1)
              & np.maximum.accumulate(mask[:, ::-1], axis=1)[:, ::-1])
    gap = np.any(inside & ~mask, axis=1)
    last = mask.shape[1] - np.argmax(mask[:, ::-1], axis=1)
    prefix_only = ~complete & ~gap & mask[:, 0]
    return {
        "windows": len(mask), "clips": len(set(clips)), "observed_windows_at_1s_2s_3s": mask[:, [9, 19, 29]].sum(0).tolist(),
        "all_30_steps": int(complete.sum()), "all_30_steps_fraction": float(complete.mean()),
        "clips_with_exact_3s": len(set(clips[mask[:, -1]])), "continuous_prefix_shorter_than_30": int(prefix_only.sum()),
        "missing_first_step": int((~mask[:, 0]).sum()), "internal_gap_windows": int(gap.sum()),
        "scored_future_positions": int(mask.sum()), "last_valid_lead_range_s": [int(last.min()) / 10, int(last.max()) / 10],
    }


def clip_interval(differences, sites, clips):
    """Fix the eligible window mixture by site; resample entire supporting clips within each site."""
    rng = np.random.Generator(np.random.PCG64(2026))
    draws = np.zeros(20_000)
    for site in sorted(set(sites)):
        selected = sites == site
        groups = sorted(set(clips[selected]))
        counts = np.array([np.count_nonzero(selected & (clips == c)) for c in groups])
        totals = np.array([differences[selected & (clips == c)].sum() for c in groups])
        sampled = rng.integers(len(groups), size=(len(draws), len(groups)))
        draws += selected.mean() * totals[sampled].sum(1) / counts[sampled].sum(1)
    return np.quantile(draws, [.025, .975]).tolist()


def analyze(path):
    data, archive_summary = read_archive(path)
    mask, sites, clips = data["future_mask"], data["site"], data["clip"]
    if mask.dtype != bool or mask.shape != (497, 30) or not np.all(mask.any(axis=1)):
        raise ValueError("Invalid future mask")
    last = mask.shape[1] - 1 - np.argmax(mask[:, ::-1], axis=1)
    all_rows = np.ones(len(mask), dtype=bool)
    error_metrics = {}
    for arm in ARMS:
        error = np.linalg.norm(data["prediction_" + arm] - data["truth_relative"], axis=2)
        error_metrics[arm] = (np.where(mask, error, 0).sum(1) / mask.sum(1), error[np.arange(len(mask)), last], error[:, -1])

    metrics = {}
    for index, name in enumerate(METRICS):
        eligible = mask[:, -1] if name == "fde_exact_3s_m" else all_rows
        control, image = (error_metrics[arm][index][eligible] for arm in ARMS)
        difference = image - control
        interval = clip_interval(difference, sites[eligible], clips[eligible])
        means = {ARMS[0]: float(control.mean()), ARMS[1]: float(image.mean())}
        original = archive_summary["test"]["all"]
        for arm, value in means.items():
            np.testing.assert_allclose(value, original["means"][arm][name], rtol=0, atol=1e-12)
        np.testing.assert_allclose(difference.mean(), original["paired_differences"]["image_geometry_minus_availability"][name],
                                   rtol=0, atol=1e-12)
        if name == "ade_m":
            np.testing.assert_allclose(interval, archive_summary["primary_test_paired_ade"]["percentile_95_m"],
                                       rtol=0, atol=1e-12)
        metrics[name] = {
            "windows": int(eligible.sum()), "clips": len(set(clips[eligible])), "arm_means_m": means,
            "paired_difference_m": float(difference.mean()), "paired_clip_bootstrap_95_m": interval,
            "interval_status": "registered primary" if name == "ade_m" else "exploratory follow-up",
        }
    return {
        "source_kaggle_version": 373590565, "source_zip_sha256": ARCHIVE, "source_replay_sha256": REPLAY,
        "sorted_test_keys_sha256": TEST_KEYS, "interval_method": {
            "delta": "image_geometry minus availability, metres", "replicates": 20000, "seed": 2026,
            "rng": "NumPy PCG64", "percentiles": [2.5, 97.5],
            "sampling": "Within each fixed site, draw its number of eligible whole clips with replacement. "
                        "Use all eligible windows per drawn clip and retain the site's original eligible-window proportion.",
        },
        "support": support(mask, clips),
        "site_support": {s: support(mask[sites == s], clips[sites == s]) for s in sorted(set(sites))},
        "paired_metrics": metrics,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_zip", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(analyze(args.result_zip), indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(result)
    else:
        print(result, end="")
