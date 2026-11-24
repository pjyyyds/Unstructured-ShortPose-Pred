"""Break down paired M1/M0 errors by frozen five-frame pose quality in the seed-42 PINNS run."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

import numpy as np

from scripts.analyze_heldout_pairs import METRICS, SITE_COUNTS, assert_metric_match, check_keys, metric_dict
from scripts.benchmark_heldout_sites import load_site_windows
from scripts.train_pinns_heldout_sites import Protocol, file_hash, save_json, source_hash, tensorize
from src.data.features import assign_quality_bucket


BUCKETS = ("clear", "partial", "heavy")
MODELS = ("M0", "M1", "M1_minus_M0")
BUNDLE_SHA = "507c0a5decf79a4a48b982987e89c37b8c99e456f5c7f11cc3d49f34bd5741ec"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def literal_conditions(quality):
    # assign_quality_bucket compares float32 feature values after conversion to Python floats.
    q1, q2, q3 = quality[:, :3].astype(np.float64).T
    clear = (q1 >= .85) & (q2 >= .60) & (q3 >= .08)
    partial = ((q1 >= .50) & (q1 < .85)) | ((q1 >= .85) & (q2 >= .40) & (q2 < .60) & (q3 >= .04))
    heavy = (q1 < .50) | (q2 < .40) | (q3 < .04)
    return {"clear": clear, "partial": partial, "heavy": heavy,
            "partial_and_heavy": partial & heavy, "none_of_three": ~(clear | partial | heavy)}


def group_means(errors, selected):
    return {name: metric_dict(values[selected].mean(axis=0)) for name, values in errors.items()}


def analyze(data_root, bundle):
    require(file_hash(bundle) == BUNDLE_SHA, "Expected the audited seed-42 PINNS result ZIP")
    groups, _, label_hash = load_site_windows(data_root)
    require({s: len(rows) for s, rows in groups.items()} == SITE_COUNTS, "Unexpected source window counts")
    site_results = {}
    with zipfile.ZipFile(bundle) as archive, tempfile.TemporaryDirectory() as tmp:
        pose = Path(tmp) / "pose.npz"
        with archive.open("pose.npz") as source, pose.open("wb") as target:
            shutil.copyfileobj(source, target)
        cloud = json.loads(archive.read("folds/report.json"))
        identity = cloud["identity"]
        require(identity["pose_sha256"] == file_hash(pose) and identity["data_sha256"] == label_hash
                and identity["source_sha256"] == source_hash() and identity["protocol"] == asdict(Protocol()),
                "Frozen input, source or training identity mismatch")
        tensors, indices, keys, _, coverage, meta = tensorize(groups, pose)
        ids_hash = hashlib.sha256(json.dumps(keys, separators=(",", ":")).encode()).hexdigest()
        require(identity["ordered_ids_sha256"] == ids_hash and meta == cloud["pose_metadata"],
                "Window order or frozen pose metadata mismatch")
        quality = tensors["q_vec"].numpy()
        buckets = np.array([assign_quality_bucket(row) for row in quality])
        conditions = literal_conditions(quality)
        np.testing.assert_array_equal(buckets, np.where(conditions["heavy"], 2, np.where(conditions["clear"], 0, 1)))
        for site, ids in indices.items():
            fold = json.loads(archive.read(f"folds/{site}/fold.json"))
            require(fold["identity"] == identity and fold["error_columns"] == ["ADE1_m", "FDE1_m"]
                    and fold["train_sites"] == sorted(set(groups) - {site}), f"{site}: fold identity mismatch")
            order = check_keys(fold["window_keys"], [keys[i] for i in ids])
            clips = np.array([keys[i][0] for i in ids])
            errors = {}
            for model in ("M0", "M1"):
                values = np.asarray(fold["paired_window_errors"][model], dtype=np.float64)
                require(values.shape == (len(ids), 2) and np.isfinite(values).all() and (values >= 0).all(),
                        f"{site}/{model}: invalid window errors")
                errors[model] = values[order]
            errors["M1_minus_M0"] = errors["M1"] - errors["M0"]
            site_buckets = buckets[ids]
            counts = np.bincount(site_buckets, minlength=3)
            require((counts > 0).all() and counts.tolist() == coverage[site]["quality_buckets_clear_partial_heavy"]
                    == fold["coverage"]["quality_buckets_clear_partial_heavy"]
                    == cloud["sites"][site]["coverage"]["quality_buckets_clear_partial_heavy"],
                    f"{site}: disagrees with archived bucket counts")
            groups_summary = {}
            for i, name in enumerate(BUCKETS):
                selected = site_buckets == i
                groups_summary[name] = {"windows": int(counts[i]), "clips": len(set(clips[selected])),
                                        "errors_m": group_means(errors, selected)}
            # Verify the classification partitions the same paired rows used for the published site means.
            for model in MODELS:
                means = [[groups_summary[b]["errors_m"][model][metric] for metric in METRICS] for b in BUCKETS]
                reconstructed = np.average(means, axis=0, weights=counts)
                np.testing.assert_allclose(reconstructed, errors[model].mean(axis=0), rtol=0, atol=1e-12)
                if model in ("M0", "M1"):
                    assert_metric_match(reconstructed, cloud["sites"][site]["metrics"][model], atol=1e-12)
                    assert_metric_match(reconstructed, fold["metrics"][model], atol=1e-12)
            site_results[site] = {
                "windows": len(ids), "clips": len(set(clips)), "buckets": groups_summary,
                "proposal_condition_counts": {name: int(mask[ids].sum()) for name, mask in conditions.items()},
            }
    macro = {}
    for bucket in BUCKETS:
        rows = [v["buckets"][bucket] for v in site_results.values()]
        macro[bucket] = {
            "windows": sum(row["windows"] for row in rows), "clips": sum(row["clips"] for row in rows),
            "equal_site_errors_m": {model: {k: float(np.mean([row["errors_m"][model][k] for row in rows]))
                                             for k in METRICS} for model in MODELS},
        }
    return {
        "source": {"bundle_sha256": BUNDLE_SHA, "pose_sha256": identity["pose_sha256"],
                   "label_and_calibration_sha256": label_hash, "ordered_window_ids_sha256": ids_hash,
                   "training_source_sha256": identity["source_sha256"], "seed": Protocol().seed},
        "protocol": {
            "cohort": "All 1,482 windows in the four previously examined PINNS sites. Five 10 Hz observations "
                      "span 0.4 s. Score top-mode ADE over 30 future steps and FDE at 3.0 s, in metres.",
            "quality_inputs": "Historical RTMPose features. q1 is the valid keypoint fraction (confidence >= 0.20); "
                              "q2 is mean confidence over valid joints; q3 is mean relative annotated box height.",
            "proposal_section": "Section 5.3 of the original research proposal, conditions (1) through (3).",
            "proposal_clear": "q1 >= 0.85 AND q2 >= 0.60 AND q3 >= 0.08",
            "proposal_partial": "0.50 <= q1 < 0.85 OR (q1 >= 0.85 AND 0.40 <= q2 < 0.60 AND q3 >= 0.04)",
            "proposal_heavy": "q1 < 0.50 OR q2 < 0.40 OR q3 < 0.04",
            "archived_assignment": "Use src.data.features.assign_quality_bucket. Clear and heavy are disjoint. "
                                   "Heavy takes priority over literal partial in overlaps. All remaining non-clear, "
                                   "non-heavy windows are assigned partial, including those satisfying no literal rule.",
            "aggregation": "Equal window weight within each site and bucket. Give all four site means equal weight "
                           "within each bucket. Paired differences are M1 minus M0; positive means higher M1 error. "
                           "A clip can contribute windows to multiple buckets.",
            "interpretation": "Descriptive, conditional on four fixed sites and a single trained seed (42). Windows "
                              "from a clip can be correlated. The buckets summarize estimated joint quality and "
                              "image scale. Physical occlusion and blind-spot onset were not annotated in this analysis. "
                              "No bucket-wise hypothesis test or five-seed inference was performed.",
        },
        "verification": "Cache IDs and stored fold error rows matched by clip, person and anchor. Frozen bucket "
                        "counts matched each site report. Window-weighted bucket means reproduced each site's "
                        "archived M0 and M1 ADE/FDE to an absolute tolerance of 1e-12 m.",
        "proposal_condition_counts": {name: int(mask.sum()) for name, mask in conditions.items()},
        "sites": site_results, "by_bucket": macro,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path, help="Original PINNS labels, calibration and video inventory")
    parser.add_argument("bundle", type=Path, help="Verified shortpose_heldout.zip (inner result ZIP)")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] /
                        "results/pinns_heldout_pose_quality.json")
    args = parser.parse_args()
    save_json(args.output, analyze(args.data_root, args.bundle))
