"""Condition archived five-frame predictions on same-person annotation history within a clip."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from scripts.analyze_heldout_pairs import BOOTSTRAP_SEED, REPLICATES, intervals, paired_clip_draws
from scripts.audit_pinns_matched_histories import digest_keys
from scripts.audit_pinns_observations import source_frames
from scripts.benchmark_heldout_sites import HISTORY, WINDOW, load_site_windows


ROOT = Path(__file__).resolve().parents[1]
FIRST = "first_sampled_annotation"
PRIOR = "at_least_35_preceding_samples"
GROUPS = {FIRST: "removed_for_short_history", PRIOR: "complete_matched"}
MODELS = ("M0", "M1")
METHODS = ("network_top1", "velocity_selected", "last_two_velocity", "oracle_minade_mode")
METRICS = ("ade_m", "fde_m")


def classify(labels, person, start):
    times = labels[person]
    if start == min(times):
        return FIRST
    if all(t in times for t in range(start - WINDOW, start)):
        return PRIOR
    raise ValueError("Registered five-frame input satisfies neither annotation-history condition")


def cohort(data_root):
    """Assign groups from annotation timestamps before loading any scores."""
    sites, _, identity = load_site_windows(data_root)
    archived = json.loads((ROOT / "results/pinns_matched_histories.json").read_text())
    registered = json.loads((ROOT / "results/pinns_heldout_replay.json").read_text())
    if identity != registered["label_and_calibration_sha256"] or identity != archived["identity"]["labels_and_calibration_sha256"]:
        raise ValueError("Labels/calibration differ from the registered source")
    keys, site_names, flags = [], [], []
    for site, windows in sorted(sites.items()):
        labels = None
        previous_clip = None
        for row in windows:
            if row.clip != previous_clip:
                labels, *_ = source_frames(data_root / "labels" / (row.clip + ".txt"))
                previous_clip = row.clip
            keys.append((row.clip, row.person_id, row.t0))
            site_names.append(site)
            flags.append(classify(labels, row.person_id, row.t0 - HISTORY + 1))
    if len(set(keys)) != 1482 or digest_keys(keys) != registered["ordered_window_ids_sha256"]:
        raise ValueError("Changed original cohort or window order")
    flags = np.asarray(flags)
    for name, old in GROUPS.items():
        subset = [k for k, keep in zip(keys, flags == name) if keep]
        if digest_keys(subset) != archived["cohorts"][old]["ordered_window_ids_sha256"]:
            raise ValueError(f"Annotation-history window IDs differ from the earlier cohort audit: {name}")
    return keys, np.asarray(site_names), flags, identity


def errors_from_bundle(bundle, keys):
    reference = (ROOT / "results/pinns_velocity_reranking.json").read_bytes()
    report = json.loads(reference)
    with zipfile.ZipFile(bundle) as z:
        if z.read("reranked_mode_scores.json") != reference:
            raise ValueError("Archived reranking report differs from the verified Kaggle aggregate")
        paired = z.read("reranked_paired_windows.npz")
    if hashlib.sha256(paired).hexdigest() != report["paired_windows_sha256"]:
        raise ValueError("The per-window error archive differs from the independently replayed source")
    with np.load(io.BytesIO(paired), allow_pickle=False) as a:
        ids = list(zip(a["clip"].tolist(), a["person_id"].tolist(), a["t0"].tolist()))
        if (ids != keys or a["seeds"].tolist() != report["seeds"] or list(MODELS) != a["models"].tolist()
                or list(METHODS) != a["methods"].tolist() or list(METRICS) != a["metrics"].tolist()):
            raise ValueError("Key order or metric labels differ")
        errors = a["errors"]
    if errors.shape != (5, 2, len(keys), 4, 2) or not np.isfinite(errors).all():
        raise ValueError("Invalid error tensor")
    direct = errors[0, 0, :, METHODS.index("last_two_velocity")]
    if not np.all(errors[:, :, :, METHODS.index("last_two_velocity")] == direct):
        raise ValueError("Direct velocity must be identical across all seeds and models")
    return errors, report


def method_means(values):
    return {
        model: {method: dict(zip(METRICS, values[m, j].tolist())) for j, method in enumerate(METHODS)}
        for m, model in enumerate(MODELS)
    }


def contrasts(errors):
    """Paired differences per window, after averaging repeated seeds on each window."""
    values = {}
    for m, name in enumerate(MODELS):
        records = errors[:, m].mean(axis=0)
        top, reranked, velocity = (records[:, i] for i in range(3))
        values[name + "_reranked_minus_top1"] = reranked - top
        values[name + "_reranked_minus_direct_velocity"] = reranked - velocity
    between = (errors[:, 1] - errors[:, 0]).mean(axis=0)
    for j, method in enumerate(METHODS[:2]):
        values["M1_minus_M0_" + method] = between[:, j]
    return values


def group_report(keys, site_names, mask, errors, seeds, pairs):
    clips = np.array([k[0] for k in keys])
    group_sites = sorted(set(site_names))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    site_records, all_draws, seed_results, site_means = {}, [], [], []
    pair_values = np.column_stack(list(pairs.values()))
    for site in group_sites:
        selected = np.flatnonzero((site_names == site) & mask)
        by_seed = np.take(errors, selected, axis=2).mean(axis=2)
        means = by_seed.mean(axis=0)
        diffs = pair_values[selected]
        draws = paired_clip_draws(diffs, clips[selected], rng)
        all_draws.append(draws)
        seed_results.append(by_seed)
        site_means.append(diffs.mean(axis=0))
        site_records[site] = {
            "windows": len(selected), "clips": len(set(clips[selected].tolist())),
            "methods": method_means(means), "paired_differences_m": paired_summary(pairs, diffs.mean(axis=0), draws),
        }
    by_seed = np.stack(seed_results).mean(axis=0)
    return {
        "windows": int(mask.sum()), "clips": len(set(clips[mask].tolist())),
        "ordered_window_ids_sha256": digest_keys([k for k, use in zip(keys, mask) if use]),
        "by_site_five_seed_means": site_records,
        "four_site_equal_by_seed": {str(s): method_means(v) for s, v in zip(seeds, by_seed)},
        "four_site_five_seed_equal": method_means(by_seed.mean(axis=0)),
        "four_site_paired_differences_m": paired_summary(pairs, np.mean(site_means, axis=0), np.mean(all_draws, axis=0)),
    }


def paired_summary(names, means, draws):
    records = {}
    for j, name in enumerate(names):
        cols = slice(2 * j, 2 * (j + 1))
        records[name] = {"mean": dict(zip(METRICS, means[cols].tolist())), "percentile95": intervals(draws[:, cols])}
    return records


def analyze(data_root, bundle):
    keys, sites, flags, identity = cohort(data_root)
    errors, original = errors_from_bundle(bundle, keys)
    pairs = contrasts(errors)
    results = {k: group_report(keys, sites, flags == k, errors, original["seeds"], pairs) for k in GROUPS}
    for site, n in original["site_window_counts"].items():
        strata = [r["by_site_five_seed_means"][site] for r in results.values()]
        if sum(r["windows"] for r in strata) != n:
            raise ValueError("Subgroup sizes do not add up to the original site cohort")
        for model in MODELS:
            for method in METHODS:
                expected = original["five_seed_means_by_site"][site][model]["methods"][method]
                for metric in METRICS:
                    observed = sum(r["windows"] * r["methods"][model][method][metric] for r in strata) / n
                    np.testing.assert_allclose(observed, expected[metric], rtol=0, atol=1e-12)
    return {
        "scope": "Post-hoc descriptive subgroup analysis of the same 1,482 held-site windows and 40 frozen fits. "
                 "Both groups use exactly five observed inputs. Group membership is an annotation-timing property "
                 "of the same clip/person ID. Physical entry, occlusion and unseen-site performance were not measured.",
        "groups": {FIRST: "Five-frame input begins at the first sampled annotation of this person ID in this clip.",
                   PRIOR: "At least 35 consecutive 10 Hz annotations of this person ID immediately precede the five inputs."},
        "metrics": "ADE averages all 30 Euclidean displacement errors, FDE scores step 30, both in metres. "
                   "The oracle selects the minimum true ADE after observing the future; its FDE uses that same path.",
        "aggregation": "Within each subgroup, mean windows within each site; weight the four sites and five seeds equally.",
        "paired_intervals": {"replicates": REPLICATES, "rng_seed": BOOTSTRAP_SEED,
            "method": "Average paired differences over the five fixed seeds per window. Sample whole clips with "
                      "replacement within each site and include all subgroup windows in each sampled clip. Use "
                      "identical draws across methods and metrics, then take percentile intervals. These intervals "
                      "condition on these sites, fitted models and the annotation-history subgroup."},
        "seeds": original["seeds"], "label_and_calibration_sha256": identity,
        "reranking_pairs_sha256": original["paired_windows_sha256"],
        "weights_archive_sha256": original["archive_sha256"], "strata": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("reranked_bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.data_root, args.reranked_bundle)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
