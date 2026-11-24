"""Evaluate frozen PINNS forecasts on the proposal's sub-half-metre five-frame histories."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.analyze_heldout_pairs import intervals, metric_dict, paired_clip_draws
from scripts.audit_pinns_matched_histories import digest_keys
from scripts.benchmark_heldout_sites import HISTORY, load_site_windows
from scripts.evaluate_heldout_implied_actions import ACTIONS, events
from scripts.score_pinns_annotation_history import FIRST, PRIOR, cohort, errors_from_bundle

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("M0", "M1", "last_two_velocity")
LIMIT_M = 0.5


def distances(history):
    history = np.asarray(history, dtype=np.float64)
    if history.ndim != 3 or history.shape[1:] != (HISTORY, 2) or not np.isfinite(history).all():
        raise ValueError("Expected five finite two-dimensional historical positions per window")
    path_length = np.linalg.norm(np.diff(history, axis=1), axis=2).sum(axis=1)
    net_distance = np.linalg.norm(history[:, -1] - history[:, 0], axis=1)
    if (net_distance > path_length + 1e-10).any():
        raise ValueError("Ground samples violate the triangle inequality")
    return path_length, net_distance


def distance_summary(values):
    quartiles = np.quantile(values, [0.25, 0.5, 0.75])
    return {"below_half_metre": int((values < LIMIT_M).sum()), "q25_median_q75_m": quartiles.tolist()}


def describe(keys, positions, sites, flags):
    route, net = distances(positions)
    strata = {}
    for name, flag in (("first", FIRST), ("prior", PRIOR)):
        for kind, condition in (("short", route < LIMIT_M), ("long", route >= LIMIT_M)):
            strata[f"{name}_{kind}"] = (flags == flag) & condition
    if not np.all(np.stack(list(strata.values())).sum(0) == 1):
        raise ValueError("The four registered cohorts must partition the input")
    design = {"ordered_keys_sha256": digest_keys(keys), "history_only_groups": {}, "site_distances": {}}
    for name, mask in strata.items():
        selected = [k for k, keep in zip(keys, mask) if keep]
        design["history_only_groups"][name] = {
            "windows": len(selected), "distinct_clips": len({k[0] for k in selected}),
            "clip_persons": len({k[:2] for k in selected}), "ordered_keys_sha256": digest_keys(selected),
            "by_site": {s: int((mask & (sites == s)).sum()) for s in sorted(set(sites))}}
    for name in ["all", *sorted(set(sites))]:
        use = np.ones(len(keys), dtype=bool) if name == "all" else sites == name
        design["site_distances"][name] = {"windows": int(use.sum()), "cumulative": distance_summary(route[use]),
                                          "net": distance_summary(net[use])}
    return design, strata


def group_scores(mask, clips, sites, truth, forecasts, seeds):
    # forecasts has axes [seed, method, window, metric].
    by_site, seed_scores, site_bootstraps = {}, [], []
    rng = np.random.default_rng(2026)
    for site in sorted(set(sites)):
        use = np.flatnonzero(mask & (sites == site))
        if not len(use):
            by_site[site] = {"windows": 0, "clips": 0, "scores": None}
            continue
        values = forecasts[:, :, use, :]
        seed_means = values.mean(axis=2)
        seed_scores.append(seed_means)
        diff = (values[:, 1] - values[:, 0]).mean(axis=0)
        site_bootstraps.append(paired_clip_draws(diff, clips[use], rng))
        by_site[site] = {"windows": len(use), "clips": len(set(clips[use])),
                         "scores": {m: metric_dict(v) for m, v in zip(METHODS, seed_means.mean(0))},
                         "M1_minus_M0": metric_dict(diff.mean(0)),
                         "events": dict(zip(ACTIONS, truth[use].sum(axis=0).tolist()))}
    result = {"sites": by_site, "contributing_sites": len(seed_scores), "four_site_ade_difference_m": None,
              "events": dict(zip(ACTIONS, truth[mask].sum(axis=0).tolist()))}
    if seed_scores:
        mean_by_seed = np.mean(seed_scores, axis=0)
        diffs = mean_by_seed[:, 1] - mean_by_seed[:, 0]
        result.update({"supported_site_equal": {m: metric_dict(v) for m, v in zip(METHODS, mean_by_seed.mean(0))},
                       "M1_minus_M0": metric_dict(diffs.mean(0)),
                       "conditional_paired_clip_95": intervals(np.mean(site_bootstraps, axis=0)),
                       "seed_M1_minus_M0": {str(seed): metric_dict(val) for seed, val in zip(seeds, diffs)}})
        if len(seed_scores) == len(by_site):
            result["four_site_ade_difference_m"] = float(diffs[:, 0].mean())
    return result


def evaluate(root, output, bundle=None):
    keys, sites, flags, data_hash = cohort(root)
    groups, _, same_hash = load_site_windows(root)
    rows = [w for name in sorted(groups) for w in groups[name]]
    if data_hash != same_hash or keys != [(w.clip, w.person_id, w.t0) for w in rows]:
        raise ValueError("Position records do not match the frozen history-group keys")
    xy = np.stack([row.xy for row in rows])
    design, strata = describe(keys, xy[:, :HISTORY], sites, flags)
    result = {"protocol_sha256": hashlib.sha256((ROOT / "reviews/pinns_short_displacement/PROTOCOL.md").read_bytes()).hexdigest(),
              "source_labels_and_calibration_sha256": data_hash, "design": design}
    if bundle is not None:
        # Only now read future event labels and existing forecast errors.
        truth = events(xy[:, :HISTORY], xy[:, None, HISTORY:])[:, 0]
        raw, source = errors_from_bundle(bundle, keys)
        forecasts = np.stack([raw[:, 0, :, 0], raw[:, 1, :, 0], raw[:, 0, :, 2]], axis=1)
        clips = np.array([key[0] for key in keys])
        result["groups"] = {name: group_scores(mask, clips, sites, truth, forecasts, source["seeds"])
                            for name, mask in strata.items()}
        result["window_errors_sha256"] = source["paired_windows_sha256"]
        for site, n in source["site_window_counts"].items():
            pieces = [v["sites"][site] for v in result["groups"].values() if v["sites"][site]["windows"]]
            if sum(s["windows"] for s in pieces) != n:
                raise ValueError("Group counts do not recover the original held-site cohort")
            for name, model, method in (("M0", "M0", "network_top1"), ("M1", "M1", "network_top1"),
                                        ("last_two_velocity", "M0", "last_two_velocity")):
                base = source["five_seed_means_by_site"][site][model]["methods"][method]
                for metric, ref in base.items():
                    actual = sum(g["windows"] * g["scores"][name][metric] for g in pieces) / n
                    np.testing.assert_allclose(actual, ref, rtol=0, atol=1e-12)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reranked-bundle", type=Path, help="Leave out to inspect only group membership")
    args = parser.parse_args()
    evaluate(args.data_root, args.output, args.reranked_bundle)
