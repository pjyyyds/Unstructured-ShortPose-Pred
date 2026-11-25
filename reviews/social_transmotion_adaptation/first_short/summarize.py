"""Score returned PINNS paths within the four previously frozen five-frame history groups."""
import argparse
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from scripts.analyze_heldout_pairs import intervals, metric_dict, paired_clip_draws


ROOT = Path(__file__).resolve().parents[3]
HISTORY_SHA = "caa8f0959f28f2d69a6f16cc9df115ab6222d0b6eb3297edd575615fef81b94c"
RETURN_SHA = "38f1777735602623006a27ebbe2c041c79d9339af89b3992d5912a8681dc7d94"
ARMS = ("last_two", "traj", "traj_box")
PAIRS = (("traj_box", "last_two"), ("traj", "last_two"), ("traj_box", "traj"))


def keys(rows):
    return list(zip(*(rows[name].tolist() for name in ("clip", "person_id", "t0"))))


def key_digest(rows):
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def summarize(histories, bundle):
    for path, expected in ((histories, HISTORY_SHA), (bundle, RETURN_SHA)):
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, "Unexpected input digest")
    frozen = json.loads((ROOT / "reviews/pinns_short_displacement/history_support.json").read_text())["design"]
    with np.load(histories, allow_pickle=False) as h:
        roster = keys(h)
        clips, sites = h["clip"], h["site"]
        observed, future = h["history_20_xy"][:, -5:], h["future_30_xy"]
        first = h["t0"] - 4 == h["first_person_label_t"]
        require(h["history_20_mask"][:, -5:].all(), "Five complete observations required")
    require(key_digest(roster) == frozen["ordered_keys_sha256"], "History keys changed")
    small = np.linalg.norm(np.diff(observed, axis=1), axis=-1).sum(axis=1) < 0.5
    groups = {prefix + "_" + name: history & length
              for prefix, history in (("first", first), ("prior", ~first))
              for name, length in (("short", small), ("long", ~small))}
    for name, mask in groups.items():
        selected = [key for key, keep in zip(roster, mask) if keep]
        require(key_digest(selected) == frozen["history_only_groups"][name]["ordered_keys_sha256"],
                "Group membership changed in " + name)

    scores = {arm: np.empty((len(roster), 2)) for arm in ARMS}
    max_abs = 0.0
    with ZipFile(bundle) as archive:
        for site in sorted(set(sites)):
            idx = np.flatnonzero(sites == site)
            with np.load(io.BytesIO(archive.read(site + "/private_windows.npz")), allow_pickle=False) as saved:
                require(keys(saved) == [roster[i] for i in idx], "Unmatched windows in " + site)
                history = observed[idx]
                velocity = history[:, -1:] - history[:, -2:-1]
                paths = {"last_two": history[:, -1:] + velocity * np.arange(1, 31)[None, :, None],
                         "traj": saved["traj_paths"], "traj_box": saved["traj_box_paths"]}
                for arm, pred in paths.items():
                    require(pred.shape == (len(idx), 30, 2), "Incomplete paths")
                    distance = np.linalg.norm(pred.astype(np.float64) - future[idx], axis=2)
                    metrics = np.column_stack((distance.mean(1), distance[:, -1]))
                    stored = saved[arm + "_errors"]
                    require(np.isfinite(metrics).all() and np.isfinite(stored).all(), "Nonfinite score")
                    np.testing.assert_allclose(metrics, stored, rtol=0, atol=1e-9)
                    max_abs = max(max_abs, float(np.abs(metrics - stored).max()))
                    scores[arm][idx] = metrics

    earlier = json.loads((ROOT / "reviews/pinns_short_displacement/short_displacement_scores.json").read_text())
    results = {}
    for name, mask in groups.items():
        by_site, macro, contrasts = {}, {}, {}
        for site in sorted(set(sites)):
            keep = mask & (site == sites)
            by_site[site] = {"windows": int(keep.sum()), "clips": len(set(clips[keep])),
                             "means": {a: metric_dict(scores[a][keep].mean(0)) for a in ARMS}}
        for arm in ARMS:
            macro[arm] = {metric: float(np.mean([s["means"][arm][metric] for s in by_site.values()]))
                          for metric in ("ade_m", "fde_m")}
        for metric, value in macro["last_two"].items():
            baseline = earlier["groups"][name]["supported_site_equal"]["last_two_velocity"][metric]
            np.testing.assert_allclose(value, baseline, rtol=0, atol=1e-12)
        for arm, reference in PAIRS:
            rng = np.random.default_rng(2026)
            delta = scores[arm] - scores[reference]
            draws = []
            for site in sorted(by_site):
                keep = mask & (site == sites)
                draws.append(paired_clip_draws(delta[keep], clips[keep], rng))
            contrasts[arm + "_minus_" + reference] = {
                "difference_m": {m: macro[arm][m] - macro[reference][m] for m in macro[arm]},
                "conditional_clip_95_m": intervals(np.mean(draws, axis=0))}
        results[name] = {"windows": int(mask.sum()), "clips": len(set(clips[mask])), "by_site": by_site,
                         "four_site_equal": macro, "paired": contrasts}
    return {"version": 373628213, "history_sha256": HISTORY_SHA, "return_sha256": RETURN_SHA,
            "scope": "Secondary subgroup analysis, one seed, four previously studied PINNS sites. "
                     "Groups were defined for an earlier study. No model selection, training or forward call.",
            "uncertainty": "Percentile 95% intervals, 20,000 paired within-site whole-clip resamples, "
                           "PCG64 seed 2026. Fixed sites and model states. Unadjusted for multiple comparisons.",
            "max_abs_discrepancy_with_archived_window_error_m": max_abs, "groups": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("histories", type=Path)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(summarize(args.histories, args.bundle), indent=2, allow_nan=False) + "\n")
