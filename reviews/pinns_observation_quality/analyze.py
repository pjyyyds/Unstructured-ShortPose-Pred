"""Summarize frozen Kaggle errors using pose quality over the available history."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from reviews.pinns_matched_multiseed.replay_helpers import inputs, verify_sources

SEEDS = (42, 1024, 2025, 3407, 4096)
LENGTHS = (5, 10, 20)
ARMS = ("M0", "M1", "M1b", "M1-concat")
GROUPS = ("clear", "partial", "heavy")
METRICS = ("ade_m", "fde_m")


def checked_zip(data, digest):
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("Archive differs from the published Kaggle return")
    return ZipFile(io.BytesIO(data))


def load_errors(work, reference, sites, keys):
    """All arrays come from the already verified checkpoints. No model is instantiated."""
    values = np.full((len(SEEDS), len(LENGTHS), len(ARMS), len(keys), 2), np.nan)
    source = reference["provenance"]
    repeat = json.loads((ROOT / "results/pinns_matched_same_worker_t20.json").read_text())

    def read_fold(archive, seed, t, arms, family):
        if seed == 42:
            fold = json.loads(archive.read("fold_ids.json"))
        else:
            fold = json.loads(archive.read("folds.json"))
        for site, idx in sites.items():
            field = "test_ids" if family == "base" else "test_keys"
            assert list(map(tuple, fold[site][field])) == [keys[i] for i in idx]
            for arm in arms:
                prefix = "" if family == "fusion" else f"T{t}/"
                raw = archive.read(f"{prefix}{site}/{arm}.pt")
                checkpoint = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
                record = checkpoint if family == "base" else checkpoint["predictions"]
                errors = record["errors"].numpy()
                assert errors.shape == (len(idx), 2) and np.isfinite(errors).all()
                values[SEEDS.index(seed), LENGTHS.index(t), ARMS.index(arm), idx] = errors

    with checked_zip((work / "matched_same_worker_T20.zip").read_bytes(),
                     source["seed42_source_T20"]["bundle_sha256"]) as repeat_zip:
        for t in LENGTHS:
            for family, arms in (("base", ARMS[:2]), ("fusion", ARMS[2:])):
                name = f"cpu_T{t}.zip" if family == "base" else f"matched_fusion_cpu_T{t}.zip"
                if t == 20:
                    data = repeat_zip.read(name)
                    digest = repeat["source"]["inner_sha256"][name]
                else:
                    data = (work / name).read_bytes()
                    digest = source["seed42_sources_T5_T10"][str(t)][family]["archive_sha256"]
                with checked_zip(data, digest) as archive:
                    read_fold(archive, 42, t, arms, family)
    for seed in SEEDS[1:]:
        name = f"multiseed_s{seed}_all.zip"
        digest = source["verified_archives_and_replay"][str(seed)][0]["archive_sha256"]
        with checked_zip((work / name).read_bytes(), digest) as archive:
            for t in LENGTHS:
                read_fold(archive, seed, t, ARMS, "multiseed")
    assert np.isfinite(values).all()
    return values


def metric_pair(values):
    return dict(zip(METRICS, np.asarray(values).tolist()))


def literal_coverage(quality):
    # Use Python's double-precision thresholds, as in assign_quality_bucket.
    valid, confidence, box_height = quality[:, :3].astype(float).T
    clear = (valid >= .85) & (confidence >= .60) & (box_height >= .08)
    partial = ((valid >= .50) & (valid < .85)) | (
        (valid >= .85) & (confidence >= .40) & (confidence < .60) & (box_height >= .04))
    heavy = (valid < .50) | (confidence < .40) | (box_height < .04)
    matches = np.stack([clear, partial, heavy]).sum(axis=0)
    return {str(i): int((matches == i).sum()) for i in range(4)}


def summarize_cell(errors, cv, clips, groups):
    # errors has axes seed, model, window, and metric.
    rng = np.random.default_rng(2026)
    boot = np.zeros((20000, 2, 2))
    site_means = []
    cv_means = []
    details = {}
    counts = {}
    for site, idx in sorted(groups.items()):
        names, inverse, window_counts = np.unique(clips[idx], return_inverse=True, return_counts=True)
        counts[site] = {"windows": len(idx), "clips": len(names)}
        if not len(idx):
            continue
        observed = errors[:, :, idx].mean(axis=2)
        site_means.append(observed)
        cv_means.append(cv[idx].mean(axis=0))
        details[site] = {arm: metric_pair(observed[:, i].mean(axis=0)) for i, arm in enumerate(ARMS)}
        details[site]["last_pair_cv"] = metric_pair(cv_means[-1])
        paired = errors[:, 1].mean(axis=0)[idx, None] - errors[:, [0, 2]].mean(axis=0)[:, idx].transpose(1, 0, 2)
        totals = np.zeros((len(names), 2, 2))
        np.add.at(totals, inverse, paired)
        draws = rng.integers(len(names), size=(len(boot), len(names)))
        bootstrap_means = totals[draws].sum(axis=1) / window_counts[draws].sum(axis=1)[:, None, None]
        boot += bootstrap_means / len(groups)
    result = {"counts": counts, "by_site": details, "windows": sum(len(idx) for idx in groups.values()),
              "clips": sum(c["clips"] for c in counts.values()), "four_site": None}
    if len(site_means) < len(groups):
        return result
    per_seed = np.mean(site_means, axis=0)
    model_means = per_seed.mean(axis=0)
    macro = {arm: metric_pair(model_means[i]) for i, arm in enumerate(ARMS)}
    macro["last_pair_cv"] = metric_pair(np.mean(cv_means, axis=0))
    contrasts = {}
    for col, arm in enumerate(("M0", "M1b")):
        contrasts["M1_minus_" + arm] = {
            "mean": metric_pair(model_means[1] - model_means[ARMS.index(arm)]),
            "clip_95": metric_pair(np.percentile(boot[:, col], (2.5, 97.5), axis=0).T)}
    result["four_site"] = {"models": macro, "paired": contrasts, "per_seed": {
        str(seed): {arm: metric_pair(per_seed[j, i]) for i, arm in enumerate(ARMS)}
        for j, seed in enumerate(SEEDS)}}
    return result


def analyze(work):
    reference = json.loads((ROOT / "results/pinns_matched_fusion_five_seed.json").read_text())
    with ZipFile(work / "cpu_T5.zip") as archive:
        cache, sites, keys = verify_sources(archive.read("matched_observations.npz"),
                                            work / "pinns_matched_histories.npz")
    errors = load_errors(work, reference, sites, keys)
    xy = cache["xy"].astype(float)
    steps = np.arange(1, 31)[None, :, None]
    cv_paths = xy[:, 19:20] + (xy[:, 19:20] - xy[:, 18:19]) * steps
    distances = np.linalg.norm(cv_paths - xy[:, 20:], axis=-1)
    cv = np.column_stack([distances.mean(axis=1), distances[:, -1]])
    hist_groups = {}
    result = {"seeds": SEEDS, "model_order": ARMS, "quality_order": GROUPS,
              "windows": len(keys), "by_duration": {}, "transitions": {}}
    for index, t in enumerate(LENGTHS):
        features, bins = inputs(cache, t)
        hist_groups[t] = bins
        length = {"literal_match_counts": literal_coverage(features["q_vec"].numpy()), "groups": {}}
        for b, name in enumerate((*GROUPS, "overall")):
            subset = {site: idx if name == "overall" else idx[bins[idx] == b] for site, idx in sites.items()}
            cell = summarize_cell(errors[:, index], cv, cache["clip"], subset)
            length["groups"][name] = cell
            if t == 5 or name == "overall":
                expected = reference["group_results"][name]
                assert cell["windows"] == expected["windows"]
                for seed in SEEDS:
                    for arm in ARMS:
                        expected_model = expected["durations"][str(t)]["seed_means"][str(seed)][arm]
                        for m in METRICS:
                            np.testing.assert_allclose(cell["four_site"]["per_seed"][str(seed)][arm][m],
                                                       expected_model[m], atol=1e-10, rtol=0)
            if name == "overall":
                for m in METRICS:
                    np.testing.assert_allclose(cell["four_site"]["models"]["last_pair_cv"][m],
                                               reference["baseline_four_site_macro"]["CV_OLS_2"][m], atol=1e-6, rtol=0)
        result["by_duration"][str(t)] = length
    for a, b in ((5, 10), (5, 20), (10, 20)):
        transitions = {}
        for site, idx in sites.items():
            mat = np.zeros((3, 3), dtype=int)
            np.add.at(mat, (hist_groups[a][idx], hist_groups[b][idx]), 1)
            transitions[site] = mat.tolist()
        result["transitions"][f"{a}_to_{b}"] = transitions
    result["protocol_sha256"] = hashlib.sha256(Path(__file__).with_name("PROTOCOL.md").read_bytes()).hexdigest()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path, help="Directory with the nine Kaggle returns and census")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(analyze(args.work), indent=2, allow_nan=False) + "\n")
