"""Verify the five-seed Kaggle archive, replay each held-out checkpoint and compare matched windows."""
import argparse
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from scripts.analyze_heldout_pairs import BOOTSTRAP_SEED, REPLICATES, intervals, metric_dict, paired_clip_draws
from scripts.run_heldout_seed_sensitivity import (
    FIXED_SEEDS, MODELS, NORMALIZATION, Protocol, SITE_COUNTS, aggregate, file_hash,
    json_digest, pilot_inputs, split_indices, validate_checkpoint,
)
from scripts.train_pinns_heldout_sites import save_json
from scripts.verify_heldout_bundle import ATOL_M, RTOL, parameter_digest, run_cpu_forward, validate_zip
from src.data.features import assign_quality_bucket
from src.models.predictor import UnstructuredShortPosePredictor


def check_summary(calculated, reported, path):
    if isinstance(calculated, dict):
        expected = {str(k): v for k, v in calculated.items()}
        if set(expected) != set(reported):
            raise ValueError(f"Different summary keys at {path}")
        for key, value in expected.items():
            check_summary(value, reported[key], f"{path}/{key}")
    else:
        np.testing.assert_allclose(calculated, reported, atol=1e-10, rtol=0, err_msg=path)


def verify(bundle, data_root, pilot_dir):
    tensors, index, keys, pilot, seed42_scores, originals = pilot_inputs(data_root, pilot_dir)
    runner_hash = file_hash(Path(__file__).with_name("run_heldout_seed_sensitivity.py"))
    expected_files = {"report.json", "sha256.json", "pose.npz"}
    expected_files.update(f"seed_42/{key}" for key in originals)
    for seed in FIXED_SEEDS[1:]:
        for site in SITE_COUNTS:
            expected_files.update(f"seed_{seed}/{site}/{name}.pt" for name in MODELS)

    scores, discrepancies, parameter_ids = {}, {}, set()
    with zipfile.ZipFile(bundle) as archive:
        validate_zip(archive)
        if set(archive.namelist()) != expected_files:
            raise ValueError("Incomplete or unexpected five-seed archive members")
        manifest = json.loads(archive.read("sha256.json"))
        hashes = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in expected_files - {"sha256.json"}}
        if hashes != manifest or hashes["pose.npz"] != pilot["identity"]["pose_sha256"]:
            raise ValueError("Manifest or pose digest mismatch")
        if any(hashes[f"seed_42/{name}"] != file_hash(path) for name, path in originals.items()):
            raise ValueError("Seed 42 differs from the original accepted pilot")
        report = json.loads(archive.read("report.json"))
        if (report["seeds"] != list(FIXED_SEEDS) or report["runner_sha256"] != runner_hash
                or report["site_window_counts"] != SITE_COUNTS or report["protocol_seed_42"] != pilot["identity"]):
            raise ValueError("Unexpected run protocol, site manifest or producer revision")
        for seed in FIXED_SEEDS:
            scores[seed] = {}
            for site, ids in index.items():
                scores[seed][site] = {}
                training_sites, train, _ = split_indices(index, keys, site)
                test_keys = [list(keys[i]) for i in ids]
                for name in MODELS:
                    path = f"seed_{seed}/{site}/{name}.pt"
                    ck = torch.load(io.BytesIO(archive.read(path)), map_location="cpu", weights_only=True)
                    if seed == 42:
                        saved = seed42_scores[site][name]
                    else:
                        expected = {
                            **pilot["identity"], "protocol": asdict(Protocol(seed=seed)), "model": name,
                            "test_site": site, "training_sites": training_sites, "train_windows": len(train),
                            "test_windows": len(ids), "train_ids_sha256": json_digest([keys[i] for i in train]),
                            "test_ids_sha256": json_digest(test_keys), "runner_sha256": runner_hash,
                            "normalization": NORMALIZATION,
                        }
                        saved = validate_checkpoint(ck, expected, test_keys)
                    weight_id = parameter_digest(ck["model_state"])
                    if weight_id in parameter_ids:
                        raise ValueError(f"Repeated network parameters for {path}")
                    parameter_ids.add(weight_id)
                    replay = run_cpu_forward(UnstructuredShortPosePredictor, name, ck["model_state"], tensors, ids, Protocol().batch_size)
                    np.testing.assert_allclose(replay, saved, atol=ATOL_M, rtol=RTOL, err_msg=path)
                    discrepancies[path] = metric_dict(np.abs(replay - saved).max(axis=0))
                    scores[seed][site][name] = saved

    summary = aggregate(scores)
    for key, value in summary.items():
        check_summary(value, report[key], key)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws, per_site, buckets = [], {}, ("clear", "partial", "heavy")
    quality = np.array([assign_quality_bucket(q) for q in tensors["q_vec"].numpy()])
    for site, ids in index.items():
        deltas = np.stack([scores[seed][site]["M1"] - scores[seed][site]["M0"] for seed in FIXED_SEEDS])
        clips = np.asarray([keys[i][0] for i in ids])
        # Keep all five model pairs on the same sampled windows and in the same sampled clips.
        draws.append(paired_clip_draws(deltas.mean(axis=0), clips, rng))
        per_site[site] = {}
        for category, bucket in enumerate(buckets):
            selected = quality[ids] == category
            if not selected.any():
                raise ValueError(f"Empty {site}/{bucket} cell")
            per_site[site][bucket] = {
                "windows": int(selected.sum()), "clips": int(np.unique(clips[selected]).size),
                "by_seed_m1_minus_m0_m": {str(seed): metric_dict(row[selected].mean(axis=0))
                                          for seed, row in zip(FIXED_SEEDS, deltas)},
            }
    quality_macro = {}
    for bucket in buckets:
        quality_macro[bucket] = {
            str(seed): {metric: float(np.mean([site[bucket]["by_seed_m1_minus_m0_m"][str(seed)][metric]
                                                for site in per_site.values()]))
                        for metric in ("ade_m", "fde_m")} for seed in FIXED_SEEDS
        }
    return {
        "archive_sha256": file_hash(bundle), "runner_sha256": runner_hash,
        "pose_sha256": pilot["identity"]["pose_sha256"], "verified_checkpoints": len(parameter_ids),
        "verified_windows_per_seed": len(keys), "cpu_torch": str(torch.__version__),
        "cpu_vs_cloud_max_abs_by_checkpoint_m": discrepancies,
        "five_seed_site_macro": summary["five_seed_site_macro"],
        "four_site_macro_by_seed": summary["four_site_macro_by_seed"],
        "five_seed_average_m1_minus_m0_clip_interval": {
            "percentile_95_m": intervals(np.mean(draws, axis=0)), "replicates": REPLICATES, "seed": BOOTSTRAP_SEED,
            "scope": "Fixed five seeds and four previously examined locations. Within each location resample entire clips, retaining every window and all five fits together. Marginal intervals conditional on these seeds and sites.",
        },
        "quality_by_site": per_site, "quality_four_site_macro_by_seed_m1_minus_m0_m": quality_macro,
        "scope": "Five-frame complete windows. Quality categories use the original mutually exclusive model rules. No physical occlusion adjudication, no independent camera sites, and no 10/20-frame extension in this run.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("pilot_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    save_json(args.output, verify(args.bundle, args.data_root, args.pilot_dir))
