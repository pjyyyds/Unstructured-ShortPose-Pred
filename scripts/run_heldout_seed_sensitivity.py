"""Resume the five fixed PINNS LOSO seeds from the accepted seed-42 Kaggle run."""
import argparse
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import zipfile

import numpy as np
import torch

from scripts.benchmark_heldout_sites import load_site_windows
from scripts.train_pinns_heldout_sites import (
    MODELS, Protocol, atomic_write, file_hash, fit_predict, mean_metrics, save_json, source_hash, tensorize,
)
from src.data.constants import FIXED_SEEDS


# SHA256 of canonical JSON mapping the original 13 relative fold paths to their SHA256.
PILOT_FOLDS_SHA = "66186fe8d21be32992a5c80fd10ddac2030ea18e31ee156fa7cfe133f5f9a351"
SITE_COUNTS = {"america_crossroad": 816, "italy_park": 84, "japan_crossroad": 432, "thailand_street": 150}
NORMALIZATION = (
    "Frozen tensorize and feature builders. Motion/targets in metres relative to the last observed position; "
    "velocity in metres/second. Each pose frame uses hip midpoint or box centre and divides by box height. "
    "No dataset-fitted feature or target scaler."
)
PAIR = "M1_minus_M0"
METRICS = ("ade_m", "fde_m")


def json_digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def split_indices(index, keys, site):
    training_sites = sorted(set(index) - {site})
    train = np.concatenate([index[name] for name in training_sites])
    test = index[site]
    if not np.array_equal(np.sort(np.concatenate([train, test])), np.arange(len(keys))):
        raise ValueError(f"{site}: train/test indices must partition every window exactly once")
    if {keys[i][0] for i in train} & {keys[i][0] for i in test}:
        raise ValueError(f"{site}: a clip appears on both sides of the split")
    return training_sites, train, test


def pilot_inputs(data_root, pilot):
    folds = pilot / "folds"
    files = {"report.json": folds / "report.json"}
    for site in SITE_COUNTS:
        for name in ("fold.json", "M0.pt", "M1.pt"):
            files[f"{site}/{name}"] = folds / site / name
    if json_digest({name: file_hash(path) for name, path in files.items()}) != PILOT_FOLDS_SHA:
        raise ValueError("The original seed-42 folds do not match the accepted Kaggle package")
    report = json.loads(files["report.json"].read_text())
    pose = pilot / "pose.npz"
    groups, _, data_sha = load_site_windows(data_root)
    signature = report["identity"]
    if (signature["protocol"] != asdict(Protocol()) or signature["pose_sha256"] != file_hash(pose)
            or signature["source_sha256"] != source_hash() or signature["data_sha256"] != data_sha):
        raise ValueError("Seed-42 protocol, labels/calibration, pose cache or frozen source mismatch")
    tensors, index, keys, cv, _, metadata = tensorize(groups, pose)
    if ({s: len(i) for s, i in index.items()} != SITE_COUNTS or json_digest(keys) != signature["ordered_ids_sha256"]
            or metadata["label_and_calibration_digest_sha256"] != data_sha):
        raise ValueError("Window identities, physical sites or cache provenance mismatch")
    pilot_scores = {}
    for site in SITE_COUNTS:
        training_sites, train, test = split_indices(index, keys, site)
        fold = json.loads(files[f"{site}/fold.json"].read_text())
        ids = [list(keys[i]) for i in test]
        if (fold["train_sites"] != training_sites or fold["train_windows"] != len(train)
                or fold["window_keys"] != ids or fold["identity"] != signature):
            raise ValueError(f"{site}: seed-42 fold identity mismatch")
        np.testing.assert_allclose(cv[test], fold["paired_window_errors"]["constant_velocity"], rtol=0, atol=1e-10)
        pilot_scores[site] = {}
        for model in MODELS:
            checkpoint = torch.load(files[f"{site}/{model}.pt"], weights_only=True, map_location="cpu")
            expected = {**signature, "model": model, "test_site": site, "training_sites": training_sites}
            if checkpoint["identity"] != expected or checkpoint["window_keys"] != ids:
                raise ValueError(f"{site} {model}: seed-42 checkpoint identity mismatch")
            errors = checkpoint["errors"].numpy()
            np.testing.assert_array_equal(errors, fold["paired_window_errors"][model])
            pilot_scores[site][model] = errors
    return tensors, index, keys, report, pilot_scores, files


def payload_digest(state, errors):
    digest = hashlib.sha256()
    for name, tensor in [*sorted(state.items()), ("__errors__", errors)]:
        values = tensor.detach().cpu().contiguous().numpy()
        digest.update(json.dumps([name, values.dtype.str, values.shape]).encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


def validate_checkpoint(checkpoint, expected, ids):
    if checkpoint["identity"] != expected or checkpoint["window_keys"] != ids:
        raise ValueError("Checkpoint identity or ordered test window IDs mismatch")
    state, errors = checkpoint["model_state"], checkpoint["errors"]
    if not state or not all(torch.isfinite(tensor).all() for tensor in state.values()):
        raise ValueError("Invalid model weights")
    if (tuple(errors.shape) != (len(ids), 2) or not torch.isfinite(errors).all() or (errors < 0).any()
            or payload_digest(state, errors) != checkpoint["payload_sha256"]):
        raise ValueError("Checkpoint weights or paired errors failed integrity checks")
    return errors.numpy()


def aggregate(scores):
    site_metrics, macro = {}, {}
    for seed in FIXED_SEEDS:
        site_metrics[seed] = {}
        for site in SITE_COUNTS:
            errors = scores[seed][site]
            metrics = {model: mean_metrics(errors[model]) for model in MODELS}
            metrics[PAIR] = mean_metrics(errors["M1"] - errors["M0"])
            site_metrics[seed][site] = metrics
        macro[seed] = {
            model: {k: float(np.mean([s[model][k] for s in site_metrics[seed].values()])) for k in METRICS}
            for model in (*MODELS, PAIR)
        }
    five_seeds = {}
    for model in (*MODELS, PAIR):
        five_seeds[model] = {}
        for k in METRICS:
            values = [macro[seed][model][k] for seed in FIXED_SEEDS]
            five_seeds[model][k] = {"mean": float(np.mean(values)), "sample_sd": float(np.std(values, ddof=1))}
    return {"sites_by_seed": site_metrics, "four_site_macro_by_seed": macro, "five_seed_site_macro": five_seeds}


def write_package(out, members, report):
    hashes = {name: file_hash(path) for name, path in members.items()}
    save_json(out / "report.json", report)
    save_json(out / "sha256.json", {**hashes, "report.json": file_hash(out / "report.json")})
    files = {**members, "report.json": out / "report.json", "sha256.json": out / "sha256.json"}

    def write_checked(temp):
        with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for name, path in sorted(files.items()):
                archive.write(path, arcname=name)
        with zipfile.ZipFile(temp) as archive:
            if set(archive.namelist()) != set(files) or archive.testzip() is not None:
                raise OSError("Incomplete five-seed archive")

    path = out / "shortpose_heldout_five_seeds.zip"
    atomic_write(path, write_checked)
    return path


def run(args):
    tensors, index, keys, pilot, seed42_scores, original = pilot_inputs(args.data_root, args.pilot_dir)
    if args.check_only:
        print(f"Verified {len(keys)} windows, four disjoint folds and eight original seed-42 checkpoints.", flush=True)
        return
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".run.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_locked(args, tensors, index, keys, pilot, seed42_scores, original)


def configure_training(device, torch_version):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if str(torch.__version__) != torch_version or device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"Training requires CUDA and the original PyTorch {torch_version}")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def run_locked(args, tensors, index, keys, pilot, seed42_scores, original):
    signature = pilot["identity"]
    runner_hash = file_hash(Path(__file__).resolve())
    device = torch.device(args.device)
    if not args.summarize_only:
        configure_training(device, signature["torch_version"])
    members = {f"seed_42/{name}": path for name, path in original.items()}
    members["pose.npz"] = args.pilot_dir / "pose.npz"
    scores = {42: seed42_scores}
    for seed in FIXED_SEEDS[1:]:
        scores[seed] = {}
        protocol = Protocol(seed=seed)
        for site in SITE_COUNTS:
            training_sites, train, test = split_indices(index, keys, site)
            ids = [list(keys[i]) for i in test]
            scores[seed][site] = {}
            for model in MODELS:
                expected = {
                    **signature, "protocol": asdict(protocol), "model": model, "test_site": site,
                    "training_sites": training_sites, "train_windows": len(train), "test_windows": len(test),
                    "train_ids_sha256": json_digest([keys[i] for i in train]), "test_ids_sha256": json_digest(ids),
                    "runner_sha256": runner_hash, "normalization": NORMALIZATION,
                }
                name = f"seed_{seed}/{site}/{model}.pt"
                path = args.output_dir / name
                if path.is_file():
                    checkpoint = torch.load(path, weights_only=True, map_location="cpu")
                    errors = validate_checkpoint(checkpoint, expected, ids)
                    print(f"Verified {name}", flush=True)
                else:
                    if args.summarize_only:
                        raise FileNotFoundError(f"Missing {name}; rerun training to resume")
                    print(f"Training seed {seed}, {model}, held out {site} ({len(train)} train, {len(test)} test)", flush=True)
                    state, measured = fit_predict(model, tensors, train, test, device, protocol)
                    err_tensor = torch.from_numpy(measured)
                    checkpoint = {"identity": expected, "model_state": state, "window_keys": ids, "errors": err_tensor,
                                  "device": torch.cuda.get_device_name(device), "payload_sha256": payload_digest(state, err_tensor)}
                    errors = validate_checkpoint(checkpoint, expected, ids)
                    atomic_write(path, lambda temp: torch.save(checkpoint, temp))
                scores[seed][site][model] = errors
                members[name] = path
    report = {
        "seeds": list(FIXED_SEEDS), "original_fits": 8, "additional_fits": 32,
        "site_window_counts": SITE_COUNTS, "protocol_seed_42": signature, "runner_sha256": runner_hash,
        "pilot_fold_manifest_sha256": PILOT_FOLDS_SHA, "normalization": NORMALIZATION,
        "error_columns": ["ADE1_m", "FDE1_m"], "window_key_fields": pilot["window_key_fields"],
        "window_protocol": pilot["window_protocol"], "selection": pilot["selection"],
        "metric_definitions": pilot["metric_definitions"],
        "scope": "Fixed five seeds, each with independent M0/M1 fits in all four folds. Reused sites and windows. "
                 "Sample SD describes between-seed variation of four-site macro results, conditional on these sites.",
        **aggregate(scores),
    }
    path = write_package(args.output_dir, members, report)
    print(f"Complete: {path}  SHA256 {file_hash(path)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/tmp/shortpose_heldout_cache/pinns_original"))
    parser.add_argument("--pilot-dir", type=Path, default=Path("/kaggle/working/shortpose_heldout"))
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/shortpose_heldout_seeds"))
    parser.add_argument("--device", default="cuda:0")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true", help="Verify pilot, source, labels and fold IDs on CPU.")
    mode.add_argument("--summarize-only", action="store_true", help="Verify all fits and rebuild the complete archive on CPU.")
    run(parser.parse_args())
