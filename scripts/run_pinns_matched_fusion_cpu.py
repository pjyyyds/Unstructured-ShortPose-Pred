"""Run one 623-window matched fusion duration on Kaggle CPU, or check/replay locally.

python -m scripts.run_pinns_matched_fusion_cpu METADATA PILOT.npz MATCHED.npz OUTPUT_DIR --duration 5
Use a separate CPU output directory. --check-only does not fit a model or create any artifact.
--replay-only validates a previously extracted bundle. The original GPU runner is unchanged.
"""

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import zipfile

import numpy as np
import torch

from scripts import run_pinns_matched_fusion as frozen

FUSION_SHA = "251145ae0e33f47cfae858ddfe787595844dea23beb3455b607139e81c594606"
REPEAT = "matched_fusion_cpu_seed42_v1"
HERE = Path(__file__).resolve()
DURATIONS = (5, 10, 20)


def configure_cpu():
    if torch.cuda.is_available():
        raise RuntimeError("Select a CPU session or hide CUDA before importing torch")
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    limit = Path("/sys/fs/cgroup/cpu.max")
    if limit.is_file():
        quota, period = limit.read_text().split()
        if quota != "max":
            cpus = min(cpus, max(1, int(quota) // int(period)))
    torch.set_num_threads(min(4, cpus))
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")


def backend_info():
    cpu_file = Path("/proc/cpuinfo")
    model = platform.machine()
    if cpu_file.exists():
        model = next(
            (
                r.partition(":")[2].strip()
                for r in cpu_file.read_text().splitlines()
                if r.startswith("model name")
            ),
            model,
        )
    rev = subprocess.run(
        ["git", "-C", str(frozen.ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    return {
        "repeat": REPEAT,
        "device": "cpu",
        "revision": rev,
        "driver_sha256": frozen.file_hash(HERE),
        "fusion_factory_training_replay_sha256": FUSION_SHA,
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_model": model,
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "mkldnn": torch.backends.mkldnn.enabled,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "env": {
            n: os.environ.get(n)
            for n in (
                "CUDA_VISIBLE_DEVICES",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "PYTHONHASHSEED",
            )
        },
        "torch_build_config": torch.__config__.show(),
    }


def pack(output, archive, files):
    hashes = {p.relative_to(output).as_posix(): frozen.file_hash(p) for p in files}
    frozen.save_json(output / "sha256.json", hashes)

    def write(temp):
        with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
            for p in [*files, output / "sha256.json"]:
                z.write(p, p.relative_to(output).as_posix())
        with zipfile.ZipFile(temp) as z:
            if z.testzip() is not None or set(z.namelist()) != set(hashes) | {"sha256.json"}:
                raise ValueError("Bad CPU result archive")

    frozen.atomic_write(archive, write)


def run(data, pilot, cache, parent, duration, check_only=False, replay_only=False):
    protocol = frozen.Protocol()
    if (protocol.seed, protocol.epochs) != (42, 28) or duration not in DURATIONS:
        raise ValueError("Expected frozen seed 42, 28 epochs and T=5,10,20")
    ref, arrays, base = frozen.load_inputs(data, pilot, cache)
    if base["runner_and_concat_sha256"] != FUSION_SHA:
        raise ValueError("Frozen fusion implementation changed")
    # load_inputs records the intended GPU version; a CPU-repeat checkpoint records the actual training build.
    base.pop("training_torch")
    tensors = frozen.tensors_for(arrays, duration)
    if tensors["x_m"].shape != (623, duration, 9) or tensors["x_p"].shape != (623, duration, 17, 6):
        raise ValueError("Unexpected matched observation shape")
    if (
        tensors["fut_xy"].shape != (623, 30, 2)
        or not tensors["fut_mask"].all()
        or not all(torch.isfinite(tensors[k]).all() for k in (*frozen.INPUTS, "fut_xy"))
    ):
        raise ValueError("Incomplete input or forecast labels")
    splits = {site: frozen.split_indices(ref.index, ref.keys, site) for site in sorted(ref.index)}
    folds = {
        site: {
            "train_sites": train_sites,
            "train_keys": [list(ref.keys[i]) for i in train],
            "test_keys": [list(ref.keys[i]) for i in test],
        }
        for site, (train_sites, train, test) in splits.items()
    }
    summary = {
        "experiment": REPEAT,
        "duration": duration,
        "windows": len(ref.keys),
        "site_counts": {s: len(v[2]) for s, v in splits.items()},
        "history_person_frames_inferred": int(arrays["pose_inferred"][:, -duration:].sum()),
        "parameters": {
            name: sum(p.numel() for p in frozen.make_model(name).parameters()) for name in frozen.VARIANTS
        },
        "scope": "CPU backend, one seed at four previously studied sites. Scores use the highest-probability of six modes.",
    }
    if check_only:
        return {**summary, "identity": base, "fit_started": False}
    if not replay_only and not (Path("/kaggle/working").is_dir() and Path("/kaggle/input").is_dir()):
        raise RuntimeError(
            "Actual fitting requires a Kaggle CPU session; use --check-only or --replay-only locally"
        )
    out = parent / f"matched_fusion_cpu_T{duration}"
    clock = out / "backend.json"
    fold_path = out / "fold_ids.json"
    report_path = out / "report.json"
    driver = out / "cpu_driver.py"
    checkpoints = [out / site / (name + ".pt") for site in splits for name in frozen.VARIANTS]
    files = [clock, fold_path, report_path, driver, *checkpoints]
    if replay_only:
        frozen.check_manifest(out, files)
        backend = json.loads(clock.read_text())
        if (
            backend["device"] != "cpu"
            or backend["repeat"] != REPEAT
            or backend["fusion_factory_training_replay_sha256"] != FUSION_SHA
            or backend["driver_sha256"] != frozen.file_hash(driver)
            or frozen.file_hash(driver) != frozen.file_hash(HERE)
        ):
            raise ValueError("Unexpected stored CPU backend or driver")
    else:
        backend = backend_info()
        out.mkdir(parents=True, exist_ok=True)
        if clock.exists() and json.loads(clock.read_text()) != backend:
            raise ValueError("Existing result was created with a different backend")
        frozen.save_json(clock, backend)
        frozen.atomic_write(driver, lambda tmp: tmp.write_bytes(HERE.read_bytes()))
    if fold_path.exists() and json.loads(fold_path.read_text()) != folds:
        raise ValueError("Stored folds disagree with the original held-out keys")
    if not replay_only:
        frozen.save_json(fold_path, folds)
    identity = {
        **base,
        "training_torch": backend["torch"],
        "training_device": "cpu",
        "backend_sha256": frozen.json_digest(backend),
        "duration": duration,
        "driver_sha256": backend["driver_sha256"],
        "experiment": REPEAT,
    }
    seen = set()
    site_metrics = {}
    max_gap = 0.0
    for site, (source, train, test) in splits.items():
        site_metrics[site] = {}
        fold = folds[site]
        for name in frozen.VARIANTS:
            expected = {
                **identity,
                "held_site": site,
                "training_sites": source,
                "variant": name,
                "train_ids_sha256": frozen.json_digest(fold["train_keys"]),
                "test_ids_sha256": frozen.json_digest(fold["test_keys"]),
            }
            p = out / site / (name + ".pt")
            if not replay_only and not p.is_file():
                print(
                    f"CPU fit T={duration} {site} {name}: {len(train)} train, {len(test)} held out", flush=True
                )
                state, detail = frozen.fit_variant(name, tensors, train, test, torch.device("cpu"), protocol)
                frozen.check_predictions(detail, tensors["fut_xy"][test].numpy())
                payload = {
                    "identity": expected,
                    "test_window_keys": fold["test_keys"],
                    "model_state": state,
                    "predictions": detail,
                }
                frozen.atomic_write(p, lambda tmp: torch.save(payload, tmp))
            record = torch.load(p, map_location="cpu", weights_only=True)
            if record["identity"] != expected or record["test_window_keys"] != fold["test_keys"]:
                raise ValueError(f"Stale CPU checkpoint: {p}")
            state = record["model_state"]
            h = frozen.parameter_digest(state)
            if h in seen:
                raise ValueError("Repeated weights across distinct CPU fits")
            seen.add(h)
            detail = record["predictions"]
            saved = frozen.check_predictions(detail, tensors["fut_xy"][test].numpy())
            replay = frozen.replay_saved(name, state, detail, tensors, test, protocol.batch_size)
            np.testing.assert_allclose(replay, saved, atol=frozen.ATOL_M, rtol=frozen.RTOL)
            max_gap = max(max_gap, float(np.max(abs(replay - saved))))
            site_metrics[site][name] = frozen.mean_metrics(saved.astype(float))
    summary.update(
        identity=identity,
        site_metrics=site_metrics,
        checkpoints_replayed=len(seen),
        max_cpu_replay_abs_m=max_gap,
        four_site_equal={
            name: {m: float(np.mean([v[name][m] for v in site_metrics.values()])) for m in ("ade_m", "fde_m")}
            for name in frozen.VARIANTS
        },
    )
    archive = out / (out.name + ".zip")
    if replay_only:
        published = json.loads(report_path.read_text())
        if any(summary[k] != published[k] for k in summary if k != "max_cpu_replay_abs_m"):
            raise ValueError("Stored summary differs from the reconstructed CPU checkpoints")
    else:
        frozen.save_json(report_path, summary)
        pack(out, archive, files)
        print(f"Packaged six trajectories and scores from eight CPU checkpoints: {archive}", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("metadata", "pilot", "cache", "output_parent"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--duration", type=int, required=True, choices=DURATIONS)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--replay-only", action="store_true")
    args = parser.parse_args()
    configure_cpu()
    report = run(
        args.metadata,
        args.pilot,
        args.cache,
        args.output_parent,
        args.duration,
        args.check_only,
        args.replay_only,
    )
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
