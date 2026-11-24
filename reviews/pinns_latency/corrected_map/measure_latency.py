"""Latency of the frozen PINNS Full model on two observed synchronous scenes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import re
import shutil
import subprocess
import sys
import time
import tempfile

import numpy as np
import torch


TIMES = (5, 10, 20)
PROTOCOL_HASH = "933949fa5a8df237f785027499a18da55bb43966d4191dfc16d4c2d30fa51bda"
CHECKPOINT = "checkpoints/PINNS_T5_Ours_Full_seed42.pt"
CHECKPOINT_HASH = "5069c5be520f2308dd190c01cd68427ba8bdb321d405fc963f5436e911d30689"
SOURCES = {
    "src/__init__.py": "a2098deb66d5b29ad22e792b1b49888eecd7b67304b40ae362c182e71a3e6de1",
    "src/models/__init__.py": "29c4d3c2f39d83e75e4688d93497f2917d852ad8ebfbffbb9ab86bff157ac5dd",
    "src/data/__init__.py": "e084ddfd18d41470ee8ab9e1d7c70d25d3462e059392d2bcc172b30af2ca32fe",
    "src/models/predictor.py": "5f1ece42560027ac13ca0504187eeb4949de2d46793f8592c3998cd8ae907e8f",
    "src/models/action_and_decoder.py": "c0d6d744bc062ab03d08ef0ac85ef838e9e1935ffd1e313aa8175304abfb8ecf",
    "src/models/env_and_gatv2.py": "ea5363d5d9f5b04d798b6ee1c5da1a314ca9c0612853906862f6eea4edeaea27",
    "src/models/motion_encoder.py": "657747c941bbcb44232946c11ddf84ca94140939dfd19d529582ad55f72f4031",
    "src/models/pose_stgcn.py": "0b63b89c6e04921274023268cf0e62e21a8856d7af8351194b9717b84daba3eb",
    "src/models/quality_gate.py": "6df0a8aa1e50cbc8cb8aede8adf76e2dedaa7772661a10f8bf0aa894a72b5784",
    "src/data/constants.py": "1586cd8dab2868a39bc0969de4408b2d11e23afd673ee4b8c1e229ce948029df",
}
ORDER_SEED = 42


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def shapes(t):
    return {"x_m": (t, 9), "x_p": (t, 17, 6), "q_vec": (4,), "q_part": (6,),
            "p_env": (16, 10, 11), "m_map": (1,), "ego_type_oh": (8,),
            "nbr_edges": (12, 9), "nbr_type_oh": (12, 8), "nbr_mask": (12,)}


def validate_arrays(data, t):
    n = 24
    require(set(data) == set(shapes(t)) | {"clip", "site", "t0", "person_id"}, "fixture fields differ")
    for key, tail in shapes(t).items():
        x = data[key]
        require(x.dtype == np.float32 and x.shape == (n, *tail) and np.isfinite(x).all(), f"invalid {key}")
    for key in ("clip", "site", "t0", "person_id"):
        valid_kinds = "U" if key in ("clip", "site") else "iu"
        require(data[key].shape == (n,) and data[key].dtype.kind in valid_kinds, f"invalid {key}")
    for key in ("m_map", "nbr_mask"):
        require(np.isin(data[key], (0, 1)).all(), f"invalid {key}")
    groups = {}
    for i, (clip, tick) in enumerate(zip(data["clip"], data["t0"])):
        groups.setdefault((str(clip), int(tick)), []).append(i)
    require(len(groups) == 2, "expected two distinct decision instants")
    for k, rows in groups.items():
        require(len(rows) == len(set(data["person_id"][rows])) == 12, "need exactly twelve distinct persons per instant")
        require(len(set(data["site"][rows])) == 1, "site changes within group")
        groups[k] = np.array(sorted(rows, key=lambda r: int(data["person_id"][r])), dtype=int)
    return groups


def support(d, rows):
    mask = d["nbr_mask"][rows] > 0
    has_map = d["m_map"][rows, 0].astype(bool) & (d["p_env"][rows, :, :, 10] > .5).any(axis=(1, 2))
    return {"site": str(d["site"][rows[0]]), "targets": len(rows),
            "map_rows_with_points": int(has_map.sum()), "neighbor_slots": int(mask.sum()),
            "car_slots": int(((d["nbr_type_oh"][rows, :, 6] > .5) & mask).sum())}


def load_fixture(folder):
    raw = (folder / "protocol.json").read_bytes()
    require(sha(raw) == PROTOCOL_HASH, "fixture protocol SHA mismatch")
    protocol = json.loads(raw)
    require(protocol["checkpoint_sha256"] == CHECKPOINT_HASH, "checkpoint disagrees with fixture")
    observed, groups = {}, {}
    for t in TIMES:
        p = folder / f"sync_T{t}.npz"
        require(sha(p.read_bytes()) == protocol["files_sha256"][p.name], f"{p.name} SHA mismatch")
        with np.load(p, allow_pickle=False) as buf:
            observed[t] = {k: buf[k] for k in buf.files}
        groups[t] = validate_arrays(observed[t], t)
    for t in TIMES:
        require(groups[t].keys() == groups[5].keys(), "different physical moments across histories")
        for key, ix in groups[t].items():
            ref = groups[5][key]
            for col in ("person_id", "site", "p_env", "m_map", "nbr_edges", "nbr_type_oh", "nbr_mask"):
                require(np.array_equal(observed[t][col][ix], observed[5][col][ref]), f"{col} differs across history lengths")
            actual = support(observed[t], ix)
            require(any(all(report[k] == v for k, v in actual.items()) for report in protocol["scene_support"]),
                    "measured support differs from fixture manifest")
    return protocol, observed, groups


def load_model(repo):
    repo = repo.resolve()
    for rel, digest in SOURCES.items():
        require(sha((repo / rel).read_bytes()) == digest, f"frozen source SHA mismatch: {rel}")
    path = repo / CHECKPOINT
    require(sha(path.read_bytes()) == CHECKPOINT_HASH, "frozen checkpoint SHA mismatch")
    sys.path.insert(0, str(repo))
    from src.models.predictor import UnstructuredShortPosePredictor
    for rel in SOURCES:
        module = sys.modules[rel[:-3].replace("/", ".").removesuffix(".__init__")]
        require(Path(module.__file__).resolve() == repo / rel, f"module loaded from another checkout: {rel}")
    model = UnstructuredShortPosePredictor("Ours_Full").eval()
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    require(sum(p.numel() for p in model.parameters()) == 471442 and len(state) == 161,
            "model parameter or state count mismatch")
    return model


def scene_token(key):
    return sha(json.dumps(key, ensure_ascii=True, separators=(",", ":")).encode())[:16]


def plan(groups, counts):
    rng = random.Random(ORDER_SEED)
    shapes_to_visit = [(t, n) for t in TIMES for n in counts]
    rng.shuffle(shapes_to_visit)
    cells = []
    for t, n in shapes_to_visit:
        keys = sorted(groups[t])
        rng.shuffle(keys)
        cells.extend((t, n, key) for key in keys)
    return cells


def target(name, arch, board_model, board_compatible):
    clean = " ".join(name.lower().split())
    if clean in ("nvidia geforce rtx 4090", "geforce rtx 4090", "nvidia rtx 4090"):
        return {"hardware": "desktop RTX 4090", "strict_target_ms": 10}
    evidence = (board_model + " " + board_compatible).lower()
    if ("orin" in clean and arch.lower() in ("arm64", "aarch64")
            and ("jetson orin nx" in evidence or re.search(r"p3767-000[01](?![0-9])", evidence))):
        return {"hardware": "Jetson Orin NX", "strict_target_ms": 25}
    return None


def textfile(path):
    try:
        return Path(path).read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        return ""


def probe(command):
    if not shutil.which(command[0]):
        return None
    try:
        out = subprocess.run(command, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def telemetry():
    fields = "name,driver_version,pstate,clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu"
    return {"nvidia_smi_all_visible_rows": probe(["nvidia-smi", "--query-gpu=" + fields, "--format=csv"]),
            "jetson_active_power_mode": probe(["nvpmodel", "-q"])}


def cuda_device(label):
    device = torch.device(label)
    require(device.type == "cuda" and torch.cuda.is_available() and torch.version.cuda is not None,
            "real timing requires an actual NVIDIA CUDA device and a CUDA-enabled PyTorch build")
    torch.cuda.set_device(device.index if device.index is not None else torch.cuda.current_device())
    return torch.device("cuda", torch.cuda.current_device())


def hardware(device):
    prop = torch.cuda.get_device_properties(device)
    board = textfile("/sys/firmware/devicetree/base/model")
    compatible = textfile("/sys/firmware/devicetree/base/compatible")
    arch = platform.machine()
    return {"selected_torch_device": str(device), "device_name": prop.name,
            "compute_capability": [prop.major, prop.minor], "total_memory_bytes": prop.total_memory,
            "architecture": arch, "system": platform.system(), "kernel": platform.release(),
            "python": platform.python_version(), "pytorch": str(torch.__version__), "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "device_tree_model": board or None,
            "device_tree_compatible": compatible or None, "tegra_release": textfile("/etc/nv_tegra_release") or None,
            "proposal_budget": target(prop.name, arch, board, compatible)}


def stats(samples, budget=None):
    a = np.array(samples, dtype=np.float64)
    require(a.ndim == 1 and a.size and np.isfinite(a).all() and (a > 0).all(), "invalid latency sample")
    out = {"count": a.size, "mean_ms": float(a.mean()), "median_ms": float(np.median(a)),
           "p95_ms": float(np.percentile(a, 95)), "min_ms": float(a.min()), "max_ms": float(a.max())}
    if budget:
        limit = budget["strict_target_ms"]
        out.update(strict_target_ms=limit, fraction_below_target=float((a < limit).mean()),
                   median_below_target=bool(out["median_ms"] < limit), p95_below_target=bool(out["p95_ms"] < limit))
    return out


def check_output(out, n):
    expected = {"trajs": (n, 6, 30, 2), "mode_probs": (n, 6), "act_probs": (n, 5)}
    require(all(k in out and tuple(out[k].shape) == v for k, v in expected.items()), "prediction output shape mismatch")
    require(all(v.dtype == torch.float32 and bool(torch.isfinite(v).all()) for v in out.values()), "non-finite or non-FP32 output")
    require(bool(torch.allclose(out["mode_probs"].sum(-1), torch.ones_like(out["mode_probs"][:, 0]), atol=1e-5)),
            "mode probabilities do not sum to one")


def check_trace_outputs(raw, traced, n):
    check_output(raw, n)
    check_output(traced, n)
    require(set(raw) == set(traced), "traced output keys changed")
    for k in raw:
        torch.testing.assert_close(traced[k], raw[k], rtol=1e-4, atol=1e-5, msg=lambda msg: f"{k}: {msg}")


def digest_outputs(out):
    return {k: sha(v.detach().contiguous().cpu().numpy().tobytes()) for k, v in sorted(out.items())}


def timed(model, batch, device, warmup, repeats):
    for _ in range(warmup):
        model(batch)
    torch.cuda.synchronize(device)
    wall = []
    for _ in range(repeats):
        torch.cuda.synchronize(device)
        start = time.perf_counter_ns()
        model(batch)
        torch.cuda.synchronize(device)
        wall.append((time.perf_counter_ns() - start) / 1e6)
    # Separate series: excludes host timer hooks from the wall-clock sample.
    first = torch.cuda.Event(enable_timing=True)
    last = torch.cuda.Event(enable_timing=True)
    stream = torch.cuda.current_stream(device)
    first.record(stream)
    last.record(stream)
    last.synchronize()
    events = []
    for _ in range(repeats):
        first.record(stream)
        model(batch)
        last.record(stream)
        last.synchronize()
        events.append(first.elapsed_time(last))
    return {"synchronized_host_wall": wall, "cuda_event_stream": events}


@torch.inference_mode()
def measure(model, data, groups, cells, args):
    device = cuda_device(args.device)
    hw = hardware(device)
    require(os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") != "1", "unset forced TF32 for this FP32 benchmark")
    require(os.environ.get("CUDA_LAUNCH_BLOCKING") != "1", "unset CUDA_LAUNCH_BLOCKING for comparable timing")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model = model.to(device=device, dtype=torch.float32)
    result = []
    traced, traced_shape = None, None
    before = telemetry()
    for t, n, key in cells:
        rows = groups[t][key][:n]
        batch = {k: torch.from_numpy(np.ascontiguousarray(data[t][k][rows])).to(device) for k in shapes(t)}
        reference = model(batch)
        check_output(reference, n)
        active = model
        if args.backend == "torchscript":
            if traced_shape != (t, n):
                traced = torch.jit.trace(model, (batch,), strict=False, check_trace=False)
                traced_shape = (t, n)
            active = traced
            predicted = active(batch)
            check_trace_outputs(reference, predicted, n)
        else:
            predicted = reference
        record = {"t_obs": t, "agents": n, "scene": scene_token(key), "input_support": support(data[t], rows),
                  "output_sha256": digest_outputs(predicted)}
        del reference, predicted
        series = timed(active, batch, device, args.warmup, args.repetitions)
        record["statistics"] = {k: stats(values, hw["proposal_budget"] if k == "synchronized_host_wall" else None)
                                for k, values in series.items()}
        record["raw_ms"] = series
        result.append(record)
    return {"hardware": hw, "telemetry_before": before, "telemetry_after": telemetry(), "cells": result}


def save_new(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"output already exists: {path}")
    raw = (json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(raw)
    try:
        os.link(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--fixture-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="New JSON file")
    p.add_argument("--backend", choices=("eager", "torchscript"), default="eager")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--repetitions", type=int, default=100)
    p.add_argument("--agent-counts", default="1,2,3,4,5,6,7,8,9,10,11,12")
    p.add_argument("--check-only", action="store_true", help="Validate inputs and strictly load weights on CPU; skip all forwards")
    args = p.parse_args()
    counts = [int(x) for x in args.agent_counts.split(",")]
    require(bool(counts) and len(counts) == len(set(counts)) and all(1 <= n <= 12 for n in counts), "invalid agent counts")
    require(args.warmup > 0 and args.repetitions > 0, "need warm-up and measured repetitions")
    require(not args.output.exists(), "choose a new output file")
    protocol, data, groups = load_fixture(args.fixture_dir)
    model = load_model(args.repo)
    cells = plan(groups, sorted(counts))
    report = {"status": "check_only_no_forwards" if args.check_only else "cuda_timing",
              "start_utc": datetime.now(timezone.utc).isoformat(), "runner_sha256": sha(Path(__file__).read_bytes()),
              "protocol_sha256": PROTOCOL_HASH, "protocol": protocol, "model_sources_sha256": SOURCES,
              "model": "Ours_Full, original PINNS T5 seed42 weights shared for all durations",
              "parameter_count": 471442, "state_entries": 161, "backend": args.backend,
              "evaluation_scope": "Model forward only, warmed resident FP32 devices and features. Each row is one complete synchronous multi-target forward. Neither feature preparation/transfer nor output copying and selection is timed.",
              "numpy": np.__version__, "agent_counts": sorted(counts), "schedule_seed": ORDER_SEED,
              "warmup_per_cell": args.warmup, "repeats_per_series_per_cell": args.repetitions,
              "arithmetic": "FP32. Eager calls the original model with two internally scripted loops. Optional torchscript additionally traces the entire module per T and N. Matmul/cuDNN TF32 and cuDNN benchmark off.",
              "percentile": "per physical instant, NumPy linear interpolation at 0.95; wall target applied to the actual hardware only",
              "cell_count": len(cells), "scene_target_selection": "ascending distinct person ID nested within each clip and tick; background neighbors are held intact",
              "plan": [{"t_obs": t, "agents": n, "scene": scene_token(k), "input_support": support(data[t], groups[t][k][:n])}
                       for t, n, k in cells]}
    if not args.check_only:
        report.update(measure(model, data, groups, cells, args))
    report["finish_utc"] = datetime.now(timezone.utc).isoformat()
    save_new(args.output, report)
    print(f"{report['status']}; {len(cells)} cells; saved {args.output}")


if __name__ == "__main__":
    main()
