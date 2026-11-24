"""Paired CUDA audit of the archived full predictor and its joint/channel reassociation."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import itertools
import json
import os
from pathlib import Path
import random
import tempfile
import time
import zipfile

import numpy as np
import torch


BUNDLE = "pinns_sync_latency_private.bin"
BUNDLE_SHA = "362efbaa576340495749b0820f398d768b82e000e714492265b91183ef061873"
RUNNER_SHA = "5de7c4993380ced7341bc04e4667a51fb7d4d81ce12c5296dac4e7b8df464e61"
TRANSFORM_SHA = "3059a50208a07b76294f763f500c1043fce638b4902657380e9a6aa5188bfe23"
PROB_TOL = 2e-5
TRAJ_TOL = 2e-4
NEAR_GAP = 2 * PROB_TOL
REPEATS = 100
WARMUPS = 25
SEED = 20251125
EXPECTED = {"trajs": (6, 30, 2), "mode_probs": (6,), "mode_logits": (6,),
            "act_probs": (5,), "act_logits": (5,), "gate_weights": (3,), "nbr_attn": (12,)}


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def stage(mount, folder, transform):
    options = sorted(mount.rglob(BUNDLE))
    require(options and all(sha(p.read_bytes()) == BUNDLE_SHA for p in options),
            "attach the original private .bin; all matching files must have its registered SHA")
    require(sha(transform.read_bytes()) == TRANSFORM_SHA, "unexpected pose inference source")
    with zipfile.ZipFile(options[0]) as z:
        require(z.testzip() is None, "input CRC failure")
        z.extractall(folder)
    frozen = folder / "measure_latency.py"
    require(sha(frozen.read_bytes()) == RUNNER_SHA, "unexpected original runner")
    spec = importlib.util.spec_from_file_location("frozen_sync_latency", frozen)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    (folder / "model/src/models/pose_inference.py").write_bytes(transform.read_bytes())
    return original


def check_states(original, optimized):
    names = {}
    for name, a, b in [("archived_state", original.state_dict(), optimized.state_dict()),
                       ("all_buffers", dict(original.named_buffers()), dict(optimized.named_buffers()))]:
        require(a.keys() == b.keys(), f"{name} names changed")
        for k in a:
            require(a[k].shape == b[k].shape and a[k].dtype == b[k].dtype and torch.equal(a[k], b[k]), f"{name}: {k} changed")
        names[name] = len(a)
    require(names["archived_state"] == 161, "unexpected checkpoint state size")
    return names


def summarize_tensor(x, y, tolerance):
    if x.shape != y.shape:
        return {"original_shape": list(x.shape), "optimized_shape": list(y.shape), "within_tolerance": False}
    a, b = x.astype(np.float64), y.astype(np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    d = np.abs(a[finite] - b[finite])
    return {"values": a.size, "nonfinite_pairs": int((~finite).sum()),
            "different_finite_values": int((d != 0).sum()), "above_tolerance": int((d > tolerance).sum()),
            "max_abs": float(d.max()) if d.size else None, "tolerance": tolerance,
            "within_tolerance": bool(finite.all() and (d <= tolerance).all())}


def ranks(x, y):
    ans = {"valid_rows": 0, "invalid_rows": 0, "top1_changed": 0, "full_order_changed": 0,
           "original_exact_top_ties": 0, "optimized_exact_top_ties": 0, "original_near_top_ties": 0,
           "optimized_near_top_ties": 0, "top1_changed_within_near_gap": 0,
           "top1_changed_within_observed_2x_max_change": 0, "inverted_pairs": 0, "inverted_pairs_within_near_gap": 0}
    for a, b in zip(x.astype(np.float64), y.astype(np.float64)):
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            ans["invalid_rows"] += 1
            continue
        ra, rb = np.argsort(-a, kind="stable"), np.argsort(-b, kind="stable")
        da, db = a[ra[0]] - a[ra[1]], b[rb[0]] - b[rb[1]]
        ans["valid_rows"] += 1
        ans["original_exact_top_ties"] += int(da == 0)
        ans["optimized_exact_top_ties"] += int(db == 0)
        ans["original_near_top_ties"] += int(da <= NEAR_GAP)
        ans["optimized_near_top_ties"] += int(db <= NEAR_GAP)
        ans["full_order_changed"] += int(not np.array_equal(ra, rb))
        if ra[0] != rb[0]:
            actual_gap = a[ra[0]] - a[rb[0]]
            ans["top1_changed"] += 1
            ans["top1_changed_within_near_gap"] += int(actual_gap <= NEAR_GAP)
            ans["top1_changed_within_observed_2x_max_change"] += int(actual_gap <= 2 * np.max(np.abs(a-b)))
        qa, qb = np.argsort(ra), np.argsort(rb)
        for i, j in itertools.combinations(range(len(a)), 2):
            if (qa[i] < qa[j]) != (qb[i] < qb[j]):
                ans["inverted_pairs"] += 1
                ans["inverted_pairs_within_near_gap"] += int(abs(a[i] - a[j]) <= NEAR_GAP)
    return ans


def compare(a, b, n):
    errors, by_field = [], {}
    for name in sorted(set(a) | set(b) | set(EXPECTED)):
        if name not in a or name not in b or name not in EXPECTED:
            errors.append(f"{name}: missing or unexpected output")
            continue
        x, y = a[name], b[name]
        if any(z.dtype != np.float32 or z.shape != (n, *EXPECTED[name]) for z in (x, y)):
            errors.append(f"{name}: dtype/shape mismatch")
        by_field[name] = summarize_tensor(x, y, TRAJ_TOL if name == "trajs" else PROB_TOL)
        if name in ("mode_probs", "act_probs"):
            for z in (x, y):
                if not np.isfinite(z).all() or not ((z >= 0) & (z <= 1)).all():
                    errors.append(f"{name}: nonfinite or outside [0,1]")
            if name == "mode_probs" and any(z.ndim != 2 or not np.allclose(z.sum(1), 1, rtol=0, atol=1e-5) for z in (x, y)):
                errors.append("mode_probs: invalid normalization")
    result = {"checks": sorted(set(errors)), "by_field": by_field,
              "all_values_within_bound": bool(not errors and all(s["within_tolerance"] for s in by_field.values()))}
    for name, out in (("mode_probs", "ranking"), ("act_probs", "actions_at_0_5")):
        if name in a and name in b and all(z[name].shape == (n, *EXPECTED[name]) for z in (a, b)):
            if name == "mode_probs":
                result[out] = ranks(a[name], b[name])
            else:
                finite = np.isfinite(a[name]) & np.isfinite(b[name])
                result[out] = {"changed_by_class": (finite & ((a[name] >= .5) != (b[name] >= .5))).sum(0).tolist()}
    return result


def as_cpu(out):
    return {k: v.detach().cpu().contiguous().numpy() for k, v in out.items()}


def order_for(scene, t, n, pairs=REPEATS):
    require(pairs > 0 and pairs % 2 == 0, "paired schedule must be even")
    seed = int(sha(f"{SEED}:{scene}:{t}:{n}".encode()), 16)
    order = ["AB"] * (pairs // 2) + ["BA"] * (pairs // 2)
    random.Random(seed).shuffle(order)
    return order


def measure_pairs(models, batch, order, synchronize, clock=time.perf_counter_ns):
    series = []
    for sequence in order:
        row = {"order": sequence}
        for name in sequence:
            synchronize()
            start = clock()
            models[name](batch)
            synchronize()
            row[name] = (clock() - start) / 1e6
            require(np.isfinite(row[name]) and row[name] > 0, "invalid wall-clock sample")
        series.append(row)
    return series


def distribution(values):
    x = np.asarray(values, dtype=np.float64)
    return {"mean": float(x.mean()), "median": float(np.median(x)), "p05": float(np.percentile(x, 5)),
            "p95": float(np.percentile(x, 95)), "minimum": float(x.min()), "maximum": float(x.max())}


def paired_summary(rows):
    a, b = (np.array([r[c] for r in rows], dtype=np.float64) for c in "AB")
    delta = a - b
    return {"original_ms": distribution(a), "optimized_ms": distribution(b),
            "paired_saving_ms": distribution(delta), "paired_saving_percent": distribution(100 * delta / a),
            "pairs": len(rows), "optimized_faster_fraction": float((delta > 0).mean()),
            "order_strata_median_saving_ms": {order: float(np.median([r["A"] - r["B"] for r in rows if r["order"] == order]))
                                               for order in ("AB", "BA")}}


def primary(cells):
    keep = [c for c in cells if c["t_obs"] == 20 and c["agents"] == 12]
    require(len(keep) == 2 and len({c["scene"] for c in keep}) == 2, "incomplete primary scenes")
    return {"scope": "T20, N12. Median within each scene, arithmetic mean of the two scene medians.",
            "positive_is_faster": True,
            "by_scene": {c["scene"]: {"site": c["input_support"]["site"], "time": c["time"]} for c in keep},
            "equal_scene_paired_saving_ms": float(np.mean([c["time"]["paired_saving_ms"]["median"] for c in keep])),
            "equal_scene_paired_saving_percent": float(np.mean([c["time"]["paired_saving_percent"]["median"] for c in keep]))}


def overall(cells, key):
    records = [c[key] for c in cells]
    fields = {}
    for field in EXPECTED:
        stats = [r["by_field"][field] for r in records if field in r["by_field"] and "values" in r["by_field"][field]]
        maxes = [s["max_abs"] for s in stats if s["max_abs"] is not None]
        fields[field] = {k: sum(s[k] for s in stats) for k in ("values", "nonfinite_pairs", "different_finite_values", "above_tolerance")}
        fields[field].update(max_abs=max(maxes, default=None), cells_compared=len(stats))
    ranking = [r["ranking"] for r in records if "ranking" in r]
    return {"cells": len(records), "cells_passing_continuous_bounds": sum(r["all_values_within_bound"] for r in records),
            "fields": fields, "ranking": {k: sum(r[k] for r in ranking) for k in (ranking[0] if ranking else ())},
            "appearances": sum(c["agents"] for c in cells), "distinct_person_moments": 24}


def store_array(z, name, a):
    buf = io.BytesIO()
    np.save(buf, a, allow_pickle=False)
    z.writestr(name + ".npy", buf.getvalue())


@torch.inference_mode()
def run(args, temp):
    frozen = stage(args.input_root, temp, args.transform)
    protocol, arrays, groups = frozen.load_fixture(temp / "sync")
    a = frozen.load_model(temp / "model")
    from src.models.pose_inference import prepare_pose_inference
    b = prepare_pose_inference(a)
    model_states = check_states(a, b)
    device = frozen.cuda_device(args.device)
    require(os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") != "1" and os.environ.get("CUDA_LAUNCH_BLOCKING") != "1",
            "unset forced TF32 or CUDA launch blocking")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    models = {"A": a.to(device), "B": b.to(device)}
    require(check_states(*models.values()) == model_states, "device transfer changed model inventory")
    report = {"start_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": sha(Path(__file__).read_bytes()),
              "input_bundle_sha256": BUNDLE_SHA, "original_runner_sha256": RUNNER_SHA, "transform_sha256": TRANSFORM_SHA,
              "frozen_sources_sha256": frozen.SOURCES, "fixture_protocol_sha256": frozen.PROTOCOL_HASH,
              "inputs": protocol["files_sha256"], "checkpoint_sha256": frozen.CHECKPOINT_HASH,
              "state_equality": model_states, "hardware": frozen.hardware(device), "telemetry_before": frozen.telemetry(),
              "settings": {"warmups_per_arm_per_cell": WARMUPS, "measured_pairs_per_cell": REPEATS, "ab_and_ba": [50, 50],
                           "ordering_seed": SEED, "cell_order_seed": frozen.ORDER_SEED, "dtype": "FP32", "TF32": False,
                           "cudnn_benchmark": False, "top_level": "eager original and eager with three FX block transforms",
                           "near_original_prob_gap": NEAR_GAP, "trajectory_abs_tol_m": TRAJ_TOL, "other_abs_tol": PROB_TOL},
              "scope": "Resident-input predictive forwards only, two original grouped moments. Pilot deployment audit. Identical row inputs by construction. Every paired host latency is synchronized before and after a single complete model call.",
              "cells": []}
    with zipfile.ZipFile(temp / "private_outputs.npz", "w", zipfile.ZIP_DEFLATED) as z:
        for t, n, key in frozen.plan(groups, range(1, 13)):
            ix = groups[t][key][:n]
            scene = frozen.scene_token(key)
            batch = {k: torch.from_numpy(np.ascontiguousarray(arrays[t][k][ix])).to(device) for k in frozen.shapes(t)}
            for j in range(WARMUPS):
                for name in ("AB" if j % 2 == 0 else "BA"):
                    models[name](batch)
            torch.cuda.synchronize(device)
            snapshots = {}
            for tag in ("A", "B", "A_repeat", "B_repeat"):
                snapshots[tag] = as_cpu(models[tag[0]](batch))
                for field, arr in snapshots[tag].items():
                    store_array(z, f"T{t}_N{n}_{scene}_{tag}_{field}", arr)
            record = {"t_obs": t, "agents": n, "scene": scene, "input_support": frozen.support(arrays[t], ix),
                      "comparison": compare(snapshots["A"], snapshots["B"], n),
                      "within_original_repeat": compare(snapshots["A"], snapshots["A_repeat"], n),
                      "within_optimized_repeat": compare(snapshots["B"], snapshots["B_repeat"], n)}
            rows = measure_pairs(models, batch, order_for(scene, t, n), lambda: torch.cuda.synchronize(device))
            record.update(time=paired_summary(rows), pairs_ms=rows)
            report["cells"].append(record)
    require(len(report["cells"]) == 72 and sum(c["agents"] for c in report["cells"]) == 468, "incomplete cells")
    report["primary_speed"] = primary(report["cells"])
    report["all_cells_numerical"] = {k: overall(report["cells"], k) for k in
        ("comparison", "within_original_repeat", "within_optimized_repeat")}
    report["comparisons_passing_all_continuous_bounds"] = sum(c["comparison"]["all_values_within_bound"] for c in report["cells"])
    report["top1_row_appearances_changed"] = sum(c["comparison"].get("ranking", {}).get("top1_changed", 0) for c in report["cells"])
    report["row_appearances_with_valid_ranking"] = sum(c["comparison"].get("ranking", {}).get("valid_rows", 0) for c in report["cells"])
    report["telemetry_after"] = frozen.telemetry()
    report["finish_utc"] = datetime.now(timezone.utc).isoformat()
    (temp / "paired_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with tempfile.NamedTemporaryFile(dir=args.output.parent, suffix=".zip", delete=False) as f:
        output = Path(f.name)
    try:
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
            for name in ("paired_report.json", "private_outputs.npz"):
                z.write(temp / name, name)
        with zipfile.ZipFile(output) as z:
            require(z.testzip() is None, "invalid output CRC")
        os.link(output, args.output)
    finally:
        output.unlink(missing_ok=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    p.add_argument("--transform", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    require(args.output.parent.is_dir() and not args.output.exists(), "supply a new output file")
    with tempfile.TemporaryDirectory() as tmp:
        report = run(args, Path(tmp))
    print("Selected GPU:", report["hardware"]["device_name"])
    print("Primary equal-scene saving, milliseconds:", report["primary_speed"]["equal_scene_paired_saving_ms"])
    print("Cells passing continuous bounds:", report["comparisons_passing_all_continuous_bounds"], "/72")
    print("Saved", args.output)


if __name__ == "__main__":
    main()
