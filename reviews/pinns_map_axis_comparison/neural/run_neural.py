"""Same-worker M2a geometry trial. Call run(aligned_samples, outdir) on Kaggle CUDA."""

from collections import Counter
from contextlib import ExitStack
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import socket
import tempfile
from unittest.mock import patch
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile

import numpy as np
import torch

from src.data.dataset_builder import materialize_window_split
from src.data.pinns_causal_input import (
    FULL_WINDOW_SHA256,
    MAP_SHA256,
    WINDOW_SHA256,
    full_motion_fingerprint,
    motion_fingerprint,
    window_key,
)
from src.engine import trainer


PROTOCOL_SHA = "1baef784110b6cc70a4f61a2aeba371191a4c314516524b61fde7198b1390a9b"
EXPECTED_MOTION = "03cd41ab844d52cb13c67ebf7a5b9ed243163681715ffbe60cbd4b569f70c6fa"
EXPECTED_FULL_MOTION = "67b1c8036ccf2378aadc6f9a08cd0aaf0ea3dff98db4ff95a36b42a0110800e3"
EXPECTED_MAP = "0d4abbe1ac37e4d6974601c40ad65fd3bcaf60b81b4bdca63878925657a32150"
SORTED_KEYS = "c086c690aff485ab3d51671371ed33368e9cfb274b7e15c4f0ae6236bc500f7a"
SORTED_GEOMETRY = "3199ef10fe0a90dc0dacb2c8bca298870f7278d81a9848f1f7e4dad7170cf27f"
SPLITS = {"train": 5686, "val": 1100, "test": 497}
SITES = ("america_crossroad", "italy_park", "japan_crossroad", "thailand_street")
ARMS = ("zero_xy", "real_xy")
METRICS = ("ade_m", "fde_last_valid_m", "fde_exact_3s_m")
CFG = dict(
    model_name="M2a",
    seed=42,
    device_str="cuda:0",
    epochs=28,
    batch_size=512,
    lr=0.002,
    dt=0.1,
    fut_steps=30,
    balanced_buckets=True,
)
OUTPUT_ZIP = "pinns_neural_image_geometry_pair.zip"


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def sha(x):
    return hashlib.sha256(x).hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def tensor_hash(tensors):
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        arr = np.ascontiguousarray(tensor.detach().cpu().numpy())
        digest.update(canonical([name, arr.dtype.str, list(arr.shape)]) + b"\0")
        digest.update(memoryview(arr).cast("B"))
    return digest.hexdigest()


def map_hash(pairs):
    digest = hashlib.sha256()
    for env, flag in pairs:
        env = np.asarray(env, dtype="<f4", order="C")
        flag = np.asarray(flag).reshape(-1)
        require(
            env.shape == (16, 10, 11) and flag.shape == (1,) and flag[0] in (0, 1),
            "Unexpected map shape or source availability",
        )
        digest.update(memoryview(env).cast("B"))
        digest.update(bytes([int(flag[0])]))
    return digest.hexdigest()


def site_from_clip(clip):
    found = [site for site in SITES if clip.startswith(site + "_")]
    require(len(found) == 1, "Unknown site in clip name")
    return found[0]


def check_samples(samples):
    """Pure identity and support check; no model construction, fitting or scoring."""
    data = sorted(samples, key=window_key)
    keys = [window_key(s) for s in data]
    require(
        len(keys) == len(set(keys)) == 7283 and sha(canonical(keys)) == SORTED_KEYS, "Wrong window roster"
    )
    require(Counter(s["split"] for s in data) == SPLITS, "Incorrect split")
    require(
        MAP_SHA256 == EXPECTED_MAP
        and WINDOW_SHA256 == EXPECTED_MOTION
        and motion_fingerprint(data) == EXPECTED_MOTION,
        "Wrong window motion or source-map registration",
    )
    require(
        FULL_WINDOW_SHA256 == EXPECTED_FULL_MOTION and full_motion_fingerprint(data) == EXPECTED_FULL_MOTION,
        "Actual source positions or masks changed",
    )
    require(
        map_hash((s["p_env"], s["m_map"]) for s in data) == SORTED_GEOMETRY,
        "Actual image geometry or availability changed",
    )
    splits = {sp: [r for r in data if r["split"] == sp] for sp in SPLITS}
    summary = {}
    for sp, rows in splits.items():
        source = sum(int(np.asarray(r["m_map"]).item()) for r in rows)
        local = sum(bool(np.any(r["p_env"][..., 10])) for r in rows)
        summary[sp] = {
            "windows": len(rows),
            "clips": len({window_key(r)[0] for r in rows}),
            "historical_reference": source,
            "local_image": local,
            "no_reference": len(rows) - source,
            "no_local_image": len(rows) - local,
        }
    require(
        summary["test"]["local_image"] == summary["test"]["historical_reference"] == 461
        and sum(bool(r["fut_mask"][-1]) for r in splits["test"]) == 233,
        "Test support differs from protocol",
    )
    return data, summary


def arm_tensors(base, arm):
    require(arm in ARMS, "Unknown geometry arm")
    if arm == "real_xy":
        return base
    out = dict(base)
    env = base["p_env"].clone()
    env[..., :2] = 0
    out["p_env"] = env
    return out


def history_fields(rows):
    keys = [window_key(s) for s in rows]
    return {
        "clip": np.asarray([k[0] for k in keys]),
        "person_id": np.asarray([k[1] for k in keys]),
        "t0": np.asarray([k[2] for k in keys]),
        "site": np.asarray([site_from_clip(k[0]) for k in keys]),
        "hist_mask": np.stack([s["hist_mask_20"][-5:] for s in rows]).astype(bool),
        "geometry_source_available": np.asarray([bool(np.asarray(s["m_map"]).item()) for s in rows]),
        "geometry_point_mask": np.stack([s["p_env"][..., 10] for s in rows]).astype(bool),
    }


def top_errors(paths, probs, truth, valid):
    n, steps = valid.shape
    require(
        paths.shape == (n, 6, steps, 2) and probs.shape == (n, 6) and truth.shape == (n, steps, 2),
        "Prediction or target dimensions",
    )
    require(
        np.isfinite(paths).all()
        and np.isfinite(probs).all()
        and (probs >= 0).all()
        and valid.any(axis=1).all(),
        "Invalid paths, probabilities or target support",
    )
    np.testing.assert_allclose(probs.sum(axis=1), 1, atol=3e-6, rtol=0)
    require(np.isfinite(truth[valid]).all(), "Nonfinite scored target")
    index = probs.argmax(axis=1)
    delta = paths[np.arange(n), index].astype(np.float64) - np.asarray(truth, dtype=np.float64)
    dist = np.hypot(delta[:, :, 0], delta[:, :, 1])
    end = steps - 1 - np.argmax(valid[:, ::-1], axis=1)
    return np.column_stack(
        (
            np.where(valid, dist, 0).sum(axis=1) / valid.sum(axis=1),
            dist[np.arange(n), end],
            np.where(valid[:, -1], dist[:, -1], np.nan),
        )
    )


def estimate(deltas, sites, clips):
    rng = np.random.Generator(np.random.PCG64(2026))
    draws = np.zeros(20000)
    for site in sorted(set(sites)):
        keep = sites == site
        names, group = np.unique(clips[keep], return_inverse=True)
        totals = np.bincount(group, weights=deltas[keep])
        sizes = np.bincount(group)
        selection = rng.integers(len(names), size=(len(draws), len(names)))
        draws += keep.mean() * totals[selection].sum(1) / sizes[selection].sum(1)
    return {
        "estimate_m": float(deltas.mean()),
        "clip_bootstrap_percentile_95_m": np.quantile(draws, [0.025, 0.975]).tolist(),
        "replicates": len(draws),
        "seed": 2026,
    }


def group_summary(mask, scores, fields):
    items = {}
    for name, error in scores.items():
        metrics = {}
        for col, metric in enumerate(METRICS):
            vals = error[mask, col]
            vals = vals[np.isfinite(vals)]
            metrics[metric] = {
                "sum_m": float(vals.sum()),
                "windows": len(vals),
                "mean_m": float(vals.mean()) if len(vals) else None,
            }
        items[name] = metrics
    return {
        "windows": int(mask.sum()),
        "clips": len(set(fields["clip"][mask])),
        "no_source": int((~fields["geometry_source_available"][mask]).sum()),
        "no_local_image": int((~fields["geometry_point_mask"][mask].any(axis=(1, 2))).sum()),
        "scored_future_steps": int(fields["fut_mask"][mask].sum()),
        "errors": items,
    }


def npz_bytes(values):
    stream = io.BytesIO()
    np.savez_compressed(stream, **values)
    return stream.getvalue()


def validated_archive(path, members):
    with ZipFile(path) as z:
        require(
            len(z.namelist()) == len(set(z.namelist()))
            and set(z.namelist()) == set(members) | {"sha256.json"},
            "Archive roster",
        )
        require(z.testzip() is None, "Archive CRC")
        wanted = json.loads(z.read("sha256.json"))
        require(set(wanted) == set(members), "Hash roster")
        values = {k: z.read(k) for k in members}
        require(all(sha(v) == wanted[k] for k, v in values.items()), "Archive fingerprint")
    return values


def write_archive(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
        partial = Path(tmp) / "new.zip"
        with ZipFile(partial, "w", ZIP_DEFLATED, compresslevel=1) as z:
            for name, raw in values.items():
                z.writestr(name, raw)
            z.writestr("sha256.json", canonical({k: sha(v) for k, v in values.items()}))
        validated_archive(partial, values)
        partial.replace(path)


def get_identity(data, tensors):
    dev = torch.cuda.get_device_properties(0)
    token = f"{socket.gethostname()}:{getattr(dev, 'uuid', '')}"
    os_boot = Path("/proc/sys/kernel/random/boot_id")
    if os_boot.exists():
        token += ":" + os_boot.read_text().strip()
    source = Path(trainer.__file__).resolve().parents[1]
    code = {
        str(p.relative_to(source)): sha(p.read_bytes())
        for folder in ("models", "engine", "data", "losses", "metrics")
        for p in (source / folder).rglob("*.py")
    }
    return {
        "protocol_sha256": PROTOCOL_SHA,
        "source_map_sha256": EXPECTED_MAP,
        "observed_image_payload_sha256": SORTED_GEOMETRY,
        "source_motion_sha256": EXPECTED_MOTION,
        "full_source_motion_sha256": EXPECTED_FULL_MOTION,
        "ordered_window_keys_sha256": SORTED_KEYS,
        "tensor_sha256": {sp: tensor_hash(blob) for sp, blob in tensors.items()},
        "split_key_sha256": {
            sp: sha(canonical([window_key(s) for s in data if s["split"] == sp])) for sp in SPLITS
        },
        "recipe": CFG,
        "runner_sha256": sha(Path(__file__).read_bytes()),
        "source_sha256": code,
        "execution": {
            "gpu": dev.name,
            "capability": list(torch.cuda.get_device_capability(0)),
            "device_index": 0,
            "worker_device_sha256": sha(token.encode()),
            "gpu_bytes": dev.total_memory,
            "torch": str(torch.__version__),
            "cuda": str(torch.version.cuda),
            "cudnn": torch.backends.cudnn.version(),
            "numpy": str(np.__version__),
            "python": platform.python_version(),
            "deterministic_algorithms": True,
            "tf32": False,
            "cublas_workspace": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        },
    }


def train_arm(tensors, arm, identity):
    """Observe the unmodified trainer's actual initialization, draws and selected test forward."""
    material = {s: arm_tensors(t, arm) for s, t in tensors.items()}
    original_build = trainer.build_any_model
    original_slice = trainer._slice_gpu_batch
    original_eval = trainer.evaluate_split_tensors
    initial = []
    sampled = hashlib.sha256()
    draws = []
    validation = []
    paths = []
    probabilities = []

    def observe_build(*args, **kwargs):
        model, with_action = original_build(*args, **kwargs)
        initial.append(tensor_hash(model.state_dict()))
        return model, with_action

    def observe_batch(data, index):
        idx = index.detach().cpu().numpy().astype("<i8", copy=False)
        sampled.update(idx.tobytes())
        draws.append(len(idx))
        return original_slice(data, index)

    def observe_eval(model, split, device, **kwargs):
        if split is material["val"]:
            result = original_eval(model, split, device, **kwargs)
            validation.append(float(result["slices"]["Overall"]["ADE_1"]))
            return result
        require(split is material["test"] and not paths, "Unexpected test forward")

        def save_outputs(mod, inputs, output):
            forbidden = {"fut_xy", "fut_mask", "y_act", "m_act"}
            require(not (forbidden & inputs[0].keys()), "Target passed into forecast")
            paths.append(output["trajs"].detach().cpu().numpy().copy())
            probabilities.append(output["mode_probs"].detach().cpu().numpy().copy())

        with model.register_forward_hook(save_outputs):
            return original_eval(model, split, device, **kwargs)

    with ExitStack() as stack:
        for key, fn in (
            ("build_any_model", observe_build),
            ("_slice_gpu_batch", observe_batch),
            ("evaluate_split_tensors", observe_eval),
        ):
            stack.enter_context(patch.object(trainer, key, fn))
        result = trainer.train_and_eval_seed(
            **CFG, train_data=material["train"], val_data=material["val"], test_data=material["test"]
        )
    checks = [ep for ep in range(1, CFG["epochs"] + 1) if ep % 4 == 0 or ep == CFG["epochs"]]
    require(
        len(initial) == 1
        and len(validation) == len(checks)
        and sum(draws) == CFG["epochs"] * len(material["train"]["x_m"]),
        "Training trace differs from registration",
    )
    require(
        np.isfinite(validation).all() and result["best_val_ade1"] == min(validation),
        "Validation selection mismatch",
    )
    predictions = np.concatenate(paths)
    probs = np.concatenate(probabilities)
    test = material["test"]
    errors = top_errors(predictions, probs, test["fut_xy"].numpy(), test["fut_mask"].numpy() > 0.5)
    np.testing.assert_allclose(errors[:, 0], result["test"]["sample_ade1"], rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(errors[:, 1], result["test"]["sample_fde1"], rtol=2e-6, atol=2e-6)
    state = result["best_state_cpu"]
    require(state is not None, "Required seed-42 weights were not returned")
    meta = {
        "arm": arm,
        "identity_sha256": identity,
        "initial_state_sha256": initial[0],
        "sampled_indices_sha256": sampled.hexdigest(),
        "sampled_windows": sum(draws),
        "minibatches": len(draws),
        "validation_ade_by_checkpoint_m": validation,
        "best_epoch": checks[int(np.argmin(validation))],
        "best_val_ade_m": float(result["best_val_ade1"]),
        "parameters": int(result["params"]),
        "train_seconds": float(result["train_sec"]),
        "best_state_sha256": tensor_hash(state),
    }
    weights = io.BytesIO()
    torch.save(state, weights)
    return {
        "fit.json": canonical(meta),
        "best_state.pt": weights.getvalue(),
        "predictions.npz": npz_bytes({"paths": predictions, "probabilities": probs}),
    }


def read_or_quarantine(path, names, identity, arm=None):
    if not path.is_file():
        return None
    try:
        packet = validated_archive(path, names)
        meta = json.loads(packet["fit.json" if arm else "aggregate.json"])
        require(
            meta["identity_sha256"] == identity and (arm is None or meta["arm"] == arm),
            "Other input, code or worker identity",
        )
        return packet
    except (ValueError, KeyError, TypeError, BadZipFile, OSError) as exc:
        fingerprint = sha(path.read_bytes())
        preserved = path.with_name(path.name + ".replaced-" + fingerprint)
        path.replace(preserved)
        print("Preserved incompatible or incomplete artifact", preserved.name, type(exc).__name__, flush=True)
        return None


def run(samples, outdir):
    """Run the two frozen arms serially. Writes a verified private result ZIP and returns its aggregate."""
    if not Path("/kaggle/working").is_dir() or not torch.cuda.is_available():
        raise RuntimeError("Real training and forward passes require Kaggle with CUDA enabled")
    import fcntl

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.cuda.set_device(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    root = Path(outdir)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".neural_image_pair.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _locked_run(samples, root)


def _locked_run(samples, root):
    rows, support = check_samples(samples)
    tensors = {sp: materialize_window_split(rows, sp, t_obs=5) for sp in SPLITS}
    for sp, t in tensors.items():
        require(
            all(torch.isfinite(t[k]).all().item() for k in ("x_m", "x_p", "q_vec", "p_env", "m_map")),
            "Nonfinite observed features",
        )
        has_joints = t["x_p"][..., 3] > 0.5
        support[sp]["windows_with_any_valid_pose_joint"] = int(has_joints.any(dim=(1, 2)).sum().item())
        support[sp]["frames_with_any_valid_pose_joint"] = int(has_joints.any(dim=2).sum().item())
    require(
        all(support[sp]["windows_with_any_valid_pose_joint"] > 0 for sp in SPLITS),
        "An entire registered split has no usable pose. Check the video and RTMPose extraction.",
    )
    info = get_identity(rows, tensors)
    identity = sha(canonical(info))
    test_rows = [s for s in rows if s["split"] == "test"]
    fields = history_fields(test_rows)
    fields.update(
        fut_xy=tensors["test"]["fut_xy"].numpy(), fut_mask=tensors["test"]["fut_mask"].numpy() > 0.5
    )
    members = ("aggregate.json", "identity.json", "test_rows.npz", *(n + ".zip" for n in ARMS))
    final_path = root / OUTPUT_ZIP
    previous = read_or_quarantine(final_path, members, identity)
    if previous is not None:
        return json.loads(previous["aggregate.json"])
    scores = {}
    training = {}
    arm_files = {}
    checkpoint_dir = root / "arm_checkpoints" / identity
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        name = checkpoint_dir / (arm + ".zip")
        parts = read_or_quarantine(name, ("fit.json", "predictions.npz", "best_state.pt"), identity, arm)
        if parts is None:
            print("Training registered M2a condition", arm, flush=True)
            parts = train_arm(tensors, arm, identity)
            write_archive(name, parts)
            torch.cuda.empty_cache()
        fit = json.loads(parts["fit.json"])
        with np.load(io.BytesIO(parts["predictions.npz"]), allow_pickle=False) as z:
            scores[arm] = top_errors(z["paths"], z["probabilities"], fields["fut_xy"], fields["fut_mask"])
        arm_files[arm + ".zip"] = name.read_bytes()
        training[arm] = fit
    for key in (
        "initial_state_sha256",
        "sampled_indices_sha256",
        "parameters",
        "minibatches",
        "sampled_windows",
    ):
        require(training[ARMS[0]][key] == training[ARMS[1]][key], "Two arms differ in " + key)
    delta = scores["real_xy"] - scores["zero_xy"]
    scores["real_minus_zero"] = delta
    sites, clips = fields["site"], fields["clip"]
    public = {
        "identity_sha256": identity,
        "protocol_sha256": PROTOCOL_SHA,
        "registered_sample_support": support,
        "fits": training,
        "backend": info["execution"],
        "primary_test_paired_ade": estimate(delta[:, 0], sites, clips),
        "pooled": group_summary(np.ones(len(sites), bool), scores, fields),
        "by_site": {s: group_summary(sites == s, scores, fields) for s in sorted(set(sites))},
        "by_clip": {c: group_summary(clips == c, scores, fields) for c in sorted(set(clips))},
    }
    write_archive(
        final_path,
        {
            "aggregate.json": canonical(public),
            "identity.json": canonical(info),
            "test_rows.npz": npz_bytes(fields),
            **arm_files,
        },
    )
    return public
