"""Independently audit the registered M2a return against earlier Kaggle truth and frozen maps."""

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch


STUDY = Path(__file__).resolve().parent
ARMS = ("zero_xy", "real_xy")
METRICS = ("ade_m", "fde_last_valid_m", "fde_exact_3s_m")
RIDGE_ARCHIVE = "093a4d1901fe392e9181897287b86cddf686bd46d1a063a706c733cf7ca58fda"
RELEASE_RUNNER = "7f0612ac16044c0c92017450e096270d3f4b1522e825c05715d1ddf163a7f0d0"
RELEASE_SOURCES = "d98fc3449cfc628b03d976497c4853d21a46d5a1b81ca573a87355bae8c271e5"
PROTOCOL = "1baef784110b6cc70a4f61a2aeba371191a4c314516524b61fde7198b1390a9b"
RECIPE = dict(
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


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def ensure_equal(expected, observed, label):
    if isinstance(expected, dict):
        require(isinstance(observed, dict) and set(observed) == set(expected), label + " fields")
        for key, value in expected.items():
            ensure_equal(value, observed[key], label + "." + key)
    elif isinstance(expected, list):
        require(isinstance(observed, list) and len(observed) == len(expected), label + " length")
        for i, value in enumerate(expected):
            ensure_equal(value, observed[i], f"{label}[{i}]")
    elif isinstance(expected, float):
        require(
            type(observed) in (float, int)
            and math.isfinite(observed)
            and math.isclose(expected, observed, rel_tol=1e-10, abs_tol=1e-10),
            label,
        )
    else:
        require(type(expected) is type(observed) and observed == expected, label)


def read_packet(raw_bytes, names):
    with ZipFile(io.BytesIO(raw_bytes)) as archive:
        roster = set(names)
        require(
            len(archive.namelist()) == len(roster) + 1
            and set(archive.namelist()) == roster | {"sha256.json"},
            "ZIP member roster",
        )
        values = {name: archive.read(name) for name in roster}
        ensure_equal(
            {key: sha(data) for key, data in values.items()},
            json.loads(archive.read("sha256.json")),
            "ZIP member hashes",
        )
    return values


def trajectory_errors(paths, probabilities, truth, mask):
    n = len(truth)
    require(
        n > 0 and truth.shape == (n, 30, 2) and mask.shape == (n, 30) and mask.dtype == np.bool_,
        "30-step truth and mask",
    )
    require(
        paths.shape == (n, 6, 30, 2)
        and probabilities.shape == (n, 6)
        and np.isfinite(paths).all()
        and np.isfinite(probabilities).all()
        and (probabilities >= 0).all(),
        "six finite candidates with nonnegative probabilities",
    )
    require(np.allclose(probabilities.sum(axis=1), 1.0, atol=3e-6, rtol=0), "probabilities must sum to one")
    result = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        available = np.flatnonzero(mask[i])
        require(available.size and np.isfinite(truth[i, available]).all(), "missing scoring truth")
        best = int(np.argmax(probabilities[i]))
        delta = paths[i, best, available].astype(np.float64) - truth[i, available]
        errors = np.sqrt(np.sum(delta**2, axis=1))
        result[i] = errors.mean(), errors[-1], errors[-1] if available[-1] == 29 else np.nan
    return result


def clip_interval(delta, sites, clips):
    delta, sites, clips = map(np.asarray, (delta, sites, clips))
    require(
        delta.ndim == 1
        and len(delta) > 0
        and sites.shape == clips.shape == delta.shape
        and np.isfinite(delta).all(),
        "paired bootstrap inputs",
    )
    rng = np.random.Generator(np.random.PCG64(2026))
    draws = np.zeros(20_000, dtype=float)
    for site in sorted(set(sites)):
        site_mask = sites == site
        names = sorted(set(clips[site_mask]))
        members = [delta[site_mask & (clips == name)] for name in names]
        sums = np.asarray([x.sum() for x in members])
        sizes = np.asarray([len(x) for x in members])
        choices = rng.integers(len(names), size=(len(draws), len(names)))
        draws += (sum(sizes) / len(delta)) * (sums[choices].sum(axis=1) / sizes[choices].sum(axis=1))
    return tuple(float(x) for x in np.percentile(draws, [2.5, 97.5]))


def checked_npz(raw, names):
    with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:
        require(len(arrays.files) == len(names) and set(arrays.files) == set(names), "NPZ member roster")
        return {key: arrays[key] for key in names}


def checkpoint_fingerprint(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        require(isinstance(name, str) and isinstance(tensor, torch.Tensor), "tensor-only checkpoint")
        value = np.ascontiguousarray(tensor.detach().cpu().numpy())
        require(np.isfinite(value).all(), "finite checkpoint tensors")
        digest.update(canonical([name, value.dtype.str, list(value.shape)]) + b"\0" + value.tobytes())
    return digest.hexdigest()


def verify_fit(raw, arm, identity, public_fit, *, registered_model=False):
    packet = read_packet(raw, ("fit.json", "best_state.pt", "predictions.npz"))
    fit = json.loads(packet["fit.json"])
    ensure_equal(public_fit, fit, arm + " public fit")
    require(fit["arm"] == arm and fit["identity_sha256"] == identity, "arm identity")
    val = np.asarray(fit["validation_ade_by_checkpoint_m"], dtype=float)
    require(val.shape == (7,) and np.isfinite(val).all() and (val >= 0).all(), "seven validation checks")
    require(
        fit["best_epoch"] == 4 * (1 + int(np.argmin(val))) and fit["best_val_ade_m"] == float(val.min()),
        "first best validation epoch",
    )
    require(
        fit["minibatches"] == 28 * math.ceil(5686 / 512) and fit["sampled_windows"] == 28 * 5686,
        "registered training budget",
    )
    require(
        all(
            len(fit[k]) == 64 and all(ch in "0123456789abcdef" for ch in fit[k])
            for k in ("sampled_indices_sha256", "initial_state_sha256")
        ),
        "training trace fingerprints",
    )
    state = torch.load(io.BytesIO(packet["best_state.pt"]), map_location="cpu", weights_only=True)
    require(
        isinstance(state, dict) and state and checkpoint_fingerprint(state) == fit["best_state_sha256"],
        "returned best checkpoint fingerprint",
    )
    if registered_model:
        schema = [
            [key, value.detach().numpy().dtype.str, list(value.shape)] for key, value in sorted(state.items())
        ]
        require(
            fit["parameters"] == 324102
            and sha(canonical(schema)) == "96c7989c053d823abd3d707d5d79fe0b9f7dbe8507297c96b99b068af0f4ea23",
            "registered M2a state schema and parameter count",
        )
    pred = checked_npz(packet["predictions.npz"], ("paths", "probabilities"))
    return fit, pred["paths"], pred["probabilities"]


def keys(arrays):
    return list(zip(arrays["clip"].tolist(), arrays["person_id"].tolist(), arrays["t0"].tolist()))


def truth_and_maps(ridge, maps, enrollment):
    raw = Path(ridge).read_bytes()
    require(sha(raw) == RIDGE_ARCHIVE, "earlier real Kaggle return signature")
    with (
        ZipFile(io.BytesIO(raw)) as archive,
        np.load(io.BytesIO(archive.read("causal_geometry_replay.npz"))) as prior,
    ):
        past_keys = keys(prior)
        index = sorted(range(len(past_keys)), key=past_keys.__getitem__)
        split_names = prior["split"].tolist()
        split_keys = {
            s: [past_keys[i] for i in index if split_names[i] == s] for s in ("train", "val", "test")
        }
        require(
            len(set(past_keys)) == len(past_keys) == 7283
            and sha(canonical(sorted(past_keys))) == enrollment["ordered_keys_sha256"],
            "reference identity",
        )
        sel = [i for i in index if split_names[i] == "test"]
        fields = {
            out: prior[source][sel]
            for out, source in (
                ("clip", "clip"),
                ("person_id", "person_id"),
                ("t0", "t0"),
                ("site", "site"),
                ("hist_mask", "historical_mask"),
                ("fut_mask", "future_mask"),
                ("fut_xy", "truth_relative"),
            )
        }
    map_bytes = Path(maps).read_bytes()
    require(sha(map_bytes) == enrollment["source_label_maps_sha256"], "registered corrected map")
    with np.load(io.BytesIO(map_bytes)) as data:
        map_keys = keys(data)
        map_index = {key: i for i, key in enumerate(map_keys)}
        require(len(map_index) == len(map_keys) and set(map_index) == set(past_keys), "map keys")
        rows = [map_index[key] for key in split_keys["test"]]
        require((data["split"][rows] == "test").all(), "map split")
        fields["geometry_source_available"] = data["source_eligible"][rows].astype(bool)
        fields["geometry_point_mask"] = data["p_env_image"][rows, :, :, 10].astype(bool)
    return fields, {s: sha(canonical(k)) for s, k in split_keys.items()}


def group(mask, scores, fields):
    errors = {}
    for arm, values in scores.items():
        errors[arm] = {}
        for i, metric in enumerate(METRICS):
            valid = values[mask, i]
            valid = valid[np.isfinite(valid)]
            errors[arm][metric] = dict(
                windows=len(valid),
                sum_m=float(valid.sum()),
                mean_m=float(valid.mean()) if len(valid) else None,
            )
    return dict(
        windows=int(mask.sum()),
        clips=len(set(fields["clip"][mask])),
        no_source=int(np.count_nonzero(~fields["geometry_source_available"][mask])),
        no_local_image=int(np.count_nonzero(~fields["geometry_point_mask"][mask].any(axis=(1, 2)))),
        scored_future_steps=int(fields["fut_mask"][mask].sum()),
        errors=errors,
    )


def verify_return(path, reference_ridge, corrected_maps):
    raw = Path(path).read_bytes()
    top = read_packet(raw, ("test_rows.npz", "aggregate.json", "identity.json", "zero_xy.zip", "real_xy.zip"))
    reported = json.loads(top["aggregate.json"])
    identity = json.loads(top["identity.json"])
    fingerprint = sha(canonical(identity))
    enrollment = json.loads((STUDY / "input_check.json").read_bytes())
    full = json.loads((STUDY / "input_identity/verified.json").read_bytes())
    expected = {
        "protocol_sha256": PROTOCOL,
        "source_map_sha256": enrollment["source_label_maps_sha256"],
        "ordered_window_keys_sha256": enrollment["ordered_keys_sha256"],
        "source_motion_sha256": enrollment["motion_and_masks_sha256"],
        "full_source_motion_sha256": full["full_20_frame_absolute_sha256"],
        "observed_image_payload_sha256": enrollment["geometry_and_source_availability_sha256"],
        "runner_sha256": RELEASE_RUNNER,
        "recipe": RECIPE,
    }
    for key, val in expected.items():
        ensure_equal(val, identity[key], key)
    require(sha(canonical(identity["source_sha256"])) == RELEASE_SOURCES, "released source hash list")
    require(
        reported["identity_sha256"] == fingerprint and reported["protocol_sha256"] == PROTOCOL, "run identity"
    )
    ensure_equal(identity["execution"], reported["backend"], "backend")
    execution = reported["backend"]
    require(
        execution["device_index"] == 0
        and execution["deterministic_algorithms"] is True
        and execution["tf32"] is False
        and execution["cublas_workspace"] == ":4096:8",
        "registered GPU settings",
    )
    require(
        isinstance(identity["tensor_sha256"], dict)
        and set(identity["tensor_sha256"]) == {"train", "val", "test"}
        and all(
            isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)
            for v in identity["tensor_sha256"].values()
        ),
        "split tensor digest form",
    )
    fields, split_hashes = truth_and_maps(reference_ridge, corrected_maps, enrollment)
    ensure_equal(split_hashes, identity["split_key_sha256"], "all split keys")
    claimed = checked_npz(top["test_rows.npz"], list(fields))
    for key, value in fields.items():
        require(np.array_equal(claimed[key], value), "external ground truth or map differs: " + key)
    for sp, counts in enrollment["splits"].items():
        support = reported["registered_sample_support"][sp]
        windows, frames = (
            support[k] for k in ("windows_with_any_valid_pose_joint", "frames_with_any_valid_pose_joint")
        )
        require(0 < windows <= counts["windows"] and windows <= frames <= 5 * windows, "pose support bounds")
        for key, val in counts.items():
            ensure_equal(val, reported["registered_sample_support"][sp][key], "split support")
    scores, traces = {}, {}
    for arm in ARMS:
        trace, paths, probs = verify_fit(
            top[arm + ".zip"], arm, fingerprint, reported["fits"][arm], registered_model=True
        )
        traces[arm] = trace
        scores[arm] = trajectory_errors(paths, probs, fields["fut_xy"], fields["fut_mask"])
    for key in ("initial_state_sha256", "sampled_indices_sha256", "parameters"):
        require(traces["zero_xy"][key] == traces["real_xy"][key], "unpaired " + key)
    scores["real_minus_zero"] = scores["real_xy"] - scores["zero_xy"]
    delta = scores["real_minus_zero"][:, 0]
    lo, hi = clip_interval(delta, fields["site"], fields["clip"])
    primary = dict(
        estimate_m=float(delta.mean()), clip_bootstrap_percentile_95_m=[lo, hi], replicates=20000, seed=2026
    )
    ensure_equal(primary, reported["primary_test_paired_ade"], "primary ADE")
    ensure_equal(group(np.ones(len(delta), dtype=bool), scores, fields), reported["pooled"], "pooled")
    for key in ("site", "clip"):
        expected = {name: group(fields[key] == name, scores, fields) for name in sorted(set(fields[key]))}
        ensure_equal(expected, reported["by_" + key], "by " + key)
    return dict(
        archive_sha256=sha(raw),
        protocol_sha256=PROTOCOL,
        test_windows=len(delta),
        exact_3s_windows=int(fields["fut_mask"][:, -1].sum()),
        primary_test_paired_ade=primary,
        scope="Stored predictions, checkpoint digests, reported selection, and independently sourced targets and map masks. No model forward or retraining.",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("archive", "prior_ridge", "corrected_maps"):
        p.add_argument(arg, type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = verify_return(args.archive, args.prior_ridge, args.corrected_maps)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
