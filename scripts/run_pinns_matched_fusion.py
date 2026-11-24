"""Compare true projected concatenation and the archived implicit gate on 623 matched PINNS windows.

From the repository root: python -m scripts.run_pinns_matched_fusion DATA_ROOT PILOT_POSE MATCHED_CACHE OUT
--check-only checks inputs, source and folds. --replay-only independently scores an extracted result bundle on CPU.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import zipfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.run_heldout_seed_sensitivity import configure_training, json_digest, split_indices
from scripts.run_pinns_matched_durations import (
    DURATIONS,
    PARENT_SHA,
    load_reference,
    read_arrays,
    tensors_for,
    validate_cache,
)
from scripts.train_pinns_heldout_sites import (
    Protocol,
    ROOT,
    atomic_write,
    file_hash,
    mean_metrics,
    save_json,
    source_hash,
)
from scripts.verify_heldout_bundle import ATOL_M, RTOL, errors_by_window, parameter_digest
from src.engine.trainer import set_deterministic_seed
from src.losses.losses import JointPredictionLoss
from src.metrics.metrics_and_stats import compute_sample_trajectory_metrics
from src.models.predictor import UnstructuredShortPosePredictor


VARIANTS = ("M1b", "M1-concat")
INPUTS = ("x_m", "ego_type_oh", "x_p", "q_vec")
CACHE_SHA = "59e59e3dfa6a2b5583ce4c3fbcb010866e793a3595e43cf691dbefbfcddc64f1"
KEY_SHA = "11ec9713c26fdd3db6a9a887b4ac8b79fbbf13c6389b5924fcd61c315a38aa34"
DATA_SHA = "0b276b34ab1b01dae91f67c56bfaea5cc931511b05b4fc1c92686a891e6f6801"
LEGACY_TREE = "e61a2310793e900c6188588c750e47c44e23fbf41f429fb94ccf8b2ea88fd094"
SOURCES = {
    "scripts/run_pinns_matched_durations.py": "d6aa83473a11740848430ca0701d6b73b0eda8f4eef3d9fb354faf5066e6d1ec",
    "scripts/audit_pinns_matched_histories.py": "b3ec0dd5eeed8072498f8962f4f97361d88bd04834eed36b4c187b64d219ce18",
    "scripts/run_heldout_seed_sensitivity.py": "e894ca8228f69e0f768ff80cc00ad7d240903c3c3a013e577f5603b0f242bf8d",
    "scripts/verify_heldout_bundle.py": "852d9108870aa8dcc161a76e6e2309d7a791f6513f869a1b631da24a6dbdaad5",
    "scripts/cache_pinns_frame_offset.py": "a6faed485afd537492841941627ce8f1aaf6fdd2ce026615409b62e937de7800",
}
TORCH_TRAIN = "2.11.0+cu130"


class ConcatenatedFusion(nn.Module):
    """Bias-free projection of concatenated channels with an exact missing-pose bypass."""

    def __init__(self, norm):
        super().__init__()
        self.out_norm = norm
        (width,) = norm.normalized_shape
        eye = torch.eye(width, device=norm.weight.device, dtype=norm.weight.dtype)
        # Construct a Parameter directly. An nn.Linear default initialization would advance the RNG.
        self.weight = nn.Parameter(torch.cat((eye, eye), dim=1))

    def forward(self, h_m, h_p=None, q_vec=None, q_part=None, part_feats=None):
        if h_p is None or q_vec is None:
            return self.out_norm(h_m), h_m.new_zeros((len(h_m), 1))
        present = q_vec[:, :1] > 0
        pose = torch.where(present, h_p, 0.0)
        projected = F.linear(torch.cat((h_m, pose), dim=-1), self.weight)
        return self.out_norm(torch.where(present, projected, h_m)), present.to(h_m.dtype)


def make_model(name):
    if name == "M1b":
        return UnstructuredShortPosePredictor(name)
    if name != "M1-concat":
        raise ValueError(f"Unknown matched fusion variant {name}")
    model = UnstructuredShortPosePredictor("M1a")
    model.quality_gate = ConcatenatedFusion(model.quality_gate.out_norm)
    model.variant = name
    return model


def fit_variant(name, tensors, train, test, device, protocol):
    """Same seeded optimizer/loss/batches/final epoch as archived fit_predict; instantiate the selected factory."""
    set_deterministic_seed(protocol.seed, dev=device)
    model = make_model(name).to(device)
    loss = JointPredictionLoss(use_action_loss=False).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=protocol.learning_rate, weight_decay=protocol.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=protocol.epochs, eta_min=protocol.learning_rate * protocol.min_lr_ratio
    )
    rng = torch.Generator().manual_seed(protocol.seed + 7)
    train = torch.as_tensor(train, dtype=torch.long)
    for _ in range(protocol.epochs):
        model.train()
        ordered = train[torch.randperm(len(train), generator=rng)]
        for idx in ordered.split(protocol.batch_size):
            batch = {key: tensors[key][idx].to(device) for key in INPUTS}
            optimizer.zero_grad(set_to_none=True)
            losses = loss(model(batch), tensors["fut_xy"][idx].to(device), tensors["fut_mask"][idx].to(device))
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), protocol.gradient_clip, error_if_nonfinite=True)
            optimizer.step()
        scheduler.step()
    model.eval()
    trajectories, probabilities, scores = [], [], []
    with torch.no_grad():
        for idx in torch.as_tensor(test, dtype=torch.long).split(protocol.batch_size):
            out = model({key: tensors[key][idx].to(device) for key in INPUTS})
            paths, probs = out["trajs"].cpu(), out["mode_probs"].cpu()
            metrics = compute_sample_trajectory_metrics(
                paths.numpy(), probs.numpy(), tensors["fut_xy"][idx].numpy(), tensors["fut_mask"][idx].numpy()
            )
            trajectories.append(paths)
            probabilities.append(probs)
            scores.append(torch.from_numpy(np.stack([metrics["ADE_1"], metrics["FDE_1"]], axis=1)))
    return model.cpu().state_dict(), {
        k: torch.cat(parts) for k, parts in (("trajs", trajectories), ("probs", probabilities), ("errors", scores))
    }


def load_inputs(data_root, parent, cache):
    if (
        source_hash() != LEGACY_TREE
        or {rel: file_hash(ROOT / rel) for rel in SOURCES} != SOURCES
        or file_hash(cache) != CACHE_SHA
    ):
        raise ValueError("Frozen training/model sources or matched pose cache changed")
    ref = load_reference(data_root, parent)
    arrays = read_arrays(cache)
    meta = validate_cache(arrays, ref)
    if (
        ref.label_digest != DATA_SHA
        or json_digest(ref.keys) != KEY_SHA
        or len(ref.keys) != 623
        or not arrays["pose_inferred"].all()
        or len(meta["zero_offset_controls"]["clips"]) != 47
    ):
        raise ValueError("Frozen window population or 12,460 pose observations changed")
    base = {
        "protocol": asdict(Protocol()),
        "data_sha256": DATA_SHA,
        "pilot_pose_sha256": PARENT_SHA,
        "matched_pose_sha256": CACHE_SHA,
        "ordered_keys_sha256": KEY_SHA,
        "core_source_sha256": LEGACY_TREE,
        "dependency_sha256": SOURCES,
        "runner_and_concat_sha256": file_hash(Path(__file__)),
        "training_torch": TORCH_TRAIN,
    }
    return ref, arrays, base


def check_predictions(detail, y):
    paths, probs, scores = (detail[key].numpy() for key in ("trajs", "probs", "errors"))
    n = len(y)
    if (
        paths.shape != (n, 6, 30, 2)
        or probs.shape != (n, 6)
        or scores.shape != (n, 2)
        or not all(np.isfinite(v).all() for v in (paths, probs, scores))
        or np.any(probs < 0)
    ):
        raise ValueError("Invalid archived trajectories, probabilities or errors")
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=2e-6, rtol=0)
    winner = paths[np.arange(n), probs.argmax(axis=1)]
    d = np.linalg.norm(winner.astype(float) - y.astype(float), axis=2)
    np.testing.assert_allclose(scores, np.column_stack([d.mean(axis=1), d[:, -1]]), atol=5e-6, rtol=1e-6)
    return scores


@torch.inference_mode()
def replay_saved(name, state, detail, tensors, test, batch_size):
    """Replay all six paths/probabilities as well as the independently computed top-mode metrics on CPU."""
    model = make_model(name).cpu().eval()
    model.load_state_dict(state, strict=True)
    if not all(torch.isfinite(value).all().item() for value in state.values()):
        raise ValueError(f"Nonfinite parameters in {name}")
    replay = []
    for start in range(0, len(test), batch_size):
        idx = torch.as_tensor(test[start : start + batch_size], dtype=torch.long)
        pred = model({key: tensors[key][idx] for key in INPUTS})
        paths, probs = pred["trajs"].numpy(), pred["mode_probs"].numpy()
        block = slice(start, start + len(idx))
        np.testing.assert_allclose(paths, detail["trajs"][block].numpy(), atol=ATOL_M, rtol=RTOL)
        np.testing.assert_allclose(probs, detail["probs"][block].numpy(), atol=2e-5, rtol=RTOL)
        winner = paths[np.arange(len(idx)), probs.argmax(axis=1)]
        replay.append(errors_by_window(winner, tensors["fut_xy"][idx].numpy()))
    return np.concatenate(replay)


def bundle_members(out, sites):
    return [
        out / "report.json",
        *[out / f"T{t}" / site / (name + ".pt") for t in DURATIONS for site in sorted(sites) for name in VARIANTS],
    ]


def check_manifest(out, files):
    expected = {p.relative_to(out).as_posix(): file_hash(p) for p in files}
    manifest = json.loads((out / "sha256.json").read_text())
    if manifest != expected:
        raise ValueError("Extracted bundle differs from its per-file SHA-256 manifest")


def run(data_root, parent, cache, out, check_only=False, replay_only=False):
    ref, arrays, base = load_inputs(data_root, parent, cache)
    protocol = Protocol()
    summary = {
        "identity": base,
        "site_counts": {s: len(ids) for s, ids in ref.index.items()},
        "windows": len(ref.keys),
        "model_definitions": {
            "M1b": "Original implicit gate with continuous quality inputs replaced by zeros, pose presence retained.",
            "M1-concat": "Original M1a backbone and initial non-fusion parameters; 256-to-128 trainable bias-free "
            "concatenation projection initialized [I I]. Missing pose bypasses projection to LN(h_m).",
        },
        "parameters": {name: sum(p.numel() for p in make_model(name).parameters()) for name in VARIANTS},
        "scope": "Single seed 42, four previously examined sites. Each fold trains on the other three sites; "
        "no model selection or five-seed inference. Holdout target is the top-probability mode.",
    }
    for site in ref.index:
        split_indices(ref.index, ref.keys, site)
    if check_only:
        print(json.dumps({k: summary[k] for k in ("windows", "site_counts", "parameters")}), flush=True)
        return summary
    device = torch.device("cuda:0")
    torch.set_num_threads(2)
    files = bundle_members(out, ref.index)
    if replay_only:
        check_manifest(out, files)
    else:
        if not Path("/kaggle/working").is_dir():
            raise RuntimeError("Real fitting requires a Kaggle GPU; use --check-only or --replay-only locally")
        configure_training(device, TORCH_TRAIN)
        out.mkdir(parents=True, exist_ok=True)
    metrics, seen_states, max_replay = {}, set(), 0.0
    for t in DURATIONS:
        tensors = tensors_for(arrays, t)
        metrics[str(t)] = {}
        for site in ref.index:
            source_sites, train, test = split_indices(ref.index, ref.keys, site)
            held_keys = [list(ref.keys[i]) for i in test]
            expected = {
                **base,
                "duration": t,
                "held_site": site,
                "training_sites": source_sites,
                "train_ids_sha256": json_digest([ref.keys[i] for i in train]),
                "test_ids_sha256": json_digest(held_keys),
            }
            site_metrics = {}
            for name in VARIANTS:
                path = out / f"T{t}" / site / (name + ".pt")
                identity = {**expected, "variant": name}
                if not path.exists() and not replay_only:
                    print(f"Fit T={t} site={site} variant={name} train={len(train)} test={len(test)}", flush=True)
                    state, detail = fit_variant(name, tensors, train, test, device, protocol)
                    check_predictions(detail, tensors["fut_xy"][test].numpy())
                    payload = {
                        "identity": identity,
                        "test_window_keys": held_keys,
                        "model_state": state,
                        "predictions": detail,
                    }
                    atomic_write(path, lambda tmp: torch.save(payload, tmp))
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                if checkpoint["identity"] != identity or checkpoint["test_window_keys"] != held_keys:
                    raise ValueError(f"Stale or incorrectly assigned checkpoint {path}")
                state = checkpoint["model_state"]
                state_hash = parameter_digest(state)
                if state_hash in seen_states:
                    raise ValueError("Repeated weights across distinct fitted experiments")
                seen_states.add(state_hash)
                saved = check_predictions(checkpoint["predictions"], tensors["fut_xy"][test].numpy())
                replay = replay_saved(name, state, checkpoint["predictions"], tensors, test, protocol.batch_size)
                np.testing.assert_allclose(replay, saved, atol=ATOL_M, rtol=RTOL)
                max_replay = max(max_replay, float(np.max(abs(replay - saved))))
                site_metrics[name] = mean_metrics(saved.astype(float))
            metrics[str(t)][site] = site_metrics
    summary.update(
        site_metrics=metrics,
        cpu_replayed=len(seen_states),
        max_cpu_replay_abs_m=max_replay,
        four_site_equal={
            t: {
                name: {k: float(np.mean([v[name][k] for v in by_site.values()])) for k in ("ade_m", "fde_m")}
                for name in VARIANTS
            }
            for t, by_site in metrics.items()
        },
    )
    if replay_only:
        published = json.loads((out / "report.json").read_text())
        for key in (
            "identity",
            "windows",
            "site_counts",
            "parameters",
            "cpu_replayed",
            "site_metrics",
            "four_site_equal",
        ):
            if published[key] != summary[key]:
                raise ValueError(f"Replayed published summary disagrees: {key}")
        return summary
    save_json(out / "report.json", summary)
    save_json(out / "sha256.json", {p.relative_to(out).as_posix(): file_hash(p) for p in files})

    def write_bundle(temp):
        with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as output:
            for path in [*files, out / "sha256.json"]:
                output.write(path, path.relative_to(out).as_posix())

    atomic_write(out / "matched_fusion.zip", write_bundle)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data_root", "pilot_pose", "matched_cache", "output"):
        parser.add_argument(name, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--replay-only", action="store_true")
    args = parser.parse_args()
    result = run(
        args.data_root, args.pilot_pose, args.matched_cache, args.output, args.check_only, args.replay_only
    )
    if not args.check_only:
        print(json.dumps(result["four_site_equal"]), flush=True)
