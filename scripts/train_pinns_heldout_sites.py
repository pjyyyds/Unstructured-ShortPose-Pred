"""Fit and evaluate M0/M1 separately for each held-out PINNS site (seed 42)."""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from scripts.benchmark_heldout_sites import HISTORY, constant_velocity, load_site_windows
from src.data.constants import DT, T_FUT
from src.data.features import build_motion_features, build_pose_and_quality_features
from src.engine.trainer import set_deterministic_seed
from src.losses.losses import JointPredictionLoss
from src.metrics.metrics_and_stats import compute_sample_trajectory_metrics
from src.models.predictor import UnstructuredShortPosePredictor


ROOT = Path(__file__).resolve().parent.parent
MODELS = ("M0", "M1")


@dataclass(frozen=True)
class Protocol:
    seed: int = 42
    epochs: int = 28
    batch_size: int = 64
    learning_rate: float = 0.002
    min_lr_ratio: float = 0.05
    weight_decay: float = 0.0001
    gradient_clip: float = 2.0


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        for block in iter(lambda: reader.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hash():
    paths = [Path(__file__).resolve(), ROOT / "scripts/benchmark_heldout_sites.py", *sorted((ROOT / "src").rglob("*.py"))]
    identity = [(p.relative_to(ROOT).as_posix(), file_hash(p)) for p in paths]
    return hashlib.sha256(json.dumps(identity).encode()).hexdigest()


def window_key(row):
    return row.clip, row.person_id, row.t0


def tensorize(groups, pose_path):
    rows = [r for site in sorted(groups) for r in groups[site]]
    keys = [window_key(r) for r in rows]
    n = len(rows)
    shapes = {
        "clip": (n,), "person_id": (n,), "t0": (n,),
        "hist_kpts": (n, HISTORY, 17, 3), "hist_bbox": (n, HISTORY, 4),
        "img_h": (n,), "pose_inferred": (n, HISTORY),
        "hist_pos_relative": (n, HISTORY, 2), "hist_mask": (n, HISTORY),
        "ego_type": (n,), "metadata_json": (),
    }
    with np.load(pose_path, allow_pickle=False) as cache:
        data = {k: cache[k] for k in shapes}
    for k, shape in shapes.items():
        if data[k].shape != shape:
            raise ValueError(f"Pose cache {k} expected {shape}, got {data[k].shape}")
    if data["clip"].dtype.kind != "U" or any(data[k].dtype.kind not in "iu" for k in ("person_id", "t0")):
        raise ValueError("Pose cache identities require Unicode clip stems and integer person_id/t0")
    cache_keys = list(zip(data["clip"].tolist(), data["person_id"].tolist(), data["t0"].tolist()))
    lookup = {k: i for i, k in enumerate(cache_keys)}
    if len(lookup) != n or len(set(keys)) != n or set(keys) != set(lookup):
        raise ValueError("Pose cache must contain exactly one record per label-derived window ID")

    order = [lookup[k] for k in keys]
    xy = np.stack([r.xy for r in rows])
    if not np.allclose(data["hist_pos_relative"][order], xy[:, :HISTORY], rtol=0, atol=1e-4):
        raise ValueError("Cached motion does not match the source clip, ID, history, or coordinate convention")
    if not np.all(data["hist_mask"] == 1):
        raise ValueError("All reference windows must have complete motion observations")
    if not all(np.isfinite(data[k]).all() for k in ("hist_kpts", "hist_bbox", "img_h")) or np.any(data["img_h"] <= 0):
        raise ValueError("Expected finite keypoints, boxes and positive image heights")
    inferred = data["pose_inferred"]
    if not np.isin(inferred, (0, 1)).all() or np.any(data["hist_kpts"][inferred == 0] != 0):
        raise ValueError("Missing inference frames must have zero pose")
    metadata = json.loads(data["metadata_json"].item())

    observed = np.ones(HISTORY, dtype=np.float32)
    xm = np.stack([build_motion_features(hist, observed, dt=DT) for hist in xy[:, :HISTORY]])
    pose = [build_pose_and_quality_features(data["hist_kpts"][i], data["hist_bbox"][i],
                                            observed, float(data["img_h"][i])) for i in order]
    xp, q, _, buckets = map(np.stack, zip(*pose))
    if not np.any(q[:, 0] > 0):
        raise ValueError("No usable pose joints in the cache")
    ego = np.zeros((n, 8), dtype=np.float32)
    standing = np.array([r.fixed == 1 for r in rows]) | (xm[:, -1, 4] < 0.20)
    ego[np.arange(n), np.where(standing, 2, 0)] = 1.0
    if not np.array_equal(ego.argmax(axis=1), data["ego_type"][order]):
        raise ValueError("Cached pedestrian types differ from the observed PINNS labels and velocity")
    arrays = {
        "x_m": xm, "x_p": xp, "q_vec": q, "ego_type_oh": ego,
        "fut_xy": xy[:, HISTORY:].astype(np.float32),
        "fut_mask": np.ones((n, T_FUT), dtype=np.float32),
    }
    tensors = {k: torch.from_numpy(v) for k, v in arrays.items()}

    cv = constant_velocity(xy[:, :HISTORY])
    distances = np.linalg.norm(cv - xy[:, HISTORY:], axis=2)
    cv_errors = np.stack([distances.mean(axis=1), distances[:, -1]], axis=1)
    index, coverage, start = {}, {}, 0
    for site in sorted(groups):
        ids = np.arange(start, start + len(groups[site]))
        start += len(ids)
        index[site] = ids
        coverage[site] = {
            "history_frames": len(ids) * HISTORY,
            "frames_inferred": int(inferred[order][ids].sum()),
            "frames_with_any_valid_joint": int((xp[ids, :, :, 3] > 0).any(axis=-1).sum()),
            "windows_without_valid_joints": int((q[ids, 0] == 0).sum()),
            "quality_buckets_clear_partial_heavy": np.bincount(buckets[ids], minlength=3).tolist(),
        }
    return tensors, index, keys, cv_errors, coverage, metadata


def fit_predict(variant, tensors, train_ids, test_ids, device, protocol):
    """Shuffle the three training sites uniformly each epoch; score the final epoch once."""
    set_deterministic_seed(protocol.seed, dev=device)
    model = UnstructuredShortPosePredictor(variant).to(device)
    loss_fn = JointPredictionLoss(use_action_loss=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol.learning_rate, weight_decay=protocol.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=protocol.epochs, eta_min=protocol.learning_rate * protocol.min_lr_ratio
    )
    rng = torch.Generator().manual_seed(protocol.seed + 7)
    train_ids = torch.as_tensor(train_ids, dtype=torch.long)
    inputs = ("x_m", "ego_type_oh") if variant == "M0" else ("x_m", "ego_type_oh", "x_p", "q_vec")
    for _ in range(protocol.epochs):
        model.train()
        ordered = train_ids[torch.randperm(len(train_ids), generator=rng)]
        for idx in ordered.split(protocol.batch_size):
            batch = {k: tensors[k][idx].to(device) for k in inputs}
            optimizer.zero_grad(set_to_none=True)
            losses = loss_fn(model(batch), tensors["fut_xy"][idx].to(device), tensors["fut_mask"][idx].to(device))
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), protocol.gradient_clip, error_if_nonfinite=True)
            optimizer.step()
        scheduler.step()

    errors = []
    model.eval()
    with torch.no_grad():
        for idx in torch.as_tensor(test_ids, dtype=torch.long).split(protocol.batch_size):
            out = model({k: tensors[k][idx].to(device) for k in inputs})
            metrics = compute_sample_trajectory_metrics(out["trajs"].cpu().numpy(), out["mode_probs"].cpu().numpy(),
                                                        tensors["fut_xy"][idx].numpy(), tensors["fut_mask"][idx].numpy())
            errors.append(np.stack([metrics["ADE_1"], metrics["FDE_1"]], axis=1))
    errors = np.concatenate(errors)
    if not np.isfinite(errors).all():
        raise ValueError(f"Non-finite predictions for {variant}")
    return model.cpu().state_dict(), errors


def atomic_write(path, writer):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
    try:
        writer(temp)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def save_json(path, data):
    atomic_write(path, lambda temp: temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n"))


def mean_metrics(errors):
    return dict(zip(("ade_m", "fde_m"), errors.mean(axis=0).tolist()))


def run(args):
    protocol = Protocol()
    groups, counts, digest = load_site_windows(args.data_root)
    reference = json.loads((ROOT / "results/pinns_heldout_sites.json").read_text())
    if digest != reference["protocol"]["label_and_calibration_digest_sha256"]:
        raise ValueError("PINNS labels/calibrations differ from the archived 1,482-window motion benchmark")
    tensors, index, keys, cv, coverage, metadata = tensorize(groups, args.pose_cache)
    if metadata.get("label_and_calibration_digest_sha256") != digest:
        raise ValueError("Pose cache provenance does not match the registered label and calibration files")
    if set(index) != set(reference["sites"]) or len(keys) != reference["test_windows_total"]:
        raise ValueError("Site list or window count differs from the registered motion benchmark")
    for site, ids in index.items():
        ref = reference["sites"][site]
        if len(ids) != ref["test_windows"] or any(abs(v - ref["constant_velocity"][k]) > 1e-10
                                                  for k, v in mean_metrics(cv[ids]).items()):
            raise ValueError(f"{site} does not reproduce the archived CV score and sample count")
    signature = {
        "protocol": asdict(protocol), "data_sha256": digest, "pose_sha256": file_hash(args.pose_cache),
        "ordered_ids_sha256": hashlib.sha256(json.dumps(keys, separators=(",", ":")).encode()).hexdigest(),
        "source_sha256": source_hash(), "torch_version": str(torch.__version__),
    }
    if args.site and args.site not in groups:
        raise ValueError(f"Unknown site {args.site}. Available: {sorted(groups)}")
    if not args.summarize_only:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("The full experiment requires a CUDA device")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    summaries = {}
    for site in ([args.site] if args.site else sorted(groups)):
        test = index[site]
        training_sites = sorted(set(groups) - {site})
        train = np.concatenate([index[name] for name in training_sites])
        ids = [list(keys[i]) for i in test]
        results = {"constant_velocity": cv[test]}
        devices = {}
        for variant in MODELS:
            path = args.output_dir / site / f"{variant}.pt"
            expected = {**signature, "model": variant, "test_site": site, "training_sites": training_sites}
            if path.exists():
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                if checkpoint["identity"] != expected or checkpoint["window_keys"] != ids:
                    raise ValueError(f"Stale fold checkpoint {path}; choose a separate output directory")
                errors = checkpoint["errors"].numpy()
            elif args.summarize_only:
                raise FileNotFoundError(f"Missing fold {path}")
            else:
                print(f"Training {variant}, held out {site}, {len(train)} train / {len(test)} test", flush=True)
                state, errors = fit_predict(variant, tensors, train, test, device, protocol)
                checkpoint = {"identity": expected, "window_keys": ids, "model_state": state,
                              "errors": torch.from_numpy(errors), "device": torch.cuda.get_device_name(device)}
                atomic_write(path, lambda temp: torch.save(checkpoint, temp))
            if errors.shape != (len(test), 2) or not np.isfinite(errors).all():
                raise ValueError(f"Invalid saved metrics in {path}")
            results[variant] = errors
            devices[variant] = checkpoint["device"]
        summary = {"train_sites": training_sites, "train_windows": len(train), "test_windows": len(test),
                   **counts[site], "coverage": coverage[site], "training_devices": devices,
                   "metrics": {m: mean_metrics(err) for m, err in results.items()}}
        summaries[site] = summary
        fold = {**summary, "identity": signature, "window_keys": ids, "error_columns": ["ADE1_m", "FDE1_m"],
                "paired_window_errors": {m: err.tolist() for m, err in results.items()}}
        save_json(args.output_dir / site / "fold.json", fold)
    if not args.site:
        macro = {m: {k: float(np.mean([s["metrics"][m][k] for s in summaries.values()])) for k in ("ade_m", "fde_m")}
                 for m in ("constant_velocity", *MODELS)}
        report = {
            "identity": signature, "pose_metadata": metadata, "total_windows": len(keys),
            "window_key_fields": ["clip", "person_id", "t0_10hz"],
            "window_protocol": reference["protocol"]["windows"],
            "constant_velocity": reference["protocol"]["constant_velocity"],
            "selection": "Final epoch of the fixed schedule. Uniform shuffle of training windows without replacement.",
            "metric_definitions": "Top-probability trajectory: 30-step mean distance (ADE), last-step distance "
                                  "at 3.0 s (FDE), both in metres. Average windows within each site, then average sites.",
            "scope": "One seed. Each held-out site was seen in earlier project development.",
            "sites": summaries, "macro": macro,
        }
        save_json(args.output_dir / "report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("pose_cache", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--site", help="Run just this physical site (one fold); omit to run all four and aggregate.")
    parser.add_argument("--device", default="cuda:0", help="CUDA device for training.")
    parser.add_argument("--summarize-only", action="store_true", help="Validate saved models and regenerate reports without training.")
    run(parser.parse_args())
