"""Leave one physical PINNS site out. Run python -m scripts.benchmark_heldout_sites DATA_ROOT."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import cv2  # The project parser uses OpenCV to determine ground-plane handedness.
import numpy as np

from src.data.constants import DT, T_FUT
from src.data.pinns_extractor import _check_right_handed_homography, _parse_pinns_label_file


HISTORY = 5
WINDOW = HISTORY + T_FUT
RIDGE_ALPHA = 1.0


def clip_windows(label_path, calib):
    frames = _parse_pinns_label_file(
        label_path, _check_right_handed_homography(calib), calib["y_min"], calib["y_max"]
    )
    tracks = defaultdict(dict)
    for frame, objects in frames.items():
        for obj in objects:
            if obj["type"].lower().startswith("per"):
                positions = tracks[obj["id"]]
                if frame in positions:
                    raise ValueError(f"Duplicate person {obj['id']} at frame {frame} in {label_path.name}")
                xy = (obj["wx"], obj["wy"])
                if np.isfinite(xy).all():
                    positions[frame] = xy

    windows = []
    used_tracks = 0
    for positions in tracks.values():
        start_count = len(windows)
        times = np.array(sorted(positions))
        for run in np.split(times, np.flatnonzero(np.diff(times) != 1) + 1):
            for start in range(0, len(run) - WINDOW + 1, WINDOW):
                indices = run[start : start + WINDOW]
                xy = np.array([positions[t] for t in indices], dtype=np.float64)
                windows.append(xy - xy[HISTORY - 1])
        used_tracks += len(windows) > start_count
    return windows, used_tracks


def load_sites(root):
    members_path = root / "archive_members.txt"
    members = set(members_path.read_text().splitlines()) if members_path.exists() else None
    groups, counts, inputs = defaultdict(list), {}, set()
    for labels in sorted((root / "labels").glob("*.txt")):
        if labels.stem.lower().startswith("readme") or "_" not in labels.stem:
            continue
        condition, clip = labels.stem.rsplit("_", 1)
        video = Path("videos") / condition / "videos" / f"ori_{clip}.mp4"
        if not ((video.as_posix() in members) if members is not None else (root / video).is_file()):
            raise ValueError(f"Missing source video entry for {labels.name}")
        calib_file = root / "videos" / condition / "calib.json"
        calib = json.loads(calib_file.read_text())
        if not np.isclose(calib["fps"], 30):
            raise ValueError(f"Unexpected frame rate in {calib_file}")
        site = f"{calib['location']}_{calib['scene']}"
        windows, tracks = clip_windows(labels, calib)
        groups[site].extend(windows)
        entry = counts.setdefault(site, {"clips": 0, "clips_with_windows": 0, "clip_person_tracks_with_windows": 0})
        entry["clips"] += 1
        entry["clips_with_windows"] += int(bool(windows))
        entry["clip_person_tracks_with_windows"] += tracks
        inputs.update([labels, calib_file])
    if len(groups) != 4 or not all(groups.values()):
        raise ValueError("Expected four physical sites, each with complete person windows")
    digest = hashlib.sha256()
    for path in sorted(inputs):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return {site: np.stack(windows) for site, windows in groups.items()}, counts, digest.hexdigest()


def constant_velocity(history):
    times = DT * np.arange(HISTORY)
    centered = times - times.mean()
    velocity = np.einsum("t,ntd->nd", centered, history) / (centered @ centered)
    return velocity[:, None, :] * (DT * np.arange(1, T_FUT + 1))[None, :, None]


def ridge_predictions(train, test_history):
    # Positions already refer to the last observed frame, whose coordinates are identically zero.
    x = train[:, : HISTORY - 1].reshape(len(train), -1)
    y = train[:, HISTORY:].reshape(len(train), -1)
    center, scale, intercept = x.mean(axis=0), x.std(axis=0), y.mean(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    z = (x - center) / scale
    weights = np.linalg.solve(z.T @ z + RIDGE_ALPHA * np.eye(z.shape[1]), z.T @ (y - intercept))
    test = test_history[:, :-1].reshape(len(test_history), -1)
    return (((test - center) / scale) @ weights + intercept).reshape(-1, T_FUT, 2)


def score(prediction, future):
    errors = np.linalg.norm(prediction - future, axis=2)
    return {"ade_m": float(errors.mean()), "fde_m": float(errors[:, -1].mean())}


def benchmark(root):
    groups, counts, digest = load_sites(root)
    results = {}
    for site, test in sorted(groups.items()):
        training_sites = sorted(set(groups) - {site})
        train = np.concatenate([groups[name] for name in training_sites])
        history, future = test[:, :HISTORY], test[:, HISTORY:]
        results[site] = {
            **counts[site], "train_sites": training_sites, "train_windows": len(train), "test_windows": len(test),
            "constant_velocity": score(constant_velocity(history), future),
            "ridge": score(ridge_predictions(train, history), future),
        }
    macro = {
        model: {metric: float(np.mean([r[model][metric] for r in results.values()])) for metric in ("ade_m", "fde_m")}
        for model in ("constant_velocity", "ridge")
    }
    return {
        "protocol": {
            "task": "Motion-only. Refit per held-out physical site. All four sites were seen by the wider project; no independent pose-network validation.",
            "coordinates": "Annotated pedestrian ground positions in metres, downsampled from 30 to 10 Hz without interpolation, with the project's calibration-based Y orientation.",
            "windows": "Five complete observations (0.4 s span), then 30 complete targets (3.0 s). Within each clip/person, split at missing or nonfinite frames and take consecutive, disjoint 35-frame blocks from each run's start. Drop short remainders.",
            "constant_velocity": "Least-squares velocity from all five history positions; extrapolate from the last observed point.",
            "ridge": "Eight coordinates of four observed positions relative to the last position predict 60 future coordinates. Z-score features and center targets using only the three training sites. Unpenalized intercept; fixed ridge alpha 1.0 on summed squared loss. No validation or tuning.",
            "metrics": "Both models scored on the identical test windows. ADE is mean Euclidean distance over all 30 steps; FDE is distance at 3.0 s. Means within site weight windows equally; macro weights the four sites equally. No independence assumption or significance test.",
            "numpy_version": np.__version__, "opencv_version": cv2.__version__,
            "label_and_calibration_digest_sha256": digest,
        },
        "test_windows_total": sum(r["test_windows"] for r in results.values()), "sites": results, "macro": macro,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path, help="PINNS labels/ and videos/<condition>/calib.json; metadata-only copies need archive_members.txt.")
    parser.add_argument("--output", type=Path, default=Path("results/pinns_heldout_sites.json"))
    args = parser.parse_args()
    output = benchmark(args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
