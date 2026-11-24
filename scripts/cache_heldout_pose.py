"""Cache frozen RTMPose detections for the five observed frames of the PINNS held-site windows.

Run from the project root with python -m scripts.cache_heldout_pose DATA_ROOT MODEL.onnx OUTPUT.npz.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from scripts.benchmark_heldout_sites import HISTORY, load_site_windows
from src.data.pinns_extractor import (
    RTMPoseONNXExtractor, _check_right_handed_homography, _parse_pinns_label_file,
)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_npz(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".npz.part")
    with tmp.open("wb") as writer:
        np.savez_compressed(writer, **arrays)
    tmp.replace(path)


def build_clip(labels, calib, video, windows, extractor):
    frames = _parse_pinns_label_file(labels, _check_right_handed_homography(calib), calib["y_min"], calib["y_max"])
    n = len(windows)
    people = np.asarray([w.person_id for w in windows], dtype=np.int64)
    times = np.asarray([w.t0 for w in windows], dtype=np.int64)
    targets = defaultdict(list)
    boxes = np.zeros((n, HISTORY, 4), dtype=np.float32)
    xy = np.zeros((n, HISTORY, 2), dtype=np.float64)
    pose = np.zeros((n, HISTORY, 17, 3), dtype=np.float32)
    inferred = np.zeros((n, HISTORY), dtype=np.bool_)
    ego_type = np.zeros(n, dtype=np.int64)
    for row, window in enumerate(windows):
        for t, frame in enumerate(range(window.t0 - HISTORY + 1, window.t0 + 1)):
            detections = [obj for obj in frames[frame] if obj["id"] == window.person_id
                          and obj["type"].lower().startswith("per")]
            if len(detections) != 1:
                raise ValueError(f"Expected one person for {(labels.stem, window.person_id, frame)}")
            det = detections[0]
            xy[row, t] = det["wx"], det["wy"]
            box = [det["ox"], det["oy"] - 0.5 * det["h"], det["w"], det["h"]]
            if np.isfinite(box).all() and box[2] > 0 and box[3] > 0:
                boxes[row, t] = box
                targets[frame].append((row, t))
        # Match the original ego-type rule using only the last two observed positions and t0 tag.
        speed = np.linalg.norm(xy[row, -1] - xy[row, -2]) / 0.1
        ego_type[row] = 2 if window.fixed == 1 or speed < 0.20 else 0
    xy -= xy[:, -1:, :].copy()
    np.testing.assert_allclose(xy, [w.xy[:HISTORY] for w in windows], rtol=0, atol=1e-10)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"Cannot open {video}")
    try:
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if height != calib["v_size"] or not np.isclose(fps, 30, atol=0.05):
            raise ValueError(f"Unexpected video geometry or rate in {video}: height={height}, fps={fps}")
        # Only t0-4 through t0 frames supply model inputs. No temporal pose interpolation.
        if targets:
            for raw in range(3 * max(targets) + 1):
                if not cap.grab():
                    break
                if raw % 3 or raw // 3 not in targets:
                    continue
                ok, image = cap.retrieve()
                if not ok:
                    continue
                row_ids, offsets = np.asarray(targets[raw // 3], dtype=int).T
                uv = extractor.extract_crops(image, boxes[row_ids, offsets])
                if not np.isfinite(uv).all():
                    raise ValueError(f"Nonfinite RTMPose output in {labels.stem}, frame {raw}")
                pose[row_ids, offsets] = uv
                inferred[row_ids, offsets] = True
    finally:
        cap.release()
    return {
        "clip": np.full(n, labels.stem, dtype=f"U{len(labels.stem)}"), "person_id": people, "t0": times,
        "hist_kpts": pose, "hist_bbox": boxes, "img_h": np.full(n, height, dtype=np.float32),
        "hist_pos_relative": xy, "hist_mask": np.ones((n, HISTORY), dtype=np.float32), "ego_type": ego_type,
        "pose_inferred": inferred,
    }


def run(root, model, output):
    if not model.is_file():
        raise FileNotFoundError(model)
    cv2.setNumThreads(1)
    model_digest = file_sha256(model)
    if model_digest != "5c0a4bf67953e6d2ac43ce15e77dc9d5d354ae18430a47d2c5963a7bc5683e3c":
        raise ValueError("Expected the frozen OpenMMLab RTMPose-m release used for this experiment")
    groups, _, data_digest = load_site_windows(root)
    by_clip = defaultdict(list)
    for windows in groups.values():
        for win in windows:
            by_clip[win.clip].append(win)
    if sum(len(windows) for windows in by_clip.values()) != 1482:
        raise ValueError("Source data do not reproduce the 1,482 registered motion windows")
    extractor = RTMPoseONNXExtractor(onnx_path=model, batch_size=32)
    if extractor.sess is None and extractor.torch_model is None:
        raise RuntimeError("A real RTMPose inference backend is required")
    parts = defaultdict(list)
    videos = {}
    # Keep verified per-clip shards so an interrupted extraction resumes without redoing earlier clips.
    shard_dir = output.parent / (output.stem + "_clips")
    script_digest = file_sha256(Path(__file__))
    for i, (clip, windows) in enumerate(sorted(by_clip.items()), start=1):
        condition, index = clip.rsplit("_", 1)
        labels = root / "labels" / f"{clip}.txt"
        calib = json.loads((root / "videos" / condition / "calib.json").read_text())
        video = root / "videos" / condition / "videos" / f"ori_{index}.mp4"
        videos[clip] = file_sha256(video)
        fingerprint = hashlib.sha256("|".join((script_digest, data_digest, model_digest, videos[clip])).encode()).hexdigest()
        shard = shard_dir / f"{clip}.npz"
        arrays = None
        if shard.exists():
            with np.load(shard, allow_pickle=False) as saved:
                if str(saved["fingerprint"]) == fingerprint:
                    arrays = {k: saved[k] for k in saved.files if k != "fingerprint"}
        if arrays is None:
            arrays = build_clip(labels, calib, video, windows, extractor)
            save_npz(shard, {**arrays, "fingerprint": np.array(fingerprint)})
        for key, data in arrays.items():
            parts[key].append(data)
        print(f"[{i}/{len(by_clip)}] {clip}: {len(windows)} windows, "
              f"{arrays['pose_inferred'].sum()}/{len(windows) * HISTORY} frames inferred", flush=True)

    all_data = {key: np.concatenate(values) for key, values in parts.items()}
    expected = {(w.clip, w.person_id, w.t0) for ws in groups.values() for w in ws}
    keys = list(zip(all_data["clip"], all_data["person_id"], all_data["t0"]))
    if set(keys) != expected or len(keys) != len(expected):
        raise ValueError("Pose cache keys differ from the reference motion windows")
    if not all_data["pose_inferred"].any():
        raise ValueError("Zero frames were inferred. Check video/annotation alignment before training")
    metadata = {
        "source": "PINNS labels, calibration and original 30 Hz videos. One frozen RTMPose-m pass on each requested person crop.",
        "model_sha256": model_digest, "label_and_calibration_digest_sha256": data_digest,
        "video_sha256": videos, "providers": extractor.active_providers,
        "boxes": "Center (ox, oy - h/2) from observed PINNS person labels, width and height in pixels.",
        "frame_mapping": "Five 10 Hz observations t0-4..t0 inclusive. Raw zero-based frame = 3 * observation index.",
        "missing_pose": "Missing or invalid crop/frame is a zero 17x3 array and pose_inferred=False. No window is dropped.",
        "site_counts": {site: len(windows) for site, windows in sorted(groups.items())},
        "schema": 1,
    }
    save_npz(output, {**all_data, "metadata_json": np.array(json.dumps(metadata, sort_keys=True))})
    print(f"Saved {len(keys)} windows to {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("onnx_path", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.data_root, args.onnx_path, args.output)
