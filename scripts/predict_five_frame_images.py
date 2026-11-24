"""Predict six ground-plane paths and five supervised action probabilities from five images.

Boxes identify the same pedestrian in each frame. Express boxes and the homography
in the stored image's pixel coordinates, before EXIF orientation. The supplied
homography maps those pixels directly to the intended ground-plane axes in metres.
"""
import argparse
import json
from pathlib import Path
from pickle import UnpicklingError
import tempfile

import cv2
import numpy as np
import torch

from scripts.cache_heldout_pose import file_sha256
from scripts.predict_five_frame_pedestrian import finite_array, forecast
from scripts.run_heldout_cloud import MODEL_SHA
from src.data.constants import DT
from src.data.pinns_extractor import RTMPoseONNXExtractor


def camera_frames(camera_file):
    """Validate five image references and project the measured bottom centres without clipping."""
    document = json.loads(camera_file.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("Camera input must be a JSON object")
    homography = finite_array(document.get("image_to_ground_homography"), (3, 3), "image_to_ground_homography")
    scale = np.abs(homography).max()
    if scale == 0 or np.linalg.matrix_rank(homography / scale) != 3:
        raise ValueError("image_to_ground_homography must be nonsingular")
    homography = homography / scale
    size = document.get("image_size_px")
    if not isinstance(size, list) or len(size) != 2 or any(type(v) is not int or v <= 0 for v in size):
        raise ValueError("image_size_px must contain positive integer [width, height]")
    width, height = size
    frames = document.get("frames")
    if not isinstance(frames, list) or len(frames) != 5 or not all(isinstance(f, dict) for f in frames):
        raise ValueError("frames must contain five measurements for the same pedestrian")
    times = finite_array([f.get("timestamp_s") for f in frames], (5,), "frames[].timestamp_s")
    if not np.allclose(np.diff(times), DT, atol=0.001, rtol=0):
        raise ValueError("The five frames must be ordered and spaced 0.1 seconds apart (1 ms tolerance)")
    paths, boxes = [], []
    for i, frame in enumerate(frames):
        name = frame.get("image")
        if not isinstance(name, str) or not name.strip() or Path(name).is_absolute():
            raise ValueError(f"frames[{i}].image must be a path relative to the camera JSON")
        paths.append((camera_file.parent / name).resolve())
        box = finite_array(frame.get("bbox_center_xywh_px"), (4,), f"frames[{i}].bbox_center_xywh_px")
        cx, cy, bw, bh = box
        if bw <= 0 or bh <= 0:
            raise ValueError(f"frames[{i}] requires positive measured box dimensions")
        if cx + bw / 2 <= 0 or cy + bh / 2 <= 0 or cx - bw / 2 >= width or cy - bh / 2 >= height:
            raise ValueError(f"frames[{i}] measured box is entirely outside image_size_px")
        boxes.append(box)
    if len(set(paths)) != 5:
        raise ValueError("Supply five distinct image files, one per measurement")
    boxes = np.stack(boxes)
    pixels = np.column_stack((boxes[:, 0], boxes[:, 1] + boxes[:, 3] / 2, np.ones(5)))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        projected = pixels @ homography.T
        # Scale-invariant check for a point on (or numerically inseparable from) the horizon.
        tolerance = 32 * np.finfo(float).eps * (np.abs(pixels) @ np.abs(homography[2]))
        bad = ~np.isfinite(projected).all(1) | (np.abs(projected[:, 2]) <= tolerance)
        if bad.any():
            raise ValueError(f"Box bottom centre in frames[{np.flatnonzero(bad)[0]}] has no finite ground projection")
        xy = projected[:, :2] / projected[:, 2:3]
    if not np.isfinite(xy).all():
        raise ValueError("Box bottom centres must produce finite ground coordinates")
    return paths, boxes, times, xy, (width, height)


def read_image(path, size, index):
    if not path.is_file():
        raise ValueError(f"frames[{index}].image does not exist: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError(f"OpenCV could not decode frames[{index}]: {path}")
    actual = (image.shape[1], image.shape[0])
    if actual != size:
        raise ValueError(f"frames[{index}] decodes to {actual} pixels; expected image_size_px {size}")
    return image


def measured_observations(paths, boxes, times, xy, size, onnx):
    if not onnx.is_file() or file_sha256(onnx) != MODEL_SHA:
        raise ValueError(f"Expected the official RTMPose-m 256x192 ONNX file with SHA-256 {MODEL_SHA}")
    extractor = RTMPoseONNXExtractor(onnx_path=onnx, batch_size=1)
    if extractor.sess is None:
        raise RuntimeError("RTMPose ONNX Runtime session is unavailable")
    poses = []
    for i, (path, box) in enumerate(zip(paths, boxes)):
        image = read_image(path, size, i)
        joints = extractor.extract_crops(image, box.astype(np.float32)[None, :])
        if (joints.shape != (1, 17, 3) or not np.isfinite(joints).all()
                or np.any((joints[:, :, 2] < 0) | (joints[:, :, 2] > 1))):
            raise ValueError(f"Invalid RTMPose joints or confidences for frames[{i}]")
        poses.append({"bbox_center_xywh_px": box.tolist(), "keypoints_xyc": joints[0].tolist()})
    return {"timestamps_s": times.tolist(), "ground_xy_m": xy.tolist(), "image_height_px": size[1],
            "pose_frames": poses, "image_measurements": {
                "ground_position": "Measured box bottom centre projected by the supplied image-to-ground homography",
                "rtmpose_onnx_sha256": MODEL_SHA, "onnxruntime_providers": extractor.sess.get_providers()}}


def write_json(path, value):
    payload = json.dumps(value, allow_nan=False, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temp:
        temporary = Path(temp.name)
    try:
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run(checkpoint_path, camera_file, onnx, output, observations_out=None):
    paths, boxes, times, xy, size = camera_frames(camera_file)
    sources = {path.resolve() for path in (checkpoint_path, camera_file, onnx, *paths)}
    outputs = [path.resolve() for path in (output, observations_out) if path is not None]
    if len(set(outputs)) != len(outputs) or sources.intersection(outputs):
        raise ValueError("Forecast and observation outputs must be distinct from each other and the input files")
    if any(path.is_dir() for path in outputs):
        raise ValueError("Forecast and observations outputs must be file paths")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    identity = checkpoint.get("identity", {}) if isinstance(checkpoint, dict) else {}
    if not isinstance(identity, dict) or identity.get("arm") not in ("auxiliary", "end_to_end", "detached"):
        raise ValueError("Select a registered auxiliary, end_to_end or detached checkpoint with a supervised action head")
    observation = measured_observations(paths, boxes, times, xy, size, onnx)
    prediction = forecast(checkpoint, observation)
    if "observable_action_probabilities" not in prediction:
        raise ValueError("Expected five supervised action probabilities from the selected checkpoint")
    if observations_out is not None:
        write_json(observations_out, observation)
    write_json(output, prediction)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("camera_input", type=Path)
    parser.add_argument("rtmpose_onnx", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observations-out", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    try:
        run(args.checkpoint, args.camera_input, args.rtmpose_onnx, args.output, args.observations_out)
    except (OSError, EOFError, UnpicklingError, ImportError, ValueError, TypeError, KeyError, RuntimeError, cv2.error) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")


if __name__ == "__main__":
    main()
