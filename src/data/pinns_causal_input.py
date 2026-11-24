"""Join the frozen strictly historical PINNS image contours to their source windows."""
from collections import Counter
import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np


MAP_SHA256 = "0d4abbe1ac37e4d6974601c40ad65fd3bcaf60b81b4bdca63878925657a32150"
WINDOW_SHA256 = "03cd41ab844d52cb13c67ebf7a5b9ed243163681715ffbe60cbd4b569f70c6fa"
FULL_WINDOW_SHA256 = "67b1c8036ccf2378aadc6f9a08cd0aaf0ea3dff98db4ff95a36b42a0110800e3"
SPLIT_COUNTS = {"train": 5686, "val": 1100, "test": 497}
PROTOCOL = "source_label_causal_image"


def window_key(sample):
    clip, person, end = sample["seq_id"], sample["person_id"], sample["t0_frame_10hz"]
    if not isinstance(clip, str) or any(
        not isinstance(value, (int, np.integer)) or isinstance(value, bool) for value in (person, end)
    ):
        raise ValueError("PINNS key must contain the clip name, integer person ID and integer 10 Hz frame")
    return clip, int(person), int(end)


def _identity_header(sample):
    key = [*window_key(sample), sample["split"]]
    return (json.dumps(key, ensure_ascii=True, separators=(",", ":")) + "\n").encode("ascii")


def motion_fingerprint(samples):
    """Encode source identity, split, observed track and scoring truth. Keep invalid slots out."""
    digest = hashlib.sha256()
    for sample in sorted(samples, key=window_key):
        digest.update(_identity_header(sample))
        hmask = np.asarray(sample["hist_mask_20"][-5:], dtype=bool)
        fmask = np.asarray(sample["fut_mask"], dtype=bool)
        if hmask.shape != (5,) or fmask.shape != (30,):
            raise ValueError("Expected five observations and 30 scoring steps")
        past = sample["hist_pos_20"][-5:][hmask] - sample["p_t0"]
        for mask, coordinates in ((hmask, past), (fmask, sample["fut_xy"][fmask])):
            digest.update(mask.astype("u1").tobytes())
            digest.update(np.ascontiguousarray(coordinates, dtype="<f4").tobytes())
    return digest.hexdigest()


def _absolute_track(sample, mask_name, xy_name, length):
    mask = np.asarray(sample[mask_name])
    xy = np.asarray(sample[xy_name])
    if mask.shape != (length,) or not np.all((mask == 0) | (mask == 1)):
        raise ValueError(f"{mask_name} must contain {length} binary entries")
    valid = mask.astype(bool)
    if xy.shape != (length, 2) or not np.isfinite(xy).all() or np.any(xy[~valid] != 0):
        raise ValueError(f"{xy_name} must contain finite XY positions and zero-padded missing steps")
    return valid, xy


def full_motion_fingerprint(samples):
    """Bind all 20 history and 30 future absolute positions, including their masks and padding."""
    digest = hashlib.sha256()
    for sample in sorted(samples, key=window_key):
        hmask, hist = _absolute_track(sample, "hist_mask_20", "hist_pos_20", 20)
        fmask, future = _absolute_track(sample, "fut_mask", "fut_pos_abs", 30)
        if not hmask[-1] or not np.array_equal(hist[-1], sample["p_t0"]):
            raise ValueError("PINNS anchor differs from the final observed ground position")
        relative = (future - hist[-1]) * fmask[:, None]
        if not np.array_equal(relative, sample["fut_xy"]):
            raise ValueError("PINNS relative scoring truth differs from the absolute future")
        digest.update(_identity_header(sample))
        for mask, xy in ((hmask, hist), (fmask, future)):
            digest.update(mask.astype("u1").tobytes())
            digest.update(np.asarray(xy, dtype="<f4").tobytes(order="C"))
    return digest.hexdigest()


def attach_causal_image(samples, path):
    """Return new sample dictionaries. The other-person hull is reserved for its separate control."""
    archive = Path(path).read_bytes()
    if hashlib.sha256(archive).hexdigest() != MAP_SHA256:
        raise ValueError("The private source-label image map does not match its registered bytes")
    if (Counter(s["split"] for s in samples) != SPLIT_COUNTS
            or full_motion_fingerprint(samples) != FULL_WINDOW_SHA256
            or motion_fingerprint(samples) != WINDOW_SHA256):
        raise ValueError("PINNS identity, motion or split changed before joining the image contours")
    with np.load(BytesIO(archive), allow_pickle=False) as data:
        keys = list(zip(data["clip"].tolist(), data["person_id"].tolist(), data["t0"].tolist()))
        index = {key: i for i, key in enumerate(keys)}
        if len(index) != len(samples) or set(index) != {window_key(s) for s in samples}:
            raise ValueError("Image map and target windows do not have a one-to-one match")
        splits, lines, available = data["split"], data["p_env_image"], data["source_eligible"]
        rows = [index[window_key(s)] for s in samples]
        if splits[rows].tolist() != [s["split"] for s in samples]:
            raise ValueError("Image-map subset and prediction-window subset differ")
    return [
        dict(s, p_env=lines[i], m_map=np.asarray([available[i]], dtype=np.float32))
        for s, i in zip(samples, rows)
    ]
