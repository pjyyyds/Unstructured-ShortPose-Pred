"""Build historical PINNS map tensors or load the same masked five-frame motion windows.

Run `python -m scripts.build_pinns_causal_maps DATA_ROOT BEV_ROOT OUTPUT_DIRECTORY`.
The original 47 clips use their archived window splits. A map source must precede
the trigger and belong to the same clip's training partition.
"""
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np

from scripts.audit_pinns_map_sources import digest, split_for_clip, target_id
from scripts.stage_pinns_metadata import metadata_identity
from src.data.pinns_extractor import (
    _assemble_sequence_windows,
    _check_right_handed_homography,
    _parse_pinns_label_file,
)
from src.data import pinns_bev
from src.data.pinns_bev import CONVENTIONS, LEGACY_RASTER, SOURCE_LABEL, bev_to_track_xy
from src.data.scene_graph import build_local_polylines
from src.data.splits import assign_site_clip_splits


REPLAY = Path(__file__).resolve().parents[1] / "results/pinns_heldout_replay.json"
SPLIT_COUNTS = {"train": 5686, "val": 1100, "test": 497}


@dataclass
class Clip:
    label: Path
    calibration_file: Path
    calibration: dict
    site: str
    flipped: bool
    frames: dict
    triggers: dict

    def windows(self):
        return keyed_windows(self.label.stem, self.frames, self.triggers, self.calibration["v_size"])


def edges_from_frames(video, selected, calib, flipped, *, bev_convention=SOURCE_LABEL):
    """Same Canny recipe as the frozen implementation; read only the nominated source frames."""
    cap = cv2.VideoCapture(str(video))
    images = []
    try:
        if abs(cap.get(cv2.CAP_PROP_FPS) - 30) > 0.05:
            raise ValueError(f"Expected 30 Hz: {video}")
        for index in selected:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = cap.read()
            if not ok or image is None:
                raise ValueError(f"Cannot decode source {video}, frame {index}")
            images.append(image)
    finally:
        cap.release()
    gray = cv2.cvtColor(np.median(images, axis=0).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 1.2), 60, 150)
    candidates, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    height, width = gray.shape
    result = []
    for contour in sorted(candidates, key=lambda c: cv2.arcLength(c, False), reverse=True)[:10]:
        if len(contour) < 4 or cv2.arcLength(contour, False) < 0.12 * max(height, width):
            continue
        uv = contour.reshape(-1, 2).astype(np.float32)
        xy = bev_to_track_xy(uv, width, height, calib, flipped, bev_convention)
        attrs = np.zeros(8, dtype=np.float32)
        attrs[[0, 7]] = 1
        result.append((xy, attrs))
    return result


def hull_from_other_people(reference, target):
    positions = [xy for pid, xy in reference if pid != target]
    if len(positions) < 20:
        return []
    hull = cv2.convexHull(np.asarray(positions, dtype=np.float32)).reshape(-1, 2)
    if len(hull) < 3:
        return []
    attr = np.zeros(8, dtype=np.float32)
    attr[[5, 6]] = 1
    return [(np.vstack([hull, hull[0]]), attr)]


def keyed_windows(clip, frames, triggers, height):
    tracks = defaultdict(dict)
    for t, objects in frames.items():
        for obj in objects:
            if obj["type"].lower().startswith("per"):
                tracks[obj["id"]][t] = np.asarray([obj["wx"], obj["wy"]], dtype=np.float32)
    for sample in _assemble_sequence_windows(clip, frames, {}, [], triggers, height, 4):
        yield target_id(sample, tracks), sample


def clips_in_order(data_root):
    expected = json.loads(REPLAY.read_text())["label_and_calibration_sha256"]
    files, _ = metadata_identity(data_root, expected)
    folders = defaultdict(dict)
    for member in files:
        if not member.startswith("labels/"):
            continue
        label = data_root / member
        condition, number = label.stem.rsplit("_", 1)
        folders[condition][int(number)] = label
    for condition, paths in sorted(folders.items()):
        calib_path = data_root / "videos" / condition / "calib.json"
        calib = json.loads(calib_path.read_text())
        site = calib["location"] + "_" + calib["scene"]
        flipped = _check_right_handed_homography(calib)
        frames = {path.stem: _parse_pinns_label_file(path, flipped, calib["y_min"], calib["y_max"])
                  for path in paths.values()}
        condition_split = assign_site_clip_splits(paths)
        for index, path in sorted(paths.items()):
            clip = path.stem
            triggers, *_ = split_for_clip(condition, index, frames, condition_split)
            yield Clip(path, calib_path, calib, site, flipped, frames[clip], triggers)


def masked_motion(sample):
    """Use the native float32 anchor and mark missing coordinates explicitly with zeros."""
    hmask = sample["hist_mask_20"][-5:].astype(bool)
    past = sample["hist_pos_20"][-5:]
    hist = np.zeros_like(past)
    hist[hmask] = past[hmask] - sample["p_t0"]
    fmask = sample["fut_mask"].astype(bool)
    future = np.zeros_like(sample["fut_xy"])
    future[fmask] = sample["fut_xy"][fmask]
    return hist, hmask, future, fmask


def load_motion_windows(data_root):
    """Map (clip, numeric person ID, t0 at 10 Hz) to site, split, relative history/mask and future/mask."""
    root = Path(data_root)
    windows = {}
    counts = Counter()
    for clip in clips_in_order(root):
        for person, sample in clip.windows():
            key = (clip.label.stem, person, sample["t0_frame_10hz"])
            if key in windows:
                raise ValueError(f"Duplicate PINNS motion window {key}")
            split = sample["split"]
            windows[key] = (clip.site, split, *masked_motion(sample))
            counts[split] += 1
    if counts != SPLIT_COUNTS:
        raise ValueError(f"Unexpected frozen cohort counts: {counts}")
    return windows


def build(data_root, bev_root, destination, *, bev_convention=SOURCE_LABEL):
    if bev_convention not in CONVENTIONS:
        raise ValueError(f"Unknown PINNS BEV convention: {bev_convention}")
    if destination.exists():
        raise FileExistsError(f"Use a new output directory: {destination}")
    cv2.setNumThreads(1)
    keys, splits, source_eligible, edges, hulls = [], [], [], [], []
    counts = defaultdict(Counter)
    manifest = {}
    blank = np.zeros((16, 10, 11), dtype=np.float32)
    for item in clips_in_order(data_root):
        path, calib_path, calib = item.label, item.calibration_file, item.calibration
        clip = path.stem
        condition, index = clip.rsplit("_", 1)
        site, flipped, frames, triggers = item.site, item.flipped, item.frames, item.triggers
        train = sorted(t for t, split in triggers.items() if split == "train")
        held = [t for t, split in triggers.items() if split != "train"]
        # The only rows eligible to construct assets are in the train partition.
        # Exclude the earliest held validation/test input history as an independent guard.
        source = [t for t in train if t <= train[0] + 60 and (not held or t < min(held) - 19)]
        if len(source) < 7:
            source = []
        end = source[-1] if source else None
        reference_set = set(source)
        raw = [3 * source[j] for j in np.linspace(0, len(source) - 1, 7, dtype=int)] if source else []
        if len(set(raw)) != len(raw) or any(triggers[f // 3] != "train" for f in raw):
            raise ValueError(f"Invalid reference frame in {clip}")
        video = bev_root / "videos" / condition / "videos" / f"bev_{index}.mp4"
        image_lines = edges_from_frames(video, raw, calib, flipped, bev_convention=bev_convention) if source else []
        reference = []
        for frame, objects in frames.items():
            if frame in reference_set:
                for obj in objects:
                    if obj["type"].lower().startswith("per"):
                        reference.append((obj["id"], np.array([obj["wx"], obj["wy"]], dtype=np.float32)))
        manifest[clip] = {
            "site": site, "reference_10hz": source, "raw_image_frames": raw, "reference_ped_rows": len(reference),
            "earliest_held_trigger_10hz": min(held) if held else None, "label_sha256": digest(path),
            "calibration_sha256": digest(calib_path), "bev_sha256": digest(video) if source else None,
        }
        hull_cache = {}
        for pid, sample in item.windows():
            t0 = sample["t0_frame_10hz"]
            eligible = end is not None and end < t0
            edge, hull = blank, blank
            if eligible:
                if max(raw) > 3 * (t0 - 1):
                    raise ValueError("Non-historical BEV source")
                if pid not in hull_cache:
                    hull_cache[pid] = hull_from_other_people(reference, pid)
                edge, _ = build_local_polylines(image_lines, sample["p_t0"], True)
                hull, _ = build_local_polylines(hull_cache[pid], sample["p_t0"], True)
            keys.append((clip, pid, t0))
            splits.append(sample["split"])
            source_eligible.append(eligible)
            edges.append(edge)
            hulls.append(hull)
            edge_used, hull_used = bool(edge[:, :, 10].any()), bool(hull[:, :, 10].any())
            for group in ("all", sample["split"], site + "/" + sample["split"]):
                counts[group].update(windows=1, eligible=int(eligible), image=int(edge_used),
                                     hull=int(hull_used), both=int(edge_used and hull_used))
    expected = SPLIT_COUNTS
    if len(set(keys)) != 7283 or len(keys) != 7283 or {k: counts[k]["windows"] for k in expected} != expected:
        raise ValueError("Frozen window IDs or split totals are inconsistent")
    edge = np.stack(edges)
    hull = np.stack(hulls)
    destination.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        destination / "maps.npz", clip=np.asarray([k[0] for k in keys]), person_id=np.asarray([k[1] for k in keys]),
        t0=np.asarray([k[2] for k in keys]), split=np.asarray(splits), source_eligible=np.asarray(source_eligible),
        p_env_image=edge, p_env_hull=hull, m_map_image=edge[:, :, :, 10].any(axis=(1, 2)).astype(np.float32),
        m_map_hull=hull[:, :, :, 10].any(axis=(1, 2)).astype(np.float32),
    )
    protocol = "causal_same_clip_training_reference_v1" if bev_convention == LEGACY_RASTER else "causal_same_clip_training_reference_v2_source_label"
    fields = {"protocol": protocol, "bev_convention": bev_convention,
              "bev_converter_sha256": digest(Path(pinns_bev.__file__)), "counts": counts, "sources": manifest,
              "script_sha256": digest(Path(__file__)), "npz_sha256": digest(destination / "maps.npz"),
              "opencv": cv2.__version__, "numpy": np.__version__}
    (destination / "manifest.json").write_text(json.dumps(fields, indent=2) + "\n")
    print(json.dumps({sp: counts[sp] for sp in ("all", *expected)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("bev_root", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--bev-convention", choices=CONVENTIONS, default=SOURCE_LABEL,
                        help="Use legacy_raster to reconstruct the original frozen Canny experiment.")
    args = parser.parse_args()
    build(args.data_root, args.bev_root, args.destination, bev_convention=args.bev_convention)
