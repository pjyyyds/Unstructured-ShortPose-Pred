"""Independently recover the neural map comparison's motion identity from PINNS text labels."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


SOURCE = "0b276b34ab1b01dae91f67c56bfaea5cc931511b05b4fc1c92686a891e6f6801"
SHORT = "03cd41ab844d52cb13c67ebf7a5b9ed243163681715ffbe60cbd4b569f70c6fa"
FULL = "67b1c8036ccf2378aadc6f9a08cd0aaf0ea3dff98db4ff95a36b42a0110800e3"
MAP = "0d4abbe1ac37e4d6974601c40ad65fd3bcaf60b81b4bdca63878925657a32150"


def verify(condition, detail):
    if not condition:
        raise ValueError(detail)


def read_label(data):
    tracks = defaultdict(dict)
    frames = set()
    for line in data.splitlines():
        row = line.split()
        if len(row) < 11:
            continue
        raw = int(row[0])
        if raw < 1 or raw % 3 != 1:
            continue
        frame = (raw - 1) // 3
        frames.add(frame)
        if row[1].lower() == b"person":
            track = tracks[int(row[2])]
            verify(frame not in track, "Duplicate source person and frame")
            track[frame] = (float(row[8]), float(row[9]))
    return tracks, sorted(frames)


def assign_splits(frames, number, siblings):
    # Three or more clips in one condition permit ordered whole-clip splits.
    if len(siblings) >= 3:
        held = max(1, round(0.15 * len(siblings)))
        val_start, test_start = len(siblings) - 2 * held, len(siblings) - held
        index = siblings.index(number)
        name = "train" if index < val_start else "val" if index < test_start else "test"
        first = frames[0] + (60 if index in (val_start, test_start) else 0)
        return {f: name for f in frames if f >= first}

    start = frames[0]
    span = max(3, max(frames[-1] - 15, start + 3) - start - 120)
    train = max(1, round(0.70 * span))
    val = max(1, round(0.50 * max(span - train, 2)))
    last_train = start + train
    last_val = last_train + 60 + val
    splits = {}
    for f in frames:
        if f <= last_train:
            splits[f] = "train"
        elif last_train + 60 <= f <= last_val:
            splits[f] = "val"
        elif f >= last_val + 60:
            splits[f] = "test"
    return splits


def historical_source_end(splits):
    training = sorted(f for f, group in splits.items() if group == "train")
    if not training:
        return None
    held = [f for f, group in splits.items() if group != "train"]
    last = training[0] + 60
    if held:
        last = min(last, min(held) - 20)
    frames = [f for f in training if f <= last]
    return frames[-1] if len(frames) >= 7 else None


def sampled_positions(track, frames):
    valid = np.array([f in track for f in frames])
    xy = np.array([track.get(f, (0.0, 0.0)) for f in frames], dtype="<f4")
    return valid, xy


def rebuild(root):
    groups = defaultdict(list)
    for path in sorted((root / "labels").glob("*.txt")):
        condition, number = path.stem.rsplit("_", 1)
        groups[condition].append((int(number), path))
    verify(len(groups) == 22 and sum(map(len, groups.values())) == 47, "Unexpected source coverage")
    sources, windows = {}, {}
    for condition, clips in groups.items():
        calibration = root / "videos" / condition / "calib.json"
        data = calibration.read_bytes()
        sources[calibration.relative_to(root).as_posix()] = hashlib.sha256(data).digest()
        cal = json.loads(data)
        verify((cal["u_axis"], cal["v_axis"]) == ("x", "-y"), "Unknown raster convention")
        siblings = sorted(number for number, _ in clips)
        for number, path in clips:
            data = path.read_bytes()
            sources[path.relative_to(root).as_posix()] = hashlib.sha256(data).digest()
            tracks, frames = read_label(data)
            splits = assign_splits(frames, number, siblings)
            source_end = historical_source_end(splits)
            for end in sorted(splits)[::4]:
                eligible = source_end is not None and source_end < end
                for person, track in tracks.items():
                    if end not in track:
                        continue
                    hm, history = sampled_positions(track, range(end - 19, end + 1))
                    fm, future = sampled_positions(track, range(end + 1, end + 31))
                    if sum(hm[-5:]) >= 2 and sum(fm) >= 15:
                        windows[path.stem, person, end] = (splits[end], eligible, hm, history, fm, future)
    meta = hashlib.sha256()
    for name, raw_digest in sorted(sources.items()):
        meta.update(name.encode() + b"\0" + raw_digest)
    verify(meta.hexdigest() == SOURCE, "Original label/calibration bytes differ")
    return windows


def summarize(windows, maps):
    relative, absolute = hashlib.sha256(), hashlib.sha256()
    totals, eligible = Counter(), Counter()
    for key, (split, can_use, hm, history, fm, future) in sorted(windows.items()):
        header = (json.dumps([*key, split], separators=(",", ":")) + "\n").encode("ascii")
        relative.update(header)
        absolute.update(header)
        for mask, xy in ((hm[-5:], history[-5:]), (fm, future)):
            relative.update(bytes(mask.tolist()))
            relative.update((xy[mask] - history[-1]).astype("<f4").tobytes())
        for mask, xy in ((hm, history), (fm, future)):
            absolute.update(bytes(mask.tolist()))
            absolute.update(xy.tobytes())
        totals[split] += 1
        eligible[split] += int(can_use)
    verify(totals == {"train": 5686, "val": 1100, "test": 497}, "Window roster differs")
    verify(relative.hexdigest() == SHORT and absolute.hexdigest() == FULL, "Motion fingerprint differs")
    result = {
        "source_files": {"labels": 47, "calibrations": 22},
        "label_and_calibration_sha256": SOURCE, "five_frame_relative_sha256": SHORT,
        "full_20_frame_absolute_sha256": FULL, "unique_windows": len(windows),
        "by_split": {s: {"windows": totals[s], "eligible_historical_image_source": eligible[s]}
                     for s in ("train", "val", "test")},
    }
    if maps is not None:
        verify(hashlib.sha256(maps.read_bytes()).hexdigest() == MAP, "Unregistered map archive")
        with np.load(maps, allow_pickle=False) as z:
            keys = list(zip(z["clip"].tolist(), z["person_id"].tolist(), z["t0"].tolist()))
            verify(len(keys) == len(windows) and set(keys) == set(windows), "Archive key mismatch")
            verify(z["split"].tolist() == [windows[k][0] for k in keys], "Archive split mismatch")
            expected = [windows[k][1] for k in keys]
            verify(np.array_equal(z["source_eligible"], expected), "Archive source-eligibility mismatch")
            maps_xy = z["p_env_image"]
            verify(maps_xy.shape == (len(keys), 16, 10, 11), "Unexpected image map dimensions")
            observed = maps_xy[:, :, :, 10].any(axis=(1, 2))
            verify(np.array_equal(observed, expected) and np.array_equal(observed, z["m_map_image"]),
                   "Nonempty image contours differ from recorded availability")
        result["map_comparison"] = {"maps_sha256": MAP, "matched_keys_splits_and_source_flags": len(keys),
                                    "nonempty_image_map_windows": int(observed.sum())}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", type=Path, help="Root containing labels/ and videos/<condition>/calib.json")
    parser.add_argument("--maps", type=Path, help="Optionally check the private registered maps.npz")
    parser.add_argument("--output", type=Path, help="Write a small aggregate JSON file")
    args = parser.parse_args()
    report = json.dumps(summarize(rebuild(args.metadata), args.maps), indent=2) + "\n"
    if args.output:
        args.output.write_text(report)
    else:
        print(report, end="")
