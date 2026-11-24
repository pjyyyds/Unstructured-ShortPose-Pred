"""Check the registered original camera videos against the PINNS neural extractor's frame requests."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

import cv2


VIDEO_SHA = "703beaf58dc695a23a8523c40f81ff295a64e1df8ef6bf16d7ccb4b513ba555c"
METADATA_SHA = "0b276b34ab1b01dae91f67c56bfaea5cc931511b05b4fc1c92686a891e6f6801"
COUNTS = (
    "clips", "declared_frames", "grabbed_within_declared_span", "eof_at_declared_count",
    "requested_person_frames", "retrieved_person_frames", "matching_decoded_dimensions",
    "person_frame_requests_outside_span", "early_grab_failures", "required_retrieve_failures",
    "unexpected_frames_after_declared_end", "decoded_size_mismatches",
    "clips_with_calibration_size_mismatch", "clips_with_calibration_fps_mismatch",
)


def counts_dict(counter):
    return {name: counter[name] for name in COUNTS}


def file_sha(path):
    with path.open("rb") as stream:
        h = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
        return h.digest()


def frame_requests(label_bytes):
    targets = set()
    selected_ticks = set()
    for line in label_bytes.splitlines():
        fields = line.split()
        if len(fields) < 11:
            continue
        raw = int(float(fields[0])) - 1
        if raw >= 0 and raw % 3 == 0:
            selected_ticks.add(raw)
            if fields[1].lower().startswith(b"per"):
                targets.add(raw)
    if len(selected_ticks) < 40:
        raise ValueError("A registered clip does not meet the extractor's minimum duration")
    return targets


def scan(video, targets, cal):
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise ValueError(f"Could not open {video.name}")
        backend = cap.getBackendName()
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if not all(math.isfinite(v) and v >= 1 and int(v) == v for v in (count, width, height)):
            raise ValueError("Invalid declared frame count or pixel size")
        count, width, height = int(count), int(width), int(height)
        fps = cap.get(cv2.CAP_PROP_FPS)
        totals = Counter(clips=1, declared_frames=count, requested_person_frames=len(targets))
        totals["person_frame_requests_outside_span"] = sum(t >= count for t in targets)
        totals["clips_with_calibration_size_mismatch"] = int((width,height) != (cal["u_size"],cal["v_size"]))
        totals["clips_with_calibration_fps_mismatch"] = int(not math.isclose(fps,cal["fps"],abs_tol=.05,rel_tol=0))
        for t in range(count):
            if not cap.grab():
                totals["early_grab_failures"] += 1
                break
            totals["grabbed_within_declared_span"] += 1
            if t in targets:
                ok, frame = cap.retrieve()
                if not ok or frame is None or not frame.size:
                    totals["required_retrieve_failures"] += 1
                    continue
                totals["retrieved_person_frames"] += 1
                if frame.shape == (height,width,3):
                    totals["matching_decoded_dimensions"] += 1
                else:
                    totals["decoded_size_mismatches"] += 1
        else:
            # OpenCV has no separate EOF flag. A failed next grab after all declared frames is the extractor's EOF test.
            if cap.grab():
                totals["unexpected_frames_after_declared_end"] += 1
                while cap.grab():
                    totals["unexpected_frames_after_declared_end"] += 1
            else:
                totals["eof_at_declared_count"] += 1
        return totals, backend, (width, height, fps)
    finally:
        cap.release()


def audit(root):
    cv2.setNumThreads(1)
    labels = sorted((root / "labels").glob("*.txt"))
    if len(labels) != 47:
        raise ValueError("Expected all 47 PINNS original clips and label files")
    metadata = {}
    video_identity = hashlib.sha256()
    by_site = defaultdict(Counter)
    backends = set()
    formats = Counter()
    for label in labels:
        condition, number = label.stem.rsplit("_",1)
        calib = root / "videos" / condition / "calib.json"
        cal_bytes = calib.read_bytes()
        cal = json.loads(cal_bytes)
        label_bytes = label.read_bytes()
        metadata[label.relative_to(root).as_posix()] = hashlib.sha256(label_bytes).digest()
        metadata[calib.relative_to(root).as_posix()] = hashlib.sha256(cal_bytes).digest()
        video = calib.parent / "videos" / f"ori_{number}.mp4"
        video_identity.update(video.relative_to(root).as_posix().encode() + b"\0" + file_sha(video))
        counts, backend, size_and_fps = scan(video, frame_requests(label_bytes), cal)
        formats[size_and_fps] += 1
        by_site[cal["location"] + "_" + cal["scene"]].update(counts)
        backends.add(backend)
    meta = hashlib.sha256()
    for name, digest in sorted(metadata.items()):
        meta.update(name.encode() + b"\0" + digest)
    if meta.hexdigest() != METADATA_SHA or video_identity.hexdigest() != VIDEO_SHA:
        raise ValueError("Input bytes differ from registered PINNS metadata or original video fingerprints")
    total = counts_dict(sum(by_site.values(), Counter()))
    ok = (total["declared_frames"] == total["grabbed_within_declared_span"]
          and total["eof_at_declared_count"] == total["clips"]
          and total["requested_person_frames"] == total["retrieved_person_frames"] == total["matching_decoded_dimensions"]
          and all(total[k] == 0 for k in ("person_frame_requests_outside_span", "early_grab_failures",
                                         "required_retrieve_failures", "unexpected_frames_after_declared_end",
                                         "decoded_size_mismatches", "clips_with_calibration_size_mismatch",
                                         "clips_with_calibration_fps_mismatch")))
    return {"passed":ok, "opencv":cv2.__version__, "backends":sorted(backends),
            "video_inventory_sha256":VIDEO_SHA, "label_and_calibration_sha256":METADATA_SHA,
            "container_formats": [{"width_px":w, "height_px":h, "fps":fps, "clips":n}
                                  for (w,h,fps),n in sorted(formats.items())],
            "all":total, "by_site":{site:counts_dict(c) for site,c in sorted(by_site.items())}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path, help="PINNS root with original videos, labels and calibration")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.data_root)
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    else:
        print(encoded,end="")
    if not report["passed"]:
        raise SystemExit("Original camera video preflight found discrepancies")
