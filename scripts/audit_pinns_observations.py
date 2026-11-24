"""Audit track history and the source video/annotation indices used by the five-frame inputs."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import cv2

from scripts.audit_pinns_short_window import reconstruct
from scripts.benchmark_heldout_sites import HISTORY, WINDOW, load_site_windows


def source_frames(labels):
    sampled = defaultdict(set)
    first_raw = {}
    label_numbers = set()
    person_numbers = set()
    with labels.open() as reader:
        for line in reader:
            fields = line.split()
            if len(fields) < 11:
                continue
            raw = int(float(fields[0]))
            label_numbers.add(raw)
            if not fields[1].lower().startswith("per"):
                continue
            person_numbers.add(raw)
            person = int(float(fields[2]))
            first_raw[person] = min(raw, first_raw.get(person, raw))
            if raw >= 1 and (raw - 1) % 3 == 0:
                sampled[person].add((raw - 1) // 3)
    return sampled, first_raw, label_numbers, person_numbers


def summarize(by_site):
    return {"all": dict(sum(by_site.values(), Counter())), "by_site": dict(sorted(by_site.items()))}


def audit(root):
    groups, _, digest = load_site_windows(root)
    heldout = defaultdict(Counter)
    video_counts = {}
    for site, rows in groups.items():
        by_clip = defaultdict(list)
        for row in rows:
            by_clip[row.clip].append(row)
        for clip, windows in by_clip.items():
            frames, first_raw, label_numbers, person_numbers = source_frames(root / "labels" / f"{clip}.txt")
            condition, index = clip.rsplit("_", 1)
            video = root / "videos" / condition / "videos" / f"ori_{index}.mp4"
            cap = cv2.VideoCapture(str(video))
            try:
                if not cap.isOpened():
                    raise ValueError(f"Cannot inspect original video: {video}")
                nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
            finally:
                cap.release()
            video_counts[clip] = {
                "video_header_frames": nframes, "video_fps": fps,
                "source_label_min_all_objects": min(label_numbers),
                "source_label_max_all_objects": max(label_numbers),
                "source_label_distinct_frame_numbers": len(label_numbers),
                "person_label_min": min(person_numbers),
                "person_label_max": max(person_numbers),
                "person_label_distinct_frame_numbers": len(person_numbers),
                "four_site_windows": len(windows), "archived_test_windows": 0,
            }
            for row in windows:
                start = row.t0 - HISTORY + 1
                times = frames[row.person_id]
                previous_run = 0
                while start - previous_run - 1 in times:
                    previous_run += 1
                c = heldout[site]
                c["windows"] += 1
                c["at_start_of_contiguous_sampled_track"] += int(previous_run == 0)
                c["at_first_sampled_label_for_id_in_clip"] += int(start == min(times))
                c["at_first_raw_label_for_id_in_clip"] += int(3 * start + 1 == first_raw[row.person_id])
                c["at_least_35_consecutive_prior_samples"] += int(previous_run >= WINDOW)
    members_file = root / "archive_members.txt"
    members = set(members_file.read_text().splitlines()) if members_file.exists() else None
    old, names, _ = reconstruct(root, members)
    archived = defaultdict(Counter)
    for row in old:
        mask = row["hist_mask_20"]
        c = archived[names[row["seq_id"]]]
        video_counts[row["seq_id"]]["archived_test_windows"] += 1
        c["windows"] += 1
        c["complete_five_frame_input"] += int(all(mask[-HISTORY:]))
        c["any_observation_in_previous_15_samples"] += int(any(mask[:-HISTORY]))
        c["all_previous_15_samples_observed"] += int(all(mask[:-HISTORY]))
    return {"label_and_calibration_digest_sha256": digest,
            "definition": "Start is the first of the five model input observations. Earlier annotations refer to the same person ID within the same recording. A track start is an annotation event; physical visibility and prior occlusion require separate evidence.",
            "archived_test": summarize(archived), "four_site_windows": summarize(heldout),
            "video_indexing": {
                "method": "Use all raw frame labels including frame zero. The frozen extractor associates label 1 with decoded frame 0 and samples every third frame. Header counts and labeling endpoints flag candidate offsets; annotation-to-pixel alignment requires an independent check.",
                "clips": dict(sorted(video_counts.items()))}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.dataset), indent=2))
