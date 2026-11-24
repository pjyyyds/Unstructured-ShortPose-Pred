"""Construct paired 5/10/20-frame PINNS histories on verified five-frame anchors.

python -m scripts.audit_pinns_matched_histories DATA_ROOT --output report.json --manifest windows.npz
The optional non-pickle NPZ contains per-window keys, historical masks, relative ground
coordinates, the common 30-step truth and flags for both cohorts. No videos are decoded.
"""
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.benchmark_heldout_sites import load_site_windows
from src.data.constants import DT, T_FUT
from src.data.pinns_extractor import _check_right_handed_homography, _parse_pinns_label_file


ROOT = Path(__file__).resolve().parents[1]
LENGTHS = (5, 10, 20)
LONGEST = max(LENGTHS)
FULL_BLOCK = LONGEST + T_FUT


def digest_keys(keys):
    return hashlib.sha256(json.dumps(keys, separators=(",", ":")).encode()).hexdigest()


@dataclass
class MatchedHistories:
    keys: list
    sites: np.ndarray
    history: np.ndarray
    mask: np.ndarray
    future: np.ndarray
    first_label_t: np.ndarray
    first_possible_20_anchor: np.ndarray
    strict: np.ndarray
    source_digest: str


def track_positions(root, clip):
    condition = clip.rsplit("_", 1)[0]
    calib = json.loads((root / "videos" / condition / "calib.json").read_text())
    parsed = _parse_pinns_label_file(root / "labels" / (clip + ".txt"), _check_right_handed_homography(calib),
                                     calib["y_min"], calib["y_max"])
    people = defaultdict(dict)
    for t, objects in parsed.items():
        for obj in objects:
            xy = (obj["wx"], obj["wy"])
            if obj["type"].lower().startswith("per") and np.isfinite(xy).all():
                track = people[obj["id"]]
                if t in track:
                    raise ValueError(f"Duplicate person/frame in {clip}: {obj['id']}, {t}")
                track[t] = xy
    return people


def grouped_indices(keys, selected):
    by_track = defaultdict(list)
    for i in np.flatnonzero(selected):
        clip, pid, _ = keys[i]
        by_track[clip, pid].append(int(i))
    for indices in by_track.values():
        yield sorted(indices, key=lambda i: keys[i][2])


def greedy_disjoint(keys, complete):
    """Earliest-finish greedy among equal-length [t0-19,t0+30] intervals within a clip/person."""
    keep = np.zeros(len(keys), dtype=bool)
    for indices in grouped_indices(keys, complete):
        last = None
        for i in indices:
            t = keys[i][2]
            if last is None or t - last >= FULL_BLOCK:
                keep[i], last = True, t
    return keep


def load_matched_histories(data_root):
    root = Path(data_root)
    groups, _, data_hash = load_site_windows(root)
    rows = [row for s in sorted(groups) for row in groups[s]]
    keys = [(r.clip, r.person_id, r.t0) for r in rows]
    accepted = json.loads((ROOT / "results/pinns_heldout_replay.json").read_text())
    if (data_hash != accepted["label_and_calibration_sha256"] or
            digest_keys(keys) != accepted["ordered_window_ids_sha256"] or len(set(keys)) != len(keys) or len(keys) != 1482):
        raise ValueError("The registered label/calibration or ordered five-frame cohort has changed")
    history = np.zeros((len(keys), LONGEST, 2), dtype=np.float64)
    mask = np.zeros((len(keys), LONGEST), dtype=bool)
    future = np.empty((len(keys), T_FUT, 2), dtype=np.float64)
    first = np.empty(len(keys), dtype=int)
    possible = np.full(len(keys), -1, dtype=int)
    by_clip = defaultdict(list)
    for i,(clip, _, _) in enumerate(keys):
        by_clip[clip].append(i)
    for clip, indices in by_clip.items():
        tracks = track_positions(root, clip)
        first_complete = {}
        for person, track in tracks.items():
            times = np.asarray(sorted(track))
            runs = np.split(times, np.flatnonzero(np.diff(times) != 1) + 1)
            first_complete[person] = next((int(run[LONGEST-1]) for run in runs if len(run) >= FULL_BLOCK), -1)
        for i in indices:
            _, person, t0 = keys[i]
            track = tracks[person]
            anchor = np.asarray(track[t0])
            first[i] = min(track)
            possible[i] = first_complete[person]
            for j,t in enumerate(range(t0-LONGEST+1,t0+1)):
                if t in track:
                    history[i,j] = np.asarray(track[t])-anchor
                    mask[i,j] = True
            future[i] = np.stack([track[t] for t in range(t0+1,t0+T_FUT+1)]) - anchor
            np.testing.assert_array_equal(np.vstack([history[i,-5:],future[i]]),rows[i].xy)
    if not mask[:, -5:].all() or not np.isfinite(future).all():
        raise ValueError("Accepted 5+30 observation support lost")
    sites = np.array([s for s in sorted(groups) for _ in groups[s]])
    return MatchedHistories(keys, sites, history, mask, future, first, possible, greedy_disjoint(keys, mask.all(axis=1)), data_hash)


def strict_keys(data_root):
    cohort = load_matched_histories(data_root)
    return {cohort.keys[i] for i in np.flatnonzero(cohort.strict)}


def support(c, chosen):
    keys = [c.keys[i] for i in np.flatnonzero(chosen)]
    time_counts = Counter((clip,t) for clip, _, t in keys)
    first = np.array([t for _,_,t in keys]) - 4 == c.first_label_t[chosen]
    speed = np.linalg.norm(c.history[chosen,-1] - c.history[chosen,-2], axis=1) / DT
    return {
        "windows": len(keys), "clips": len({k[0] for k in keys}), "clip_person_tracks": len({k[:2] for k in keys}),
        "distinct_clip_anchor_times": len(time_counts),
        "windows_sharing_clip_anchor_time_with_another_person": sum(n for n in time_counts.values() if n>1),
        "five_frame_history_starts_at_first_sampled_label_of_person": int(first.sum()),
        "mean_last_two_point_speed_mps": float(speed.mean()) if len(speed) else None,
        "median_last_two_point_speed_mps": float(np.median(speed)) if len(speed) else None,
    }


def overlaps(keys, chosen, length):
    """Count distinct pairs; inclusive t0 shared by history, first future is t0+1."""
    repeated = Counter()
    touched = set()
    gaps = Counter()
    same_future = 0
    for indices in grouped_indices(keys, chosen):
        times = [keys[i][2] for i in indices]
        gaps.update(b-a for a,b in zip(times,times[1:]))
        for j,i in enumerate(indices):
            for h in range(j+1,len(indices)):
                b = indices[h]
                gap = keys[b][2] - keys[i][2]
                if gap >= length + T_FUT:
                    break
                repeated[length+T_FUT-gap] += 1
                touched.update((i,b))
                same_future += int(gap < T_FUT)
    return {"same_track_overlapping_window_pairs": sum(repeated.values()), "windows_in_an_overlapping_pair": len(touched),
            "shared_history_plus_future_steps_per_pair": dict(sorted(repeated.items())),
            "pairs_with_overlapping_scored_futures": same_future,
            "adjacent_same_track_anchor_pairs": sum(gaps.values()),
            "adjacent_anchor_gap_min_steps": min(gaps,default=None),
            "adjacent_anchor_gap_35_pairs": gaps[35], "adjacent_anchor_gap_50_pairs": gaps[50],
            "adjacent_anchor_gap_70_pairs": gaps[70]}


def grid_selection(c, subset):
    """Inspect the opportunity lost by inheriting the old grid, independently of any model."""
    ever_possible, represented = 0, 0
    delays = Counter()
    for indices in grouped_indices(c.keys, subset):
        possible = c.first_possible_20_anchor[indices[0]]
        if possible >= 0:
            ever_possible += 1
            complete = [i for i in indices if c.mask[i].all()]
            if complete:
                represented += 1
                delays[c.keys[complete[0]][2] - int(possible)] += 1
    return {"original_tracks_with_any_complete_20_plus_30_off_or_on_grid": ever_possible,
            "of_these_represented_on_frozen_grid": represented,
            "complete_tracks_missed_by_frozen_anchor_phase": ever_possible-represented,
            "delay_from_first_possible_to_first_retained_20_anchor_steps_by_track": dict(sorted(delays.items()))}


def archived_test(root):
    # Original V2 materialization slices the same master-window records at all three lengths.
    from scripts.audit_pinns_short_window import reconstruct
    manifest = root / "archive_members.txt"
    samples, _, _ = reconstruct(root, set(manifest.read_text().splitlines()) if manifest.exists() else None)
    if len(samples) != 497:
        raise ValueError("Original V2 split reconstruction did not return 497 rows")
    masks = np.stack([s["hist_mask_20"] for s in samples]).astype(bool)
    targets = np.stack([s["fut_mask"] for s in samples]).astype(bool)
    return {"windows_on_each_materialized_length": len(samples),
            "clips": len({s["seq_id"] for s in samples}),
            "complete_historical_positions_by_length": {str(t):int(masks[:,-t:].all(axis=1).sum()) for t in LENGTHS},
            "complete_30_step_future": int(targets.all(axis=1).sum()), "valid_at_3s": int(targets[:,-1].sum()),
            "method": "Original V2 uses the same raw rows at T=5/10/20 and slices the last T frames with masks. "
                      "The original window filter requires >=2 of the final 5 history points and >=15 of 30 future positions."}


def write_manifest(path, c):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as out:
        np.savez_compressed(out, clip=np.array([k[0] for k in c.keys]), person_id=[k[1] for k in c.keys],
                            t0=[k[2] for k in c.keys], site=c.sites, history_20_xy=c.history,
                            history_20_mask=c.mask, future_30_xy=c.future, complete_20=c.mask.all(axis=1),
                            strict_disjoint_20=c.strict, first_person_label_t=c.first_label_t)


def summarize(data_root, c):
    all_rows = np.ones(len(c.keys),dtype=bool)
    complete = c.mask.all(axis=1)
    groups = {"frozen_five": all_rows, "complete_matched": complete, "strict_disjoint": c.strict,
              "removed_for_short_history": ~complete, "removed_for_50_step_overlap": complete & ~c.strict}
    metrics = {}
    for name, chosen in groups.items():
        metrics[name] = {
            "all": support(c,chosen), "sites": {s:support(c,chosen & (c.sites==s)) for s in sorted(set(c.sites))},
            "ordered_window_ids_sha256": digest_keys([k for k, use in zip(c.keys,chosen) if use])}
    timing = {}
    for name in ("frozen_five", "complete_matched", "strict_disjoint"):
        applicable = (5,) if name == "frozen_five" else LENGTHS
        timing[name] = {str(t):overlaps(c.keys,groups[name],t) for t in applicable}
    if any(v["same_track_overlapping_window_pairs"] for v in timing["strict_disjoint"].values()):
        raise ValueError("Greedy cohort retained a same-track interval overlap")
    return {
        "protocol": {"scope": "Coordinate-only selection audit at four previously examined sites; no model fit or pose inference.",
                     "anchor": "Verified seed-42 reference keys. Frames sampled at 10 Hz (raw annotation number 3*t+1). "
                               "No interpolation. Each history ends at t0 and all three lengths share the exact "
                               "person, t0 and complete 30 future points. Coordinates are meters relative to t0.",
                     "complete_matched": "All original anchors with 20 consecutive observations ending at t0.",
                     "strict_disjoint": "Among complete_matched only, keep earliest per clip/person, then the earliest "
                                        "t0 at least 50 steps after the previous retained anchor. The combined "
                                        "[t0-19,t0+30] interval contains 50 position labels (4.9 s endpoint span); "
                                        "independence across people or clips is not implied.",
                     "overlap": "At gap d=35, each pair shares 5 positions at T=10 and 15 at T=20: the earlier "
                                "window's future overlaps the later window's history. Two 30-step scored futures "
                                "are already disjoint at d>=30. Counts refer to same typed person within a clip.",
                     "selection": "The original 35-step tiling anchors the first window at the fifth point of each "
                                  "contiguous run. Full 20-frame eligibility preferentially retains people with longer "
                                  "continuous histories; greedy thinning further changes the weighting of long tracks. "
                                  "An annotated track start or an eligible anchor does not establish "
                                  "first emergence from occlusion. No pose or target error comparison is implied.",
                     "manifest": "In NPZ take the last T positions/mask from history_20_xy/history_20_mask, the "
                                 "same future_30_xy for all T, and the desired complete_20 or strict_disjoint_20 flag. "
                                 "The private manifest includes all base keys for attrition checks."},
        "identity": {"labels_and_calibration_sha256": c.source_digest, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "complete_history_counts_within_frozen_five": {str(t):int(c.mask[:,-t:].all(axis=1).sum()) for t in LENGTHS},
        "cohorts": metrics, "timing": timing,
        "inherited_anchor_phase": {"all": grid_selection(c,all_rows),
                                   "sites": {s:grid_selection(c,c.sites==s) for s in sorted(set(c.sites))}},
        "archived_v2_test": archived_test(data_root)}


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("data_root",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--manifest",type=Path,help="Optional local per-anchor coordinate NPZ; aggregate report excludes individual rows")
    args=p.parse_args()
    cohort=load_matched_histories(args.data_root)
    report=summarize(args.data_root,cohort)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    if args.manifest: write_manifest(args.manifest,cohort)
    print({k:v["all"]["windows"] for k,v in report["cohorts"].items()})
