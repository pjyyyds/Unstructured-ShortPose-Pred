"""Audit neighbor inputs from the original 497 PINNS five-frame test windows.

Run from the repository root: python -m scripts.audit_pinns_neighbor_inputs DATA_ROOT
Use --members ARCHIVE_MEMBERS.txt with metadata-only data. Full source archives
need just the original labels, calibration and video filenames; video decoding,
pose caches, network weights and previously serialized graph tensors are unnecessary.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np

from scripts.audit_pinns_short_window import reconstruct
from src.data.constants import DT
from src.data.dataset_builder import materialize_window_split
from src.data.pinns_extractor import _check_right_handed_homography, _parse_pinns_label_file
from src.data.scene_graph import build_neighbor_edges


ROOT = Path(__file__).resolve().parents[1]
TYPE_NAMES = ("walking_person", "assisted_person", "standing_person", "bicycle_rider", "scooter_rider",
              "utility_cart", "passenger_car", "shuttle_van")


class Edge(NamedTuple):
    source_type: str
    neighbor_id: int
    encoded_type: str
    valid_speed: bool
    radial_speed: float
    used_fifth_lag: bool


def neighbors_from_history(past, t0, sample):
    """Receives only the six 10 Hz frames ending at t0; IDs never use a future position."""
    if not all(t0 - 5 <= t <= t0 for t in past):
        raise ValueError("Unexpected frame outside the causal graph history")
    obs = past[t0]
    positions = np.array([(o["wx"], o["wy"]) for o in obs], dtype=np.float32)
    ped = np.array([o["type"].lower().startswith("per") for o in obs])
    hits = np.flatnonzero(ped & np.all(positions == sample["p_t0"], axis=1))
    if len(hits) != 1:
        raise ValueError(f"Ambiguous current ego position in {sample['seq_id']} at {t0}: {hits}")
    i = hits[0]
    prior = {t: {(o["type"], o["id"]): o for o in group} for t, group in past.items()}
    velocity = np.zeros_like(positions)
    gaps = np.zeros(len(obs), dtype=int)
    ratio = np.zeros(len(obs), dtype=np.float32)
    types = np.zeros(len(obs), dtype=int)
    for j, o in enumerate(obs):
        key = (o["type"], o["id"])
        ratio[j] = sum(key in prior.get(t, {}) for t in range(t0 - 4, t0 + 1)) / 5.0
        for back in range(1, 6):
            previous = prior.get(t0 - back, {}).get(key)
            if previous is not None:
                displacement = [o["wx"] - previous["wx"], o["wy"] - previous["wy"]]
                velocity[j] = np.asarray(displacement, dtype=np.float32) / (back * DT)
                gaps[j] = back
                break
        if ped[j]:
            standing = o["fixed"] == 1 or (gaps[j] > 0 and np.linalg.norm(velocity[j]) < 0.20)
            types[j] = 2 if standing else 0
        else:
            types[j] = 6
    other = np.flatnonzero(np.arange(len(obs)) != i)
    edges, one_hot, mask = build_neighbor_edges(
        positions[i], velocity[i], gaps[i] > 0, ratio[i],
        positions[other], velocity[other], gaps[other] > 0, ratio[other], types[other],
    )
    for name, computed in (("nbr_edges", edges), ("nbr_type_oh", one_hot), ("nbr_mask", mask)):
        np.testing.assert_array_equal(computed, sample[name])
    # Independently recover the source ID for each slot from the radius/nearest-neighbor rule.
    delta = positions[other].astype(float) - positions[i]
    distance = np.hypot(delta[:, 0], delta[:, 1])
    candidates = np.flatnonzero((distance > 1e-5) & (distance <= 10))
    keep = candidates[np.argsort(distance[candidates])[:12]]
    selected = other[keep]
    np.testing.assert_equal(len(keep), int(mask.sum()))
    np.testing.assert_allclose(distance[keep], edges[:len(keep), 4], rtol=1e-6, atol=1e-6)
    detail = []
    for slot, j in enumerate(selected):
        speed_ok = bool(gaps[i] and gaps[j])
        vr = 0.0
        if speed_ok:
            dv = velocity[j].astype(float) - velocity[i]
            vr = -float(delta[keep[slot]] @ dv) / (distance[keep[slot]] + 1e-6)
        np.testing.assert_allclose(vr, edges[slot, 5], atol=1e-6, rtol=1e-6)
        np.testing.assert_equal(int(speed_ok), edges[slot, 6])
        np.testing.assert_equal(int(one_hot[slot].argmax()), types[j])
        detail.append(Edge(obs[j]["type"], obs[j]["id"], TYPE_NAMES[types[j]], speed_ok, float(edges[slot, 5]),
                       bool(gaps[i] == 5 or gaps[j] == 5)))
    return {"clip": sample["seq_id"], "ego": obs[i]["id"], "t0": int(t0), "candidates": len(candidates), "edges": detail}


def summarize(rows):
    all_edges = [edge for row in rows for edge in row["edges"]]
    per_class = {}
    for name in ("Person", "Car", "Bicycle/Scooter"):
        typed = [(row, edge) for row in rows for edge in row["edges"] if edge.source_type == name]
        pairs = {(row["clip"], row["ego"], edge.neighbor_id) for row, edge in typed}
        windows = {(row["clip"], row["ego"], row["t0"]) for row, _ in typed}
        fast_approach = {(row["clip"], row["ego"], row["t0"]) for row, edge in typed
                         if edge.valid_speed and edge.radial_speed >= 1.0}
        valid = np.array([edge.radial_speed for _, edge in typed if edge.valid_speed])
        per_class[name] = {
            "edge_slots": len(typed), "distinct_directed_clip_ego_neighbor_pairs": len(pairs),
            "windows_with_one_or_more": len(windows),
            "valid_radial_edges": len(valid), "missing_radial_edges": len(typed) - len(valid),
            "approaching_positive": int((valid > 0).sum()), "receding_negative": int((valid < 0).sum()),
            "zero_radial": int((valid == 0).sum()), "approaching_at_least_1_mps": int((valid >= 1).sum()),
            "windows_with_approach_at_least_1_mps": len(fast_approach),
            "radial_speed_p05_p50_p95_mps": np.percentile(valid, [5, 50, 95]).tolist() if len(valid) else None,
        }
    return {
        "windows": len(rows), "clips": len({r["clip"] for r in rows}),
        "distinct_clip_trigger_times": len({(r["clip"], r["t0"]) for r in rows}),
        "windows_without_neighbors": sum(not r["edges"] for r in rows),
        "windows_at_12_slots": sum(len(r["edges"]) == 12 for r in rows),
        "windows_truncated_by_12": sum(r["candidates"] > 12 for r in rows),
        "edge_slots_dropped_by_12": sum(max(0, r["candidates"] - 12) for r in rows),
        "edge_slots": len(all_edges), "available_radial_edges": sum(e.valid_speed for e in all_edges),
        "edges_using_t0_minus_5": sum(e.used_fifth_lag for e in all_edges),
        "encoded_types": {k: sum(e.encoded_type == k for e in all_edges) for k in TYPE_NAMES},
        "source_neighbor_types": per_class,
    }


def audit(data, members, reference):
    labels = sorted((data / "labels").glob("*.txt"))
    frames, raw_counts, paths = {}, Counter(), set()
    for label in labels:
        if not label.stem.rsplit("_", 1)[-1].isdecimal():
            continue
        condition = label.stem.rsplit("_", 1)[0]
        config = data / "videos" / condition / "calib.json"
        c = json.loads(config.read_text())
        paths.update((label, config))
        with label.open() as f:
            raw_counts.update(fields[1] for line in f if len(fields := line.split()) >= 11)
        frames[label.stem] = _parse_pinns_label_file(label, _check_right_handed_homography(c), c["y_min"], c["y_max"])
    digest = hashlib.sha256()
    for p in sorted(paths):
        digest.update(p.relative_to(data).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(p.read_bytes()).digest())
    if digest.hexdigest() != "0b276b34ab1b01dae91f67c56bfaea5cc931511b05b4fc1c92686a891e6f6801":
        raise ValueError("Unexpected label or calibration inputs")
    inventory = set(members.read_text().splitlines()) if members is not None else None
    samples, sites, _ = reconstruct(data, inventory)
    saved = json.loads(reference.read_text())
    observed_clips = Counter(s["seq_id"] for s in samples)
    expected_clips = {clip: x["windows"] for clip, x in saved["by_clip_with_test_anchors"].items() if x["windows"]}
    if dict(observed_clips) != expected_clips or len(samples) != 497:
        raise ValueError("Reconstructed window counts differ from the saved 497-window cohort")
    site_rows = defaultdict(list)
    if set(raw_counts) != {"Person", "Car"}:
        raise ValueError(f"Unexpected raw PINNS classes: {sorted(raw_counts)}")
    materialized = materialize_window_split(samples, "test", t_obs=5)
    for name in ("nbr_edges", "nbr_type_oh", "nbr_mask"):
        np.testing.assert_array_equal(materialized[name].numpy(), np.stack([sample[name] for sample in samples]))
    for sample in samples:
        clip = sample["seq_id"]
        t = sample["t0_frame_10hz"]
        past = {ft: frames[clip][ft] for ft in range(t - 5, t + 1) if ft in frames[clip]}
        site_rows[sites[clip]].append(neighbors_from_history(past, t, sample))
    rows = [r for site in sorted(site_rows) for r in site_rows[site]]
    source_files = ("src/data/pinns_extractor.py", "src/data/scene_graph.py", "src/data/dataset_builder.py",
                    "scripts/audit_pinns_short_window.py")
    return {
        "protocol": {
            "reference_window_counts_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
            "population": "Original 497 five-frame PINNS test windows, already used throughout development. 21 clips, four previously examined sites. Graph inputs apply to graph-enabled M2b/M2/M3 and full models. M0 and M1 in the separate four-site retraining use no neighbor graph.",
            "edge": "Directed neighbor-to-ego slots, at t0. Keep 0.00001 < ground distance <= 10 metres, nearest first, at most 12. A window can have multiple slots, and consecutive windows can observe the same pair.",
            "type": "Raw Person or Car labels. Assigned tensor slots use eight-dimensional one-hot: person standing when fixed flag==1 or valid speed <0.20 m/s; otherwise walking-person code, even if velocity missing. All non-person objects map to passenger-car code. Physical riders, disability/assistance or vehicle subtypes cannot be identified from these labels.",
            "time": "Separate known test-window selection (requires at least 15 annotated future frames) from graph construction. Recompute every retained input tensor from source detections limited to t0-5..t0. Five-frame observation ratios cover t0-4..t0; velocities choose nearest available same-type/ID detection among t0-1..t0-5. No coordinates after t0 enter edges. This is an input audit, not an online trigger or static-map validation.",
            "radial": "If both velocities exist, v_rad = -dot(p_j-p_i,v_j-v_i)/(distance+1e-6); positive means distance decreasing. The 1 m/s cutoff is descriptive. Invalid velocities yield zero plus mask=0; valid zero means no measured radial motion. All positions/IDs are label-supplied, not detector estimates.",
            "ablation": "The M2b_norad model masks the explicit radial scalar before graph processing. Relative position, distance and relative velocity remain, so the underlying radial relation can still be derived from the other channels.",
            "verification": "Compared all 12x9 edges, 12x8 neighbor one-hot and 12 masks exactly with the unmodified archived input builder on every original held-out window. Verified materialize_window_split(t_obs=5) forwards all three arrays unchanged. Independently checked distance, radial sign/magnitude and typed slot order against raw actor IDs.",
            "source_sha256": {k: hashlib.sha256((ROOT / k).read_bytes()).hexdigest() for k in source_files},
            "label_calibration_digest_sha256": digest.hexdigest(), "numpy_version": np.__version__,
        },
        "all_47_source_clips_raw_30hz_class_rows": dict(raw_counts), "total": summarize(rows),
        "sites": {k: summarize(v) for k, v in sorted(site_rows.items())},
        "clips": {c: summarize([r for r in rows if r["clip"] == c]) for c in sorted({r["clip"] for r in rows})},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path, help="PINNS data root containing labels/ and videos/")
    parser.add_argument("--members", type=Path, help="Original ZIP member list for a metadata-only copy")
    parser.add_argument("--reference", type=Path, default=ROOT / "results/pinns_motion_audit.json",
                        help="Published per-clip test-window counts")
    parser.add_argument("--output", type=Path, default=ROOT / "results/pinns_neighbor_inputs.json")
    args = parser.parse_args()
    member_path = args.members
    if member_path is None and (args.data_root / "archive_members.txt").exists():
        member_path = args.data_root / "archive_members.txt"
    report = audit(args.data_root, member_path, args.reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
