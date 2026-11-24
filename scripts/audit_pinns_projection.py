"""Check PINNS image/ground coordinate consistency without video, detection or trajectory models."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


# Each offset is expressed relative to the stored image point, using the label's pixel width/height.
ANCHORS = {
    "stored_xy": (0.0, 0.0),
    "down_half_height": (0.0, 0.5),
    "up_half_height": (0.0, -0.5),
    "down_one_height": (0.0, 1.0),
    "top_left_to_bottom_center": (0.5, 1.0),
    "top_left_to_center": (0.5, 0.5),
    "right_half_width": (0.5, 0.0),
    "left_half_width": (-0.5, 0.0),
    "left_one_width": (-1.0, 0.0),
}


def map_points(xy, matrix):
    raw = np.column_stack((xy, np.ones(len(xy)))) @ matrix.T
    if np.any(np.abs(raw[:, 2]) < 1e-12):
        raise ValueError("Ground-plane projection reaches infinity")
    return raw[:, :2] / raw[:, 2:]


def errors_summary(errors):
    return {"n": len(errors), "median_m": float(np.median(errors)), "p95_m": float(np.percentile(errors, 95)),
            "mean_m": float(np.mean(errors)), "fraction_below_0_01_m": float(np.mean(errors < 0.01))}


def summarize(parts):
    keys = parts[0]["distances"]
    offsets = np.concatenate([p["uv_offset"] for p in parts])
    norm_offsets = np.concatenate([p["normalized_offset"] for p in parts])
    sample_count = sum(p["rows"] for p in parts)
    return {
        "clips": len(parts), "person_observations": sample_count,
        "distances": {key: errors_summary(np.concatenate([p["distances"][key] for p in parts])) for key in keys},
        "inverse_projection_offsets": {
            "pixel_du_dv_q05_median_q95": np.percentile(offsets, [5, 50, 95], axis=0).tolist(),
            "du_over_width_dv_over_height_q05_median_q95": np.percentile(norm_offsets, [5, 50, 95], axis=0).tolist(),
        },
    }


def score_rows(rows, calib):
    # Columns: original raw frame, ox, oy, w, h, wx, wy.
    xy, size, target = rows[:, 1:3], rows[:, 3:5], rows[:, 5:7]
    if len(rows) == 0 or not np.isfinite(rows).all() or np.any(size <= 0):
        raise ValueError("Expected finite populated rows, positive width and height across all anchor comparisons")
    matrix, _ = cv2.findHomography(np.asarray(calib["pts_2d"], dtype=float),
                                   np.asarray(calib["pts_3d"], dtype=float), method=0)
    if matrix is None:
        raise ValueError("Unable to fit the four supplied calibration correspondences")
    y_sum = calib["y_min"] + calib["y_max"]
    native_xy = map_points(xy, matrix)
    distances = {"stored_xy_unreflected": np.linalg.norm(native_xy - target, axis=1)}
    for name, offset in ANCHORS.items():
        projected = map_points(xy + np.asarray(offset) * size, matrix)
        projected[:, 1] = y_sum - projected[:, 1]
        distances[name] = np.linalg.norm(projected - target, axis=1)
    conjugate = target.copy()
    conjugate[:, 1] = y_sum - conjugate[:, 1]
    offset = map_points(conjugate, np.linalg.inv(matrix)) - xy
    return {"rows": len(rows), "distances": distances, "uv_offset": offset, "normalized_offset": offset / size}


def read_rows(path):
    first = None
    rows = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) < 11:
            continue
        t = int(float(fields[0]))
        first = t if first is None else min(first, t)
        if fields[1].lower().startswith("per"):
            rows.append([t, *[float(fields[i]) for i in (3, 4, 6, 7, 8, 9)]])
    if first is None or first < 0:
        raise ValueError(f"Unexpected first source frame {first} in {path}")
    return first, np.asarray(rows, dtype=float)


def run(root):
    sources, per_site, per_origin = set(), defaultdict(list), defaultdict(list)
    all_parts, zero_native, per_clip = [], [], {}
    for label in sorted((root / "labels").glob("*.txt")):
        condition, separator, clip_number = label.stem.rpartition("_")
        if not separator or not clip_number.isdecimal():
            continue
        config = root / "videos" / condition / "calib.json"
        calib = json.loads(config.read_text())
        sources.update((label, config))
        first, person = read_rows(label)
        project = person[(person[:, 0] >= 1) & ((person[:, 0] - 1) % 3 == 0)]
        measured = score_rows(project, calib)
        site = f"{calib['location']}_{calib['scene']}"
        entry = {"site": site, "first_label_frame_all_objects": first, "all_person_rows_30hz": len(person),
                 "project_10hz": summarize([measured])}
        if first == 0:
            native = score_rows(person[(person[:, 0] % 3) == 0], calib)
            zero_native.append(native)
            entry["zero_origin_10hz"] = summarize([native])
        per_clip[label.stem] = entry
        all_parts.append(measured)
        per_site[site].append(measured)
        per_origin["zero" if first == 0 else "positive"].append(measured)
    digest = hashlib.sha256()
    for path in sorted(sources):
        digest.update(path.relative_to(root).as_posix().encode() + b"\x00")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return {
        "protocol": {
            "purpose": "Coordinate reconstruction diagnostic. Rows are ground-truth person annotations; repeated "
                       "observations are descriptive and do not measure detection, tracking or forecast accuracy.",
            "units": "Euclidean XY residual in metres, pooled across 10 Hz person/frame rows, NumPy linear quantiles.",
            "frames": "Project phase uses raw frame >=1 and (raw frame-1) modulo 3 == 0, regardless of clip. Three "
                      "clips start at 0; a separate comparison uses frame modulo 3 == 0 for those clips. "
                      "Every target and candidate within each comparison comes from the same label row. A positive minimum "
                      "annotated frame alone does not determine the video frame origin.",
            "geometry": "Fit map H from each condition's four pts_2d to pts_3d world correspondences, using OpenCV "
                        "findHomography(method=0). For each person point (ox+a*w, oy+b*h), divide H*[u,v,1] by "
                        "its last coordinate and reflect world Y as y_min+y_max-Y. Compare to raw label (wx,wy). "
                        "stored_xy_unreflected uses the original point and omits the Y reflection.",
            "anchor_offsets_width_height": {k: list(v) for k,v in ANCHORS.items()},
            "inverse": "Map raw world (wx, y_min+y_max-wy) through H inverse to image, subtract stored (ox,oy), "
                       "and report q05, median, q95 of pixel offsets and offsets divided by label width/height.",
            "aggregation": "All observations contribute equally within each reported group; observations of a "
                           "longer clip have more weight. The archive stores all per-clip results separately.",
            "input_label_and_calibration_sha256": digest.hexdigest(),
            "numpy_version": np.__version__, "opencv_version": cv2.__version__,
        },
        "project_10hz": {
            "all": summarize(all_parts), "by_site": {k: summarize(v) for k,v in sorted(per_site.items())},
            "by_min_annotated_frame_category": {k: summarize(v) for k,v in sorted(per_origin.items())},
        },
        "zero_origin_10hz_three_clips": summarize(zero_native) if zero_native else None,
        "clips": per_clip,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("pinns_projection_audit.json"))
    args = parser.parse_args()
    args.output.write_text(json.dumps(run(args.data_root), indent=2, allow_nan=False) + "\n")
