"""Replay only trajectory, split, map and mask enrollment using the original PINNS metadata."""

import argparse
import json
from pathlib import Path

from scripts.build_pinns_causal_maps import clips_in_order
from src.data.pinns_causal_input import attach_causal_image
from reviews.pinns_map_axis_comparison.neural.run_neural import (
    check_samples,
    EXPECTED_MAP,
    EXPECTED_MOTION,
    SORTED_GEOMETRY,
    SORTED_KEYS,
)


def design(metadata, corrected_maps):
    # Upstream windows assemble no detector/model outputs here. Only enrollment fields are retained.
    keep = (
        "seq_id",
        "t0_frame_10hz",
        "split",
        "hist_pos_20",
        "hist_mask_20",
        "p_t0",
        "fut_xy",
        "fut_mask",
        "fut_pos_abs",
    )
    raw = []
    for item in clips_in_order(metadata):
        for person, row in item.windows():
            raw.append(dict({k: row[k] for k in keep}, person_id=int(person)))
    _, output = check_samples(attach_causal_image(raw, corrected_maps))
    return {
        "scope": "Source enrollment and causal image geometry only; no pose estimation, fitting or prediction.",
        "ordered_keys_sha256": SORTED_KEYS,
        "geometry_and_source_availability_sha256": SORTED_GEOMETRY,
        "motion_and_masks_sha256": EXPECTED_MOTION,
        "source_label_maps_sha256": EXPECTED_MAP,
        "heldout_exact_3s_target_windows": 233,
        "splits": output,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("metadata", type=Path)
    p.add_argument("corrected_maps", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    res = design(a.metadata, a.corrected_maps)
    a.output.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res["splits"]))
