"""Check whether the PINNS pedestrian type input can be recovered from observed positions."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from scripts.audit_pinns_short_window import reconstruct
from scripts.benchmark_heldout_sites import HISTORY, load_site_windows
from src.data.constants import DT
from src.data.features import build_motion_features


def raw_flags(root):
    flags = Counter()
    clips = 0
    for path in sorted((root / "labels").glob("*.txt")):
        if path.stem.lower().startswith("readme") or "_" not in path.stem:
            continue
        clips += 1
        with path.open() as lines:
            for line in lines:
                parts = line.split()
                if len(parts) >= 11 and parts[1].lower().startswith("per"):
                    flags[int(float(parts[10]))] += 1
    return {"source_clips": clips, "pedestrian_rows": sum(flags.values()),
            "column_11_histogram": dict(sorted(flags.items()))}


def speed_type(hist, mask):
    end = build_motion_features(hist, mask, dt=DT)[-1]
    return int(end[6] > 0.5 and end[4] < 0.20)


def compare(pairs):
    return {"windows": len(pairs), "standing_by_model_rule": sum(model for model, _ in pairs),
            "standing_by_observed_speed": sum(speed for _, speed in pairs),
            "different_type": sum(model != speed for model, speed in pairs)}


def audit(root):
    groups, _, digest = load_site_windows(root)
    heldout, archived = defaultdict(list), defaultdict(list)
    observed = np.ones(HISTORY, dtype=np.float32)
    for site, rows in groups.items():
        for row in rows:
            speed = speed_type(row.xy[:HISTORY], observed)
            heldout[site].append((int(row.fixed == 1 or speed), speed))

    archive_members = root / "archive_members.txt"
    members = set(archive_members.read_text().splitlines()) if archive_members.exists() else None
    samples, names, _ = reconstruct(root, members)
    for sample in samples:
        hist, mask = sample["hist_pos_20"][-HISTORY:], sample["hist_mask_20"][-HISTORY:]
        if not np.all(mask == 1):
            raise ValueError("The archived comparison expects five complete position observations")
        model = int(sample["ego_type_oh"][2] > 0.5)
        archived[names[sample["seq_id"]]].append((model, speed_type(hist, mask)))

    def summary(by_site):
        return {"all": compare([pair for pairs in by_site.values() for pair in pairs]),
                "by_site": {name: compare(pairs) for name, pairs in sorted(by_site.items())}}

    return {
        "label_and_calibration_digest_sha256": digest,
        "rule": "Encode a pedestrian as standing if column 11 (fixed) equals 1 or a valid recent observed speed is below 0.20 m/s. Each audited window has complete five-frame position history.",
        "source_labels": raw_flags(root), "archived_test": summary(archived),
        "four_site_windows": summary(heldout),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.dataset), indent=2, allow_nan=False))
