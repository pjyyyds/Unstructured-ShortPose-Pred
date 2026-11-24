"""Evaluate the fixed last-two-observation baseline only on ETH University (seq_eth)."""
from collections import Counter, defaultdict
import hashlib
import io
import json
import math
from pathlib import Path
import tarfile
import urllib.request


URL = "https://ethz.ch/content/dam/ethz/special-interest/itet/cvl/vision-dam/documents/ewap_dataset_light.tgz"
ARCHIVE_SHA = "3e057d7c8631c9b5ae638b0d74d454d7be567532a51cd5e06f0323bfe863ce98"
MEMBERS = ("ewap_dataset/README.txt", "ewap_dataset/seq_eth/info.txt", "ewap_dataset/seq_eth/obsmat.txt")
FRAMES_PER_OBSERVATION = 6
WINDOW = 9


def fetch_source():
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0 (ETH development reproducibility check)"})
    with urllib.request.urlopen(req, timeout=30) as response:
        data = response.read(500_001)
    if hashlib.sha256(data).hexdigest() != ARCHIVE_SHA:
        raise ValueError("Official archive differs from the checked version")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        # Do not extract the archive; the held-out recording is never opened.
        return {name: archive.extractfile(name).read() for name in MEMBERS}


def benchmark(content):
    readme, info, table = (content[k] for k in MEMBERS)
    if (b"pos_x pos_z pos_y" not in readme or b"positions and velocities are in meters" not in readme
            or b"2.5 fps" not in info or b"0.4 seconds" not in info):
        raise ValueError("Check official documentation before interpreting time or coordinate units")
    persons = defaultdict(dict)
    nonfinite_positions = duplicates = rows = 0
    for line in table.splitlines():
        if not line.strip():
            continue
        rows += 1
        fields = line.split()
        if len(fields) != 8:
            raise ValueError("Unexpected obsmat format")
        frame, person = map(float, fields[:2])
        if not frame.is_integer() or not person.is_integer():
            raise ValueError("Expected integer-valued frame and pedestrian IDs")
        xy = (float(fields[2]), float(fields[4]))
        if not all(map(math.isfinite, xy)):
            nonfinite_positions += 1
            continue
        points = persons[int(person)]
        if int(frame) in points:
            if points[int(frame)] != xy:
                raise ValueError("Conflicting coordinates for one pedestrian and frame")
            duplicates += 1
        points[int(frame)] = xy

    ade, fde = [], []
    people_with_windows = runs = unused_tail_points = 0
    deltas = Counter()
    for track in persons.values():
        observed = sorted(track)
        before = len(ade)
        boundaries = [0]
        for i in range(1, len(observed)):
            delta = observed[i] - observed[i-1]
            deltas[delta] += 1
            if delta != FRAMES_PER_OBSERVATION:
                boundaries.append(i)
        boundaries.append(len(observed))
        for left, right in zip(boundaries, boundaries[1:]):
            runs += 1
            unused_tail_points += (right - left) % WINDOW
            for first in range(left, right - WINDOW + 1, WINDOW):
                frames = observed[first:first+WINDOW]
                p0, p1 = track[frames[0]], track[frames[1]]
                dx, dy = p1[0]-p0[0], p1[1]-p0[1]
                distances = []
                for k, frame in enumerate(frames[2:], 1):
                    x, y = track[frame]
                    distances.append(math.hypot(p1[0]+k*dx-x, p1[1]+k*dy-y))
                ade.append(math.fsum(distances)/7)
                fde.append(distances[-1])
        people_with_windows += len(ade) > before
    unique_points = sum(map(len, persons.values()))
    if not ade or unique_points != WINDOW*len(ade) + unused_tail_points:
        raise ValueError("Window accounting failed")
    if deltas.most_common(1)[0][0] != FRAMES_PER_OBSERVATION:
        raise ValueError("Sample cadence differs from the pre-specified six-frame interval")
    return {
        "scope": "ETH University seq_eth development sequence only; manually annotated tracks. No model fitting, interpolation or tuning.",
        "protocol": {"dt_s": 0.4, "raw_frame_step": FRAMES_PER_OBSERVATION,
                     "history_observations": 2, "history_span_s": 0.4, "future_steps": 7, "last_horizon_s": 2.8,
                     "xy_columns_one_based": [3,5],
                     "windows": "Group by original pedestrian ID, sort by frame and start new runs at every delta other than 6. Use disjoint blocks of nine from the beginning of each run. Ignore tail remainders. Collapse identical position duplicates, reject conflicting duplicates, omit nonfinite positions.",
                     "prediction": "Last observed ground position plus 1..7 times the last displacement; supplied velocity columns and target positions are not model inputs.",
                     "aggregation": "Equal weight per window; ADE averages all seven future Euclidean distances and FDE uses the seventh."},
        "source": {"url": URL, "archive_sha256": ARCHIVE_SHA,
                   "opened_members_sha256": {k:hashlib.sha256(v).hexdigest() for k,v in content.items()}},
        "counts": {"input_rows": rows, "nonfinite_position_rows": nonfinite_positions,
                   "identical_duplicate_rows": duplicates, "unique_frame_person_points": unique_points,
                   "people_in_annotations": len(persons), "people_with_windows": people_with_windows,
                   "exact_six_frame_intervals": deltas[FRAMES_PER_OBSERVATION],
                   "other_intervals_within_person": sum(deltas.values())-deltas[FRAMES_PER_OBSERVATION],
                   "continuous_runs": runs, "unused_tail_points": unused_tail_points,
                   "scored_windows": len(ade), "scored_future_positions": 7*len(ade)},
        "last_two_constant_velocity": {"ade_m": math.fsum(ade)/len(ade), "fde_m_at_2_8s": math.fsum(fde)/len(fde)},
    }


if __name__ == "__main__":
    result = benchmark(fetch_source())
    destination = Path(__file__).resolve().with_name("eth_development_cv.json")
    destination.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"counts":result["counts"],"errors_m":result["last_two_constant_velocity"]}, indent=2))
