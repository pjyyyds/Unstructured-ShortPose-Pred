"""Check source occlusion grades against archived RTMPose scores. NumPy and standard library."""
import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
from zipfile import ZipFile

import numpy as np


ARCHIVE_SHA = '8d4d78336868155a660a457e230024c3f5c6acc54ba8965be416bb39f2ccbf5b'
PROTOCOL_SHA = 'e38274320ebe02a2f87c106dbc297732ab7d23b56ee086f1976d36dd35d56779'
ROSTER_SHA = '7f5771e9cd540259b7a059bfef8794600cd563adf96d733e71503925b94b2c71'
PREFIX = 'V2X-Seq-SPD/infrastructure-side/'
GRADES = (0, 1, 2)
THRESHOLD = 0.20  # >=, in frozen predictor source 032800e94bb0828c45d3f5195ac5cc17aa93bc4c.


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


@dataclass
class Design:
    grades: np.ndarray
    sites: np.ndarray
    sequences: np.ndarray
    actors: np.ndarray


def read_design(root, z, records):
    named = json.loads(z.read('source_members.json'))
    require(sha(z.read('source_members.json')) == json.loads(z.read('sha256.json'))['source_members.json'], 'Source manifest')
    checked_files = set()

    def verified(name):
        data = (root / name).read_bytes()
        require(sha(data) == named[PREFIX + name], 'Source changed: ' + name)
        checked_files.add(name)
        return json.loads(data)

    keys = json.loads(records['keys_json'].item())
    key_sha = sha(json.dumps(keys, separators=(',', ':')).encode())
    require(key_sha == ROSTER_SHA and len(keys) == len(set(map(tuple, keys))) == 54, 'Original roster')
    actors = records['actors']
    np.testing.assert_array_equal(actors, [f'{s}:{p}' for s,p,t in keys])
    clocks = records['timestamps_us'][:, :5]
    raw_meta = verified('data_info.json')
    metadata = {(r['sequence_id'], int(r['frame_id'])): r for r in raw_meta}
    require(len(metadata) == len(raw_meta), 'Duplicate metadata identity')
    raw_labels = {}
    sites, sequences, states = [], [], []
    people_seen, images_seen = set(), set()
    truncation = Counter()
    for i, (sequence, target, end) in enumerate(keys):
        values, corners = [], set()
        for j, tick in enumerate(range(end - 4, end + 1)):
            meta = metadata[sequence, tick]
            stamp = int(meta['image_timestamp'])
            require(stamp == clocks[i, j], 'Camera timestamp differs from pose crop')
            camera = meta['intersection_loc'], meta['camera_ip']
            image_key = (*camera, stamp)
            truth_key = (*image_key, target)
            require(truth_key not in people_seen, 'Repeated same physical person-frame')
            people_seen.add(truth_key)
            images_seen.add(image_key)
            corners.add(camera)
            label = meta['label_camera_std_path']
            if label not in raw_labels:
                raw_labels[label] = verified(label)
            matches = [o for o in raw_labels[label] if o['type'] == 'Pedestrian' and int(o['track_id']) == target]
            require(len(matches) == 1, 'Ambiguous target among camera annotations')
            grade = int(matches[0]['occluded_state'])
            require(grade in GRADES, 'Unrecognized occlusion grade')
            truncation[int(matches[0]['truncated_state'])] += 1
            values.append(grade)
        require(len(corners) == 1, 'Camera change within a window')
        sites.append(corners.pop()[0])
        sequences.append(sequence)
        states.append(values)
    design = Design(np.array(states), np.array(sites), np.array(sequences), np.asarray(actors))
    require(len(people_seen) == 270 and len(images_seen) == 210, 'Original image-person support')
    np.testing.assert_array_equal(np.bincount(design.grades.ravel(), minlength=3), [172, 96, 2])
    return design, {'source_annotation_and_metadata_members_verified': len(checked_files),
                    'unique_camera_images': len(images_seen), 'unique_person_frames': len(people_seen),
                    'truncation_is_separate_field': dict(sorted(truncation.items()))}


def support(design, mask):
    windows = mask.any(axis=1)
    return {'frames': int(mask.sum()), 'windows': int(windows.sum()),
            'sequence_local_people': len(set(design.actors[windows])),
            'sequences': len(set(design.sequences[windows])), 'intersections': len(set(design.sites[windows]))}


def get_measurements(raw):
    require(raw.shape[-3:] == (5, 17, 3), 'Expected five 17-joint x/y/score arrays')
    # A crop with no usable detector output remains in place as a zero record.
    cleaned = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    no_output = np.all(cleaned == 0, axis=(-1, -2))
    conf = np.clip(cleaned[..., 2], 0, 1)
    valid = conf >= np.asarray(THRESHOLD, dtype=conf.dtype)
    return conf, valid, no_output


def statistics(design, conf=None, valid=None, no_output=None):
    def grouped(windows):
        grades = {}
        for grade in GRADES:
            mask = windows[:, None] & (design.grades == grade)
            item = support(design, mask)
            if conf is not None:
                scores = conf[mask]
                item.update({'joints': scores.size, 'joint_confidence_mean': float(scores.mean(dtype=np.float64)) if scores.size else None,
                             'valid_joints': int(valid[mask].sum()),
                             'valid_joint_fraction_at_0_20': float(valid[mask].mean()) if scores.size else None,
                             'whole_person_no_output': int(no_output[mask].sum())})
            grades[str(grade)] = item
        clear = design.grades == 0
        mixed = windows & clear.any(axis=1) & (~clear).any(axis=1)
        pair = support(design, np.broadcast_to(mixed[:, None], clear.shape))
        pair['clear_frames'] = int(clear[mixed].sum())
        pair['occluded_frames'] = int((~clear[mixed]).sum())
        if conf is not None:
            # Unrounded equal-weight joint averages, then grade-group means within each window.
            frame_scores = conf.mean(axis=2, dtype=np.float64)
            deltas = np.asarray([row[~low].mean() - row[low].mean() for row, low in zip(frame_scores[mixed], clear[mixed])])
            pair.update(mean_occluded_minus_clear=float(deltas.mean()) if len(deltas) else None,
                        median_occluded_minus_clear=float(np.median(deltas)) if len(deltas) else None,
                        lower=int((deltas < 0).sum()), equal=int((deltas == 0).sum()), higher=int((deltas > 0).sum()))
        return {'windows': int(windows.sum()), 'grades': grades, 'mixed_windows': pair}

    return {'pooled': grouped(np.ones(len(design.sites), dtype=bool)),
            'by_intersection': {s: grouped(design.sites == s) for s in sorted(set(design.sites))},
            'support_by_sequence': {
                s: {'windows': int((design.sequences == s).sum()),
                    'mixed_windows': int(((design.sequences == s) & (design.grades == 0).any(1) & (design.grades > 0).any(1)).sum()),
                    'frames_grade_0_1_2': [int((design.grades[design.sequences == s] == g).sum()) for g in GRADES]}
                for s in sorted(set(design.sequences))}}


def crosscheck(design, conf, valid, metrics):
    """Recompute pooled estimates by scalar iteration on joints and equal-weight windows."""
    totals = {g: [] for g in GRADES}
    n_valid = Counter()
    diffs = []
    for codes, frames in zip(design.grades, conf):
        baseline, occluded = [], []
        for grade, sample in zip(codes, frames):
            score = math.fsum(float(c) for c in sample) / 17
            totals[int(grade)].append(score)
            n_valid[int(grade)] += sum(c >= np.float32(THRESHOLD) for c in sample)
            (baseline if grade == 0 else occluded).append(score)
        if baseline and occluded:
            diffs.append(math.fsum(occluded) / len(occluded) - math.fsum(baseline) / len(baseline))
    for grade in GRADES:
        g = metrics['grades'][str(grade)]
        np.testing.assert_allclose(g['joint_confidence_mean'], math.fsum(totals[grade]) / len(totals[grade]), rtol=0, atol=1e-14)
        assert g['valid_joints'] == n_valid[grade]
    pair = metrics['mixed_windows']
    assert (pair['lower'], pair['equal'], pair['higher']) == (sum(x < 0 for x in diffs), diffs.count(0), sum(x > 0 for x in diffs))
    np.testing.assert_allclose(pair['mean_occluded_minus_clear'], math.fsum(diffs) / len(diffs), rtol=0, atol=1e-14)


def run(archive, source, protocol, support_only):
    require(sha(archive.read_bytes()) == ARCHIVE_SHA, 'Cloud archive identity')
    require(sha(protocol.read_bytes()) == PROTOCOL_SHA, 'Registered endpoints')
    with ZipFile(archive) as z:
        require(z.testzip() is None, 'Archive CRC')
        packed = z.read('private.npz')
        require(sha(packed) == json.loads(z.read('sha256.json'))['private.npz'], 'Archived NPZ fingerprint')
        with np.load(io.BytesIO(packed), allow_pickle=False) as record:
            design, provenance = read_design(source, z, record)
            if support_only:
                out = statistics(design)
            else:
                raw = record['rtmpose_kpts']
                require(raw.shape == (54, 5, 17, 3), 'Original person-joint array')
                conf, valid, empty = get_measurements(raw)
                # All five underlying boxes passed the source feature constructor's height gate.
                np.testing.assert_allclose(valid.mean(axis=(1, 2)), record['quality'][:, 0], atol=4e-8, rtol=0)
                out = statistics(design, conf, valid, empty)
                crosscheck(design, conf, valid, out['pooled'])
                provenance.update({'nonfinite_source_confidence': int((~np.isfinite(raw[..., 2])).sum()),
                                   'score_outside_unit_interval': int(((raw[..., 2] < 0) | (raw[..., 2] > 1)).sum()),
                                   'whole_person_all_zero_records': int(empty.sum()),
                                   'frames_with_all_zero_confidences': int((conf == 0).all(axis=2).sum()),
                                   'archived_q1_denominators_rebuilt': len(valid)})
    return {'scope': '54 previously enrolled windows. Equal weight for 17 archived RTMPose detector confidences per human-frame; same-window contrasts weight each mixed window equally. No trajectory scores or new models.',
            'registration_git': '238b5f0', 'protocol_sha256': PROTOCOL_SHA, 'archive_sha256': ARCHIVE_SHA,
            'ordered_window_keys_sha256': ROSTER_SHA, 'verifier_sha256': sha(Path(__file__).read_bytes()),
            'source_predictor_git': '032800e94bb0828c45d3f5195ac5cc17aa93bc4c', 'numpy': np.__version__,
            'provenance': provenance, 'support_only': support_only, 'statistics': out}


def test():
    g = np.array([[0, 1, 0, 0, 2], [0, 0, 0, 0, 0], [1, 0, 1, 1, 1]])
    d = Design(g, np.array(['A','B','A']), np.array(['a','b','a']), np.array(['a:1','b:1','a:2']))
    f = np.ones((3, 5, 17, 3), dtype=np.float32)
    f[..., 2] = 1
    f[0, 1] = 0
    f[0, 4, :, 2] = -0.1
    f[2, 1, :, 2] = np.float32(0.2)
    f[1, 1, 0, 2] = 2.9
    c, v, e = get_measurements(f)
    assert e.sum() == 1 and v[2, 1].all() and c[1, 1, 0] == 1
    s = statistics(d, c, v, e)['pooled']
    assert s['mixed_windows']['lower'] == 1 and s['mixed_windows']['higher'] == 1
    crosscheck(d, c, v, s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--support-only', action='store_true')
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--metadata', type=Path, help='Root containing data_info.json and label/camera/')
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.test:
        test()
        print('Synthetic clipping, missing retention, exact threshold, and equal-window checks passed')
    else:
        if any(x is None for x in (args.archive, args.metadata, args.protocol, args.out)):
            parser.error('Supply --archive, --metadata, --protocol, --out')
        result = run(args.archive, args.metadata, args.protocol, args.support_only)
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
        print(json.dumps(result['statistics']['pooled']))
