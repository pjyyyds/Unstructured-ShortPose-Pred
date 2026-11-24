"""Standard-library-only, independent replay of the registered DAIR pose confidence comparison."""
import argparse
import ast
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import struct
from zipfile import ZipFile

ARCHIVE = '8d4d78336868155a660a457e230024c3f5c6acc54ba8965be416bb39f2ccbf5b'
ROSTER = '7f5771e9cd540259b7a059bfef8794600cd563adf96d733e71503925b94b2c71'
PROTOCOL = 'e38274320ebe02a2f87c106dbc297732ab7d23b56ee086f1976d36dd35d56779'
PREFIX = 'V2X-Seq-SPD/infrastructure-side/'
GRADE_VALUES = (0, 1, 2)


def sha(buf):
    return hashlib.sha256(buf).hexdigest()


def require(ok, detail):
    if not ok:
        raise ValueError(detail)


def npy(zip_file, field, shape_expected, dtype_expected):
    """Read the small set of fixed, C-ordered NPY 1.0 fields found in the frozen archive."""
    with zip_file.open(field + '.npy') as source:
        require(source.read(8) == b'\x93NUMPY\x01\x00', 'Unexpected NPY signature/version')
        length, = struct.unpack('<H', source.read(2))
        meta = ast.literal_eval(source.read(length).decode('latin1').strip())
        require(meta['shape'] == shape_expected and meta['fortran_order'] is False, field + ' shape/order')
        dtype = meta['descr']
        buffer = source.read()
    elements = math.prod(shape_expected)
    if dtype_expected == 'unicode':
        require(dtype.startswith('<U') and dtype[2:].isdigit(), field + ' dtype')
        width = int(dtype[2:]) * 4
        require(len(buffer) == width * elements, field + ' byte length')
        return [buffer[i:i+width].decode('utf-32-le').rstrip('\x00') for i in range(0, len(buffer), width)]
    require(dtype == dtype_expected, field + ' dtype')
    fmt = {'<f4': '<f', '<i8': '<q'}[dtype]
    require(len(buffer) == elements * struct.calcsize(fmt), field + ' byte length')
    return [entry[0] for entry in struct.iter_unpack(fmt, buffer)]


def load_windows(archive, source):
    require(sha(archive.read_bytes()) == ARCHIVE, 'Kaggle artifact identity')
    with ZipFile(archive) as outer:
        require(outer.testzip() is None, 'Dataset CRC')
        manifest = json.loads(outer.read('sha256.json'))
        names_bytes = outer.read('source_members.json')
        raw_npz = outer.read('private.npz')
        require(sha(names_bytes) == manifest['source_members.json'] and sha(raw_npz) == manifest['private.npz'], 'Member fingerprints')
        named = json.loads(names_bytes)
    with ZipFile(io.BytesIO(raw_npz)) as matrices:
        keys = json.loads(npy(matrices, 'keys_json', (), 'unicode')[0])
        observer_ids = npy(matrices, 'actors', (54,), 'unicode')
        clocks = npy(matrices, 'timestamps_us', (54, 35), '<i8')
        raw = npy(matrices, 'rtmpose_kpts', (54, 5, 17, 3), '<f4')
        q = npy(matrices, 'quality', (54, 4), '<f4')
    require(sha(json.dumps(keys, separators=(',', ':')).encode()) == ROSTER, 'Original cohort')
    require(len(keys) == len(set(map(tuple, keys))) == 54, 'Duplicated window')

    loaded = {}
    def member(path):
        if path not in loaded:
            buf = (source / path).read_bytes()
            require(sha(buf) == named[PREFIX + path], 'Source member ' + path)
            loaded[path] = json.loads(buf)
        return loaded[path]

    anchor_frames = {(seq, tick) for seq, _, end in keys for tick in range(end - 4, end + 1)}
    source_rows = {}
    for camera in member('data_info.json'):
        key = camera['sequence_id'], int(camera['frame_id'])
        if key in anchor_frames:
            require(key not in source_rows, 'Ambiguous enrolled camera frame')
            source_rows[key] = camera
    require(set(source_rows) == anchor_frames, 'Missing camera frame')
    windows = []
    unique_person_images = set()
    unique_images = set()
    truncated_codes = []
    no_output = 0
    for w, (seq, pid, end) in enumerate(keys):
        actor = (seq, pid)
        require(observer_ids[w] == f'{seq}:{pid}', 'Pose-crop actor')
        window = {'sequence': seq, 'actor': actor, 'grades': [], 'means': [], 'valid': [], 'empty': []}
        site_set = set()
        for j, tick in enumerate(range(end - 4, end + 1)):
            camera = source_rows[seq, tick]
            site_set.add(camera['intersection_loc'])
            time = int(camera['image_timestamp'])
            require(clocks[35*w+j] == time, 'Frame/camera clock')
            location = (camera['intersection_loc'], camera['camera_ip'], time)
            require((*location, pid) not in unique_person_images, 'Duplicate physical source target')
            unique_person_images.add((*location, pid))
            unique_images.add(location)
            matches = [o for o in member(camera['label_camera_std_path']) if o['type'] == 'Pedestrian' and int(o['track_id']) == pid]
            require(len(matches) == 1, 'Nonunique person at observed tick')
            grade = int(matches[0]['occluded_state'])
            require(grade in GRADE_VALUES, 'Unknown level')
            truncated_codes.append(int(matches[0]['truncated_state']))
            base = (5*w + j) * 17 * 3
            triples = raw[base:base + 17*3]
            require(len(triples) == 51 and all(math.isfinite(v) for v in triples), 'Source cohort contains nonfinite entries')
            scores = [min(1.0, max(0.0, value)) for value in triples[2::3]]
            absent = all(v == 0.0 for v in triples)
            no_output += absent
            window['grades'].append(grade)
            window['means'].append(statistics.fmean(scores))
            window['valid'].append(sum(score >= 0.20 for score in scores))
            window['empty'].append(absent)
        require(len(site_set) == 1, 'Location changed in a single window')
        window['site'], = site_set
        require(abs(sum(window['valid']) / 85 - q[w*4]) < 4e-8, 'Threshold mismatch with original q1')
        windows.append(window)
    require(len(unique_person_images) == 270 and len(unique_images) == 210 and len(loaded) == 211, 'Source support')
    return windows, {'unique_human_frames': len(unique_person_images), 'distinct_camera_images': len(unique_images),
                     'verified_label_and_camera_metadata_files': len(loaded), 'zero_filled_person_outputs': no_output,
                     'untruncated_human_frames': truncated_codes.count(0), 'original_per_window_q1_checked': len(windows)}


def subset_summary(windows):
    grade_groups = {}
    for grade in GRADE_VALUES:
        frames = [(w, j) for w in windows for j, g in enumerate(w['grades']) if g == grade]
        valid = sum(w['valid'][j] for w, j in frames)
        n = len(frames)
        grade_groups[str(grade)] = {
            'frames': n, 'windows': sum(grade in w['grades'] for w in windows),
            'sequence_local_people': len({w['actor'] for w, _ in frames}),
            'sequences': len({w['sequence'] for w, _ in frames}), 'intersections': len({w['site'] for w, _ in frames}),
            'joints': 17*n, 'valid_joints': valid, 'whole_person_no_output': sum(w['empty'][j] for w, j in frames),
            'valid_joint_fraction_at_0_20': valid / (17*n) if n else None,
            'joint_confidence_mean': statistics.fmean(w['means'][j] for w, j in frames) if n else None}

    deltas, mixed = [], []
    for w in windows:
        clear = [v for g, v in zip(w['grades'], w['means']) if g == 0]
        occ = [v for g, v in zip(w['grades'], w['means']) if g in (1, 2)]
        if clear and occ:
            mixed.append(w)
            deltas.append(statistics.fmean(occ) - statistics.fmean(clear))
    f = 5 * len(mixed)
    n_clear = sum(w['grades'].count(0) for w in mixed)
    return {'windows': len(windows), 'grades': grade_groups, 'mixed_windows': {
        'frames': f, 'windows': len(mixed), 'sequence_local_people': len({w['actor'] for w in mixed}),
        'sequences': len({w['sequence'] for w in mixed}), 'intersections': len({w['site'] for w in mixed}),
        'clear_frames': n_clear, 'occluded_frames': f - n_clear, 'lower': sum(x < 0 for x in deltas),
        'equal': deltas.count(0.0), 'higher': sum(x > 0 for x in deltas),
        'mean_occluded_minus_clear': statistics.fmean(deltas) if deltas else None,
        'median_occluded_minus_clear': statistics.median(deltas) if deltas else None}}


def replay(archive, root, protocol, comparison):
    require(sha(protocol.read_bytes()) == PROTOCOL, 'Registered definition')
    spec = json.loads(comparison.read_bytes())
    require(spec['protocol_sha256'] == PROTOCOL and spec['archive_sha256'] == ARCHIVE
            and spec['ordered_window_keys_sha256'] == ROSTER and spec['support_only'] is False, 'Comparison cohort or endpoint')
    windows, inputs = load_windows(archive, root)
    pooled = subset_summary(windows)
    by_site = {}
    by_clip = {}
    for site in sorted({w['site'] for w in windows}):
        by_site[site] = subset_summary([w for w in windows if w['site'] == site])
    for clip in sorted({w['sequence'] for w in windows}):
        v = subset_summary([w for w in windows if w['sequence'] == clip])
        by_clip[clip] = {'windows': v['windows'], 'mixed_windows': v['mixed_windows']['windows'],
                         'frames_grade_0_1_2': [v['grades'][str(g)]['frames'] for g in GRADE_VALUES]}
    summary = {'pooled': pooled, 'by_intersection': by_site, 'support_by_sequence': by_clip}
    checks = {'numeric_values': 0, 'count_values': 0, 'undefined_values': 0, 'max_abs_difference': 0.0}

    def compare(own, published, path='statistics'):
        if isinstance(own, dict):
            require(isinstance(published, dict) and own.keys() == published.keys(), path + ' fields')
            for key in own:
                compare(own[key], published[key], path + '.' + key)
        elif isinstance(own, list):
            require(isinstance(published, list) and len(own) == len(published), path + ' list')
            for i, (a, b) in enumerate(zip(own, published)):
                compare(a, b, path + '[' + str(i) + ']')
        elif own is None:
            require(published is None, path + ' zero-support result')
            checks['undefined_values'] += 1
        elif type(own) is int:
            require(type(published) is int and published == own, path + ' exact integer')
            checks['count_values'] += 1
        else:
            require(isinstance(published, (int, float)), path + ' numeric')
            delta = abs(own - published)
            require(delta < 1e-14, path + ' numerical disagreement')
            checks['max_abs_difference'] = max(checks['max_abs_difference'], delta)
            checks['numeric_values'] += 1

    compare(summary, spec['statistics'])
    return {'protocol_sha256': PROTOCOL, 'archive_sha256': ARCHIVE, 'ordered_cohort_key_sha256': ROSTER,
            'original_aggregate_sha256': sha(comparison.read_bytes()), 'second_verifier_sha256': sha(Path(__file__).read_bytes()),
            'method': 'Python standard library. Authenticated original camera annotations, decoded the NPY arrays with struct and computed per-frame and within-window means using statistics.fmean. No model execution or per-person publication.',
            'input_checks': inputs, 'comparison': checks, 'recomputed_aggregates': summary}


if __name__ == '__main__':
    arg = argparse.ArgumentParser(description=__doc__)
    arg.add_argument('--archive', type=Path, required=True)
    arg.add_argument('--metadata', type=Path, required=True, help='Directory with data_info.json and label/camera/')
    arg.add_argument('--protocol', type=Path, required=True)
    arg.add_argument('--reference', type=Path, required=True, help='Original aggregate from check_occlusion.py')
    arg.add_argument('--out', type=Path, required=True)
    a = arg.parse_args()
    result = replay(a.archive, a.metadata, a.protocol, a.reference)
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result['comparison']))
