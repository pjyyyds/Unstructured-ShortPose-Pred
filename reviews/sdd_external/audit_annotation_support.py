"""Audit un-interpolated 10 Hz SDD labels in all three phases before evaluating a predictor."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request

ARCHIVES = (
    ('crowdbotp/OpenTraj', '97ec1d5e1b579f26febeaa9083705952deac338a', 'datasets/SDD/'),
    ('flclain/StanfordDroneDataset', '9179f3c0dd9ac0bd67a6223703d04a7577a9fb18', ''),
)
LENGTHS = (5, 10, 20, 35, 40, 50)


def source_manifest(repo, rev, prefix):
    url = f'https://api.github.com/repos/{repo}/git/trees/{rev}?recursive=1'
    with urllib.request.urlopen(url, timeout=45) as response:
        tree = json.load(response)
    if tree['sha'] != rev or tree['truncated']:
        raise ValueError(f'Incomplete or changed source tree for {repo}')
    return {entry['path'][len(prefix):]: (entry['size'], entry['sha'])
            for entry in tree['tree'] if entry['type'] == 'blob' and entry['path'].startswith(prefix)
            and entry['path'].endswith('/annotations.txt')}


def run_lengths(frame_flags):
    counts = {phase: {length: 0 for length in LENGTHS} for phase in range(3)}
    longest = [0, 0, 0]
    for frames in frame_flags.values():
        previous, run = [-5]*3, [0]*3
        for frame in sorted(frames):
            phase = frame % 3
            run[phase] = run[phase]+1 if previous[phase]+3 == frame else 1
            previous[phase] = frame
            longest[phase] = max(longest[phase], run[phase])
            for length in LENGTHS:
                if run[phase] >= length:
                    counts[phase][length] += 1
    return {'possible_overlapping_windows': counts, 'maximum_consecutive': longest}


def inspect(content):
    present, noninterpolated = defaultdict(set), defaultdict(set)
    counts = Counter()
    for line in content.splitlines():
        row = line.split()
        if not row or row[9].strip(b'"').lower() != b'pedestrian':
            continue
        track, frame = int(row[0]), int(row[5])
        lost, occluded, generated = map(int, row[6:9])
        if not all(flag in (0, 1) for flag in (lost, occluded, generated)):
            raise ValueError('Unrecognized annotation flag')
        counts['all_pedestrian_rows'] += 1
        if lost:
            continue
        if frame in present[track]:
            raise ValueError('Repeated frame for one in-view pedestrian')
        present[track].add(frame)
        counts['in_view'] += 1
        counts['in_view_occluded'] += occluded
        counts['in_view_interpolated'] += generated
        if not generated:
            noninterpolated[track].add(frame)
            counts['in_view_noninterpolated'] += 1
    return {'counts': dict(counts), 'tracks_in_view': len(present),
            'tracks_with_noninterpolated_labels': len(noninterpolated),
            'in_view_all_rows': run_lengths(present), 'in_view_noninterpolated_only': run_lengths(noninterpolated)}


def audit(archive, manifests):
    expected, comparison = manifests
    if expected != comparison or len(expected) != 60:
        raise ValueError('The expected 60 annotation files differ between distribution mirrors')
    rows = {}
    with tarfile.open(archive, 'r:gz') as stream:
        for member in stream:
            if not member.isfile() or not member.name.endswith('/annotations.txt'):
                continue
            name = member.name.split('/', 1)[1]
            if name not in expected or name in rows:
                raise ValueError(f'Unexpected or duplicate {name}')
            content = stream.extractfile(member).read()
            git_blob = hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
            if (len(content), git_blob) != expected[name]:
                raise ValueError(f'Failed source hash for {name}')
            rows[name] = {'sha256': hashlib.sha256(content).hexdigest(), 'git_blob': git_blob, **inspect(content)}
    if set(rows) != set(expected):
        raise ValueError('Missing annotations')
    aggregate = Counter()
    for row in rows.values():
        aggregate.update(row['counts'])
    eligible = {n: sum(sum(v[n] for v in row['in_view_noninterpolated_only']['possible_overlapping_windows'].values())
                       for row in rows.values()) for n in LENGTHS}
    return {'scope': 'All 60 recordings across eight source sites. No predictions or coordinates read for scoring.',
            'nominal_clock': 'Original 30 fps video. Count every phase (0,1,2) separately at stride three original frames.',
            'definition': 'For one person every tick must have lost=0 and generated=0. An occluded annotation is retained. Lengths 35/40/50 include 30 completely labeled future positions. Counts may overlap.',
            'distribution': 'All 60 annotations match both pinned distribution mirrors by Git blob identity. The Stanford download endpoint was unavailable during collection.',
            'source_trees': [dict(repository=r, revision=s, prefix=p) for r,s,p in ARCHIVES],
            'totals': dict(aggregate),'possible_noninterpolated_windows_sum_over_phases':eligible,'recordings':dict(sorted(rows.items()))}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path, help='Tar.gz from the pinned flclain/StanfordDroneDataset revision, see README.')
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = audit(args.archive, [source_manifest(*x) for x in ARCHIVES])
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(result['totals'], result['possible_noninterpolated_windows_sum_over_phases'])
