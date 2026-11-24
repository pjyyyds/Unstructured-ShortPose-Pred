"""Replay a frozen seed-42 subgroup from archive only. Requires NumPy and the standard library."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
from zipfile import ZipFile

import numpy as np


HISTORY = 'caa8f0959f28f2d69a6f16cc9df115ab6222d0b6eb3297edd575615fef81b94c'
SIX = '388920f757f0f76d688f281e232d20f2c1afe4178e50771d6f9d0f6146f61748'
RERANKED = 'b24ac2bb1ad3d6742be2cbe16b927e42ef9a1a54a17ce2c08be3e92fa286b871'
PROTOCOL = '78ffdcfd4587399ad5d413d184a61ac08dd92642f8c69b86af6969e9d5a74480'
ALL_KEYS = 'f5ee511a54ad818499d3dde6213f7f3b49332943ff2fc93d93aed87d316dbb91'
SUBSET_KEYS = '198c634d9c5d447a14b05327ffd51d738962cf71a628dfcce571551e4e092684'
SITES = {'america_crossroad': 146, 'italy_park': 24, 'japan_crossroad': 102, 'thailand_street': 51}
EVENTS = ('start', 'stop', 'left', 'right', 'maintain')
MODELS = ('M0', 'M1')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def key_hash(keys):
    return digest(json.dumps(keys, separators=(',', ':')).encode())


def keys_of(part):
    return list(zip(part['clip'].tolist(), part['person_id'].tolist(), part['t0'].tolist()))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def event_flags(track):
    """35 complete 2D samples. Smoothed velocity at index j ends at source tick j+1."""
    track = np.asarray(track, dtype=np.float64)
    require(track.shape == (35, 2) and np.isfinite(track).all(), 'Incomplete event track')
    velocity = np.diff(track, axis=0) / 0.1
    mean_v = np.array([velocity[max(0, j - 2):j + 1].mean(axis=0) for j in range(34)])
    speed = np.hypot(*mean_v.T)
    prior = speed[1:4].mean()
    future = speed[4:]

    def triplets(flags):
        return np.flatnonzero(flags[: -2] & flags[1: -1] & flags[2:])

    start = prior <= 0.2 and bool(len(triplets(future >= 0.6)))
    stop = prior >= 0.6 and bool(len(triplets(future <= 0.2)))
    left = right = False
    # Each contiguous moving segment has its own baseline heading.
    # Candidates are encountered chronologically; earliest sustained future turn wins.
    moving = np.flatnonzero(speed[1:] >= 0.5) + 1
    for section in np.split(moving, np.flatnonzero(np.diff(moving) != 1) + 1):
        if len(section) < 3:
            continue
        angle = np.unwrap(np.arctan2(mean_v[section, 1], mean_v[section, 0]))
        turn = (angle - angle[0])[section >= 4]
        if len(turn) < 3:
            continue
        up = triplets(turn >= math.pi / 6)
        down = triplets(turn <= -math.pi / 6)
        if len(up) or len(down):
            left = bool(len(up) and (not len(down) or up[0] <= down[0]))
            right = not left
            break
    return (start, stop, left, right, not any((start, stop, left, right)))


def counts(truth, any_hit, top_hit, clips):
    by_event = {}
    for e, event in enumerate(EVENTS):
        positive = truth[:, e]
        result = {'positive': int(positive.sum()), 'negative': int((~positive).sum()),
                  'positive_clips': len(set(clips[positive])), 'negative_clips': len(set(clips[~positive]))}
        for i, model in enumerate(MODELS):
            result[model] = {}
            for name, hit in [('top1', top_hit[i, :, e]), ('any6', any_hit[i, :, e])]:
                result[model][name] = {'TP': int((positive & hit).sum()), 'FP': int((~positive & hit).sum())}
        result['neither_model_any6_true_positives_missed'] = int((positive & ~any_hit[:, :, e].any(axis=0)).sum())
        by_event[event] = result
    return {'windows': len(truth), 'clips': len(set(clips)), 'events': by_event}


def verify(root, protocol):
    named = {'pinns_matched_histories.npz': HISTORY, 'anchored_six_candidates.zip': SIX, 'reranked_modes.zip': RERANKED}
    require(digest(protocol.read_bytes()) == PROTOCOL, 'Protocol fingerprint')
    for name, sha in named.items():
        require(digest((root / name).read_bytes()) == sha, 'Source fingerprint ' + name)
    with np.load(root / 'pinns_matched_histories.npz', allow_pickle=False) as info:
        keys = keys_of(info)
        require(len(keys) == len(set(keys)) == 1482 and key_hash(keys) == ALL_KEYS, 'Full roster')
        history = info['history_20_xy'][:, -5:]
        require(info['history_20_mask'][:, -5:].all() and np.isfinite(history).all(), 'History missing')
        first = info['t0'] - 4 == info['first_person_label_t']
        length = [math.fsum(math.dist(a, b) for a, b in zip(w[:-1], w[1:])) for w in history]
        take = first & (np.asarray(length) < 0.5)
        sites, clips, future = info['site'], info['clip'], info['future_30_xy']
    subset = [key for key, keep in zip(keys, take) if keep]
    require(key_hash(subset) == SUBSET_KEYS and len(subset) == 323, 'First-short roster')
    require(len({k[:2] for k in subset}) == 323, 'Repeated clip-person target')
    require({s: int((take & (sites == s)).sum()) for s in set(sites)} == SITES, 'Site denominators')

    with ZipFile(root / 'reranked_modes.zip') as z:
        require(z.testzip() is None, 'Ranking CRC')
        with np.load(io.BytesIO(z.read('reranked_paired_windows.npz')), allow_pickle=False) as q:
            require(keys_of(q) == keys, 'Ranking keys')
            top_selection = q['mode_indices'][q['seeds'].tolist().index(42), :, :, q['index_columns'].tolist().index('network_top1')]
            require(q['models'].tolist() == list(MODELS), 'Ranking arms')
    selected = np.flatnonzero(take)
    truth = np.array([event_flags(np.vstack((history[j], future[j]))) for j in selected])
    any_hit = np.zeros((2, len(selected), 5), dtype=bool)
    top_hit = np.zeros_like(any_hit)
    checked = 0
    with ZipFile(root / 'anchored_six_candidates.zip') as z:
        require(z.testzip() is None, 'Original candidate CRC')
        require(json.loads(z.read('scores.json'))['training_seed'] == 42, 'Seed')
        registration = json.loads(z.read('run.json'))['registration']
        require(registration['ordered_keys_sha256'] == ALL_KEYS, 'Candidate roster')
        manifest = json.loads(z.read('sha256.json'))
        for site in sorted(SITES):
            fold = np.flatnonzero(sites == site)
            entry = site + '/heldout_private.npz'
            blob = z.read(entry)
            require(digest(blob) == manifest[entry], 'Candidate member fingerprint')
            with np.load(io.BytesIO(blob), allow_pickle=False) as q:
                require(keys_of(q) == [keys[i] for i in fold], 'Fold order')
                require(key_hash(keys_of(q)) == registration['folds'][site]['test_key_sha256'], 'Fold roster digest')
                local = np.flatnonzero(take[fold])
                output_rows = np.searchsorted(selected, fold[local])
                for a, model in enumerate(MODELS):
                    paths = q['original_' + model + '_paths'][local]
                    prob = q['original_' + model + '_probabilities'][local]
                    require(paths.shape == (len(local), 6, 30, 2) and prob.shape == (len(local), 6), 'Candidate dimensions')
                    require(np.isfinite(paths).all() and np.isfinite(prob).all() and (prob >= 0).all(), 'Candidate validity')
                    np.testing.assert_allclose(prob.sum(1), 1., rtol=0, atol=1e-6)
                    chosen = np.argmax(prob, axis=1)
                    np.testing.assert_array_equal(chosen, top_selection[a, fold[local]])
                    checked += len(local)
                    for j, out_row in enumerate(output_rows):
                        h = history[selected[out_row]]
                        emitted = np.array([event_flags(np.vstack((h, candidate))) for candidate in paths[j]])
                        any_hit[a, out_row] = np.any(emitted, axis=0)
                        top_hit[a, out_row] = emitted[chosen[j]]
    require(checked == 2 * len(selected), 'Ranking comparison incomplete')
    result = {}
    for site in ('pooled', *sorted(SITES)):
        rows = np.ones(len(selected), dtype=bool) if site == 'pooled' else sites[selected] == site
        result[site] = counts(truth[rows], any_hit[:, rows], top_hit[:, rows], clips[selected][rows])
    return {'registration_git': '86eeadd', 'protocol_sha256': PROTOCOL, 'script_sha256': digest(Path(__file__).read_bytes()),
            'primary_events': ['start', 'stop'], 'secondary_events': list(EVENTS[2:]), 'seed': 42,
            'input_sha256': named, 'all_keys_sha256': ALL_KEYS, 'subgroup_keys_sha256': SUBSET_KEYS,
            'ranking_indices_matched_against_separate_archive': checked, 'candidate_tracks_scored': checked * 6,
            'label_paths_scored': len(selected), 'numpy_version': np.__version__, 'counts': result}


def test():
    def track(hist_v, future_v):
        vel = np.vstack((np.tile(hist_v, (4, 1)), np.broadcast_to(future_v, (30, 2))))
        return np.vstack((np.zeros(2), np.cumsum(vel * 0.1, axis=0)))
    for name, h, f in [('maintain',(1,0),(1,0)), ('start',(0,0),(1,0)), ('stop',(1,0),(0,0)),
                       ('left',(1,0),(0,1)), ('right',(1,0),(0,-1))]:
        assert np.array_equal(event_flags(track(h, f)), np.asarray(EVENTS) == name)
    assert event_flags(track((0,0), (0,0)))[4]
    true = np.eye(5, dtype=bool)
    any_hit = np.array([true.copy(), true.copy()])
    any_hit[:, 0, 0] = False
    any_hit[0, 4, 0] = True
    top = any_hit.copy()
    top[0, 4, 0] = False
    stats = counts(true, any_hit, top, np.array(list('aabcd')))['events']['start']
    assert stats['M0'] == {'top1': {'TP': 0, 'FP': 0}, 'any6': {'TP': 0, 'FP': 1}}
    assert stats['neither_model_any6_true_positives_missed'] == 1 and stats['positive'] == 1 and stats['negative'] == 4


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.test:
        test()
        print('Synthetic event and paired counting checks passed')
    else:
        if any(v is None for v in (args.input, args.protocol, args.output)):
            parser.error('Provide --input, --protocol, --output or use --test')
        summary = verify(args.input, args.protocol)
        args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
