"""Independently check history-only DUT graph membership without loading forecast models or future positions."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def canonical(obj):
    return json.dumps(obj, separators=(',', ':'), ensure_ascii=True).encode()


def new_counts():
    return {'windows': 0, 'pool': Counter(), 'selected': Counter(), 'removed_by_limit': Counter(),
            'support': Counter(), 'pool_support': Counter(), 'velocity_valid': Counter(),
            'velocity_missing': Counter(), 'neighbor_type_codes': Counter(),
            'presence_frames_1_to_5': Counter(), 'ego_actual_speed_below_0_2_windows': 0,
            'coincident_candidates_excluded': 0, 'windows_above_12_candidates': 0,
            'maximum_pool_size': 0, 'windows_people_displace_all_candidate_vehicles': 0,
            'windows_vehicles_displace_all_candidate_people': 0, 'ego_history_5of5': 0}


def support(ped_count, car_count):
    if ped_count and car_count:
        return 'mixed'
    if ped_count:
        return 'people_only'
    if car_count:
        return 'cars_only'
    return 'empty'


def read_history(snapshot_frames, files, scale, source_hashes, source_dir):
    rows = defaultdict(dict)
    for category, path in files:
        assert sha(path.read_bytes()) == source_hashes[str(path.relative_to(source_dir))]
        cols = ('x', 'y') if category == 'Person' else ('x_c', 'y_c')
        with path.open() as src:
            for row in csv.DictReader(src):
                frame = int(row['frame'])
                if frame not in snapshot_frames:
                    continue
                ident = category, int(row['id'])
                xy = np.asarray([float(row[k]) for k in cols], dtype=np.float64) / scale
                if ident in rows[frame] or not np.isfinite(xy).all():
                    raise ValueError(f'Duplicate or invalid source record: {path.name}, {frame}, {ident}')
                rows[frame][ident] = xy
    return rows


def inspect(root):
    coverage = json.loads((root / 'coverage.json').read_bytes())
    cases = json.loads((root / 'private_cohort.json').read_bytes())
    keys = [v['key'] for v in cases]
    assert len(keys) == 3066 and len({tuple(k) for k in keys}) == 3066
    assert sha(canonical(keys)) == '02c44cf55e5ca8690dc1fe9ac9c076af44dc89f11629b8ccc9e725e49e4580ec'
    source_dir = root / 'private_source'
    by_clip = defaultdict(list)
    for case in cases:
        clip, pid, t0 = case['key']
        observed = tuple(map(int, case['frames_1based'][:5]))
        ts = np.array(case['time_s'][:5], dtype=np.float64)
        assert sorted(set(observed)) == list(observed) and observed[-1] == t0
        np.testing.assert_allclose(np.diff(ts), np.diff(observed) * 1000 / 23976, rtol=0, atol=2e-14)
        by_clip[clip].append((pid, t0, observed))
    per_clip = {}
    ordered_ids = []
    ordered_state = []
    swaps = []
    swap_by_clip = {}
    revised_members = []
    for clip, items in sorted(by_clip.items()):
        ctr = new_counts()
        swap_stats = Counter()
        swap_by_clip[clip] = swap_stats
        needed = {frame for _, _, observed in items for frame in observed}
        ratio = source_dir / 'data/ratios' / f'{clip}_ratio_pixel2meter.txt'
        assert sha(ratio.read_bytes()) == coverage['members'][str(ratio.relative_to(source_dir))]
        files = [(cat, source_dir / 'data/trajectories' / f'{clip}_traj_{label}.csv')
                 for cat, label in [('Person', 'ped'), ('Car', 'veh')]]
        snapshots = read_history(needed, files, float(ratio.read_text()), coverage['members'], source_dir)
        for person, t0, observed in items:
            ego = 'Person', person
            past = [snapshots[k] for k in observed]
            assert all(ego in s for s in past)
            current = past[-1]
            others = [ident for ident in current if ident != ego]
            delta = np.array([current[k] - current[ego] for k in others], dtype=np.float64).reshape(-1, 2)
            distance = np.hypot(delta[:, 0], delta[:, 1])
            qualified = np.flatnonzero((distance > 1e-5) & (distance <= 10.0))
            ranking = sorted(qualified, key=lambda i: (distance[i], *others[i]))
            selected = ranking[:12]
            pool = Counter(others[i][0] for i in ranking)
            kept = Counter(others[i][0] for i in selected)
            chosen_ids = [others[i] for i in selected]
            ranked_cars = [i for i in ranking if others[i][0] == 'Car']
            revised = list(chosen_ids)
            if ranked_cars and not kept['Car']:
                assert len(chosen_ids) == 12 and all(c == 'Person' for c, ident in chosen_ids)
                winner = min(ranked_cars, key=lambda i: (distance[i], others[i][1]))
                assert winner == ranked_cars[0] and winner not in selected
                incoming = others[winner]
                removed = chosen_ids[-1]
                revised[-1] = incoming
                assert revised[:11] == chosen_ids[:11] and len(set(revised)) == len(chosen_ids)
                assert [j for j in range(12) if revised[j] != chosen_ids[j]] == [11]
                past_present = [j for j in range(4) if incoming in past[j]]
                swaps.append([clip, person, t0, incoming[1], removed[1]])
                swap_stats['swapped_windows'] += 1
                swap_stats['inserted_car_with_historical_velocity'] += bool(past_present)
            else:
                assert revised == chosen_ids
                swap_stats['unchanged_windows'] += 1
            swap_stats['original_windows_with_car'] += bool(kept['Car'])
            swap_stats['radius_pool_windows_with_car'] += bool(ranked_cars)
            swap_stats['revised_windows_with_car'] += any(c == 'Car' for c, ident in revised)
            swap_stats['revised_person_edges'] += sum(c == 'Person' for c, ident in revised)
            swap_stats['revised_car_edges'] += sum(c == 'Car' for c, ident in revised)
            revised_members.append([[clip, person, t0], revised])
            ctr['windows'] += 1
            ctr['ego_history_5of5'] += 1
            ctr['pool'].update(pool)
            ctr['selected'].update(kept)
            ctr['removed_by_limit'].update(pool-kept)
            ctr['pool_support'][support(pool['Person'], pool['Car'])] += 1
            ctr['support'][support(kept['Person'], kept['Car'])] += 1
            ctr['coincident_candidates_excluded'] += int(np.count_nonzero(distance <= 1e-5))
            ctr['windows_above_12_candidates'] += int(len(ranking) > 12)
            ctr['maximum_pool_size'] = max(ctr['maximum_pool_size'], len(ranking))
            ctr['windows_people_displace_all_candidate_vehicles'] += int(bool(pool['Car'] and not kept['Car']))
            ctr['windows_vehicles_displace_all_candidate_people'] += int(bool(pool['Person'] and not kept['Person']))
            ego_vel = (current[ego] - past[-2][ego]) / ((observed[-1] - observed[-2]) * 1000 / 23976)
            ctr['ego_actual_speed_below_0_2_windows'] += int(np.linalg.norm(ego_vel) < .2)
            ids = []
            states = []
            for i in selected:
                neighbor = others[i]
                category, ident = neighbor
                presence = [j for j, previous in enumerate(past) if neighbor in previous]
                count = len(presence)
                ctr['presence_frames_1_to_5'][str(count)] += 1
                if count == 1:
                    ctr['velocity_missing'][category] += 1
                    type_code = 6 if category == 'Car' else 0
                else:
                    ctr['velocity_valid'][category] += 1
                    j = presence[-2]
                    dt = (observed[-1] - observed[j]) * 1000 / 23976
                    neighbor_vel = (current[neighbor] - past[j][neighbor]) / dt
                    relative = neighbor_vel - ego_vel
                    radial = -np.dot(relative, delta[i]) / (distance[i] + 1e-6)
                    assert np.isfinite(radial) and np.linalg.norm(relative) + 1e-10 >= abs(radial)
                    type_code = 6 if category == 'Car' else (2 if np.linalg.norm(neighbor_vel) < .2 else 0)
                ctr['neighbor_type_codes'][str(type_code)] += 1
                ids.append([category, ident])
                states.append([category, ident, count, int(count>1), type_code])
            ordered_ids.append([[clip, person, t0], ids])
            ordered_state.append([[clip, person, t0], states])
        per_clip[clip] = ctr
    total = new_counts()
    for stat in per_clip.values():
        for field, counts in stat.items():
            if isinstance(counts, Counter):
                total[field].update(counts)
            elif field == 'maximum_pool_size':
                total[field] = max(total[field], counts)
            else:
                total[field] += counts
    for stat in [*per_clip.values(), total]:
        for g in ('support', 'pool_support'):
            for name in ('empty', 'people_only', 'cars_only', 'mixed'):
                stat[g].setdefault(name, 0)
        selected = sum(stat['selected'].values())
        assert selected == sum(stat['velocity_missing'].values()) + sum(stat['velocity_valid'].values())
        assert selected == sum(stat['neighbor_type_codes'].values()) == sum(stat['presence_frames_1_to_5'].values())
        assert sum(stat['pool'].values()) == selected + sum(stat['removed_by_limit'].values())
        assert sum(stat['support'].values()) == sum(stat['pool_support'].values()) == stat['windows']
    swap_total = Counter()
    for subset in swap_by_clip.values():
        swap_total.update(subset)
    assert swap_total['swapped_windows'] == 810 and swap_total['unchanged_windows'] == 2256
    assert swap_total['revised_windows_with_car'] == swap_total['radius_pool_windows_with_car'] == 1202
    assert swap_total['original_windows_with_car'] == 392
    assert sum(swap_total[k] for k in ('revised_person_edges','revised_car_edges')) == sum(total['selected'].values())
    intervention = {'addendum_sha256':sha((Path(__file__).parent/'VEHICLE_SLOT_ADDENDUM.md').read_bytes()),
                    'all_slots_unchanged_on_ineligible_windows':True,'first_11_unchanged_on_all_replacements':True,
                    'changed_clips':sum(x['swapped_windows']>0 for x in swap_by_clip.values()),
                    'overall':dict(swap_total),'per_clip':{k:dict(v) for k,v in swap_by_clip.items()},
                    'ordered_replacement_id_schema':'Compact JSON ordered five-element lists [clip,numeric_target_id,t0,numeric_new_car_id,numeric_old_person_id] for replaced windows only.',
                    'ordered_replacement_ids_sha256':sha(canonical(swaps)),
                    'ordered_revised_memberships_sha256':sha(canonical(revised_members))}
    return {'vehicle_slot_intervention':intervention,'cohort_sha256': sha(canonical(keys)), 'protocol_sha256': sha((Path(__file__).parent / 'PROTOCOL.md').read_bytes()),
            'counts_unit':'An edge is one window/neighbor at the final history frame. An actor may be counted more than once across windows. Membership is selected before querying any future observation. Distinct local IDs include source class.',
            'ordered_memberships_sha256': sha(canonical(ordered_ids)),
            'ordered_members_and_history_sha256': sha(canonical(ordered_state)),
            'digest_schema':'Canonical compact JSON outer list in the enrolled key order. Membership each [key, [[class,numeric_id],...]]. State each [key, [[class,numeric_id,presence_count,velocity_valid_int,type_code],...]].',
            'per_clip':per_clip, 'overall':total}


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('dut_private_dir',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    report=inspect(args.dut_private_dir)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report['overall'],indent=1))
