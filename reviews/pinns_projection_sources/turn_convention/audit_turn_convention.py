"""Confirm the five frozen PINNS event bits before and after reflecting ground Y. No model calls."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ARCHIVE_SHA = 'caa8f0959f28f2d69a6f16cc9df115ab6222d0b6eb3297edd575615fef81b94c'
ROSTER_SHA = 'f5ee511a54ad818499d3dde6213f7f3b49332943ff2fc93d93aed87d316dbb91'
ANALYSIS_FIELD_SHA = '96dd49a25b07a6f6a1064218b107341186a7a821b9b0321ad88514eff696474c'
ANALYSIS_FIELDS = ('clip','person_id','t0','site','history_20_xy','history_20_mask','future_30_xy')
RULE_SHA = '4b1300c579377a6013355fb6920779ef56972781dd1b548a19f9f8f940af7646'
EVENTS = ('start','stop','positive_heading_change','negative_heading_change','maintain')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(ok, detail):
    if not ok:
        raise ValueError(detail)


def bit_counts(values):
    return {name:int(n) for name,n in zip(EVENTS,values.sum(axis=0))}


def map_bits(row):
    return row[:,[0,1,3,2,4]]


def first_hits(track):
    """Independently find first trigger of either turn sign using complete source ticks 0..34."""
    velocity = np.diff(track, axis=0) / .1
    smoothed = np.vstack([velocity[max(0,i-2):i+1].mean(axis=0) for i in range(len(velocity))])
    speeds = np.hypot(*smoothed.T)
    # A velocity at index i uses the sample at tick i+1. The turn rule starts at tick 2.
    moving = np.flatnonzero(speeds[1:] >= .5) + 1
    found = [None, None]
    for block in np.split(moving, np.flatnonzero(np.diff(moving) != 1)+1):
        if len(block)<3:
            continue
        theta = np.unwrap(np.arctan2(smoothed[block,1], smoothed[block,0]))
        future = block >= 4
        relative = (theta-theta[0])[future]
        if len(relative)<3:
            continue
        for sign in (0,1):
            hit = relative >= np.pi/6 if sign==0 else relative <= -np.pi/6
            starts = np.flatnonzero(hit[:-2] & hit[1:-1] & hit[2:])
            if len(starts):
                tick = int(block[future][starts[0]] + 1)
                if found[sign] is None or tick < found[sign]:
                    found[sign] = tick
    return tuple(found)


def read_raw(metadata, clip):
    cal = json.loads((metadata/'videos'/clip.rsplit('_',1)[0]/'calib.json').read_bytes())
    require(cal['u_axis'] in ('x','+x') and cal['v_axis']=='-y', 'Unexpected native raster convention')
    positions = {}
    p = metadata/'labels'/(clip+'.txt')
    for line in p.read_text().splitlines():
        f = line.split()
        if len(f)<11 or f[1].lower()!='person':
            continue
        raw = int(f[0])-1
        if raw<0 or raw%3:
            continue
        key = int(f[2]), raw//3
        require(key not in positions, 'Duplicate source person-frame')
        positions[key] = (float(f[8]),float(f[9]))
    return positions


def first_selected(pair):
    a,b=pair
    if a is None and b is None: return (0,0)
    if b is None or (a is not None and a<=b): return (1,0)
    return (0,1)


def audit(repo, metadata, npz):
    container_sha = sha(npz.read_bytes())
    require(sha((repo/'src/data/actions.py').read_bytes())==RULE_SHA, 'Frozen event rules')
    sys.path.insert(0,str(repo.resolve()))
    from src.data.actions import compute_action_labels
    with np.load(npz, allow_pickle=False) as z:
        digest = hashlib.sha256()
        for name in ANALYSIS_FIELDS:
            array = z[name]
            description = name.encode() + b'\0' + array.dtype.str.encode() + b'\0' + json.dumps(array.shape).encode() + b'\0'
            digest.update(description)
            digest.update(np.ascontiguousarray(array).tobytes())
        require(digest.hexdigest() == ANALYSIS_FIELD_SHA, 'Frozen analysis fields differ')
        keys=list(zip(z['clip'].tolist(),z['person_id'].tolist(),z['t0'].tolist()))
        require(len(keys)==len(set(keys))==1482 and sha(json.dumps(keys,separators=(',',':')).encode())==ROSTER_SHA,'Original cohort')
        require(z['history_20_mask'][:,-5:].all(),'Missing history')
        tracks=np.concatenate((z['history_20_xy'][:,-5:],z['future_30_xy']),axis=1)
        sites=z['site']
    require(tracks.shape==(1482,35,2) and np.isfinite(tracks).all(), 'Incomplete track')
    buckets=defaultdict(list)
    for j,(clip,p,t0) in enumerate(keys):buckets[clip].append((j,p,t0))
    conditions={clip.rsplit('_',1)[0] for clip in buckets}
    require(len(buckets)==47 and len(conditions)==22,'Source coverage')
    for clip, rows in buckets.items():
        people=read_raw(metadata,clip)
        for i,person,end in rows:
            obs=np.array([people[person,t] for t in range(end-4,end+31)])
            # Proves that the stored displacements in this cohort use raw annotation XY in every clip.
            np.testing.assert_array_equal(obs-obs[4],tracks[i])

    def label(track):
        y,m=compute_action_labels(track[:5],np.ones(5),track[5:],np.ones(30))
        require(m==1, 'Action validity')
        return y.astype(bool)

    bits=[]; chronology=[]
    for track in tracks:
        variants=(track,track*np.array([1.,-1.]))
        pair=[label(t) for t in variants]
        when=[first_hits(t) for t in variants]
        for vec,times in zip(pair,when):
            require(tuple(vec[2:4])==first_selected(times),'Earliest-event rule disagrees with frozen implementation')
        bits.append(pair);chronology.append(when)
    raw,reflected=np.asarray(bits).transpose(1,0,2)
    def summarize(idx):
        original=raw[idx];mirror=reflected[idx];timeline=[chronology[i] for i in np.flatnonzero(idx)]
        both=[a for a,b in timeline if None not in a]
        return {'windows':len(original),'raw':bit_counts(original),'declared_bev':bit_counts(mirror),
                'unchanged_entire_vector':int((original==mirror).all(1).sum()),
                'changed_entire_vector':int((original!=mirror).any(1).sum()),
                'mismatched_after_swapping_turn_columns':int((map_bits(original)!=mirror).any(1).sum()),
                'start_stop_maintain_changed':int((original[:,[0,1,4]]!=mirror[:,[0,1,4]]).any(1).sum()),
                'multiple_labels':int((original.sum(1)>1).sum()),'both_signed_turn_thresholds_sustained':len(both),
                'same_earliest_tick_for_two_signs':sum(a==b for a,b in both),
                'raw_positive_first_when_both':sum(a<b for a,b in both),'raw_negative_first_when_both':sum(a>b for a,b in both),
                'first_sustained_tick_of_either_sign_fails_exchange':sum((a[0],a[1])!=(b[1],b[0]) for a,b in timeline)}
    out={'pooled':summarize(np.ones(len(sites),bool)),
         'by_site':{s:summarize(sites==s) for s in sorted(set(sites))}}
    expected=json.loads((repo/'results/pinns_heldout_implied_actions.json').read_bytes())
    for s in ['pooled',*sorted(set(sites))]:
        head=out['pooled'] if s=='pooled' else out['by_site'][s]
        ref=expected['pooled'] if s=='pooled' else expected['sites'][s]
        require(list(head['raw'].values())==[ref['classes'][a]['positive_windows'] for a in ['start','stop','left','right','maintain']], 'Published source ground-truth mismatch')
    return {'scope':'All 1,482 original complete five-observation, 30-future PINNS trajectories. Reflection applied to every relative Y coordinate with X unchanged; a per-scene constant translation cancels in the anchored tracks.',
            'original_cache_sha256':ARCHIVE_SHA,'input_cache_sha256':container_sha,'analysis_field_sha256':ANALYSIS_FIELD_SHA,
            'original_zip_bytes_matched':container_sha==ARCHIVE_SHA,'ordered_keys_sha256':ROSTER_SHA,'action_rules_sha256':RULE_SHA,
            'verifier_sha256':sha(Path(__file__).read_bytes()),'numpy':np.__version__,'conditions_with_declared_u_x_v_minus_y':len(conditions),
            'clips_source_relative_trajectory_verified':len(buckets),'original_positions_replayed':int(tracks.shape[0]*tracks.shape[1]),
            'interpretation':'The original left/right fields index positive/negative signed changes in the stored metric XY axes. In declared BEV axes +X right/+Y up the signed classes exchange. This is a planar coordinate convention audit; the records do not independently validate anatomical or physical turn labels, image mirroring or camera heading.',
            'earliest_rule':'Three sustained future samples to reach each sign. Preserve earliest eligible tick across moving segments; the frozen code favors its positive sign on an exact tie. Both trigger times checked even when only the first is emitted.',
            'results':out}


if __name__=='__main__':
    arg=argparse.ArgumentParser(description=__doc__)
    for name in ('repo','metadata','histories','out'):
        arg.add_argument('--'+name,type=Path,required=True)
    a=arg.parse_args()
    result=audit(a.repo,a.metadata,a.histories)
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result['results']['pooled']))
