"""Independent aggregation of an existing Kaggle forecast-error tensor. No torch or model calls."""
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile
import argparse
import io
import json
import math

import numpy as np

HERE = Path(__file__).resolve().parent
FIELDS = ('ade_m','fde_m')
ARMS = ('M0','M1','last_two_velocity')


def digest(data):
    return sha256(data).hexdigest()


def key_digest(keys):
    return digest(json.dumps(keys, separators=(',',':')).encode())


def vector(values):
    return dict(zip(FIELDS,np.asarray(values).tolist()))


def verify_report(reference, histories, bundle, metadata):
    design=json.loads((HERE/'history_support.json').read_bytes())
    assert digest((HERE/'PROTOCOL.md').read_bytes()) == design['protocol_sha256']
    target=json.loads(reference.read_bytes())
    assert design=={k:target[k] for k in design}
    archived=histories
    assert digest(archived.read_bytes())=='caa8f0959f28f2d69a6f16cc9df115ab6222d0b6eb3297edd575615fef81b94c'
    assert digest(bundle.read_bytes())=='b24ac2bb1ad3d6742be2cbe16b927e42ef9a1a54a17ce2c08be3e92fa286b871'

    # Deliberately load historical fields only, and fix the partition before opening the score bundle.
    with np.load(archived,allow_pickle=False) as z:
        clips,people,end,sites=(z[k] for k in ('clip','person_id','t0','site'))
        keys=list(zip(clips.tolist(),people.tolist(),end.tolist()))
        observed=z['history_20_xy'][:,-5:]
        assert z['history_20_mask'][:,-5:].all() and np.isfinite(observed).all()
        first=(end-4)==z['first_person_label_t']
        earliest=z['first_person_label_t']
    assert len(set(keys))==len(keys)==1482 and key_digest(keys)==design['design']['ordered_keys_sha256']
    # The compressed history is only 20 frames; independently verify 35 contiguous PRE-window samples in native labels.
    for clip in set(clips):
        times=defaultdict(set)
        for line in (metadata/'labels'/(clip+'.txt')).read_text(errors='ignore').splitlines():
            v=line.split()
            if len(v)<11 or not v[1].lower().startswith('per'):continue
            raw_id=int(float(v[0]))-1
            if raw_id>=0 and raw_id%3==0:times[int(float(v[2]))].add(raw_id//3)
        for i in np.flatnonzero(clips==clip):
            start=int(end[i])-4;track=times[int(people[i])]
            assert min(track)==earliest[i] and set(range(start,start+5))<=track
            assert first[i]==(min(track)==start)
            if not first[i]:assert set(range(start-35,start))<=track
    path=np.array([math.fsum(math.dist(p,q) for p,q in zip(w[:-1],w[1:])) for w in observed])
    net=np.array([math.dist(w[-1],w[0]) for w in observed])
    masks={a+'_'+b:ann&dist for a,ann in [('first',first),('prior',~first)] for b,dist in [('short',path<.5),('long',path>=.5)]}
    assert np.all(sum(mask.astype(int) for mask in masks.values())==1)
    assert sum(first)==449 and masks['first_short'].sum()==323
    checks=0;worst=0.
    def close(actual,want):
        nonlocal checks,worst
        actual=np.asarray(actual);want=np.asarray(want)
        np.testing.assert_allclose(actual,want,rtol=0,atol=2e-12)
        checks+=actual.size;worst=max(worst,float(abs(actual-want).max()))
    def compare_metrics(actual,want):close(actual,[want[k] for k in FIELDS])
    for group,mask in masks.items():
        expected=design['design']['history_only_groups'][group]
        subset=[key for key,keep in zip(keys,mask) if keep]
        assert expected==dict(windows=len(subset),distinct_clips=len({k[0] for k in subset}),
                              clip_persons=len({k[:2] for k in subset}),ordered_keys_sha256=key_digest(subset),
                              by_site={s:int((mask&(sites==s)).sum()) for s in sorted(set(sites))})
    for s,r in design['design']['site_distances'].items():
        use=np.ones(len(keys),bool) if s=='all' else sites==s
        for name,vals in [('cumulative',path),('net',net)]:
            assert (vals[use]<.5).sum()==r[name]['below_half_metre']
            close(np.quantile(vals[use],[.25,.5,.75]),r[name]['q25_median_q75_m'])
    with ZipFile(bundle) as z:
        assert z.testzip() is None
        raw=z.read('reranked_paired_windows.npz');cloud=json.loads(z.read('reranked_mode_scores.json'))
        assert digest(raw)==target['window_errors_sha256']==cloud['paired_windows_sha256']
        with np.load(io.BytesIO(raw),allow_pickle=False) as d:
            assert keys==list(zip(d['clip'].tolist(),d['person_id'].tolist(),d['t0'].tolist()))
            seeds=d['seeds'].tolist();models=d['models'].tolist();methods=d['methods'].tolist()
            assert seeds==[42,1024,2025,3407,4096] and models==['M0','M1'] and d['metrics'].tolist()==list(FIELDS)
            all_errors=d['errors']
            assert all_errors.shape==(5,2,1482,4,2) and np.isfinite(all_errors).all()
            assert d['index_columns'].tolist()[0]=='network_top1'
            top=methods.index('network_top1');cv=methods.index('last_two_velocity')
            chosen=d['mode_indices'][:,:,:,0];assert ((chosen>=0)&(chosen<6)).all()
            direct=all_errors[:,:,:,cv,:]
            assert np.all(direct==direct[0,0][None,None])
            values=np.stack((all_errors[:,0,:,top],all_errors[:,1,:,top],direct[:,0]),axis=2) # seed,window,arm,metric
    def average(arr):return np.asarray(arr,dtype=float).mean(axis=0)
    all_result={};by_site_sums=defaultdict(lambda:np.zeros((5,3,2)))
    for group,mask in masks.items():
        ref=target['groups'][group]
        site_rows={};means=[];ensemble_boot=[];rng=np.random.Generator(np.random.PCG64(2026))
        for s in sorted(set(sites)):
            idx=np.flatnonzero(mask&(sites==s));src=ref['sites'][s];sites_clips=clips[idx];uniq=np.unique(sites_clips)
            assert len(idx)==src['windows'] and len(uniq)==src['clips']
            tmp=values[:,idx].mean(1)
            by_site_sums[s]+=values[:,idx].sum(1)
            means.append(tmp)
            for j,arm in enumerate(ARMS):compare_metrics(tmp[:,j].mean(0),src['scores'][arm])
            compare_metrics((tmp[:,1]-tmp[:,0]).mean(0),src['M1_minus_M0'])
            delta=(values[:,idx,1]-values[:,idx,0]).mean(0)
            # Count multiplicities per clip in each sampled replicate rather than indexing window error vectors.
            selected=rng.integers(len(uniq),size=(20000,len(uniq)))
            freq=np.stack([(selected==i).sum(1) for i in range(len(uniq))],axis=1)
            sums=np.stack([delta[sites_clips==c].sum(0) for c in uniq])
            size=np.array([(sites_clips==c).sum() for c in uniq])
            ensemble_boot.append((freq@sums)/(freq@size)[:,None])
            site_rows[s]=dict(windows=len(idx),clips=len(uniq),seed_means_by_method={str(seed):{arm:vector(row[j]) for j,arm in enumerate(ARMS)} for seed,row in zip(seeds,tmp)})
        mean=average(means);diff=mean[:,1]-mean[:,0]
        for j,arm in enumerate(ARMS):compare_metrics(mean[:,j].mean(0),ref['supported_site_equal'][arm])
        for seed,delta in zip(seeds,diff):compare_metrics(delta,ref['seed_M1_minus_M0'][str(seed)])
        compare_metrics(diff.mean(0),ref['M1_minus_M0']);close(diff.mean(0)[0],ref['four_site_ade_difference_m'])
        ci=np.quantile(average(ensemble_boot),[.025,.975],axis=0).T
        compare_metrics(ci,ref['conditional_paired_clip_95'])
        all_result[group]=dict(windows=int(mask.sum()),sites=site_rows,four_site_five_seed_means={arm:vector(mean[:,j].mean(0)) for j,arm in enumerate(ARMS)},
                               per_seed_difference={str(seed):vector(row) for seed,row in zip(seeds,diff)},
                               four_site_m1_minus_m0_m=vector(diff.mean(0)),paired_clip_95_m=vector(ci))
    for s,arr in by_site_sums.items():
        n=cloud['site_window_counts'][s]
        for i,seed in enumerate(seeds):
            for j,arm in enumerate(ARMS):
                model=arm if j<2 else 'M0';method='network_top1' if j<2 else 'last_two_velocity'
                compare_metrics(arr[i,j]/n,cloud['by_seed_site'][str(seed)][s][model]['methods'][method])
    return dict(scope='Existing Kaggle v373583293 network_top1 and fixed-CV saved ADE/FDE, no inference, refitting or selection. Secondary site-by-seed outputs are descriptive.',original_archive_sha256=digest(bundle.read_bytes()),
                history_sha256=digest(archived.read_bytes()),production_aggregation_sha256=digest(reference.read_bytes()),
                paired_window_error_sha256=digest(raw),source_label_calibration_sha256=design['source_labels_and_calibration_sha256'],
                verifier_sha256=digest(Path(__file__).read_bytes()),compiled_against_protocol=design['protocol_sha256'],
                history_only_windows=1482,independently_verified_prior_35_sample_windows=int((~first).sum()),history_ties_exactly_half_metre=int((path==.5).sum()),
                group_rosters_match_registration=True,scalar_summary_comparisons=checks,max_abs_numeric_difference=worst,
                groups=all_result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('histories', type=Path, help='Original private 1,482-window history census NPZ')
    parser.add_argument('bundle', type=Path, help='Original Kaggle reranked_modes.zip')
    parser.add_argument('metadata', type=Path, help='Staged official label/calibration root')
    parser.add_argument('output', type=Path, help='Write aggregate-only independent replay JSON here')
    parser.add_argument('--reference', type=Path, default=HERE/'short_displacement_scores.json')
    args = parser.parse_args()
    result = verify_report(args.reference, args.histories, args.bundle, args.metadata)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('Verified', result['scalar_summary_comparisons'], 'summary scalars; max absolute difference',
          result['max_abs_numeric_difference'])
