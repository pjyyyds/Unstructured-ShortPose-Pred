"""Verify multi-seed archives independently; pool paired within-seed contrasts in fixed T5 quality groups."""
import argparse,hashlib,io,json,sys,zipfile
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from reviews.pinns_matched_multiseed.replay_helpers import (
    PROTOCOL, forward, id_hash, ids, inputs, measure, sha, verify_sources,
)
from scripts.verify_heldout_bundle import parameter_digest

SEEDS=(42,1024,2025,3407,4096)
TS=(5,10,20)
ARMS=('M0','M1','M1-concat','M1b')
DRIVER='883a0d4fdf3207969a52de59aa8af957b6d1dc083d1808b0a44f552bae3ed63a'
QUAL=('clear','partial','heavy')
FIELDS=('ade_m','fde_m')
HERE=Path(__file__).resolve().parent


def json_digest(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def site_metric(x):
    return dict(zip(FIELDS,x.mean(0).tolist()))


def load_42(work, cache, keys, groups, features):
    """Replay the original five downloaded seed-42 archives, without a private intermediate NPZ."""
    published = json.loads((ROOT/'results/pinns_matched_same_worker_t20.json').read_text())
    combined = (work/'matched_same_worker_T20.zip').read_bytes()
    if sha(combined) != published['source']['bundle_sha256']:
        raise ValueError('Wrong same-worker T20 seed-42 archive')
    with zipfile.ZipFile(io.BytesIO(combined)) as outer:
        assert outer.testzip() is None
        embedded = {name: outer.read(name) for name in ('cpu_T20.zip', 'matched_fusion_cpu_T20.zip')}
        for name, data in embedded.items():
            assert sha(data) == published['source']['inner_sha256'][name]
    rows, provenance = {}, {}
    for t in TS:
        for family, arms, name in [('base', ('M0','M1'), f'cpu_T{t}.zip'),
                                   ('fusion', ('M1b','M1-concat'), f'matched_fusion_cpu_T{t}.zip')]:
            data = embedded[name] if t == 20 else (work/name).read_bytes()
            if t != 20:
                assert sha(data) == published['previous_runs'][str(t)][family]['archive_sha256']
            provenance[name] = sha(data)
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                manifest = json.loads(z.read('sha256.json'))
                assert z.testzip() is None and set(z.namelist()) == set(manifest)|{'sha256.json'}
                assert len(z.namelist()) == len(set(z.namelist()))
                assert all(sha(z.read(k)) == digest for k,digest in manifest.items())
                fold = json.loads(z.read('fold_ids.json'))
                for site,idx in groups.items():
                    train_sites = sorted(set(groups)-{site})
                    train_ids = [keys[i] for s in train_sites for i in groups[s]]
                    test_ids = [keys[i] for i in idx]
                    suffix = 'ids' if family == 'base' else 'keys'
                    assert fold[site]['train_sites'] == train_sites
                    assert list(map(tuple,fold[site]['train_'+suffix])) == train_ids
                    assert list(map(tuple,fold[site]['test_'+suffix])) == test_ids
                    for arm in arms:
                        filename = (f'T{t}/' if family=='base' else '') + f'{site}/{arm}.pt'
                        ck = torch.load(io.BytesIO(z.read(filename)), map_location='cpu', weights_only=True)
                        meta = ck['identity']
                        assert meta['protocol']==PROTOCOL and meta['duration']==t
                        assert meta['train_ids_sha256']==id_hash(train_ids) and meta['test_ids_sha256']==id_hash(test_ids)
                        assert meta.get('site',meta.get('held_site'))==site and meta.get('model',meta.get('variant'))==arm
                        paths, probs = forward(arm,ck['model_state'],features[t],idx)
                        truth = cache['xy'][idx,20:].astype(np.float32)
                        errors = measure(paths[np.arange(len(idx)),probs.argmax(1)],truth)
                        if family=='fusion':
                            pred = ck['predictions']
                            np.testing.assert_allclose(paths,pred['trajs'].numpy(),atol=.0002,rtol=.00002)
                            np.testing.assert_allclose(probs,pred['probs'].numpy(),atol=.00002,rtol=.00002)
                            saved = pred['errors'].numpy()
                        else:
                            saved = ck['errors'].numpy()
                        np.testing.assert_allclose(errors,saved,atol=.0002,rtol=.00002)
                        rows.setdefault((t,arm),np.full((len(keys),2),np.nan))[idx] = saved
    old = json.loads((ROOT/'results/pinns_matched_cpu_fusion.json').read_text())
    for (t,arm),errors in rows.items():
        ref = published['four_site_macro'][arm] if t==20 else old['durations'][str(t)]['four_site_macro_m'][arm]
        macro=np.stack([errors[i].mean(0) for i in groups.values()]).mean(0)
        np.testing.assert_allclose(macro,[ref[k] for k in FIELDS],atol=1e-10,rtol=0)
    return rows,provenance


def read_package(p,cache,features,keys,groups):
    with zipfile.ZipFile(p) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
        manifest=json.loads(z.read('sha256.json'));r=json.loads(z.read('report.json'));backend=json.loads(z.read('backend.json'))
        seed=r['seed'];lengths=sorted(r['lengths']); src=r['source_identity']
        design=json.loads((HERE/'real_check_only.json').read_text())
        assert seed in SEEDS and set(lengths)<=set(TS) and len(lengths)==len(set(lengths))
        expected={f'T{t}/{s}/{a}.pt' for s in groups for t in lengths for a in ARMS}
        expected|={f'truth/{s}.npz' for s in groups}|{'backend.json','driver.py','report.json','folds.json'}
        assert set(manifest)==expected and set(z.namelist())==expected|{'sha256.json'}
        for f,h in manifest.items():assert sha(z.read(f))==h,f
        assert r['driver_sha256']==manifest['driver.py']==DRIVER and r['source_identity']==design['source_identity']
        assert src['ordered_keys_sha256']==id_hash(keys)
        assert r['fixed_seed_set']==list(SEEDS) and r['families']==list(ARMS) and r['protocol']==dict(PROTOCOL,seed=seed)
        assert r['window_count']==len(keys) and r['checkpoints']==len(lengths)*16 and r['site_counts']=={s:len(i) for s,i in groups.items()}
        assert r['processor_controls']==design['processor_controls'] and r['pose_frames']==12460 and r['clips']==47
        assert r['backend_sha256']==json_digest(backend) and backend['deterministic'] and backend['device']=='cpu' and backend['torch']=='2.11.0+cpu'
        fold=json.loads(z.read('folds.json'));assert json_digest(fold)==r['folds_sha256']
        vals={(t,a):np.full((len(keys),2),np.nan) for t in lengths for a in ARMS}; maximum={k:0. for k in ('trajs','probs','errors')}
        for s,idx in groups.items():
            others=sorted(set(groups)-{s});train=np.concatenate([groups[o] for o in others])
            assert fold[s]['train_sites']==others and list(map(tuple,fold[s]['train_keys']))==[keys[i] for i in train]
            assert list(map(tuple,fold[s]['test_keys']))==[keys[i] for i in idx]
            with np.load(io.BytesIO(z.read(f'truth/{s}.npz')),allow_pickle=False) as y:
                assert ids(y)==[keys[i] for i in idx] and y['fut_mask'].all()
                np.testing.assert_array_equal(y['fut_xy'],cache['xy'][idx,20:].astype(np.float32)); np.testing.assert_array_equal(y['history_xy_20'],cache['xy'][idx,:20]);truth=y['fut_xy']
            for t in lengths:
                for a in ARMS:
                    ck=torch.load(io.BytesIO(z.read(f'T{t}/{s}/{a}.pt')),map_location='cpu',weights_only=True)
                    k=ck['identity']
                    expected={'experiment':r['experiment'],'seed':seed,'duration':t,'heldout_site':s,'model':a,
                              'protocol':dict(PROTOCOL,seed=seed),'train_keys_sha256':id_hash([keys[i] for i in train]),
                              'test_keys_sha256':id_hash([keys[i] for i in idx]),'sources_sha256':json_digest(src),
                              'driver_sha256':DRIVER,'backend_sha256':json_digest(backend)}
                    assert k==expected and parameter_digest(ck['model_state'])==ck['parameter_sha256']
                    assert ck['fit_and_collect_seconds']>0 and ck['fit_and_collect_seconds']==r['results'][str(t)]['fit_and_collect_seconds'][s][a]
                    pred=ck['predictions'];paths,probs=forward(a,ck['model_state'],features[t],idx)
                    errors=measure(paths[np.arange(len(idx)),probs.argmax(1)],truth)
                    for field,array,tol in [('trajs',paths,.0002),('probs',probs,.00002),('errors',errors,.0002)]:
                        orig=pred[field].numpy()
                        np.testing.assert_allclose(array,orig,atol=tol,rtol=.00002)
                        maximum[field]=max(maximum[field],float(np.max(abs(orig-array))))
                    vals[t,a][idx]=pred['errors'].numpy()
                    reported=r['results'][str(t)]['site_means'][s][a]
                    np.testing.assert_allclose(vals[t,a][idx].mean(0),[reported[m] for m in FIELDS],atol=1e-10,rtol=0)
                for a in ('M0','M1b','M1-concat'):
                    v=r['results'][str(t)]['paired_within_site'][s]['M1_minus_'+a]
                    np.testing.assert_allclose((vals[t,'M1'][idx]-vals[t,a][idx]).mean(0),[v[m] for m in FIELDS],atol=1e-10,rtol=0)
        for (t,a),e in vals.items():
            assert np.isfinite(e).all()
            reported=r['results'][str(t)]['four_site_macro'][a]
            np.testing.assert_allclose(np.mean([e[i].mean(0) for i in groups.values()],0),[reported[m] for m in FIELDS],atol=1e-10,rtol=0)
        return seed,vals,{'archive_sha256':sha(p.read_bytes()),'durations':lengths,'cpu_model':backend['cpu_model'],
                          'worker_fingerprint':backend['worker_fingerprint'],'max_local_vs_cloud_abs':maximum,'torch':backend['torch']}


def registered_primary(by, groups, clips):
    """Five fixed paired fits, then whole-clip resampling within each fixed site."""
    if set(by) != set(SEEDS) or any((5, 'M1') not in rows for rows in by.values()):
        return None
    per_seed = {str(seed): by[seed][5, 'M1'][:, 0] - by[seed][5, 'M0'][:, 0] for seed in SEEDS}
    means = {seed: float(np.mean([values[idx].mean() for idx in groups.values()]))
             for seed, values in per_seed.items()}
    shared = np.stack(list(per_seed.values())).mean(0)
    random = np.random.default_rng(2026)
    draws = np.zeros(20000)
    for idx in groups.values():
        names, inverse = np.unique(clips[idx], return_inverse=True)
        count = len(names)
        weights = random.multinomial(count, np.full(count, 1/count), size=len(draws))
        clip_totals = np.bincount(inverse, weights=shared[idx], minlength=count)
        clip_sizes = np.bincount(inverse, minlength=count)
        draws += (weights @ clip_totals) / (weights @ clip_sizes) / len(groups)
    return {
        'endpoint': 'T5 M1 minus M0 top-probability ADE (m)',
        'seed_differences_m': means, 'seed_range_m': [min(means.values()), max(means.values())],
        'four_site_five_seed_mean_m': float(np.mean(list(means.values()))),
        'by_site_m': {site: float(shared[idx].mean()) for site,idx in groups.items()},
        'paired_clip_percentile_95_m': np.percentile(draws, [2.5,97.5]).tolist(),
        'scope': '20,000 whole-clip draws within the four fixed sites, RNG seed 2026. Five fitted seeds fixed.'
    }


def pool(work,packs):
    torch.set_num_threads(2)
    with zipfile.ZipFile(work/'cpu_T5.zip') as z:
        cache,groups,keys=verify_sources(z.read('matched_observations.npz'),work/'pinns_matched_histories.npz')
    feats={};bins=None
    for t in TS:
        feats[t],b=inputs(cache,t)
        if t==5:bins=b
    assert np.bincount(bins).tolist()==[188,283,152]
    by={};provenance={}
    by[42],provenance['existing_42']=load_42(work,cache,keys,groups,feats)
    for package in packs:
        seed,vals,record=read_package(package,cache,feats,keys,groups)
        assert seed!=42,'Seed 42 already contributed; explicit replacement policy required for any subsequent repeat'
        previous=by.setdefault(seed,{})
        assert not set(previous)&set(vals),'Duplicate seed, duration and arm'
        previous.update(vals);provenance.setdefault(str(seed),[]).append(record)
    clips=cache['clip']
    slices={'overall':np.ones(len(keys),bool),**{name:bins==i for i,name in enumerate(QUAL)}}
    summary={}
    for q,mask in slices.items():
        siteids={s:idx[mask[idx]] for s,idx in groups.items()}
        summary[q]={'windows':int(mask.sum()),'site_support':{s:{'windows':len(i),'clips':len(set(clips[i]))} for s,i in siteids.items()},'durations':{}}
        assert all(len(i)>0 for i in siteids.values())
        for t in TS:
            paired={};perseed={};means={};model_summary={}
            for seed,rows in by.items():
                if (t,'M1') not in rows:continue
                perseed[str(seed)]={};means[str(seed)]={}
                for a in ARMS:
                    means[str(seed)][a]=site_metric(np.stack([rows[t,a][i].mean(0) for i in siteids.values()]))
                for a in (('M0','M1b','M1-concat') if q=='overall' else ('M0','M1b')):
                    v=rows[t,'M1']-rows[t,a]
                    perseed[str(seed)]['M1_minus_'+a]=site_metric(np.stack([v[i].mean(0) for i in siteids.values()]))
            if set(m for m in map(int,perseed))==set(SEEDS):
                for a in ARMS:
                    values=np.array([[means[str(seed)][a][m] for m in FIELDS] for seed in SEEDS])
                    model_summary[a]={'equal_seed_mean':dict(zip(FIELDS,values.mean(0).tolist())),'sample_sd_across_seed_means':dict(zip(FIELDS,values.std(0,ddof=1).tolist()))}
                for c in perseed['42']:
                    vals=np.array([[perseed[str(seed)][c][m] for m in FIELDS] for seed in SEEDS])
                    paired[c]={'equal_seed_mean':dict(zip(FIELDS,vals.mean(0).tolist())),'sample_sd_across_seed_differences':dict(zip(FIELDS,vals.std(0,ddof=1).tolist()))}
            summary[q]['durations'][str(t)]={'seed_means':means,'paired_by_seed':perseed,'five_seed_model_summary':model_summary or None,'five_seed_paired_summary':paired or None}
    return {'scope':'Equal site and within-site window weights. All duration models use one fixed T5 operational quality group per window; this describes joint confidence/scale rather than verified physical occlusion. Paired seed differences; old seed 42 counted once with the same-worker 20-frame rerun. Seed SD reports variability of the five specified fitted models.',
            'ingested_training_seeds':sorted(by),'source':provenance,
            'registered_primary':registered_primary(by,groups,clips),
            'registered_secondary_sha256':sha((HERE/'quality_groups_plan.json').read_bytes()),'groups':summary}

if __name__=='__main__':
    if not __debug__:
        raise RuntimeError('Verification requires Python without -O.')
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('work',type=Path);p.add_argument('output',type=Path);p.add_argument('bundles',nargs='*',type=Path)
    args=p.parse_args();result=pool(args.work,args.bundles);args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(result['ingested_training_seeds'],{k:v['windows'] for k,v in result['groups'].items()})
