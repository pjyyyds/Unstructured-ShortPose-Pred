"""Verify multi-seed archives independently; pool paired within-seed contrasts in fixed T5 quality groups."""
import argparse,hashlib,io,json,sys,zipfile
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from verify_matched_cpu_fusion import forward
from verify_matched_duration_weights import verify_sources, inputs, ids, id_hash, sha, measure, PROTOCOL
from scripts.verify_heldout_bundle import parameter_digest
from scripts.run_heldout_seed_sensitivity import json_digest

SEEDS=(42,1024,2025,3407,4096)
TS=(5,10,20)
ARMS=('M0','M1','M1-concat','M1b')
DRIVER='883a0d4fdf3207969a52de59aa8af957b6d1dc083d1808b0a44f552bae3ed63a'
QUAL=('clear','partial','heavy')
FIELDS=('ade_m','fde_m')
HERE=Path(__file__).resolve().parent


def site_metric(x):
    return dict(zip(FIELDS,x.mean(0).tolist()))


def load_42(work,cache,keys,groups):
    result={}; provenance={}
    file_map={'mixed_5_10':work/'matched_cpu_fusion_replayed.npz','same_worker_T20':work/'matched_same_worker_T20_replayed.npz'}
    # These are the independent local forward products of the accepted seed-42 packs. Check all
    # candidates against their truth, and all site-equal means against the reviewed reports.
    ref_old=json.loads((work/'matched_cpu_three_durations_verified.json').read_text())
    ref_20=json.loads((work/'matched_same_worker_T20_verified.json').read_text())
    fusion_reference={}
    for t,expected_sha in [(5,'cc5837725ac83009ec6d5256f2df76133e16f9e3cc6f6a573c9c5f55383233ae'),
                           (10,'35fafe3b55e8053f120b364a3c8fb798cb423a1871d9deec00fe397f32a2fff8')]:
        pack=work/f'matched_fusion_cpu_T{t}.zip'
        assert sha(pack.read_bytes())==expected_sha
        with zipfile.ZipFile(pack) as z:
            fusion_reference[t]=json.loads(z.read('report.json'))['four_site_equal']
    for key,path in file_map.items():
        with np.load(path,allow_pickle=False) as z:
            old=ids(z); order={k:i for i,k in enumerate(old)}
            assert len(order)==len(old)==len(keys) and set(old)==set(keys)
            take=[order[k] for k in keys]
            truth=z['truth' if key=='same_worker_T20' else 'truth_xy'][take]
            np.testing.assert_array_equal(truth,cache['xy'][:,20:].astype(np.float32))
            for t in ((20,) if key=='same_worker_T20' else (5,10)):
                for a in ARMS:
                    stem=(f'T{t}_' if key=='mixed_5_10' else '')+a
                    paths=z[stem+'_trajectories'][take];prob=z[stem+'_probabilities'][take]
                    err=z[stem+'_errors'][take]
                    np.testing.assert_allclose(measure(paths[np.arange(len(keys)),prob.argmax(1)],truth),err,atol=.0002,rtol=.00002)
                    macro=site_metric(np.stack([err[i].mean(0) for i in groups.values()]))
                    if key=='same_worker_T20':
                        np.testing.assert_allclose(list(macro.values()),[ref_20['four_site_macro'][a][m] for m in FIELDS],atol=1e-10,rtol=0)
                    elif a in ('M0','M1'):
                        np.testing.assert_allclose(list(macro.values()),[ref_old['macro'][f'{a}_{t}'][m] for m in FIELDS],atol=1e-10,rtol=0)
                    else:
                        np.testing.assert_allclose(list(macro.values()),[fusion_reference[t][a][m] for m in FIELDS],atol=1e-10,rtol=0)
                    result[t,a]=err
        provenance[key]=sha(path.read_bytes())
    return result,provenance


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


def pool(work,packs):
    torch.set_num_threads(2)
    cache,groups,keys=verify_sources(work/'matched_observations.npz',work/'pinns_matched_histories.npz')
    feats={};bins=None
    for t in TS:
        feats[t],b=inputs(cache,t)
        if t==5:bins=b
    assert np.bincount(bins).tolist()==[188,283,152]
    by={};provenance={}
    by[42],provenance['existing_42']=load_42(work,cache,keys,groups)
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
            'registered_secondary_sha256':sha((HERE/'quality_groups_plan.json').read_bytes()),'groups':summary}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('work',type=Path);p.add_argument('output',type=Path);p.add_argument('bundles',nargs='*',type=Path)
    args=p.parse_args();result=pool(args.work,args.bundles);args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(result['ingested_training_seeds'],{k:v['windows'] for k,v in result['groups'].items()})
