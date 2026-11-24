"""Fit four controlled CV/zero × M0/M1 arms on a single CPU worker."""
import argparse
from dataclasses import asdict
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import platform
import zipfile

import numpy as np
import torch

from anchored_model import SixCandidates, observed_base, training_loss
from scripts.analyze_heldout_pairs import BOOTSTRAP_SEED, REPLICATES, intervals, paired_clip_draws
from scripts.benchmark_heldout_sites import load_site_windows
from scripts.rerank_pinns_velocity_modes import last_two_future
from scripts.run_heldout_seed_sensitivity import SITE_COUNTS, NORMALIZATION, json_digest, split_indices
from scripts.train_pinns_heldout_sites import ROOT, Protocol, atomic_write, file_hash, save_json, source_hash, tensorize
from scripts.verify_heldout_bundle import ATOL_M, IDS_SHA, LABELS_SHA, RTOL, SOURCE_SHA, validate_zip
from src.models.predictor import UnstructuredShortPosePredictor

DRIVER = Path(__file__).resolve()
POSE_SHA = 'b493a47536664ddb4fd66603c1bbf6b2c2e33f553520382027b3ae96b53447d0'
BUNDLE_SHA = '0bf2745ab2d5ddcf5f5fbb5d330e6c6cc800625cec301f4c7105d031bdb49d2d'
P = Protocol()
ARMS = {f'{base}_{m}': (m, base == 'cv') for base in ('zero','cv') for m in ('M0','M1')}
ORIGINAL = ('original_M0','original_M1')
COMPARISONS = (('cv_M0','zero_M0'), ('cv_M1','zero_M1'), ('cv_M1','cv_M0'),
               ('zero_M1','zero_M0'), ('cv_M0','last_two'), ('cv_M1','last_two'),
               ('cv_M0','original_M0'), ('cv_M1','original_M1'))
OBS = ('history_xy','x_m','ego_type_oh','x_p','q_vec')
SCORES = ('top1_ade_m','top1_fde_m','minade6_m','minfde6_m','top1_is_minade',
          'mean_pair_distance_m','top1_probability','probability_entropy_nats')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def prepare(data, bundle):
    require(file_hash(bundle) == BUNDLE_SHA, 'Wrong five-seed archive')
    require(source_hash() == SOURCE_SHA, 'Original model, parser or training dependencies changed')
    with zipfile.ZipFile(bundle) as archive:
        validate_zip(archive)
        pose = archive.read('pose.npz')
        require(hashlib.sha256(pose).hexdigest() == POSE_SHA, 'Wrong pose cache')
    groups, _, data_sha = load_site_windows(data)
    require(data_sha == LABELS_SHA, 'Wrong official label/calibration set')
    tensors, index, keys, _, _, _ = tensorize(groups, io.BytesIO(pose))
    require(json_digest(keys) == IDS_SHA and len(set(keys)) == 1482, 'Wrong window identities')
    require({s:len(v) for s,v in index.items()} == SITE_COUNTS, 'Wrong site partition')
    xy = np.stack([r.xy for s in sorted(groups) for r in groups[s]])
    require(np.isfinite(xy).all() and xy.shape == (1482,35,2) and (xy[:,4] == 0).all(),
            'Expected ground positions in the last-observation coordinate system')
    require(tensors['fut_mask'].bool().all(), 'Missing registered future point')
    history = xy[:,:5].copy()
    np.testing.assert_array_equal(history.astype(np.float32), tensors['x_m'][:,:,:2].numpy())
    np.testing.assert_array_equal(xy[:,5:].astype(np.float32), tensors['fut_xy'].numpy())
    tensors['history_xy'] = torch.from_numpy(history)
    tensors['truth'] = torch.from_numpy(xy[:,5:].copy())
    canonical, _ = observed_base(tensors['history_xy'], True)
    np.testing.assert_array_equal(canonical.numpy(), last_two_future(history))
    folds = {}
    for site in sorted(index):
        training_sites, train, test = split_indices(index, keys, site)
        folds[site] = dict(training_sites=training_sites, training_windows=len(train), test_windows=len(test),
                           test_clips=len({keys[i][0] for i in test}),
                           train_key_sha256=json_digest([keys[i] for i in train]),
                           test_key_sha256=json_digest([keys[i] for i in test]))
    plan = {
        'experiment':'five_history_six_candidate_cv_anchor_seed42',
        'controlled_primary':'cv_M0 minus zero_M0, predicted highest-probability ADE in metres, four sites equally weighted.',
        'secondary_comparisons':[list(x) for x in COMPARISONS[1:]],
        'scope':'Exploratory development; the four sites previously informed this choice. One training seed. All signs retained.',
        'arms':ARMS,'optimization':asdict(P),'normalization':NORMALIZATION,'folds':folds,
        'backend':'All sixteen new fits on one deterministic Kaggle CPU worker. Same init across common parameters; seed 49 permutations and seed 42 stochastic modules reset for every fit.',
        'base':'Centered ground XY. The float64 fixed base is zero for the control or p5+(p5-p4)/0.1 times (h*0.1) for h=1..30. Float32 decoder cumulative residual is added in float64. Training and scoring use label-derived float64 target positions.',
        'assignment':'Original detached nearest-ADE winner, smooth-L1 beta .5, mode weight .25, label smoothing .04, 65/35 hard vs distance rank. Exact best-ADE ties choose the closest fixed 60-degree direction among tied queries to the largest valid future excursion from the arm base (earliest peak), rotated relative to the last observed step. Zero speed uses world +x; zero deviation and exact directional ties average the tied optima. The change of labels on completely tied batches is an intended consequence of adding the anchor.',
        'reference':'Original seed-42 M0 and explicit M1 weights, plus last-two CV; network replay of archived top1 must match saved per-window errors at fixed legacy tolerance.',
        'metrics':SCORES,'interval':f'{REPLICATES} whole-clip paired draws within each site, seed {BOOTSTRAP_SEED}, average four site means; conditional on these sites and seed.',
        'scope_of_inference':'Hypothesis concerns the starting path and its corresponding residual fit under a fixed tie rule; old independently trained networks are descriptive.',
        'frozen_source_sha256':SOURCE_SHA,'label_and_calibration_sha256':LABELS_SHA,'ordered_keys_sha256':IDS_SHA,
        'archive_sha256':BUNDLE_SHA,'pose_sha256':POSE_SHA,
        'utility_sha256':{name:file_hash(ROOT/'scripts'/name) for name in (
            'analyze_heldout_pairs.py','rerank_pinns_velocity_modes.py','run_heldout_seed_sensitivity.py',
            'verify_heldout_bundle.py')},
        'files_sha256':{p.name:file_hash(p) for p in (DRIVER, DRIVER.with_name('anchored_model.py'))}}
    return tensors, index, keys, json.loads(json.dumps(plan))


def observed(tensors, rows):
    return {k:tensors[k][rows] for k in OBS}


def fit(arm, training):
    """Only this fold's three-site tensor subset is accepted here."""
    model = SixCandidates(*ARMS[arm]).train()
    torch.manual_seed(P.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=P.learning_rate, weight_decay=P.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=P.epochs,
                                                       eta_min=P.learning_rate*P.min_lr_ratio)
    rng = torch.Generator().manual_seed(P.seed + 7)
    logs = []
    n = len(training['truth'])
    for epoch in range(1, P.epochs+1):
        sums = torch.zeros(2,6,dtype=torch.float64)
        tied_windows = 0
        total_loss = 0.
        first = None
        for ids in torch.randperm(n,generator=rng).split(P.batch_size):
            opt.zero_grad(set_to_none=True)
            outputs = model(**observed(training,ids))
            loss, weights, tied = training_loss(*outputs, training['truth'][ids],training['fut_mask'][ids])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),P.gradient_clip,error_if_nonfinite=True)
            opt.step()
            total_loss += loss.item()*len(ids)
            sums[0] += weights.sum(0)
            sums[1] += weights[tied].sum(0)
            tied_windows += int(tied.sum())
            if epoch == 1 and first is None:
                require(tied.all(), 'The first batch must start exactly on the fixed base')
                first = dict(windows=len(ids),all_tied=True,mode_credit=weights.sum(0).tolist())
        sched.step()
        require(abs(sums[0].sum().item()-n)<1e-8 and abs(sums[1].sum().item()-tied_windows)<1e-8,
                'Lost training-window assignment')
        logs.append(dict(epoch=epoch,mean_objective=total_loss/n,windows=n,tied_windows=tied_windows,
                         mode_credit=sums[0].tolist(),tied_sector_credit=sums[1].tolist()))
        if first is not None:
            logs[-1]['first_batch']=first
    return model.state_dict(), logs


@torch.inference_mode()
def forecast(model, tensors, rows, original_name=None):
    require(set(tensors)==set(OBS),'Forecast accepts observed tensors only')
    paths, probs = [], []
    model.eval()
    for ids in torch.as_tensor(rows).split(P.batch_size):
        features = observed(tensors, ids)
        if original_name is None:
            path, logits, _, _ = model(**features)
            weights = logits.softmax(1)
        else:
            names = ('x_m','ego_type_oh') if original_name == 'M0' else ('x_m','ego_type_oh','x_p','q_vec')
            result = model({k:features[k] for k in names})
            path,weights = result['trajs'],result['mode_probs']
        paths.append(path.numpy())
        probs.append(weights.numpy())
    return np.concatenate(paths),np.concatenate(probs)


def measures(paths, probs, truth):
    n=len(truth)
    require(paths.shape==(n,6,30,2) and probs.shape==(n,6), 'Bad six-candidate output')
    require(np.isfinite(paths).all() and np.isfinite(probs).all() and np.all(probs>=0), 'Invalid forecast')
    np.testing.assert_allclose(probs.sum(1),1,atol=1e-6,rtol=0)
    rows=np.arange(n)
    dist=np.linalg.norm(paths.astype(np.float64)-truth[:,None],axis=-1)
    ades, fdes=dist.mean(2),dist[:,:,-1]
    top,oracle=probs.argmax(1),ades.argmin(1)
    separation=np.stack([np.linalg.norm(paths[:,i].astype(np.float64)-paths[:,j],axis=-1).mean(1)
                         for i,j in itertools.combinations(range(6),2)]).mean(0)
    prob=probs.astype(np.float64)
    entropy=-(prob*np.log(np.maximum(prob,np.finfo(float).tiny))).sum(1)
    values=np.column_stack((ades[rows,top],fdes[rows,top],ades.min(1),fdes.min(1),
                            top==oracle,separation,probs[rows,top],entropy))
    return values,top,oracle


def summarize(values,top=None,oracle=None):
    result=dict(zip(SCORES,values.mean(0).tolist()))
    if top is not None:
        result.update(top1_mode_fraction=(np.bincount(top,minlength=6)/len(top)).tolist(),
                      minade_mode_fraction=(np.bincount(oracle,minlength=6)/len(top)).tolist())
    return result


def score_site(archive, site, test, keys, tensors, models, out):
    fold_keys=[keys[i] for i in test]
    truth=tensors['truth'][test].numpy()
    private={'clip':np.array([k[0] for k in fold_keys]),'person_id':np.array([k[1] for k in fold_keys]),
             't0':np.array([k[2] for k in fold_keys])}
    values={}
    table={}
    inputs={k:tensors[k] for k in OBS}
    for arm in list(ARMS)+list(ORIGINAL):
        if arm in ARMS:
            path,prob=forecast(models[arm],inputs,test)
        else:
            name=arm.split('_')[1]
            ck=torch.load(io.BytesIO(archive.read(f'seed_42/{site}/{name}.pt')),map_location='cpu',weights_only=True)
            require(list(map(tuple,ck['window_keys']))==fold_keys,'Archived baseline key order mismatch')
            net=UnstructuredShortPosePredictor(name)
            net.load_state_dict(ck['model_state'],strict=True)
            path,prob=forecast(net,inputs,test,name)
        scores,top,oracle=measures(path,prob,truth)
        if arm in ORIGINAL:
            np.testing.assert_allclose(scores[:,:2],ck['errors'].numpy(),atol=ATOL_M,rtol=RTOL)
        table[arm]=summarize(scores,top,oracle)
        values[arm]=scores[:,:2]
        private.update({arm+'_paths':path,arm+'_probabilities':prob,arm+'_metrics':scores})
    cv=last_two_future(tensors['history_xy'][test].numpy())
    dist=np.linalg.norm(cv-truth,axis=-1)
    values['last_two']=np.column_stack((dist.mean(1),dist[:,-1]))
    table['last_two']=summarize(values['last_two'])
    private['last_two_metrics']=values['last_two']
    def write(tmp):
        with tmp.open('wb') as f:
            np.savez_compressed(f,**private)
    atomic_write(out/'heldout_private.npz',write)
    return table,values,private['clip']


def compare(sites):
    macro={}
    for arm,row in next(iter(sites.values()))[0].items():
        macro[arm]={}
        for key in row:
            value=np.mean([r[0][arm][key] for r in sites.values()],axis=0)
            macro[arm][key]=value.tolist()
    diffs={}
    for left,right in COMPARISONS:
        deltas,draws,per_site={},[],{}
        rng=np.random.default_rng(BOOTSTRAP_SEED)
        for site,(_,values,clips) in sorted(sites.items()):
            change=values[left]-values[right]
            d=paired_clip_draws(change,clips,rng)
            deltas[site]=change.mean(0)
            draws.append(d)
            per_site[site]={'difference_ade_fde_m':deltas[site].tolist(),'interval95_m':intervals(d)}
        diffs[left+'_minus_'+right]={'by_site':per_site,'equal_site_difference_ade_fde_m':np.mean(list(deltas.values()),0).tolist(),
                    'equal_site_interval95_m':intervals(np.mean(draws,axis=0))}
    return {'by_site':{s:r[0] for s,r in sites.items()},'four_site_macro':macro,'paired':diffs}


def write_archive(out, members):
    def write(tmp):
        with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
            for p in members:
                archive.write(p,p.relative_to(out))
            manifest={p.relative_to(out).as_posix():file_hash(p) for p in members}
            archive.writestr('sha256.json',json.dumps(manifest,sort_keys=True))
        with zipfile.ZipFile(tmp) as archive:
            validate_zip(archive)
    atomic_write(out/'anchored_six_candidates.zip',write)


def main(args):
    tensors,index,keys,plan=prepare(args.data,args.bundle)
    if args.mode=='check':
        print(json.dumps({'ordered_keys':len(keys),'protocol_identity':json_digest(plan),'folds':plan['folds']}))
        return
    if args.mode=='register':
        if args.plan.exists(): require(json.loads(args.plan.read_text())==plan,'Existing registration differs')
        else: save_json(args.plan,plan)
        return
    require(json.loads(args.plan.read_text())==plan,'Driver, data or registered plan differs')
    require(Path('/kaggle/working').is_dir() and Path('/kaggle/input').is_dir(),'Fitting is reserved for Kaggle')
    require(not torch.cuda.is_available(),'Run with CUDA_VISIBLE_DEVICES empty')
    torch.set_num_threads(min(4,len(os.sched_getaffinity(0))))
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    worker={'torch':str(torch.__version__),'numpy':np.__version__,'platform':platform.platform(),
            'threads':torch.get_num_threads(),'worker_token':hashlib.sha256(platform.node().encode()).hexdigest()}
    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    record={'registration':plan,'worker':worker}
    record_path=out/'run.json'
    if record_path.exists():require(json.loads(record_path.read_text())==record,'Different worker or design in this output')
    else:save_json(record_path,record)
    members=[record_path]
    for src in (DRIVER,DRIVER.with_name('anchored_model.py')):
        dst=out/src.name
        if dst.exists():require(file_hash(dst)==file_hash(src),'Output contains different source')
        else:atomic_write(dst,lambda tmp:tmp.write_bytes(src.read_bytes()))
        members.append(dst)
    results={}
    with zipfile.ZipFile(args.bundle) as archive:
        for site in sorted(index):
            _,train,test=split_indices(index,keys,site)
            folder=out/site
            folder.mkdir(exist_ok=True)
            training={k:tensors[k][train] for k in OBS+('truth','fut_mask')}
            models={}
            for arm in ARMS:
                path=folder/(arm+'.pt')
                identity={'design_sha256':json_digest(plan),'worker':worker,'arm':arm,'site':site,**plan['folds'][site]}
                if not path.exists():
                    state,logs=fit(arm,training)
                    atomic_write(path,lambda tmp:torch.save({'identity':identity,'state':state,'epoch_logs':logs},tmp))
                ck=torch.load(path,weights_only=True,map_location='cpu')
                require(ck['identity']==identity and len(ck['epoch_logs'])==P.epochs,'Incomplete/mismatched checkpoint')
                net=SixCandidates(*ARMS[arm])
                net.load_state_dict(ck['state'],strict=True)
                models[arm]=net
                logs_path=folder/(arm+'_epochs.json')
                save_json(logs_path,ck['epoch_logs'])
                members.extend([path,logs_path])
                print(f'Completed {site} {arm}',flush=True)
            results[site]=score_site(archive,site,test,keys,tensors,models,folder)
            members.append(folder/'heldout_private.npz')
    report=out/'scores.json'
    save_json(report,{'registration_sha256':json_digest(plan),'cohort':SITE_COUNTS,'metric_columns':SCORES,
                      'training_seed':P.seed,'source_archive_sha256':BUNDLE_SHA,**compare(results)})
    write_archive(out,members+[report])
    print(out/'anchored_six_candidates.zip',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data',type=Path)
    parser.add_argument('bundle',type=Path)
    parser.add_argument('plan',type=Path)
    parser.add_argument('--mode',choices=('check','register','fit'),required=True)
    parser.add_argument('--out',type=Path,default=Path('/kaggle/working/anchored_six'))
    main(parser.parse_args())
