"""Frozen, post-decoder projection of two impossible event probabilities; no fitting."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from scripts.analyze_heldout_pairs import paired_clip_draws
from scripts.benchmark_heldout_sites import load_site_windows
from scripts.run_heldout_seed_sensitivity import SITE_COUNTS, json_digest, split_indices
from scripts.train_pinns_heldout_sites import file_hash, save_json
from src.data.actions import compute_action_labels

HERE=Path(__file__).resolve().parent
SOURCE_ARCHIVE='b84eaaadcaab4518ded15c83e9d746030e03693e89dfe9e23b15ef2f1fc8587e'
LABELS='0b276b34ab1b01dae91f67c56bfaea5cc931511b05b4fc1c92686a891e6f6801'
IDS='f5ee511a54ad818499d3dde6213f7f3b49332943ff2fc93d93aed87d316dbb91'
ARMS=('auxiliary','end_to_end','detached')
EVENTS=('start','stop')


def require(test,msg):
    if not test:raise ValueError(msg)


def eligible(histories,dt=.1):
    """Input only five finite measured ground positions, no truth, scores, IDs or threshold fitting."""
    h=np.asarray(histories,dtype=np.float64)
    require(h.ndim==3 and h.shape[1:]==(5,2) and np.isfinite(h).all() and dt==.1,'Expected five complete positions at 10 Hz')
    v=np.diff(h,axis=1)/dt
    # Labeler ignores the undefined first velocity. Smoothed velocities at h[2], h[3], h[4].
    smooth=np.stack((v[:,:2].mean(1),v[:,:3].mean(1),v[:,1:].mean(1)),axis=1)
    mean_speed=np.hypot(smooth[:,:,0],smooth[:,:,1]).mean(1)
    return np.stack((mean_speed<=.20,mean_speed>=.60),axis=1)


def project(original_probs,eligible_events):
    """Modify reported outputs after a complete frozen forward; trajectory tensors are not inputs."""
    p=np.asarray(original_probs); allow=np.asarray(eligible_events)
    require(p.ndim==2 and p.shape[1]==5 and allow.dtype==bool and allow.shape==(len(p),2) and
            np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),'Unexpected frozen probabilities')
    result=p.copy()
    result[:,:2]=np.where(allow,p[:,:2],0.)
    return result


def class_metrics(y,p):
    y=np.asarray(y,dtype=bool); p=np.asarray(p,dtype=float); pred=p>=.5
    tp=int((pred&y).sum());fp=int((pred&~y).sum())
    fn=int((~pred&y).sum());tn=int((~pred&~y).sum())
    return dict(n=len(y),positives=tp+fn,tp=tp,fp=fp,fn=fn,tn=tn,
                recall=tp/(tp+fn) if tp+fn else None,precision=tp/(tp+fp) if tp+fp else None,
                f1=(2*tp/(2*tp+fp+fn)) if (tp+fn) else None,
                brier=float(np.mean((p-y)**2)))


def load_histories(data):
    groups,_,h=load_site_windows(data)
    require(h==LABELS,'Different PINNS ground labels or calibration')
    keys=[];xy=[];index={}
    for s in sorted(groups):
        start=len(xy)
        for r in groups[s]:
            keys.append((r.clip,r.person_id,r.t0));xy.append(r.xy)
        index[s]=np.arange(start,len(xy))
    require(json_digest(keys)==IDS and {s:len(v) for s,v in index.items()}==SITE_COUNTS,'Different benchmark split')
    xy=np.stack(xy)
    return xy,keys,index,eligible(xy[:,:5])


def design(data):
    xy,keys,index,valid=load_histories(data)
    from src.data import actions
    folds={}
    for s,idx in sorted(index.items()):
        others,train,test=split_indices(index,keys,s)
        folds[s]=dict(train_sites=others,train_keys_sha256=json_digest([keys[i] for i in train]),
                      test_keys_sha256=json_digest([keys[i] for i in test]),windows=len(idx),
                      clips=len({keys[i][0] for i in idx}),eligible_start_stop=valid[idx].sum(0).tolist())
    plan=dict(source_kaggle=373610725,source_archive_sha256=SOURCE_ARCHIVE,
              labels_calibration_sha256=LABELS,five_frame_cohort_sha256=IDS,
              action_labeler_sha256=file_hash(Path(actions.__file__)),driver_sha256=file_hash(Path(__file__)),
              scope='Post-hoc intervention proposed after prior poor stop recall; its scores on this intervention are unseen. Four previously examined sites and seed42; all outcomes reported.',
              primary='Detached stop Brier: projected minus original, average windows per held-out site then sites equally. This deterministic restriction cannot worsen Brier with exact rule-derived labels.',
              secondary='Start and stop Brier, confusion counts/recall/precision/F1 at >=0.5, original false positives removed, eligible support and score changes; all three heads per site, pooled counts, equal-site Brier, paired whole-clip intervals. Other classes, six trajectories, mode probabilities and selected mode are fixed.',
              transformation='After the existing complete forecast, zero reported start probability unless v_pre<=0.20 m/s; zero stop probability unless v_pre>=0.60 m/s. Same original probabilities inside eligibility. No independent renormalization, threshold fitting, or feedback into the decoder.',
              speed='Five ground positions at 0.1 s. Four interval velocity vectors. At the final three history frames average the preceding 2,3,3 available interval vectors, respectively; average the norms of those three vectors.',
              threshold=.5,arms=ARMS,source_fits=16,scored_supervised_fits=12,seed=42,
              no_new_fits=True,interval='20000 pairs of whole-clip draws by site, PCG64 2026, percentiles conditional on these sites/fits.',
              identities=folds)
    return json.loads(json.dumps(plan)),(xy,keys,index,valid)


def calculate(bundle,plan,inputs,out):
    xy,keys,index,allow=inputs
    require(file_hash(bundle)==SOURCE_ARCHIVE,'Wrong frozen joint-model archive')
    # The evaluation labels enter only after eligibility is already fixed from observed history.
    label,valid=zip(*(compute_action_labels(row[:5],np.ones(5),row[5:],np.ones(30)) for row in xy))
    y=np.array(label,dtype=bool)
    require(np.all(valid) and y.sum(0).tolist()==[49,53,214,208,975] and not (y[:,:2]&~allow).any(),'Event rule violated')
    rows={a:{e:{} for e in EVENTS} for a in ARMS}
    score_pairs={a:{} for a in ARMS}
    raw_cols={};clips={};field_hashes={}
    with zipfile.ZipFile(bundle) as z:
        require(len(z.namelist())==len(set(z.namelist())) and z.testzip() is None,'Bad archive')
        manifest=json.loads(z.read('sha256.json'))
        require(set(manifest)|{'sha256.json'}==set(z.namelist()),'Wrong frozen member list')
        for n,d in manifest.items():require(hashlib.sha256(z.read(n)).hexdigest()==d,'Changed '+n)
        fields=('trajs','mode_logits','mode_probs','errors')
        for s,idx in sorted(index.items()):
            fold=json.loads(z.read(s+'/fold_keys.json')); expected=plan['identities'][s]
            require(json_digest(fold['train_keys'])==expected['train_keys_sha256'] and
                    [tuple(k) for k in fold['test_keys']]==[keys[i] for i in idx],'Different held-out IDs')
            with np.load(io.BytesIO(z.read(s+'/targets.npz')),allow_pickle=False) as t:
                require(list(zip(t['clip'].tolist(),t['person_id'].tolist(),t['t0'].tolist()))==[keys[i] for i in idx],'Target keys mismatch')
                np.testing.assert_array_equal(t['y_act'].astype(bool),y[idx]);require(t['m_act'].all(),'Invalid head target')
            clips[s]=np.array([keys[i][0] for i in idx])
            field_hashes[s]={}
            for arm in ('no_head',*ARMS):
                with np.load(io.BytesIO(z.read(f'{s}/{arm}.npz')),allow_pickle=False) as f:
                    require(f['trajs'].shape==(len(idx),6,30,2) and f['mode_probs'].shape==(len(idx),6),'Wrong frozen forecasts')
                    field_hashes[s][arm]={k:hashlib.sha256(f[k].tobytes()).hexdigest() for k in fields}
                    if arm=='no_head':
                        require('act_probs' not in f.files,'Unexpected unsupervised head')
                        continue
                    raw=f['act_probs']
                    np.testing.assert_allclose(raw,1/(1+np.exp(-f['act_logits'].astype(float))),atol=2e-7,rtol=0)
                    new=project(raw,allow[idx])
                    np.testing.assert_array_equal(new[:,2:],raw[:,2:])
                    np.testing.assert_array_equal(new[:,:2][allow[idx]],raw[:,:2][allow[idx]])
                    require(not new[:,:2][~allow[idx]].any(),'Impossible score survived')
                    score_pairs[arm][s]=(raw[:,:2].copy(),new[:,:2].copy())
                    for j,e in enumerate(EVENTS):
                        target=y[idx,j]; eligible_mask=allow[idx,j]
                        before,after=class_metrics(target,raw[:,j]),class_metrics(target,new[:,j])
                        require((before['tp'],before['fn'])==(after['tp'],after['fn']) and after['fp']<=before['fp'],'Wrong projection effect')
                        expected_gain=np.sum(raw[~eligible_mask,j].astype(float)**2)/len(idx)
                        np.testing.assert_allclose(before['brier']-after['brier'],expected_gain,atol=2e-15,rtol=0)
                        rows[arm][e][s]=dict(original=before,projected=after,eligible=int(eligible_mask.sum()),
                                              eligible_positives=int(target.sum()),removed_false_positives=before['fp']-after['fp'],
                                              changed_probabilities=int(np.count_nonzero(new[:,j]!=raw[:,j])))
                    for k in fields:require(hashlib.sha256(f[k].tobytes()).hexdigest()==field_hashes[s][arm][k],'A frozen trajectory field changed')
    summary={}
    for arm in ARMS:
        rng=np.random.default_rng(2026);interval_draws={};deltas={}
        for site,idx in sorted(index.items()):
            old,new=(v.astype(float) for v in score_pairs[arm][site])
            deltas[site]=(new-y[idx,:2])**2-(old-y[idx,:2])**2
            interval_draws[site]=paired_clip_draws(deltas[site],clips[site],rng)
        summary[arm]={}
        for j,e in enumerate(EVENTS):
            sites=sorted(index);counts={}; means={}
            for method,ix in [('original',0),('projected',1)]:
                true=np.concatenate([y[index[s],j] for s in sites])
                prob=np.concatenate([score_pairs[arm][s][ix][:,j] for s in sites])
                counts[method]=class_metrics(true,prob)
                means[method]=float(np.mean([rows[arm][e][s][method]['brier'] for s in sites]))
            ci=lambda x:np.quantile(x,[.025,.975]).tolist()
            for s in sites:
                rows[arm][e][s]['paired_brier_difference_95']=ci(interval_draws[s][:,j])
            summary[arm][e]=dict(total_windows=len(keys),pooled=counts,equal_site_brier=means,
                                 projected_minus_original_equal_site_brier=means['projected']-means['original'],
                                 paired_site_macro_brier95=ci(np.mean(list(interval_draws.values()),axis=0)[:,j]),
                                 removed_false_positives=counts['original']['fp']-counts['projected']['fp'],
                                 by_site=rows[arm][e])
    out.mkdir(parents=True,exist_ok=True)
    public=out/'projection_scores.json';save_json(public,dict(plan_sha256=json_digest(plan),source_sha256=SOURCE_ARCHIVE,
        site_windows=SITE_COUNTS,scope=plan['scope'],summary=summary,
        identity_audit={'frozen_trajectories_checked':4*6*len(keys),'six_mode_forecasts':4*len(keys),'trajectory_or_probability_mutations':0,
                        'threshold_selection_labels':0,'new_fits':0,'trajectory_feedback_used':False}))
    verify=out/'field_hashes.json';save_json(verify,field_hashes)
    result=out/'action_eligibility_projection.zip'
    with zipfile.ZipFile(result,'w',zipfile.ZIP_DEFLATED) as z:
        z.write(public,public.name);z.write(verify,verify.name);z.write(Path(__file__),'audit_projection.py')
        z.writestr('plan.json',json.dumps(plan,indent=2)+'\n')
    print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path);p.add_argument('plan',type=Path)
    p.add_argument('--bundle',type=Path);p.add_argument('--out',type=Path,default=Path('/kaggle/working/action_eligibility'))
    p.add_argument('--mode',choices=('check','register','score'),required=True);args=p.parse_args()
    plan,inputs=design(args.data)
    if args.mode=='register':
        if args.plan.exists():require(json.loads(args.plan.read_bytes())==plan,'Plan changed')
        else:save_json(args.plan,plan)
    elif args.mode=='score':
        require(args.bundle is not None and json.loads(args.plan.read_bytes())==plan,'Provide locked bundle and plan')
        calculate(args.bundle,plan,inputs,args.out)
    print(json_digest(plan))
