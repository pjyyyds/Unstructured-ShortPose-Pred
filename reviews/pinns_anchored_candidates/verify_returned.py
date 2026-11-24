"""Independent, forward-only CPU replay of Kaggle 373619572. No fitting or cloud imports."""
import argparse
from collections import Counter
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import re
import zipfile

import numpy as np
import torch

from scripts.benchmark_heldout_sites import load_site_windows
from scripts.train_pinns_heldout_sites import tensorize
from src.models.predictor import UnstructuredShortPosePredictor


SITE_COUNTS={'america_crossroad':816,'italy_park':84,'japan_crossroad':432,'thailand_street':150}
TRAINED=('zero_M0','zero_M1','cv_M0','cv_M1')
NEURAL=TRAINED+('original_M0','original_M1')
COLUMNS=('top1_ade_m','top1_fde_m','minade6_m','minfde6_m','top1_is_minade',
         'mean_pair_distance_m','top1_probability','probability_entropy_nats')
FIXED_PLAN='bc3ef3a92896058c12997df3c663a57814b068faa1d3eb1e01da8d92fd99f2fa'
BUNDLE='0bf2745ab2d5ddcf5f5fbb5d330e6c6cc800625cec301f4c7105d031bdb49d2d'
ATOL_PATH_M,RTOL_PATH,ATOL_PROB=2e-4,2e-5,2e-6
RAW_TOL=2e-12
CONTRASTS=(('cv_M0','zero_M0'),('cv_M1','zero_M1'),('cv_M1','cv_M0'),('zero_M1','zero_M0'),
           ('cv_M0','last_two'),('cv_M1','last_two'),('cv_M0','original_M0'),('cv_M1','original_M1'))


def require(cond,why):
    if not cond:
        raise ValueError(why)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(obj):
    return sha(json.dumps(obj,sort_keys=True,separators=(',',':')).encode())


def archive_members(z):
    require(len(set(z.namelist()))==len(z.namelist()) and z.testzip() is None,'Bad zip names or CRC')
    manifest=json.loads(z.read('sha256.json'))
    require(set(z.namelist())==set(manifest)|{'sha256.json'},'Manifest membership mismatch')
    for member,h in manifest.items():
        require(sha(z.read(member))==h,'Bad archive checksum '+member)
    return manifest


def trajectory_base(history):
    displacement=history[:,-1]-history[:,-2]
    return history[:,-1,None]+np.arange(1,31)[None,:,None]*displacement[:,None]


def first_batch(truth,hist,train,anchored):
    order=torch.randperm(len(train),generator=torch.Generator().manual_seed(49))[:64].numpy()
    ids=train[order]
    history=hist[ids]
    # Reproduce operation ordering for the registered first-batch tie assignment.
    residual=truth[ids].copy()
    dx=history[:,-1]-history[:,-2]
    if anchored:
        residual-=history[:,-1:]+(dx/.1)[:,None]*(.1*np.arange(1,31))[None,:,None]
    peak=np.hypot(residual[...,0],residual[...,1]).argmax(1)
    r=residual[np.arange(len(ids)),peak]
    credit=np.zeros(6)
    for heading,delta in zip(dx,r):
        h=complex(*heading)
        h=h/abs(h) if h else 1.
        # Six complex dot products, with opposite pairs explicitly symmetric.
        local=complex(*delta)*h.conjugate()
        c,s=.5,math.sqrt(3.)/2
        directions=(1+0j,complex(c,s),complex(-c,s),-1+0j,complex(-c,-s),complex(c,-s))
        dots=np.array([(local*d.conjugate()).real for d in directions])
        wins=dots==dots.max()
        credit+=wins/wins.sum()
    return credit


@torch.inference_mode()
def independent_rollout(model,features):
    """Use frozen encoder modules; reconstruct recurrence, residual integration and softmax."""
    xm=features['x_m']
    hm=model.motion_enc(xm)
    if model.variant=='M1':
        pose,part=model.pose_enc(features['x_p'])
        fused,_=model.quality_gate(hm,h_p=pose,q_vec=features['q_vec'],part_feats=part)
    else:
        fused,_=model.quality_gate(hm)
    n=len(xm)
    env=xm.new_zeros((n,model.env_dim))
    h,_=model.interaction(h_mp=fused,h_env=env,ego_type_oh=features['ego_type_oh'],
                          eff_map=xm.new_zeros((n,1)),nbr_edges=None,nbr_type_oh=None,nbr_mask=None)
    dec=model.decoder
    q=dec.mode_queries[None].expand(n,-1,-1)
    action=xm.new_zeros((n,6,dec.cond_dim))
    env6=env[:,None,:].expand(-1,6,-1)
    fixed=dec.ih_static(torch.cat([env6,q,action],-1).reshape(n*6,-1))
    state=(h[:,None]+dec.mode_init_proj(q)).reshape(n*6,-1)
    velocity=torch.where(xm[:,-1,6:7]>.5,xm[:,-1,2:4],0.)
    prev=(.1*velocity[:,None].expand(-1,6,-1)).reshape(n*6,2)
    pos=torch.zeros_like(prev)
    trajectory=[]
    for _ in range(30):
        ir,iz,inn=(dec.ih_delta(prev)+fixed).chunk(3,-1)
        hr,hz,hn=dec.hh_proj(state).chunk(3,-1)
        r=(ir+hr).sigmoid()
        z=(iz+hz).sigmoid()
        state=(1-z)*(inn+r*hn).tanh()+z*state
        prev=dec.delta_out(state)
        pos=pos+prev
        trajectory.append(pos.reshape(n,6,2))
    logits=dec.pi_head(torch.cat([h[:,None].expand(-1,6,-1),state.reshape(n,6,-1)-h[:,None],q,action],-1))
    logits=logits[...,0].double().numpy()
    mass=np.exp(logits-logits.max(1,keepdims=True))
    return torch.stack(trajectory,2).double().numpy(),mass/mass.sum(1,keepdims=True)


def score(path,prob,truth):
    n=len(truth)
    require(path.shape==(n,6,30,2) and prob.shape==(n,6) and np.isfinite(path).all() and
            np.isfinite(prob).all() and (prob>=0).all(),'Invalid returned candidates')
    np.testing.assert_allclose(prob.sum(1),1,rtol=0,atol=1e-6)
    path,prob=path.astype(float),prob.astype(float)
    diff=path-truth[:,None]
    dist=np.hypot(diff[...,0],diff[...,1])
    a,f=dist.mean(-1),dist[:,:,-1]
    top,oracle=prob.argmax(1),a.argmin(1)
    gap=0.
    for i in range(6):
        for j in range(i):
            d=path[:,i]-path[:,j]
            gap=gap+np.hypot(d[...,0],d[...,1]).mean(1)/15
    row=np.arange(n)
    entropy=-(prob*np.log(np.maximum(prob,np.finfo(float).tiny))).sum(1)
    values=np.stack([a[row,top],f[row,top],a.min(1),f.min(1),top==oracle,gap,prob[row,top],entropy],1)
    table=dict(zip(COLUMNS,values.mean(0).tolist()))
    for name,ids in [('top1_mode_fraction',top),('minade_mode_fraction',oracle)]:
        table[name]=[np.mean(ids==i) for i in range(6)]
    return values,table,top,oracle


def cluster_indices(clips,rng):
    unique=np.unique(clips)
    sampled=rng.integers(0,len(unique),size=(20000,len(unique)))
    return np.stack([(sampled==i).sum(1) for i in range(len(unique))],axis=1)


def paired(all_scores,all_clips):
    rng=np.random.default_rng(2026)
    draws={}
    for site,clips in sorted(all_clips.items()):
        c=cluster_indices(clips,rng)
        blocks=[clips==name for name in np.unique(clips)]
        draws[site]=(c,blocks,c@np.array([b.sum() for b in blocks]))
    result={}
    def ci(x):
        v=np.percentile(x,[2.5,97.5],axis=0).T
        return dict(zip(['ade_m','fde_m'],v.tolist()))
    for left,right in CONTRASTS:
        by_site,intervals,point={},[],[]
        for site,errors in sorted(all_scores.items()):
            c,blocks,denom=draws[site]
            differences=errors[left][:,:2]-errors[right][:,:2]
            rep=(c@np.stack([differences[b].sum(0) for b in blocks]))/denom[:,None]
            by_site[site]={'difference_ade_fde_m':differences.mean(0),'interval95_m':ci(rep)}
            intervals.append(rep)
            point.append(differences.mean(0))
        result[left+'_minus_'+right]=dict(by_site=by_site,equal_site_difference_ade_fde_m=np.mean(point,0),
                                          equal_site_interval95_m=ci(np.mean(intervals,0)))
    return result


def same_tree(local,cloud):
    if isinstance(local,dict):
        require(set(local)==set(cloud),'Different summary keys')
        return max(same_tree(v,cloud[k]) for k,v in local.items())
    a,b=np.asarray(local),np.asarray(cloud)
    require(a.shape==b.shape,'Different summary shape')
    np.testing.assert_allclose(a,b,rtol=0,atol=RAW_TOL)
    return float(np.max(abs(a-b)))


def verify(output,original,metadata,plan_path):
    require(sha(plan_path.read_bytes())==FIXED_PLAN,'Plan differs from pre-fit publication')
    require(sha(original.read_bytes())==BUNDLE,'Wrong original weights')
    plan=json.loads(plan_path.read_bytes()); abi=digest(plan)
    with zipfile.ZipFile(output) as z,zipfile.ZipFile(original) as old:
        hashes=archive_members(z)
        expected={'run.json','scores.json','anchored_model.py','run_anchor.py'}
        for s in SITE_COUNTS:
            expected.add(s+'/heldout_private.npz')
            for name in TRAINED:
                expected.update([f'{s}/{name}.pt',f'{s}/{name}_epochs.json'])
        require(set(hashes)==expected,'Unexpected result membership')
        run=json.loads(z.read('run.json')); report=json.loads(z.read('scores.json'))
        require(run['registration']==plan and report['registration_sha256']==abi and report['training_seed']==42,'Registration mismatch')
        require(report['cohort']==SITE_COUNTS and tuple(report['metric_columns'])==COLUMNS and report['source_archive_sha256']==BUNDLE,
                'Returned report schema or source differs')
        require(all(hashes[name]==h for name,h in plan['files_sha256'].items()),'Different model/driver')
        require(re.fullmatch('[0-9a-f]{64}',run['worker']['worker_token']),'Invalid worker record')
        groups,_,labels=load_site_windows(metadata)
        require(labels==plan['label_and_calibration_sha256'],'Raw ground labels/calibrations differ')
        require(old.testzip() is None,'Original archive CRC')
        pose=old.read('pose.npz');require(sha(pose)==plan['pose_sha256'],'Pose file differs')
        t,index,keys,_,_,_=tensorize(groups,io.BytesIO(pose))
        xy=np.array([r.xy for s in sorted(groups) for r in groups[s]])
        h,y=xy[:,:5],xy[:,5:]
        cv=trajectory_base(h)
        equality=np.max(abs(cv-(h[:,-1:]+((h[:,-1]-h[:,-2])/.1)[:,None]*(.1*np.arange(1,31))[None,:,None])))
        require(equality<1e-12 and np.all(h[:,-1]==0) and bool(t['fut_mask'].all()),'Invalid observed base or completeness')
        np.testing.assert_array_equal(h.astype(np.float32),t['x_m'][:,:,:2].numpy())
        require(digest(keys)==plan['ordered_keys_sha256'] and len(set(keys))==1482,'Window identities differ')
        require({s:len(v) for s,v in index.items()}==SITE_COUNTS,'Different site windows')
        features={k:t[k] for k in ('x_m','x_p','q_vec','ego_type_oh')}
        del t
        site_scores={};clips={};by_site={};max_path=max_prob=max_cloud_metric=max_replay_metric=max_baseline=0.
        max_cv=0.;new_weights=old_weights=batches=0;epoch_initial={};total_ties=Counter()
        for site,ids in sorted(index.items()):
            train=np.concatenate([index[s] for s in sorted(index) if s!=site])
            test_keys=[keys[i] for i in ids]
            train_keys=[keys[i] for i in train]
            require(not ({k[0] for k in train_keys}&{k[0] for k in test_keys}),'Clip leakage')
            fold=plan['folds'][site]
            require(digest(train_keys)==fold['train_key_sha256'] and digest(test_keys)==fold['test_key_sha256'],
                    'Unregistered split')
            require(fold['train_key_sha256']!=fold['test_key_sha256'] and fold['test_windows']==len(ids) and
                    fold['training_windows']==len(train),'Incorrect fold sizes')
            site_scores[site]={};by_site[site]={};epoch_initial[site]={}
            with np.load(io.BytesIO(z.read(site+'/heldout_private.npz')),allow_pickle=False) as saved:
                expected_cols={'clip','person_id','t0','last_two_metrics'}|{f'{a}_{b}' for a in NEURAL for b in ('paths','metrics','probabilities')}
                require(set(saved.files)==expected_cols,'Unexpected private columns')
                require(list(zip(saved['clip'].tolist(),saved['person_id'].tolist(),saved['t0'].tolist()))==test_keys,'Key mismatch')
                clips[site]=saved['clip'].copy()
                require(len(np.unique(clips[site]))==fold['test_clips'],'Clip count')
                for arm in NEURAL:
                    name=arm.rsplit('_',1)[1]
                    net=UnstructuredShortPosePredictor(name).cpu().eval()
                    if arm in TRAINED:
                        ck=torch.load(io.BytesIO(z.read(f'{site}/{arm}.pt')),map_location='cpu',weights_only=True)
                        identity=dict(design_sha256=abi,worker=run['worker'],arm=arm,site=site,**fold)
                        require(identity==ck['identity'],'Checkpoint worker, fold or training identity differs')
                        require(all(k.startswith('network.') for k in ck['state']),'Unexpected anchored module')
                        net.load_state_dict({k.removeprefix('network.'):v for k,v in ck['state'].items()},strict=True)
                        logs=ck['epoch_logs']
                        require(logs==json.loads(z.read(f'{site}/{arm}_epochs.json')) and len(logs)==28,'Epoch log mismatch')
                        for i,entry in enumerate(logs,1):
                            require(entry['epoch']==i and entry['windows']==len(train) and 0<=entry['tied_windows']<=len(train)
                                    and np.isfinite(entry['mean_objective']),'Invalid training record')
                            for key,tot in [('mode_credit',len(train)),('tied_sector_credit',entry['tied_windows'])]:
                                v=np.array(entry[key]);require(v.shape==(6,) and (v>=0).all() and abs(v.sum()-tot)<1e-8,'Mode credit mismatch')
                            total_ties[arm]+=entry['tied_windows']
                        initial=logs[0]['first_batch'];require(initial['all_tied'] and initial['windows']==64,'Missing initial tie')
                        expected_first=first_batch(y,h,train,arm.startswith('cv_'))
                        np.testing.assert_allclose(initial['mode_credit'],expected_first,rtol=0,atol=1e-10)
                        epoch_initial[site][arm]=expected_first.tolist()
                        require(torch.isfinite(net.decoder.delta_out.weight).all() and net.decoder.delta_out.weight.abs().sum()>0,'Untrained readout')
                        new_weights+=1
                    else:
                        ck=torch.load(io.BytesIO(old.read(f'seed_42/{site}/{name}.pt')),map_location='cpu',weights_only=True)
                        require(list(map(tuple,ck['window_keys']))==test_keys,'Old checkpoint keys')
                        net.load_state_dict(ck['model_state'],strict=True);old_weights+=1
                    residuals=[];lp=[]
                    for start in range(0,len(ids),64):
                        batch=ids[start:start+64]
                        residual,prob=independent_rollout(net,{k:v[batch] for k,v in features.items()})
                        residuals.append(residual);lp.append(prob);batches+=1
                    residual=np.concatenate(residuals)
                    path=residual+cv[ids,None] if arm.startswith('cv_') else residual
                    prob=np.concatenate(lp)
                    cloud_path,cloud_prob=saved[arm+'_paths'],saved[arm+'_probabilities']
                    np.testing.assert_allclose(path,cloud_path,rtol=RTOL_PATH,atol=ATOL_PATH_M)
                    np.testing.assert_allclose(prob,cloud_prob,rtol=0,atol=ATOL_PROB)
                    max_path=max(max_path,float(np.max(abs(path-cloud_path))))
                    max_prob=max(max_prob,float(np.max(abs(prob-cloud_prob))))
                    cloud_values,table,top,oracle=score(cloud_path,cloud_prob,y[ids])
                    replay,_,rt,ro=score(path,prob,y[ids])
                    np.testing.assert_array_equal(rt,top);np.testing.assert_array_equal(ro,oracle)
                    np.testing.assert_allclose(replay[:,0:6],cloud_values[:,0:6],rtol=RTOL_PATH,atol=ATOL_PATH_M)
                    np.testing.assert_allclose(replay[:,6:],cloud_values[:,6:],rtol=0,atol=4*ATOL_PROB)
                    np.testing.assert_allclose(cloud_values,saved[arm+'_metrics'],rtol=0,atol=RAW_TOL)
                    if arm in ('original_M0','original_M1'):
                        diff=float(np.max(abs(replay[:,:2]-ck['errors'].numpy())))
                        np.testing.assert_allclose(replay[:,:2],ck['errors'].numpy(),rtol=RTOL_PATH,atol=ATOL_PATH_M)
                        max_baseline=max(max_baseline,diff)
                    max_cloud_metric=max(max_cloud_metric,float(np.max(abs(saved[arm+'_metrics']-cloud_values))))
                    max_replay_metric=max(max_replay_metric,float(np.max(abs(cloud_values[:,:6]-replay[:,:6]))))
                    by_site[site][arm]=table;site_scores[site][arm]=cloud_values
                d=cv[ids]-y[ids]; dist=np.hypot(d[...,0],d[...,1])
                values=np.column_stack([dist.mean(1),dist[:,-1]])
                site_scores[site]['last_two']=values
                by_site[site]['last_two']=dict(zip(COLUMNS,values.mean(0).tolist()))
                max_cv=max(max_cv,float(np.max(abs(values-saved['last_two_metrics']))))
                np.testing.assert_allclose(values,saved['last_two_metrics'],rtol=0,atol=RAW_TOL)
            print('Replayed',site,flush=True)
        first=by_site[next(iter(by_site))]
        macro={arm:{key:np.mean([by_site[s][arm][key] for s in by_site],axis=0) for key in first[arm]} for arm in first}
        contrasts=paired(site_scores,clips)
        errors=[same_tree(actual,report[key]) for actual,key in [(by_site,'by_site'),(macro,'four_site_macro'),(contrasts,'paired')]]
        return dict(kaggle_version=373619572,preregistered_git='66fb39d',archive_sha256=sha(output.read_bytes()),
                    scores_sha256=hashes['scores.json'],registration_file_sha256=FIXED_PLAN,registration_identity=abi,
                    model_forwards=batches,windows=1482,full_six_candidate_windows=1482*6,
                    site_counts=SITE_COUNTS,learned_checkpoints=new_weights,reference_checkpoints=old_weights,
                    epochs_per_fit=28,identical_worker_tokens=1,top1_index_disagreements=0,minade_index_disagreements=0,
                    peak_abs_candidate_coordinate_error_m=max_path,peak_abs_probability_error=max_prob,
                    peak_abs_metric_diff_from_saved_paths=max_cloud_metric,peak_abs_replayed_trajectory_metric_error_m=max_replay_metric,
                    peak_abs_old_top1_error_difference_m=max_baseline,peak_abs_cv_error_m=max_cv,
                    peak_abs_analytic_anchor_difference_m=float(equality),peak_abs_aggregate_or_interval_error=max(errors),
                    first_batch_mode_credit=epoch_initial,all_training_ties_across_folds=dict(total_ties),
                    primary=report['paired']['cv_M0_minus_zero_M0'],interpretation='Single-worker matched objective and init, seed 42. Conditional intervals on four previously examined sites. No training replay.',
                    independent_replay='Bare original encoders, explicit 30-step recurrent equations, independent observed-base arithmetic, float64 probability normalization and per-candidate metric reconstruction. Bootstrap independently uses clip multiplicities.',
                    verification_thresholds={'cross_cpu_path_m_atol':ATOL_PATH_M,'cross_cpu_path_rtol':RTOL_PATH,'cross_cpu_prob_atol':ATOL_PROB,
                                             'returned_paths_to_aggregates_atol':RAW_TOL},local_torch=str(torch.__version__))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ('returned','original','metadata','plan','output'):
        parser.add_argument(field,type=Path)
    args=parser.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
    summary=verify(args.returned,args.original,args.metadata,args.plan)
    args.output.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if not isinstance(v,dict)},indent=2))
