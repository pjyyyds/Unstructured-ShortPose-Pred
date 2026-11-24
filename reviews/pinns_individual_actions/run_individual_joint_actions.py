"""Five-frame individual-only four-site joint action/trajectory development experiment.

Run with the frozen repository revision on PYTHONPATH. Register on the known development
data before fitting. Training is enabled only on Kaggle CPU. 'verify' never trains.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import zipfile

import numpy as np
import torch
from torch import nn

from scripts.benchmark_heldout_sites import load_site_windows
from scripts.evaluate_heldout_implied_actions import ACTIONS, class_scores
from scripts.run_heldout_seed_sensitivity import json_digest, split_indices
from scripts.train_pinns_heldout_sites import Protocol, atomic_write, file_hash, save_json, tensorize
from scripts.verify_heldout_bundle import ATOL_M, RTOL, SITE_COUNTS, parameter_digest
import src
from src.data.actions import compute_action_labels
from src.engine.trainer import _compute_train_action_pos_weights, set_deterministic_seed
from src.losses.losses import JointPredictionLoss
from src.models.action_and_decoder import ActionConditionedDecoder
from src.models.predictor import UnstructuredShortPosePredictor

ROOT=Path(src.__file__).resolve().parents[1]
REV='e8aafb53a74c0e0dcc10610ed9e74064c34bfaad'
POSE='b493a47536664ddb4fd66603c1bbf6b2c2e33f553520382027b3ae96b53447d0'
DATA='0b276b34ab1b01dae91f67c56bfaea5cc931511b05b4fc1c92686a891e6f6801'
COHORT='f5ee511a54ad818499d3dde6213f7f3b49332943ff2fc93d93aed87d316dbb91'
ARMS={'no_head':'none','auxiliary':'aux_only','end_to_end':'e2e','detached':'stop_grad'}
INPUTS=('x_m','x_p','q_vec','ego_type_oh')
P=Protocol()


def require(ok,message):
    if not ok: raise ValueError(message)


class ActionDropout(nn.Dropout):
    """Private repeatable mask; adding the action head does not shift trajectory dropout."""
    step_seed=0
    def forward(self,x):
        if not self.training: return x
        g=torch.Generator(device=x.device).manual_seed(self.step_seed)
        keep=torch.rand(x.shape,device=x.device,generator=g) >= self.p
        return x*keep.to(x.dtype)/(1-self.p)


def model_for(arm):
    mode=ARMS[arm]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(P.seed)
        model=UnstructuredShortPosePredictor('M1')
        if mode!='none':
            shared=model.decoder.state_dict()
            torch.manual_seed(P.seed)
            reference=ActionConditionedDecoder(action_mode='e2e').state_dict()
            dec=ActionConditionedDecoder(action_mode=mode)
            dec.load_state_dict({k:shared[k] if k in shared else reference[k] for k in dec.state_dict()})
            require(isinstance(dec.act_head[3],nn.Dropout),'Frozen head dropout layout changed')
            dec.act_head[3]=ActionDropout(dec.act_head[3].p)
            model.decoder=dec
        model.action_mode=mode
    return model


def prepare(data,pose):
    require(subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()==REV,'Wrong source revision')
    require(not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain','--untracked-files=no']),'Modified frozen source')
    require(file_hash(pose)==POSE,'Unexpected pose cache')
    groups,_,data_sha=load_site_windows(data)
    tensors,idx,keys,_,_,_=tensorize(groups,pose)
    require(data_sha==DATA and json_digest(keys)==COHORT and len(set(keys))==1482,'Wrong data, ID order or cohort')
    require({s:len(v) for s,v in idx.items()}==SITE_COUNTS,'Unexpected site counts')
    xy=np.stack([r.xy for site in sorted(groups) for r in groups[site]])
    event,valid=zip(*(compute_action_labels(tr[:5],np.ones(5),tr[5:],np.ones(30)) for tr in xy))
    labels=np.stack(event); masks=np.asarray(valid,dtype=np.float32)
    require(np.isin(labels,(0,1)).all() and np.array_equal(masks,np.ones(1482)),'Invalid action targets')
    require(labels.sum(0).tolist()==[49,53,214,208,975],'Unexpected registered event counts')
    tensors['y_act']=torch.from_numpy(labels);tensors['m_act']=torch.from_numpy(masks)
    return tensors,idx,keys


def design(tensors,index,keys):
    valid=tensors['m_act'].numpy().astype(bool);y=tensors['y_act'].numpy()
    folds={}
    for s in index:
        others,train,test=split_indices(index,keys,s)
        folds[s]={'train_sites':others,'train_windows':len(train),'test_windows':len(test),
                  'train_ids_sha256':json_digest([keys[i] for i in train]),'test_ids_sha256':json_digest([keys[i] for i in test]),
                  'test_clips':len({keys[i][0] for i in test}),'valid_test_windows':int(valid[test].sum()),
                  'test_positive_counts':y[test][valid[test]].sum(0).astype(int).tolist(),
                  'training_positive_weights':_compute_train_action_pos_weights(tensors['y_act'][train],tensors['m_act'][train]).tolist()}
    return {'experiment':'individual_only_joint_actions_loso_seed42','freeze':'Registered before evaluating this experiment. Four physical development sites previously studied.',
            'frozen_git_revision':REV,'driver_sha256':file_hash(Path(__file__)),'pose_sha256':POSE,'data_sha256':DATA,
            'ordered_keys_sha256':COHORT,'truth_sha256':hashlib.sha256(y.tobytes()+tensors['m_act'].numpy().tobytes()).hexdigest(),
            'cohort':'Original 1,482 five-frame windows, 47 clips; disjoint 35-frame blocks within each clip/person; 10 Hz, 0.4 s observed span and all 30 future ground points through 3.0 s.',
            'training':asdict(P),'models':ARMS,'inputs':list(INPUTS),
            'action_labels':list(ACTIONS),'action_support':{'by_class':y[valid].sum(0).astype(int).tolist(),'valid_windows':int(valid.sum())},
            'labels':'Unmodified five observable event rules applied to float64 label-derived history and future. Original window-level validity applies to all five classes. Targets and validity are only used for training loss and evaluation.',
            'architecture':'Common quality-gated M1 individual motion/pose/ego type; maps and neighbor graph disabled in all arms. no_head has no action parameters and no head scores. Auxiliary fits the head but supplies zero action condition. End_to_end and detached use identical logit-to-trajectory bridge; only the latter detaches predicted action probabilities before the bridge. Shared backbone still receives trajectory and action supervision.',
            'initialization':'All arms use identical seed-42 M1 backbone and common trajectory decoder state; supervised heads share initial values; conditional bridges share initial values. Same batches and backbone/decoder dropout stream; separate per-minibatch seed for head dropout (420000+epoch*10000+batch).',
            'optimization':'Frozen 28-epoch final state, AdamW and cosine schedule. Uniform shuffle of training windows, permutation seed 49. JointPredictionLoss defaults: beta=.50, mode weight=.25, supervised BCE weight=.15. Fold-specific positive weights are the project dampened ratio computed exclusively from training labels. No validation/tuning/early stopping.',
            'primary':'Detached minus end_to_end paired highest-probability ADE (metres), site means equally weighted; positive means greater error for detached. Report the difference regardless of sign.',
            'action_calibration':'Companion measure is detached minus end_to_end mean binary Brier score on valid windows, equally weighting five classes then four sites. Sigmoid probabilities are used without post-hoc calibration.',
            'secondary':'Exact 3s top1 FDE, per-class equal-score-group AP, F1 at >=0.5, class-macro and four-site macro; no positive class yields null F1/AP and is absent from that site macro. Auxiliary vs no_head trajectory, end_to_end vs auxiliary trajectory and action, and detached vs end_to_end trajectory and action. AP and F1 may be unstable for rare start and stop.',
            'interval':'Primary ADE and companion Brier: 20,000 paired within-site whole-clip bootstrap draws, seed 2026; four sites equally weighted. Conditional on these trained models; descriptive, no seed or unseen-site inference.',
            'folds':folds}


def fit(arm,tensors,ids):
    model=model_for(arm)
    set_deterministic_seed(P.seed,torch.device('cpu'))
    weight=_compute_train_action_pos_weights(tensors['y_act'][ids],tensors['m_act'][ids])
    loss_fn=JointPredictionLoss(use_action_loss=arm!='no_head',act_pos_weights=weight)
    opt=torch.optim.AdamW(model.parameters(),lr=P.learning_rate,weight_decay=P.weight_decay)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=P.epochs,eta_min=P.learning_rate*P.min_lr_ratio)
    ids=torch.as_tensor(ids,dtype=torch.long);g=torch.Generator().manual_seed(P.seed+7)
    model.train()
    for ep in range(P.epochs):
        batch_ids=ids[torch.randperm(len(ids),generator=g)]
        for j,ix in enumerate(batch_ids.split(P.batch_size)):
            if arm!='no_head':model.decoder.act_head[3].step_seed=420000+ep*10000+j
            opt.zero_grad(set_to_none=True)
            pred=model({k:tensors[k][ix] for k in INPUTS})
            loss=loss_fn(pred,tensors['fut_xy'][ix],tensors['fut_mask'][ix],tensors['y_act'][ix],tensors['m_act'][ix])
            loss['loss'].backward()
            nn.utils.clip_grad_norm_(model.parameters(),P.gradient_clip,error_if_nonfinite=True)
            opt.step()
        sched.step()
    return model.state_dict()


def predict(arm,state,data,rows,batch_size):
    model=model_for(arm).eval(); model.load_state_dict(state,strict=True)
    names=['trajs','mode_logits','mode_probs']+(['act_logits','act_probs'] if arm!='no_head' else [])
    values={k:[] for k in names}
    with torch.inference_mode():
        for ix in torch.as_tensor(rows).split(batch_size):
            pred=model({k:data[k][ix] for k in INPUTS})
            for k in names: values[k].append(pred[k].numpy())
    a={k:np.concatenate(v) for k,v in values.items()}
    require(all(np.isfinite(v).all() for v in a.values()),'Nonfinite forward')
    np.testing.assert_allclose(a['mode_probs'].sum(1),1.,atol=1e-6)
    if arm!='no_head':
        np.testing.assert_allclose(torch.tensor(a['act_logits']).sigmoid().numpy(),a['act_probs'],atol=1e-7)
    top=a['trajs'][np.arange(len(rows)),a['mode_probs'].argmax(1)].astype(np.float64)
    d=np.linalg.norm(top-data['fut_xy'][rows].numpy().astype(np.float64),axis=-1)
    a['errors']=np.column_stack([d.mean(1),d[:,-1]])
    return a


def action_summary(y,score):
    per={}
    for j,name in enumerate(ACTIONS):
        v=class_scores(y[:,j],score[:,j])
        per[name]={**v,'positives':int(y[:,j].sum()),'brier':float(np.mean((score[:,j]-y[:,j])**2))}
    return {'classes':per,'macro_ap':float(np.mean([v['ap'] for v in per.values() if v['ap'] is not None])),
        'macro_f1':float(np.mean([v['f1'] for v in per.values() if v['f1'] is not None])),
        'macro_brier':float(np.mean([v['brier'] for v in per.values()]))}


def save_npz(path,arrays):
    def write(tmp):
        with tmp.open('wb') as f: np.savez_compressed(f,**arrays)
    atomic_write(path,write)


def run_fold(out,registration,backend,data,index,keys,s,verify_only):
    others,train,test=split_indices(index,keys,s)
    folder=out/s
    truth=data['y_act'][test].numpy().astype(bool); valid=data['m_act'][test].numpy()>0.5
    id_list=[keys[i] for i in test]
    fold={'train_sites':others,'train_keys':[keys[i] for i in train],'test_keys':id_list}
    gt={'clip':np.array([k[0] for k in id_list]),'person_id':np.array([k[1] for k in id_list]),'t0':np.array([k[2] for k in id_list]),
        'y_act':truth,'m_act':valid,'valid_by_class':np.repeat(valid[:,None],len(ACTIONS),axis=1),'fut_xy':data['fut_xy'][test].numpy(),
        'fut_mask':data['fut_mask'][test].numpy()}
    if not verify_only:
        if not (folder/'fold_keys.json').exists(): save_json(folder/'fold_keys.json',fold)
        if not (folder/'targets.npz').exists(): save_npz(folder/'targets.npz',gt)
    require(json_digest(json.loads((folder/'fold_keys.json').read_text()))==json_digest(fold),'Mismatch in fold keys')
    with np.load(folder/'targets.npz') as z:
        require(set(z.files)==set(gt),'Mismatch in archived target fields')
        for k,v in gt.items(): np.testing.assert_array_equal(z[k],v)
    rows={}; summary={}
    for arm in ARMS:
        identity={'registration_sha256':json_digest(registration),'heldout':s,'arm':arm,'mode':ARMS[arm],'training_backend':backend,
                  **{k:registration['folds'][s][k] for k in ('train_ids_sha256','test_ids_sha256','training_positive_weights')},
                  'initial_state_sha256':parameter_digest(model_for(arm).state_dict())}
        path=folder/(arm+'.pt'); pred_path=folder/(arm+'.npz')
        if not path.exists() and not verify_only:
            print(f'Fitting {s}/{arm} (CPU, seed 42, 28 epochs)',flush=True)
            state=fit(arm,data,train)
            atomic_write(path,lambda tmp:torch.save({'identity':identity,'model_state':state},tmp))
        ck=torch.load(path,map_location='cpu',weights_only=True)
        stored=ck['identity']
        require({k:v for k,v in stored.items() if k!='training_backend'}=={k:v for k,v in identity.items() if k!='training_backend'},'Wrong checkpoint')
        require(stored['training_backend']['device']=='cpu','Wrong training device')
        if not verify_only: require(stored['training_backend']==backend,'Changed training backend on resume')
        if not pred_path.exists() and not verify_only: save_npz(pred_path,predict(arm,ck['model_state'],data,test,P.batch_size))
        replay=predict(arm,ck['model_state'],data,test,31)
        with np.load(pred_path) as z:
            require(set(z.files)==set(replay),'Prediction fields changed')
            saved={k:z[k] for k in z.files}
        for k in replay:
            np.testing.assert_allclose(replay[k],saved[k],atol=ATOL_M,rtol=RTOL,err_msg=f'{s}/{arm}/{k}')
        e=saved['errors']; rows[arm]={'ade':e[:,0]}
        metrics={'ade_m':float(e[:,0].mean()),'fde_m':float(e[:,1].mean()),'action_head':None}
        if arm!='no_head':
            prob=saved['act_probs'].astype(np.float64)
            metrics['action_head']=action_summary(truth[valid],prob[valid])
            rows[arm]['brier']=np.mean((prob-truth.astype(np.float64))**2,axis=1)
        summary[arm]=metrics
    return summary,rows,gt['clip']


def aggregate(by_site):
    result={};arms=list(ARMS)
    for arm in arms:
        site=[item[0][arm] for item in by_site.values()]
        result[arm]={k:float(np.mean([v[k] for v in site])) for k in ('ade_m','fde_m')}
        result[arm]['action_head']=None if arm=='no_head' else {k:float(np.mean([v['action_head'][k] for v in site])) for k in ('macro_ap','macro_f1','macro_brier')}
    summary={'sites':{s:x[0] for s,x in by_site.items()}}
    if set(by_site)!=set(SITE_COUNTS): return summary
    rng=np.random.default_rng(2026); replicates=[]; means=[]
    for s in sorted(by_site):
        _,rows,clips=by_site[s]
        v=np.column_stack([rows['detached'][k]-rows['end_to_end'][k] for k in ('ade','brier')])
        blocks=[v[clips==c] for c in sorted(set(clips.tolist()))]
        w=rng.multinomial(len(blocks),np.full(len(blocks),1/len(blocks)),size=20000)
        replicates.append((w@np.stack([b.sum(0) for b in blocks]))/(w@np.array([len(b) for b in blocks]))[:,None])
        means.append(v.mean(0))
    ci=np.quantile(np.mean(replicates,axis=0),[.025,.975],axis=0)
    comparisons={}
    for a,b in [('auxiliary','no_head'),('end_to_end','auxiliary'),('detached','end_to_end')]:
        record={k:result[a][k]-result[b][k] for k in ('ade_m','fde_m')}
        if b!='no_head':
            record.update({k:result[a]['action_head'][k]-result[b]['action_head'][k] for k in ('macro_ap','macro_f1','macro_brier')})
        comparisons[a+'_minus_'+b]=record
    summary.update(four_site_macro=result,secondary_contrasts=comparisons,primary_paired={k:{'detached_minus_e2e':float(np.mean(means,axis=0)[i]),'paired_clip_95pct':ci[:,i].tolist()} for i,k in enumerate(('ade_m','brier'))})
    return summary


def bundle(out,sites,registration,summary):
    paths=[out/'preregistration.json',out/'driver.py']
    tag=sites[0] if len(sites)==1 else 'all' if set(sites)==set(SITE_COUNTS) else '_'.join(sites)
    report=out/f'report_{tag}.json'
    save_json(report,{'registration_sha256':json_digest(registration),'sites_completed':sites,**summary})
    paths.append(report)
    for s in sites:
        paths += [out/s/k for k in ['fold_keys.json','targets.npz',*[a+ext for a in ARMS for ext in ('.pt','.npz')]]]
    def write(tmp):
        with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
            manifest={}
            for p in paths:
                name=p.relative_to(out).as_posix(); z.write(p,name); manifest[name]=file_hash(p)
            z.writestr('sha256.json',json.dumps(manifest,sort_keys=True))
        with zipfile.ZipFile(tmp) as z: require(z.testzip() is None,'Archive CRC failed')
    atomic_write(out/f'joint_actions_{tag}.zip',write)


def main(args):
    torch.set_num_threads(min(4,len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count() or 1))
    torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
    data,idx,keys=prepare(args.data,args.pose)
    plan=design(data,idx,keys)
    if args.mode=='register':
        if args.registration.exists():
            require(json.loads(args.registration.read_text())==plan,'Refusing to replace different registration')
        else: save_json(args.registration,plan)
        print('Registered',json_digest(plan),flush=True);return
    require(json.loads(args.registration.read_text())==plan,'Preregistration or frozen inputs changed')
    sites=sorted(args.sites or idx)
    require(len(sites)==len(set(sites)) and set(sites)<=set(idx),'Unexpected/duplicate site')
    v=args.mode=='verify'
    backend={'device':'cpu','torch':str(torch.__version__),'numpy':np.__version__,'platform':platform.platform(),
             'threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),'deterministic':torch.are_deterministic_algorithms_enabled(),
             'cpu':platform.machine()}
    if Path('/proc/cpuinfo').is_file():
        backend['cpu']=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),backend['cpu'])
    if not v:
        require(Path('/kaggle/working').is_dir() and Path('/kaggle/input').is_dir(),'Training is restricted to Kaggle')
        require(not torch.cuda.is_available(),'CPU-only experiment; hide GPUs')
        args.out.mkdir(parents=True,exist_ok=True)
        saved_plan=args.out/'preregistration.json'
        if saved_plan.exists(): require(json.loads(saved_plan.read_text())==plan,'Output belongs to a different registration')
        else: save_json(saved_plan,plan)
        driver=args.out/'driver.py'
        if driver.exists(): require(file_hash(driver)==plan['driver_sha256'],'Output contains a different implementation')
        else: atomic_write(driver,lambda tmp:tmp.write_bytes(Path(__file__).read_bytes()))
    results={}
    for s in sites:
        results[s]=run_fold(args.out,plan,backend,data,idx,keys,s,v)
        if not v: bundle(args.out,[s],plan,aggregate({s:results[s]}))
    summary=aggregate(results)
    if not v and len(sites)>1:bundle(args.out,sites,plan,summary)
    print(json.dumps(summary.get('primary_paired',{'checked_sites':sites})),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','pose','registration'): p.add_argument(name,type=Path)
    p.add_argument('--mode',choices=('register','fit','verify'),required=True)
    p.add_argument('--out',type=Path,default=Path('/kaggle/working/individual_joint_actions'))
    p.add_argument('--sites',choices=sorted(SITE_COUNTS),nargs='+')
    main(p.parse_args())
