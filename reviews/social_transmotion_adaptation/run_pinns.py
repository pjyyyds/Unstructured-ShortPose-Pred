"""Train the registered target-only PINNS adaptation of the authors' JRDB model."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import zipfile

import numpy as np
import torch

from check_protocol import LABELS, ORDER, POSE, SOURCE, build_observed, model_class
from scripts.analyze_heldout_pairs import intervals,paired_clip_draws
from scripts.benchmark_heldout_sites import load_site_windows
from scripts.run_heldout_seed_sensitivity import SITE_COUNTS,json_digest,split_indices
from scripts.train_pinns_heldout_sites import atomic_write,file_hash,save_json

ROOT=Path(__file__).resolve().parent
REVISION='7695ba335f7cba7f41660d0edc66a57cd52da08a'
REPO='https://github.com/vita-epfl/social-transmotion'
LOSS_SHA='7b0f40c3b3ea1efca79815c75cc1fff2abce8d5567277d113c5026cba1ab0e63'
LICENSE_SHA='8486a10c4393cee1c25392769ddd3b2d6c242d6ec7928e1414efff7dfb2f07ef'
ADAPTED_SHA='32a6f06d7d685c3558ccdcd50186ca5ba192ab3ccd0fdda90c20156aceab0d8d'
ARMS=('traj_box','traj')
PAIRS=(('traj_box','last_two'),('traj','last_two'),('traj_box','traj'))
SEED=42
BATCH=16
EPOCHS=28


def check(test,message):
    if not test:raise ValueError(message)


def prepare(data,pose,upstream):
    for file,ref in [('model_jrdb.py',SOURCE),('utils/metrics.py',LOSS_SHA),('LICENSE',LICENSE_SHA)]:
        check(file_hash(upstream/file)==ref,'Changed upstream member '+file)
    check(file_hash(pose)==POSE,'Wrong pose/box cache')
    groups,_,source_sha=load_site_windows(data)
    check(source_sha==LABELS,'Wrong PINNS source')
    windows=[row for site in sorted(groups) for row in groups[site]]
    keys=[(r.clip,r.person_id,r.t0) for r in windows]
    check(json_digest(keys)==ORDER and len(set(keys))==1482,'Wrong official window order')
    xy=np.stack([r.xy for r in windows])
    history=xy[:,:5].copy()
    truth=xy[:,5:].copy()
    with np.load(pose,allow_pickle=False) as cache:
        ck=list(zip(cache['clip'].tolist(),cache['person_id'].tolist(),cache['t0'].tolist()))
        check(len(set(ck))==len(ck)==len(keys) and set(ck)==set(keys),'Cache identity mismatch')
        lookup={k:i for i,k in enumerate(ck)}
        order=[lookup[k] for k in keys]
        np.testing.assert_array_equal(history,cache['hist_pos_relative'][order])
        check(np.all(cache['hist_mask'][order]==1),'Missing motion observation')
        boxes=cache['hist_bbox'][order]
    check(np.isfinite(truth).all() and truth.shape==(1482,30,2),'Incomplete ground-truth futures')
    tokens={arm:torch.from_numpy(build_observed(history,boxes,arm=='traj_box')) for arm in ARMS}
    check(torch.equal(tokens['traj'][:,:,0],tokens['traj_box'][:,:,0]),'Ground token changed across arms')
    index={}
    first=0
    for s in sorted(groups):
        index[s]=np.arange(first,first+len(groups[s]))
        first+=len(groups[s])
    check({s:len(v) for s,v in index.items()}==SITE_COUNTS,'Unverified site sizes')
    folds={}
    for s in sorted(index):
        others,train,test=split_indices(index,keys,s)
        folds[s]=dict(train_sites=others,train_windows=len(train),test_windows=len(test),
                      test_clips=len({keys[i][0] for i in test}),train_keys=json_digest([keys[i] for i in train]),
                      test_keys=json_digest([keys[i] for i in test]))
    plan=dict(name='PINNS target-only Social-Transmotion official JRDB code adaptation',
              code_revision=REVISION,upstream_repository=REPO,upstream_model_sha256=SOURCE,
              temporal_adapter_model_sha256=ADAPTED_SHA,upstream_loss_sha256=LOSS_SHA,license_sha256=LICENSE_SHA,
              source_notice='Social-Transmotion, Saadatnejad, Gao, Messaoud, Alahi (ICLR 2024), VITA EPFL, GNU AGPL v3. See THIRD_PARTY.md and the upstream license. The temporal adapter modifies 9 and 21 to 5 and 35. Original author datasets and pretrained weights are not used.',
              registered_with='outcomes of this external-method adaptation unseen; four-site PINNS development results for other predictors seen',
              data_sha256=LABELS,cache_sha256=POSE,keys_sha256=ORDER,folds=folds,
              inputs={'targets':1,'neighbor_people':0,'joints':0,'ground_xy':'five real samples; last-centered metres',
                      'traj_box':'five observed left/top/width/height in pixels, each token minus its final observed box times .25',
                      'traj':'identical full network with the box token values zeroed throughout fitting and evaluation'},
              fit={'seed':SEED,'epochs':EPOCHS,'batch_size':BATCH,'optimizer':'Adam defaults, no weight decay',
                   'lr':'1e-4 through epoch 23 inclusive, then 1e-5 (official 80%-of-budget rule)',
                   'loss':'Unmodified official MSE_LOSS: mean per-step Euclidean error of one target path times 100',
                   'gradient_norm_clip':1,'batch_order_seed':49,'rotation_augmentation':False,'precision':'float32, no AMP or TF32',
                   'worker':'same worker/device and common init, batch permutation, dropout/random streams in all eight fits; CPU scoring from final weights',
                   'checkpoint_selection':'28th and final epoch; no held-out tuning or early stopping'},
              adaptation='One target, no neighboring people or skeleton, no rotation augmentation, 10 Hz 5+30, 28 epochs vs native 100, seed42 vs native0. Original learned layers and author ADE loss. Three training sites per fold. This is not a native JRDB score.',
              primary='signed difference in deterministic traj_box ADE minus direct last-two-position CV ADE in metres, equal site means',
              contrasts=PAIRS,reference='last two measured positions extrapolated unchanged 30 steps over 0.1s steps, no training',
              scoring='one primary-person path per method; ADE is mean 30-step distance and FDE is step 30 at 3 seconds. No modal probabilities or oracle metrics. Keep all outcomes. No new-site generalization claim.',
              uncertainty='20,000 within-site paired whole-clip bootstrap replicates, PCG64 seed2026, combine four site means equally; intervals conditional on these four sites and fixed training seed.',
              code_sha256={f:file_hash(ROOT/f) for f in ('run_pinns.py','check_protocol.py','THIRD_PARTY.md')})
    return tokens,truth,history,index,keys,json.loads(json.dumps(plan))


def create(cls,device):
    # Reset all randomness before constructing each condition, and again before fitting.
    torch.manual_seed(SEED)
    net=cls(tok_dim=40,nhid=128,nhead=4,dim_feedfwd=1024,nlayers_local=6,nlayers_global=3,
            dropout=.1,output_scale=1.,obs_and_pred=35,num_tokens=2,device=str(device))
    return net.to(device).float()


def state_digest(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def predict(net,x):
    net.eval()
    with torch.no_grad():
        result=[net(b,torch.zeros((len(b),1),dtype=torch.bool))[:,5:,0].numpy().copy() for b in x.split(BATCH)]
    path=np.concatenate(result)
    check(path.shape==(len(x),30,2) and np.isfinite(path).all(),'Invalid path')
    return path


def fit(cls,loss_fn,device,x,y,identity,checkpoint):
    # Only the three training-site subsets are passed to this function.
    net=create(cls,device)
    initial=state_digest(net.state_dict())
    opt=torch.optim.Adam(net.parameters(),lr=1e-4)
    torch.manual_seed(SEED)
    perm=torch.Generator().manual_seed(49)
    logs=[]
    if checkpoint.exists():
        saved=torch.load(checkpoint,weights_only=True,map_location='cpu')
        check(saved['identity']==identity and saved['initial_sha256']==initial,'Resume ID or initialization differs')
        net.load_state_dict(saved['state']);opt.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['cpu_rng'])
        if device.type=='cuda':torch.cuda.set_rng_state(saved['device_rng'],device)
        perm.set_state(saved['perm']);logs=saved['logs']
        check(len(logs)<=EPOCHS,'Invalid recovery epoch')
    for epoch in range(len(logs),EPOCHS):
        net.train()
        lr=1e-4 if epoch<23 else 1e-5
        for group in opt.param_groups: group['lr']=lr
        total=0.
        for ids in torch.randperm(len(x),generator=perm).split(BATCH):
            xb=x[ids].to(device);yb=y[ids].to(device)
            mask=torch.zeros((len(ids),1),dtype=torch.bool,device=device)
            opt.zero_grad(set_to_none=True)
            out=net(xb,mask,metamask=True)[:,5:]
            loss=loss_fn(out,yb)
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step()
            total+=loss.item()*len(ids)
        logs.append({'epoch':epoch+1,'lr':lr,'windows':len(x),'objective':total/len(x)})
        payload=dict(identity=identity,initial_sha256=initial,state=net.state_dict(),optimizer=opt.state_dict(),
                     logs=logs,cpu_rng=torch.get_rng_state(),perm=perm.get_state(),
                     device_rng=torch.cuda.get_rng_state(device) if device.type=='cuda' else None)
        atomic_write(checkpoint,lambda p:torch.save(payload,p))
    return dict(identity=identity,initial_sha256=initial,model_state={k:v.cpu() for k,v in net.state_dict().items()},epochs=logs)


def metrics(path,target):
    error=np.linalg.norm(path.astype(np.float64)-target,axis=-1)
    return np.stack([error.mean(1),error[:,-1]],1)


def summarize(site_errors,clips):
    names=(*ARMS,'last_two')
    local={s:{a:dict(zip(('ade_m','fde_m'),v[a].mean(0).tolist())) for a in names} for s,v in site_errors.items()}
    macro={a:{k:float(np.mean([v[a][k] for v in local.values()])) for k in ('ade_m','fde_m')} for a in names}
    paired={}
    for a,b in PAIRS:
        rng=np.random.default_rng(2026)
        strata={}; reps=[]
        for s,values in sorted(site_errors.items()):
            d=values[a]-values[b]
            draw=paired_clip_draws(d,clips[s],rng);reps.append(draw)
            strata[s]=dict(delta_ade_fde_m=d.mean(0).tolist(),interval95_m=intervals(draw))
        paired[a+'_minus_'+b]=dict(by_site=strata,site_macro_delta_ade_fde_m=[macro[a][k]-macro[b][k] for k in ('ade_m','fde_m')],
                                   site_macro_interval95_m=intervals(np.mean(reps,0)))
    return dict( by_site=local,four_site_equal=macro,paired=paired)


def main(args):
    tokens,y,h,idx,keys,plan=prepare(args.data,args.pose,args.upstream)
    cls,modified=model_class(args.upstream/'model_jrdb.py',True)
    check(modified==ADAPTED_SHA,'Temporal adapter differs')
    if args.mode in ('check','register'):
        if args.mode=='register':
            if args.plan.exists():check(json.loads(args.plan.read_bytes())==plan,'Existing plan differs')
            else:save_json(args.plan,plan)
        print(json_digest(plan),{s:len(ids) for s,ids in idx.items()});return
    check(Path('/kaggle/working').is_dir() and json.loads(args.plan.read_bytes())==plan,'Training requires Kaggle and the fixed plan')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    device=torch.device(args.device)
    check(device.type=='cpu' or torch.cuda.is_available(),'CUDA unavailable')
    torch.set_num_threads(min(4,len(os.sched_getaffinity(0))));torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
    if device.type=='cuda':
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda,'enable_cudnn_sdp'):torch.backends.cuda.enable_cudnn_sdp(False)
    worker={'device':str(device),'torch':str(torch.__version__),'threads':torch.get_num_threads(),
            'model':torch.cuda.get_device_name(device) if device.type=='cuda' else platform.machine(),
            'worker_fingerprint':hashlib.sha256(platform.node().encode()).hexdigest()}
    spec=importlib.util.spec_from_file_location('official_metrics',args.upstream/'utils/metrics.py')
    official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    runtime=dict(plan=plan,worker=worker)
    record=out/'run.json'
    if record.exists():check(json.loads(record.read_bytes())==runtime,'Different design or worker in output')
    else:save_json(record,runtime)
    members=[record]
    for name,src in [('LICENSE',args.upstream/'LICENSE'),*[(k,ROOT/k) for k in plan['code_sha256']]]:
        dst=out/name;atomic_write(dst,lambda p:p.write_bytes(src.read_bytes()));members.append(dst)
    results={};clips={};initials=set()
    truth=torch.from_numpy(y.astype(np.float32))[:,:,None,:]
    cv=h[:,-1:]+(h[:,-1]-h[:,-2])[:,None]*np.arange(1,31)[None,:,None]
    for site in sorted(idx):
        _,train,test=split_indices(idx,keys,site)
        folder=out/site;folder.mkdir(exist_ok=True)
        clips[site]=np.array([keys[i][0] for i in test]);results[site]={'last_two':metrics(cv[test],y[test])}
        private={'clip':clips[site],'person_id':np.array([keys[i][1] for i in test]),'t0':np.array([keys[i][2] for i in test]),
                 'last_two_errors':results[site]['last_two']}
        for arm in ARMS:
            identity=dict(plan_sha256=json_digest(plan),worker=worker,site=site,arm=arm,split=plan['folds'][site])
            last=folder/(arm+'.pt');restart=folder/(arm+'.resume.pt')
            if not last.exists():
                state=fit(cls,official.MSE_LOSS,device,tokens[arm][train],truth[train],identity,restart)
                atomic_write(last,lambda p:torch.save(state,p))
            ck=torch.load(last,weights_only=True,map_location='cpu')
            check(ck['identity']==identity and len(ck['epochs'])==EPOCHS,'Unverified final checkpoint')
            net=create(cls,torch.device('cpu'))
            check(state_digest(net.state_dict())==ck['initial_sha256'],'CPU and fit initialization differ')
            net.load_state_dict(ck['model_state'],strict=True)
            pred=predict(net,tokens[arm][test]);results[site][arm]=metrics(pred,y[test])
            private[arm+'_paths']=pred;private[arm+'_errors']=results[site][arm]
            initials.add(ck['initial_sha256'])
            members.append(last)
            restart.unlink(missing_ok=True)
            print('Completed',site,arm,flush=True)
        npz=folder/'private_windows.npz'
        def write(tmp):
            with tmp.open('wb') as stream:np.savez_compressed(stream,**private)
        atomic_write(npz,write);members.append(npz)
    check(len(initials)==1,'Models lack one shared initialization')
    aggregate=out/'scores.json'
    save_json(aggregate,dict(plan_sha256=json_digest(plan),source_revision=REVISION,site_windows=SITE_COUNTS,
                             seed=SEED,fit_count=8,initial_sha256=initials.pop(),**summarize(results,clips)))
    members.append(aggregate)
    def bundle(tmp):
        with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
            for p in members:z.write(p,p.relative_to(out))
            z.writestr('sha256.json',json.dumps({p.relative_to(out).as_posix():file_hash(p) for p in members},sort_keys=True))
        with zipfile.ZipFile(tmp) as z:check(z.testzip() is None,'Bad output CRC')
    atomic_write(out/'social_transmotion_pinns.zip',bundle)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('data','pose','upstream','plan'):p.add_argument(k,type=Path)
    p.add_argument('--mode',choices=('check','register','fit'),required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    p.add_argument('--out',type=Path,default=Path('/kaggle/working/social_transmotion_pinns'))
    main(p.parse_args())
