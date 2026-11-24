"""Same-worker, fixed-cohort M1 vs M2b with and without real current neighbors."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
import zipfile
import numpy as np
import torch

from scripts.analyze_heldout_pairs import SITE_COUNTS, metric_dict
from scripts.train_pinns_heldout_sites import (Protocol, file_hash, load_site_windows, tensorize,
                                             source_hash, UnstructuredShortPosePredictor, JointPredictionLoss)
from src.data.pinns_extractor import _parse_pinns_label_file, _check_right_handed_homography
from src.data.scene_graph import build_neighbor_edges

LABEL='0b276b34ab1b01dae91f67c56bfaea5cc931511b05b4fc1c92686a891e6f6801'
POSE='b493a47536664ddb4fd66603c1bbf6b2c2e33f553520382027b3ae96b53447d0'
KEYS='f5ee511a54ad818499d3dde6213f7f3b49332943ff2fc93d93aed87d316dbb91'
TREE='e61a2310793e900c6188588c750e47c44e23fbf41f429fb94ccf8b2ea88fd094'
LOCK='51c70ae3866ddd14b327d00bfa91aed665e0e6c703da8eba1a96d9e7dbd4f057'
STRUCTURE='8f3d6c78e5e4bc40091db485729774e9b53e5402e32f65a3091084c0226506b3'
ARMS=('M1','M2b_real','M2b_empty')
CONTROLS=('real_edges_off','real_drop_car','real_drop_people')
PAIRS=(('M2b_real','M1'),('M2b_real','M2b_empty'),('M2b_empty','M1'))+tuple((c,'M2b_real') for c in CONTROLS)
COMMON=('x_m','ego_type_oh','x_p','q_vec')
GRAPH=('nbr_edges','nbr_type_oh','nbr_mask')

def digest(raw):return hashlib.sha256(raw).hexdigest()

def log_hash(x):return digest(json.dumps(x,separators=(',',':')).encode())

def array_hash(named):
    h=hashlib.sha256()
    for k,v in sorted(named.items()):
        a=np.ascontiguousarray(v)
        h.update(k.encode()+b'\0'+a.dtype.str.encode()+b'\0'+str(a.shape).encode()+b'\0'+a.tobytes())
    return h.hexdigest()

def typed_frame(rows):
    out={}
    for row in rows:
        key=row['type'],int(row['id'])
        xy=np.array([row['wx'],row['wy']],dtype=np.float64)
        if key in out or key[0] not in ('Person','Car') or not np.isfinite(xy).all():
            raise ValueError('Unexpected annotated object, position or repeated typed identity')
        out[key]=xy
    return out

def graph(history,ego_id,expect_xy):
    if len(history)!=5:raise ValueError('Exactly five history snapshots required')
    ego=('Person',ego_id)
    ego_xy=np.stack([snap[ego] for snap in history])
    np.testing.assert_allclose(ego_xy-ego_xy[-1],expect_xy,rtol=0,atol=1e-8)
    now=history[-1]
    def state(k):
        count=sum(k in snap for snap in history)
        gap=next((d for d in range(1,5) if k in history[-1-d]),0)
        v=(now[k]-history[-1-gap][k])/(.1*gap) if gap else np.zeros(2)
        return v,gap,count/5
    ego_v,ego_gap,rho=state(ego)
    pool=[];coincident=0
    for kind,ident in now:
        if (kind,ident)==ego:continue
        delta=now[kind,ident]-now[ego]
        dist=float(np.hypot(*delta))
        if dist<=1e-5:coincident+=1
        if 1e-5<dist<=10:pool.append((dist,kind,ident))
    pool.sort() # deterministic ties by source class, ID
    keep=pool[:12]
    edge=np.zeros((12,9),np.float32);types=np.zeros((12,8),np.float32);mask=np.zeros(12,np.float32)
    pids=np.full(12,-1,np.int64);source=np.full(12,-1,np.int8);lags=np.full(12,-1,np.int8)
    for j,(dist,kind,ident) in enumerate(keep):
        k=kind,ident
        v,gap,rho_j=state(k);dp=now[k]-now[ego];valid=bool(ego_gap and gap)
        dv=v-ego_v if valid else np.zeros(2)
        radial=-float(dp@dv)/(dist+1e-6) if valid else 0.
        typ=6 if kind=='Car' else 2 if gap and np.linalg.norm(v)<.20 else 0
        edge[j]=[*dp,*dv,dist,radial,float(valid),rho_j,rho]
        mask[j]=1;types[j,typ]=1;pids[j]=ident;source[j]=int(kind=='Car');lags[j]=gap
        # Frozen Eq. 7 constructor, applied to one selected neighbor to avoid unspecified ties in np.argsort.
        e,t,m=build_neighbor_edges(now[ego],ego_v,bool(ego_gap),rho,np.array([now[k]]),np.array([v]),
                                    [bool(gap)],[rho_j],[typ],max_nbrs=1)
        np.testing.assert_allclose(edge[j],e[0],rtol=1e-6,atol=1e-6)
        np.testing.assert_array_equal(types[j],t[0]);assert m[0]==1
    pre_cap=np.array([sum(k==kind for _,k,_ in pool) for kind in ('Person','Car')]+[coincident],dtype=np.int64)
    return edge,types,mask,pids,source,lags,len(pool),pre_cap

def strata(source):
    person=(source==0).any(1);car=(source==1).any(1)
    return {'empty':~person&~car,'person_only':person&~car,'car_only':~person&car,'mixed':person&car,
            'with_any_person':person,'with_any_car':car}

def prepare(root,pose):
    if file_hash(pose)!=POSE or source_hash()!=TREE:raise ValueError('Wrong pose or frozen implementation')
    groups,_,d=load_site_windows(root)
    tensors,index,keys,_,_,metadata=tensorize(groups,pose)
    if d!=LABEL or metadata['label_and_calibration_digest_sha256']!=LABEL or log_hash(keys)!=KEYS or {s:len(v) for s,v in index.items()}!=SITE_COUNTS:
        raise ValueError('Wrong cohort or source files')
    rows=[r for s in sorted(groups) for r in groups[s]]
    clips=np.asarray([r.clip for r in rows]);outputs=[None]*len(keys)
    for c in sorted(set(clips)):
        calib=json.loads((root/'videos'/c.rsplit('_',1)[0]/'calib.json').read_text())
        parsed=_parse_pinns_label_file(root/'labels'/(c+'.txt'),_check_right_handed_homography(calib),calib['y_min'],calib['y_max'])
        snapshots={t:typed_frame(v) for t,v in parsed.items()}
        for i in np.flatnonzero(clips==c):
            w=rows[i]
            outputs[i]=graph([snapshots[t] for t in range(w.t0-4,w.t0+1)],w.person_id,w.xy[:5])
    edges,types,mask,ids,source,lags,candidates,pre_cap=map(np.stack,zip(*outputs))
    audit={'neighbor_id':ids,'source_class':source,'velocity_gap_ticks':lags,'candidate_counts':candidates,
           'encoded_type':np.where(mask.astype(bool),types.argmax(-1),-1).astype(np.int8)}
    identity=array_hash(audit)
    if identity!=STRUCTURE:raise ValueError('Unexpected typed-slot census '+identity)
    audit['eligible_person_car_and_excluded_coincident_counts']=pre_cap
    tensors.update({k:torch.from_numpy(v) for k,v in zip(GRAPH,(edges,types,mask))})
    def counts(g):
        m=mask[g].astype(bool); encoded=audit['encoded_type'][g]
        return {'windows':len(g),'clips':len(set(clips[g])),'strata':{k:int(v.sum()) for k,v in strata(source[g]).items()},
                'edges':int(m.sum()),'velocity_valid_edges':int(edges[g,:,6].sum()),'truncated_windows':int((candidates[g]>12).sum()),
                'within_radius_before_cap_person_windows':int((pre_cap[g,0]>0).sum()),'within_radius_before_cap_car_windows':int((pre_cap[g,1]>0).sum()),
                'near_coincident_excluded_rows':int(pre_cap[g,2].sum()),'edges_lost_to_cap':int(np.maximum(candidates[g]-12,0).sum()),
                'observed_types':{k:int((encoded==i).sum()) for i,k in [(0,'pedestrian_or_unclassified_speed'),(2,'history_slow_pedestrian'),(6,'car')]}}
    provenance={'pose_sha256':POSE,'labels_calibration_sha256':LABEL,'ordered_keys_sha256':KEYS,'frozen_tree_sha256':TREE,
                'typed_slot_structure_sha256':identity,'runtime_graph_sha256':array_hash(dict(zip(GRAPH,(edges,types,mask)))),
                'by_site':{s:counts(g) for s,g in index.items()},'all':counts(np.arange(len(keys)))}
    return tensors,index,keys,audit,provenance

def initialize(arm):
    torch.manual_seed(42)
    ref=UnstructuredShortPosePredictor('M1')
    common=ref.state_dict()
    if arm=='M1':model=ref
    else:
        torch.manual_seed(42)
        model=UnstructuredShortPosePredictor('M2b')
        st=model.state_dict()
        if not set(common)<=set(st) or any(not k.startswith('interaction.') for k in set(st)-set(common)):
            raise ValueError('Unexpected architecture mismatch')
        for k,v in common.items():
            if v.shape!=st[k].shape:raise ValueError(k)
        model.load_state_dict({**st,**common},strict=True)
    common_hash=array_hash({k:v.numpy() for k,v in common.items()})
    full_hash=array_hash({k:v.numpy() for k,v in model.state_dict().items()})
    return model,common_hash,full_hash

def inputs(tensors,ids,arm):
    batch={k:tensors[k][ids] for k in COMMON}
    if arm=='M1':return batch
    edges,typ,mask=(tensors[k][ids] for k in GRAPH)
    if arm in ('M2b_empty','real_edges_off'): keep=torch.zeros_like(mask)
    elif arm=='real_drop_car':keep=mask*(typ[:,:,6]<.5)
    elif arm=='real_drop_people':keep=mask*(typ[:,:,6]>.5)
    elif arm=='M2b_real':keep=mask
    else:raise ValueError(arm)
    batch.update(nbr_edges=edges*keep[:,:,None],nbr_type_oh=typ*keep[:,:,None],nbr_mask=keep)
    return batch

def train(model,arm,tensors,ids,p):
    model.train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=p.learning_rate,weight_decay=p.weight_decay)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=p.epochs,eta_min=p.learning_rate*p.min_lr_ratio)
    criterion=JointPredictionLoss(use_action_loss=False)
    rng=torch.Generator().manual_seed(p.seed+7);ids=torch.as_tensor(ids)
    losses=[];steps=0;batch_hash=hashlib.sha256()
    def decoder_seed(module,ins):torch.manual_seed(p.seed+200003+steps)
    hook=model.decoder.register_forward_pre_hook(decoder_seed)
    try:
        for _ in range(p.epochs):
            order=ids[torch.randperm(len(ids),generator=rng)]
            for idx in order.split(p.batch_size):
                batch_hash.update(idx.numpy().astype('<i8').tobytes())
                torch.manual_seed(p.seed+100003+steps)
                out=model(inputs(tensors,idx,arm))
                loss=criterion(out,tensors['fut_xy'][idx],tensors['fut_mask'][idx])['loss']
                if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                optimizer.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),p.gradient_clip,error_if_nonfinite=True)
                optimizer.step();losses.append(float(loss.detach()));steps+=1
            scheduler.step()
    finally:hook.remove()
    return np.array(losses),batch_hash.hexdigest()

@torch.inference_mode()
def predict(model,arm,tensors,indices):
    model.eval();paths=[];scores=[]
    for block in torch.as_tensor(indices).split(64):
        y=model(inputs(tensors,block,arm))
        paths.append(y['trajs']);scores.append(y['mode_probs'])
    p=torch.cat(paths).numpy();q=torch.cat(scores).numpy()
    if p.shape!=(len(indices),6,30,2) or not np.isfinite(p).all() or not np.isfinite(q).all():raise ValueError('Invalid predictions')
    np.testing.assert_allclose(q.sum(1),1.,rtol=1e-5,atol=1e-6)
    gt=tensors['fut_xy'][indices].numpy().astype(float)
    dist=np.linalg.norm(p[np.arange(len(q)),q.argmax(1)].astype(float)-gt,axis=-1)
    return p,q,np.column_stack((dist.mean(1),dist[:,-1]))

def stats(index,keys,source,errors):
    clips=np.array([r[0] for r in keys]);rng=np.random.default_rng(2026)
    means={};draws={};subsets={}
    for s,g in index.items():
        means[s]={n:e[g].mean(0) for n,e in errors.items()}
        sub=strata(source[g])
        subsets[s]={k:{'windows':int(v.sum()),'paired_real_minus_m1_m':metric_dict((errors['M2b_real']-errors['M1'])[g[v]].mean(0)) if v.any() else None}
                    for k,v in sub.items()}
        groups=[g[clips[g]==c] for c in sorted(set(clips[g]))];n=len(groups)
        b=rng.multinomial(n,np.full(n,1/n),size=20000);den=b@np.array([len(h) for h in groups])
        draws[s]={k:b@np.stack([v[h].sum(0) for h in groups])/den[:,None] for k,v in errors.items()}
    pairs={}
    for a,b in PAIRS:
        d=np.mean([v[a]-v[b] for v in draws.values()],axis=0);ci=np.quantile(d,[.025,.975],axis=0)
        pairs[a+'_minus_'+b]={'four_site_m':metric_dict(np.mean([v[a]-v[b] for v in means.values()],axis=0)),
                              'paired_clip_95_m':dict(zip(('ade_m','fde_m'),ci.T.tolist()))}
    return {'four_site_m':{n:metric_dict(np.mean([v[n] for v in means.values()],axis=0)) for n in errors},
            'site_m':{s:{n:metric_dict(v) for n,v in row.items()} for s,row in means.items()},'contrasts':pairs,'strata_by_site':subsets}

def run(root,pose,out,check_only=False):
    p=Protocol()
    tensors,index,keys,audit,identity=prepare(root,pose)
    print(json.dumps(identity),flush=True)
    if check_only:return identity
    if not Path('/kaggle/working').is_dir():raise RuntimeError('Use --check-only locally; fit only on Kaggle CPU')
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError(out)
    torch.set_num_threads(min(4,len(os.sched_getaffinity(0))));torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
    backend={'device':'cpu','cpu':next(line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),
             'torch':str(torch.__version__),'torch_build':torch.__config__.show(),'numpy':np.__version__,'python':platform.python_version(),
             'threads':torch.get_num_threads(),'interop':torch.get_num_interop_threads(),'determinism':torch.are_deterministic_algorithms_enabled(),
             'affinity':sorted(os.sched_getaffinity(0)),'mkldnn':torch.backends.mkldnn.enabled}
    n=len(keys);methods=ARMS+CONTROLS
    errors={k:np.full((n,2),np.nan) for k in methods};paths={k:np.empty((n,6,30,2),np.float32) for k in methods}
    probs={k:np.empty((n,6),np.float32) for k in methods}
    fits={}
    with tempfile.TemporaryDirectory(dir=out.parent) as temp:
        temp=Path(temp); archive=temp/'output.zip'
        for s,g in index.items():
            train_ids=np.concatenate([ids for k,ids in index.items() if k!=s])
            assert not {keys[i][0] for i in g}&{keys[i][0] for i in train_ids}
            records={}
            for arm in ARMS:
                print('Fit',s,arm,len(train_ids),len(g),flush=True)
                model,shared,all_params=initialize(arm)
                losses,batch_sha=train(model,arm,tensors,train_ids,p)
                filename=s+'_'+arm+'.pt'
                state={'model_state':model.state_dict(),'loss_per_batch':torch.from_numpy(losses),'arm':arm,'site':s,
                       'train_rows':torch.as_tensor(train_ids),'test_rows':torch.as_tensor(g),'batch_order_sha256':batch_sha,
                       'shared_initial_sha256':shared,'full_initial_sha256':all_params}
                torch.save(state,temp/filename)
                # Read back each checkpoint into a fresh predictor before making stored predictions.
                reload=torch.load(temp/filename,map_location='cpu',weights_only=True)
                model=UnstructuredShortPosePredictor('M1' if arm=='M1' else 'M2b')
                model.load_state_dict(reload['model_state'],strict=True)
                for method in (arm,*CONTROLS) if arm=='M2b_real' else (arm,):
                    predictive,q,e=predict(model,method,tensors,g)
                    paths[method][g]=predictive;probs[method][g]=q;errors[method][g]=e
                records[arm]={'weights':filename,'optimizer_steps':len(losses),'batch_order_sha256':batch_sha,'shared_initial_sha256':shared,
                              'full_initial_sha256':all_params,'train_windows':len(train_ids),'test_windows':len(g)}
            assert len({v['batch_order_sha256'] for v in records.values()})==len({v['shared_initial_sha256'] for v in records.values()})==1
            assert records['M2b_real']['full_initial_sha256']==records['M2b_empty']['full_initial_sha256']
            fits[s]=records
        for change,needed in [('real_edges_off',audit['source_class']>=0),('real_drop_car',audit['source_class']==1),
                              ('real_drop_people',audit['source_class']==0)]:
            unchanged=~needed.any(1)
            np.testing.assert_array_equal(paths[change][unchanged],paths['M2b_real'][unchanged])
            np.testing.assert_array_equal(probs[change][unchanged],probs['M2b_real'][unchanged])
        assert all(np.isfinite(v).all() for v in errors.values())
        arrays={'clip':np.array([r[0] for r in keys]),'person_id':np.array([r[1] for r in keys]),'t0':np.array([r[2] for r in keys]),
                'fut_xy':tensors['fut_xy'].numpy(),**audit,**{k:tensors[k].numpy() for k in GRAPH}}
        for name in errors:
            arrays.update({name+'_paths':paths[name],name+'_probs':probs[name],name+'_errors':errors[name]})
        np.savez_compressed(temp/'private.npz',**arrays)
        aggregate={'scope':'One seed, previously studied fixed four PINNS sites. Input is label-derived tracking; exclude any future neighbor membership. Frozen five-frame pose, motion and ego-type oracle shared. No map or action objective.',
                   'protocol':asdict(p),'protocol_sha256':LOCK,'runner_sha256':file_hash(Path(__file__)),'source':identity,'backend':backend,
                   'interval':'Site-stratified paired whole-clip bootstrap, preserve all windows of each replicate clip. Four sites equal, conditional on these fits/sites, RNG 2026, 20000 marginal percentile draws.',
                   'primary':'M2b_real minus M1 ADE. M2b_empty has the identical graph architecture; all graph inputs masked in train and test. real_* rows mask only at inference.',
                   'folds':fits,**stats(index,keys,audit['source_class'],errors)}
        (temp/'aggregate.json').write_text(json.dumps(aggregate,indent=2,allow_nan=False)+'\n')
        (temp/'run_graph_control.py').write_bytes(Path(__file__).read_bytes())
        files=sorted(k for k in temp.iterdir() if k.is_file())
        manifest={k.name:file_hash(k) for k in files}
        assert len(files)==15 and len(fits)==4
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=2) as z:
            for k in files:z.write(k,k.name)
            z.writestr('sha256.json',json.dumps(manifest,sort_keys=True))
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None and all(digest(z.read(k))==h for k,h in manifest.items())
        archive.replace(out)
    print('Verified 12 fits',out,out.stat().st_size,file_hash(out),flush=True)
    return aggregate

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data',type=Path);parser.add_argument('pose',type=Path);parser.add_argument('--output',type=Path,default=Path('/kaggle/working/pinns_graph_control.zip'))
    parser.add_argument('--check-only',action='store_true');parser.add_argument('--support',type=Path)
    a=parser.parse_args();ans=run(a.data,a.pose,a.output,a.check_only)
    if a.check_only and a.support:a.support.write_text(json.dumps(ans,indent=2,allow_nan=False)+'\n')
