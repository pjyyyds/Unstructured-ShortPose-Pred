"""Four-site five-frame factorial trajectory experiment. Real fitting is Kaggle-only."""
import argparse,hashlib,json,os,platform,sys,zipfile
from pathlib import Path

import numpy as np
import torch
from equivariant_model import GroundForecaster,frame,loss_for
from scripts.benchmark_heldout_sites import load_site_windows
from scripts.train_pinns_heldout_sites import ROOT,tensorize,atomic_write,save_json
from scripts.verify_heldout_bundle import IDS_SHA,LABELS_SHA,parameter_digest
from scripts.run_heldout_seed_sensitivity import SITE_COUNTS,json_digest,split_indices

P=Path(__file__).resolve().parent
PLAN='2dd4e77f5023088cfdede6cce026c1fe52b757c0d370d025329fe3e48427bac5'
POSE='b493a47536664ddb4fd66603c1bbf6b2c2e33f553520382027b3ae96b53447d0'
ARMS={'global_motion':(False,False),'global_pose':(False,True),'local_motion':(True,False),'local_pose':(True,True)}
PAIR=[('local_pose','local_motion'),('local_pose','global_pose'),('local_motion','global_motion'),('global_pose','global_motion')]
OBS=('history','pose','quality')
FIELDS=('ade_m','fde_m','minade6_m','minfde6_m','mode0_selected','distinct_mean_pair_m')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def need(x,msg):
    if not x:raise ValueError(msg)
def inputs(d,inds):return [d[k][inds] for k in OBS]
def new(arm):
    torch.manual_seed(42);model=GroundForecaster(*ARMS[arm])
    source=P/'initial_state.npz'
    need(sha(source)=='b1ff904b3cf9b28b146c55a50dbbc5e69a2bd56d1eabf8e41b68208b7a73cd80','Frozen shared seed-42 initialization changed')
    with np.load(source,allow_pickle=False) as z:
        model.load_state_dict({k:torch.from_numpy(z[k]) for k in z.files},strict=True)
    return model
def npz(path,**items):
    def write(p):
        with p.open('wb') as f:np.savez_compressed(f,**items)
    atomic_write(path,write)

def prepare(data,pose):
    need(sha(P/'PROTOCOL.md')==PLAN and sha(pose)==POSE,'Registration or pose identity changed')
    g,_,labelsha=load_site_windows(data);need(labelsha==LABELS_SHA,'Label identity changed')
    t,idx,keys,_,_,_=tensorize(g,pose);need(json_digest(keys)==IDS_SHA and {k:len(v) for k,v in idx.items()}==SITE_COUNTS,'Wrong source cohorts')
    xy=np.stack([x.xy for s in sorted(g) for x in g[s]])
    need(xy.shape==(1482,35,2) and np.isfinite(xy).all() and np.all(xy[:,4]==0),'Invalid metric tracks')
    np.testing.assert_array_equal(xy[:,:5].astype(np.float32),t['x_m'][:,:,:2])
    np.testing.assert_array_equal(xy[:,5:].astype(np.float32),t['fut_xy'])
    d={'history':torch.from_numpy(xy[:,:5].copy()),'truth':torch.from_numpy(xy[:,5:].copy()),'pose':t['x_p'],'quality':t['q_vec']}
    valid=frame(d['history'],True)[2].numpy();split={};support={}
    for site,te in idx.items():
        others,train,test=split_indices(idx,keys,site)
        assert list(te)==list(test) and not set(te)&set(train)
        split[site]={'train_sites':others,'train_keys':[keys[i] for i in train],'test_keys':[keys[i] for i in te]}
        support[site]={'windows':len(te),'clips':len({keys[i][0] for i in te}),'heading_identifiable':int(valid[te].sum()),'degenerate':int((~valid[te]).sum()),'tracks':len({keys[i][:2] for i in te})}
    plan={'protocol_sha256':PLAN,'implementation_checks_sha256':sha(P/'IMPLEMENTATION_CHECKS.md'),'source_sha256':{n:sha(P/n) for n in ('equivariant_model.py','run_equivariant.py','test_equivariance.py')},'source_revision':'66fb39d2de3df280ac831a4b5a0c0f095f1a0779','canonical_reader_sha256':{n:sha(ROOT/n) for n in ('src/data/features.py','src/data/constants.py','src/data/pinns_extractor.py','scripts/benchmark_heldout_sites.py','scripts/train_pinns_heldout_sites.py','scripts/verify_heldout_bundle.py','scripts/run_heldout_seed_sensitivity.py','scripts/stage_pinns_metadata.py')},'label_calibration_sha256':labelsha,'pose_sha256':POSE,'ordered_keys_sha256':IDS_SHA,'folds_sha256':json_digest(split),'observed_support':support,'seed':42,'same_initial_state_sha256':parameter_digest(new('global_motion').state_dict()),'architectures':ARMS,'shared_initial_npz_sha256':sha(P/'initial_state.npz'),'parameters_allocated_in_every_arm':sum(p.numel() for p in new('global_pose').parameters()),'axes_from_history_only':True}
    return d,idx,keys,split,json.loads(json.dumps(plan))

def fit(arm,training):
    model=new(arm).train();initial=parameter_digest(model.state_dict())
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=28,eta_min=.0001)
    gen=torch.Generator().manual_seed(49);logs=[]; n=len(training['truth'])
    for epoch in range(28):
        order=torch.randperm(n,generator=gen);total=0.;trained=0
        for ids in order.split(64):
            optimizer.zero_grad(set_to_none=True)
            paths,logits,valid=model(*inputs(training,ids))
            loss=loss_for(paths,logits,training['truth'][ids],valid)
            if loss is None:continue
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.,error_if_nonfinite=True);optimizer.step()
            count=int(valid.sum());trained+=count;total+=loss.item()*count
        scheduler.step();logs.append({'epoch':epoch+1,'directional_training_windows':trained,'loss':total/trained,'ordered_indices_sha256':hashlib.sha256(order.numpy().tobytes()).hexdigest()})
    return model.eval(),initial,logs

@torch.inference_mode()
def observe(model,observed):
    assert set(observed)==set(OBS)
    values=[[],[],[]]
    for ids in torch.arange(len(observed['history'])).split(64):
        traj,lg,ok=model(*inputs(observed,ids))
        for v,o in zip(values,(traj,lg.softmax(1),ok)):v.append(o.numpy())
    return tuple(np.concatenate(v) for v in values)

def measures(paths,probs,truth):
    assert np.isfinite(paths).all() and np.isfinite(probs).all() and (probs>=0).all()
    np.testing.assert_allclose(probs.sum(1),1,atol=1e-6)
    dst=np.linalg.norm(paths-truth[:,None],axis=3);top=probs.argmax(1);ii=np.arange(len(paths))
    ade=dst.mean(2);fde=dst[:,:,-1];spread=np.mean([np.linalg.norm(paths[:,a]-paths[:,b],axis=2).mean(1) for a in range(6) for b in range(a)],axis=0)
    return np.stack((ade[ii,top],fde[ii,top],ade.min(1),fde.min(1),top==0,spread),axis=1)

def orbit(arm,model,observed,paths,probs,valid):
    out={}
    for deg in (37,90,173):
        a=np.deg2rad(deg);rot=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
        shift=np.array([13.25,-9.5]);transformed={**observed,'history':observed['history']@torch.tensor(rot.T.copy())+torch.tensor(shift)}
        p,q,s=observe(model,transformed);assert np.array_equal(s,valid)
        expected=paths@rot.T+shift
        gap=float(np.max(np.abs(p-expected)));pgap=float(np.max(np.abs(q-probs)))
        out[str(deg)]={'max_mode_indexed_coordinate_error_m':gap,'max_probability_error':pgap,'top_mode_changes':int(np.sum(q.argmax(1)!=probs.argmax(1)))}
        if arm.startswith('local'):
            np.testing.assert_allclose(p,expected,atol=1e-5,rtol=0);np.testing.assert_allclose(q,probs,atol=2e-6,rtol=0)
    return out

def summarize(scores,idx,keys,valid):
    def m(e):return dict(zip(FIELDS,e.mean(0).tolist()))
    report={}
    for label,mask in [('all',np.ones(len(keys),bool)),('identifiable',valid),('degenerate',~valid)]:
        sub={s:rows[mask[rows]] for s,rows in idx.items()};exists=[s for s,r in sub.items() if len(r)]
        report[label]={'by_site':{s:{'windows':len(r),'tracks':len({keys[i][:2] for i in r}),'clips':len({keys[i][0] for i in r}),'metrics':{a:m(v[r]) for a,v in scores.items()} if len(r) else None} for s,r in sub.items()},'contributing_sites':exists,'site_equal':{a:m(np.stack([v[sub[s]].mean(0) for s in exists])) for a,v in scores.items()} if exists else None}
    rng=np.random.default_rng(2026);boots=[]
    for s,rows in idx.items():
        clips,inv=np.unique([keys[i][0] for i in rows],return_inverse=True);n=len(clips);w=rng.multinomial(n,np.full(n,1/n),size=20000)
        boots.append((rows,inv,w,w@np.bincount(inv)))
    comparisons={}
    for a,b in PAIR+[(a,'last_two_cv') for a in ARMS]:
        diff=scores[a][:,:2]-scores[b][:,:2];rep=np.zeros((20000,2));by_site={}
        for (site,rows),(ri,inv,w,denom) in zip(idx.items(),boots):
            assert np.array_equal(rows,ri)
            totals=np.column_stack([np.bincount(inv,weights=diff[rows,i]) for i in range(2)])
            rep+=(w@totals)/denom[:,None]/4
            by_site[site]=dict(zip(FIELDS[:2],diff[rows].mean(0).tolist()))
        v=np.mean([list(d.values()) for d in by_site.values()],axis=0)
        comparisons[a+'_minus_'+b]={'mean':dict(zip(FIELDS[:2],v.tolist())),'paired_clip_95':dict(zip(FIELDS[:2],np.percentile(rep,[2.5,97.5],axis=0).T.tolist())),'by_site':by_site}
    return report,comparisons

def main(args):
    torch.set_num_threads(min(4,os.cpu_count() or 1));torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest');out=args.out;out.mkdir(exist_ok=True,parents=True)
    d,idx,keys,split,plan=prepare(args.data,args.pose)
    if args.mode=='check':save_json(out/'plan.json',plan);print(json.dumps(plan['observed_support']));return
    need(Path('/kaggle/working').is_dir() and not torch.cuda.is_available(),'Fitting restricted to Kaggle CPU')
    need(json.loads(args.plan.read_text())==plan,'Frozen plan or code changed')
    metric={k:np.full((1482,len(FIELDS)),np.nan) for k in (*ARMS,'last_two_cv')}
    valid=frame(d['history'],True)[2].numpy();orbits={};out_files=[]
    save_json(out/'folds.json',split);out_files+=[out/'folds.json']
    for site,te in idx.items():
        _,tr,_=split_indices(idx,keys,site)
        training={k:v[tr] for k,v in d.items()};observed={k:d[k][te] for k in OBS};truth=d['truth'][te].numpy();orbits[site]={}
        h=observed['history'].numpy();cv=h[:,-1:]+(h[:,-1:]-h[:,-2:-1])*np.arange(1,31)[None,:,None]
        dir_=out/site;dir_.mkdir(exist_ok=True)
        npz(dir_/'targets.npz',truth=truth,history=h,valid=valid[te],clip=[keys[i][0] for i in te],person_id=[keys[i][1] for i in te],t0=[keys[i][2] for i in te]);out_files+=[dir_/'targets.npz']
        # Expose two deterministic CV scores; one repeated trajectory has zero diversity.
        metric['last_two_cv'][te]=measures(np.broadcast_to(cv[:,None],(len(te),6,30,2)),np.full((len(te),6),1/6),truth)
        orders=None
        for arm in ARMS:
            model,initial,log=fit(arm,training);assert initial==plan['same_initial_state_sha256']
            order=[k['ordered_indices_sha256'] for k in log]
            if orders is not None:assert orders==order
            orders=order
            p,q,v=observe(model,observed);assert np.array_equal(v,valid[te])
            np.testing.assert_allclose(p[:,0],cv,atol=1e-12,rtol=0)
            values=measures(p,q,truth);metric[arm][te]=values
            assert np.all(values[:,2]<=metric['last_two_cv'][te,0]+1e-10)
            orbits[site][arm]=orbit(arm,model,observed,p,q,v)
            state=model.state_dict();save={'arm':arm,'site':site,'seed':42,'fold_sha256':json_digest(split[site]),'initial_sha256':initial,'parameter_sha256':parameter_digest(state),'model_state':state,'log':log}
            atomic_write(dir_/(arm+'.pt'),lambda tmp:torch.save(save,tmp));npz(dir_/(arm+'.npz'),trajs=p,probs=q,errors=values)
            out_files+=[dir_/(arm+'.pt'),dir_/(arm+'.npz')]
            print('Completed',site,arm,flush=True)
    assert all(np.isfinite(v).all() for v in metric.values())
    groups,pairs=summarize(metric,idx,keys,valid)
    cpu=Path('/proc/cpuinfo')
    hardware=next((x.split(':',1)[1].strip() for x in cpu.read_text().splitlines() if x.startswith('model name')),platform.machine()) if cpu.exists() else platform.machine()
    report={'scope':'PINNS four previously used locations, five observed points, one training seed, no external score. Coordinate changes keep image-derived inputs fixed. All 1482 windows retained in all four arms.','plan':plan,'models':groups,'paired':pairs,'ground_coordinate_tests':orbits,'software':{'numpy':np.__version__,'torch':str(torch.__version__),'python':sys.version,'cpu':hardware,'threads':torch.get_num_threads(),'deterministic':torch.are_deterministic_algorithms_enabled()},'primary':'local_pose_minus_local_motion top-probability ADE','fits':16}
    save_json(out/'equivariant_results.json',report);out_files+=[out/'equivariant_results.json']
    files={p.relative_to(out).as_posix():p for p in out_files}
    for n in ('equivariant_model.py','run_equivariant.py','test_equivariance.py','PROTOCOL.md','IMPLEMENTATION_CHECKS.md','initial_state.npz'):files[n]=P/n
    files['plan.json']=args.plan
    with zipfile.ZipFile(out/'equivariant_forecasts.zip','w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for n,p in files.items():z.write(p,n)
        z.writestr('sha256.json',json.dumps({n:sha(p) for n,p in files.items()},sort_keys=True))
    print(json.dumps(pairs['local_pose_minus_local_motion']))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path);p.add_argument('pose',type=Path)
    p.add_argument('--mode',choices=['check','fit'],required=True);p.add_argument('--plan',type=Path);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
