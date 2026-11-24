"""Frozen graph transfer to DUT. Model evaluation and target scoring are restricted to Kaggle CPU."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import platform
import tempfile
import zipfile
import numpy as np
import torch
from fusion_cloud import load_pose_cache
from motion_cloud import build,fetch,grouped,SITES
from observed_graph import array_sha,collect,key_sha,sha,strata
from src.data.features import build_motion_features,build_pose_and_quality_features
from src.models.predictor import UnstructuredShortPosePredictor

D=Path(__file__).resolve().parent
PINNS='0b3e590b8d7cb2bf99670fd27d3bdbf2a40fa0e77bf29f7ac0423b73535589f8'
MATRICES={k:np.asarray(v,np.float32) for k,v in
          [('0',[[1,0],[0,1]]),('90',[[0,-1],[1,0]]),('180',[[-1,0],[0,-1]]),
           ('270',[[0,1],[-1,0]]),('mirror_y',[[1,0],[0,-1]])]}
ARMS=('M1','M2b_real','M2b_empty','real_edges_off','real_drop_car','real_drop_people')
RESERVE='real_reserve_one_car'
PAIR=(('M2b_real','M1'),('M2b_real','M2b_empty'),('M2b_empty','M1'),
      ('real_edges_off','M2b_real'),('real_drop_car','M2b_real'),('real_drop_people','M2b_real'))

def read(p):return json.loads(Path(p).read_bytes())
def m(v):return dict(zip(('ade_m','fde_m'),np.asarray(v).tolist()))
def npz(path,**arr):np.savez_compressed(path,**arr)

def load_weights(path):
    if sha(path.read_bytes())!=PINNS:raise ValueError('Wrong source training archive')
    nets={};digests={}
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        manifest=json.loads(z.read('sha256.json'))
        for site in SITES:
            digests[site]={}
            for arm in ARMS[:3]:
                n=f'{site}_{arm}.pt';raw=z.read(n);enc=sha(raw);assert enc==manifest[n];digests[site][arm]=enc
                ck=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
                assert (ck['site'],ck['arm'])==(site,arm)
                model=UnstructuredShortPosePredictor('M1' if arm=='M1' else 'M2b');model.load_state_dict(ck['model_state'],strict=True)
                nets[site,arm]=model.cpu().eval()
    return nets,digests

def motion(history,rot):
    xm=np.stack([build_motion_features((h-h[-1])@rot.T,np.ones(5,np.float32),dt=.1) for h in history])
    ego=np.eye(8,dtype=np.float32)[np.where(xm[:,-1,4]<.2,2,0)]
    return torch.from_numpy(xm),torch.from_numpy(ego)

def graph_input(g,arm,rot):
    if arm=='M1':return {}
    e=g['edges'].copy();e[:,:,:2]=e[:,:,:2]@rot.T;e[:,:,2:4]=e[:,:,2:4]@rot.T
    kind=g['source'];typ=g['encoded_type'].copy();keep=kind>=0
    if arm==RESERVE:
        ch=g['reserve_changed'];rep=g['reserve_last_edge'][ch].copy()
        rep[:,:2]=rep[:,:2]@rot.T;rep[:,2:4]=rep[:,2:4]@rot.T
        e[ch,-1]=rep;typ[ch,-1]=6
    elif arm in ('M2b_empty','real_edges_off'):keep=np.zeros_like(keep)
    elif arm=='real_drop_car':keep=kind==0
    elif arm=='real_drop_people':keep=kind==1
    elif arm!='M2b_real':raise ValueError(arm)
    onehot=np.eye(8,dtype=np.float32)[np.maximum(0,typ)]*keep[:,:,None]
    return {k:torch.from_numpy(v) for k,v in dict(nbr_edges=e*keep[:,:,None],nbr_type_oh=onehot,nbr_mask=keep.astype(np.float32)).items()}

@torch.inference_mode()
def predict(net,shared,graph,rot):
    batch={**shared,**graph};paths=[];prob=[]
    for ix in torch.arange(len(batch['x_m'])).split(128):
        result=net({key:v[ix] for key,v in batch.items()})
        paths.append(result['trajs'].numpy()@rot);prob.append(result['mode_probs'].numpy())
    p=np.concatenate(paths);q=np.concatenate(prob)
    assert p.shape==(len(q),6,30,2) and q.shape==(len(p),6) and np.isfinite(p).all() and np.isfinite(q).all() and (q>=0).all()
    np.testing.assert_allclose(q.sum(1),1,atol=2e-6)
    return p,q,q.argmax(1).astype(np.int8)

def metrics(pred,y):
    distances=np.linalg.norm(pred.astype(float)-y,axis=2)
    return np.column_stack((distances.mean(1),distances[:,-1]))

def paired(diff,mask,clips):
    names=sorted(set(clips[mask]))
    if not names:return {'windows':0,'clips':0,'clip_equal':None,'paired_clip_95':None}
    vals=np.stack([diff[(clips==s)&mask].mean(0) for s in names])
    rng=np.random.Generator(np.random.PCG64(2026))
    draws=rng.integers(0,len(names),size=(20000,len(names)))
    return {'windows':int(mask.sum()),'clips':len(names),'clip_equal':m(vals.mean(0)),
            'paired_clip_95':m(np.percentile(vals[draws].mean(1),[2.5,97.5],axis=0).T),
            'by_clip':{s:{'windows':int(((clips==s)&mask).sum()),'difference':m(v)} for s,v in zip(names,vals)}}

def score(saved,keys,frames,history,g,pose_mask,root):
    # Called only after every candidate, probability and top-mode index has been written.
    orig,full_frames,ticks,xy=build(root)
    assert orig==keys;np.testing.assert_array_equal(full_frames,frames);np.testing.assert_array_equal(xy[:,:5],history)
    true=xy[:,5:]-xy[:,4:5];dt=ticks[:,4]-ticks[:,3];times=ticks[:,5:]-ticks[:,4:5]
    cv=(history[:,4]-history[:,3])[:,None,:]*(times/dt[:,None])[:,:,None]
    cv_err=metrics(cv,true);clips=np.array([k[0] for k in keys]);all_mask=np.ones(len(keys),bool)
    sub={'all':all_mask,**strata(g['source'])}
    for c in range(6):sub[f'pose_matches_{c}']=pose_mask.sum(1)==c
    angles={};p0={};rows=np.arange(len(true));private={}
    for angle in MATRICES:
        arms=ARMS+(RESERVE,) if angle=='0' else ARMS
        arrays={}
        for site in SITES:
            file=saved/f'paths_{angle}_{site}.npz'
            with np.load(file,allow_pickle=False) as npf:
                for label in arms:
                    top=npf[label+'_top'];p=npf[label+'_p']
                    np.testing.assert_array_equal(top,npf[label+'_q'].argmax(1))
                    arrays.setdefault(label,[]).append(metrics(p[rows,top],true))
                if angle=='0':
                    same=~g['reserve_changed']
                    for key in ('p','q','top'):
                        np.testing.assert_array_equal(npf[RESERVE+'_'+key][same],npf['M2b_real_'+key][same])
        means={}
        for arm,v in arrays.items():
            arr=np.stack(v);private[angle+'_'+arm]=arr;means[arm]=arr.mean(0)
        if angle=='0':p0=means
        means['cv_measured']=cv_err
        angles[angle]={'subgroups':{s:{arm:grouped(v,mask,keys) for arm,v in means.items()} for s,mask in sub.items()},
                       'by_clip':{s:{arm:m(v[clips==s].mean(0)) for arm,v in means.items()} for s in sorted(set(clips))},
                       'contrasts_clip_equal':{a+'_minus_'+b:grouped(means[a]-means[b],all_mask,keys)['clip_equal'] for a,b in PAIR}}
    by={a+'_minus_'+b:paired(p0[a]-p0[b],all_mask,clips) for a,b in PAIR}
    reserve={n:paired(p0[RESERVE]-p0['M2b_real'],mask,clips) for n,mask in
             [('all',all_mask),('affected',g['reserve_changed']),('precap_any_car',g['pre'][:,1]>0)]}
    npz(saved/'errors_and_targets.npz',truth=true,cv_measured=cv_err,**private)
    span=times[:,-1]/23976
    return {'all_axes':angles,'native_primary_and_secondary':by,'native_vehicle_slot':reserve,'actual_end_s_min_max':[float(span.min()),float(span.max())]}

def run(args):
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    cov=read(D/'coverage.json');keys,f,h,g,audit=collect(args.data,cov,fetch)
    reg=read(D/'historical_denominators.json')
    assert json.loads(json.dumps(audit))==reg,'History-only census changed'
    nets,sha_weights=load_weights(args.weights)
    if args.check:
        print(json.dumps({'histories':len(keys),'loaded_states':len(nets),'own_positions_only':True,'graph':audit['all'],'reserve':audit['vehicle_slot']}));return
    if not Path('/kaggle/working').is_dir() or torch.cuda.is_available():raise RuntimeError('External prediction restricted to Kaggle CPU')
    if not args.cache or not args.output or args.output.exists():raise ValueError('Supply one verified cache and a new output ZIP')
    detector,k,b,mask=load_pose_cache(args.cache,keys,f,cov['key_manifest_sha256'],sha((D/'motion_and_detector_freeze.json').read_bytes()))
    heights={v['clip']:v['frame_size'][1] for v in cov['clips']}
    parts=[build_pose_and_quality_features(j,bb,np.ones(5,np.float32),heights[c])[:2] for j,bb,(c,_,_) in zip(k,b,keys)]
    xp,q=(torch.from_numpy(np.stack(arr)) for arr in zip(*parts))
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temp:
        dest=Path(temp);npz(dest/'private_graph.npz',keys_json=np.array(json.dumps(keys)),frames=f[:,:5],history=h,pose_mask=mask,**g)
        for name,rot in MATRICES.items():
            xm,ego=motion(h,rot);common=dict(x_m=xm,ego_type_oh=ego,x_p=xp,q_vec=q)
            arms=ARMS+(RESERVE,) if name=='0' else ARMS
            graph={arm:graph_input(g,arm,rot) for arm in arms}
            for site in SITES:
                arrays={}
                for arm in arms:
                    weight=arm if arm in ARMS[:3] else 'M2b_real'
                    pp,qq,idx=predict(nets[site,weight],common,graph[arm],rot)
                    arrays.update({arm+'_p':pp,arm+'_q':qq,arm+'_top':idx})
                npz(dest/f'paths_{name}_{site}.npz',**arrays)
            print('All four sources and arms stored for',name,flush=True)
        report=score(dest,keys,f,h,g,mask,args.data)
        registered={n:sha((D/n).read_bytes()) for n in ('PROTOCOL.md','HANDEDNESS_ADDENDUM.md','VEHICLE_SLOT_ADDENDUM.md','observed_graph.py','run_transfer.py','motion_cloud.py','fusion_cloud.py','coverage.json','historical_denominators.json')}
        report.update(scope='DUT one camera/11 clips; fixed source seed42. Clip-equal paired values conditional on these clips. Input author vehicle and pedestrian tracks, no automatic vehicle perception.',runner_and_protocol=registered,source_weights=sha_weights,
                      source_training_archive_sha256=PINNS,pose_cache_sha256=sha(args.cache.read_bytes()),pose_detector_report=detector,
                      history=audit,conditions=list(MATRICES),arms=list(ARMS),runtime={'torch':str(torch.__version__),'numpy':np.__version__,'platform':platform.platform(),'cuda':torch.cuda.is_available(),'threads':torch.get_num_threads()})
        (dest/'aggregate.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        files=sorted(x for x in dest.iterdir() if x.is_file());manifest={}
        pending=dest/'result.tmp'
        with zipfile.ZipFile(pending,'w',zipfile.ZIP_STORED) as z:
            for path in files:z.write(path,path.name);manifest[path.name]=sha(path.read_bytes())
            for n in registered:z.write(D/n,'sources/'+n);manifest['sources/'+n]=registered[n]
            z.writestr('sha256.json',json.dumps(manifest,sort_keys=True))
        with zipfile.ZipFile(pending) as z:assert z.testzip() is None
        pending.replace(args.output)
    print('Done',args.output,args.output.stat().st_size,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True,type=Path);p.add_argument('--weights',required=True,type=Path)
    p.add_argument('--cache',type=Path);p.add_argument('--output',type=Path);p.add_argument('--check',action='store_true');run(p.parse_args())
