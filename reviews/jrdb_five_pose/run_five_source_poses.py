"""Registered 197-window JRDB development comparison, real fitting restricted to Kaggle."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
import zipfile
import numpy as np

from check_intersection import OFFSETS,SCENES,SHA
from prior_native_residual import residual_fit,infer

D=Path(__file__).resolve().parent
ARMS=('motion','last_pose','first_last_pose','five_pose')
CV='last_two_cv'; METRICS=('ade_m','fde_m')
PAIRS=(('five_pose','first_last_pose'),('five_pose','motion'),('last_pose','motion'),('first_last_pose','last_pose'))+tuple((m,CV) for m in ARMS)
FUTURE=np.arange(6,43,6);PAST=np.array(OFFSETS);LOCK='ea139fde2942c2e9847cd58165230374db36fff1b0378c94f145c32922740d6f'
def sha(b):return hashlib.sha256(b).hexdigest()
def dump(x):return (json.dumps(x,indent=2,allow_nan=False)+'\n').encode()
def metric(v):return dict(zip(METRICS,np.asarray(v).tolist()))

def stage(root):
    keys=[];past=[];future=[];joints=[];support={}
    for i,s in enumerate(SCENES):
        loc={'tracks':root/'jrdb/data/train'/(s+'_shift_0.ndjson'),'poses':root/'labels_3d_keypoints/train'/(s+'.json'),'odom':root/'odometry/train'/(s+'.json')}
        raw={k:p.read_bytes() for k,p in loc.items()};assert all(sha(b)==SHA[k][i] for k,b in raw.items())
        xy={};rates=set()
        for line in raw['tracks'].splitlines():
            record=json.loads(line)
            if 'scene' in record:rates.add(record['scene']['fps']);continue
            row=record['track'];key=(int(row['p']),int(row['f']));point=(float(row['x']),float(row['y']))
            if not np.isfinite(point).all() or point==(0.,0.):continue
            assert key not in xy and key[1]%6==0;xy[key]=point
        assert rates=={2.5}
        pose={}
        for name,people in json.loads(raw['poses'])['labels'].items():
            for obj in people:
                if not obj['label_id'].startswith('pedestrian:'):continue
                key=(int(obj['label_id'].split(':')[1]),int(Path(name).stem))
                assert key not in pose
                pose[key]=np.array(obj['keypoints'],dtype=float).reshape(33,3)
        odom={int(Path(name).stem) for name,rec in json.loads(raw['odom'])['odometry'].items() if
              np.isfinite([*rec['position'].values(),*rec['orientation'].values()]).all()}
        members=Counter();bounds={};selected=Counter()
        for (p,t) in sorted(xy):
            if (p,t-6) not in xy or not all((p,t+d) in xy for d in FUTURE):continue
            if not all(t+d in odom for d in (*OFFSETS,*FUTURE)):continue
            members['ground_eligible_pre_greedy']+=1
            if (p,t) not in pose:members['last_pose_absent']+=1;continue
            if not np.isfinite(pose[p,t]).all():members['last_pose_present_incomplete']+=1;continue
            if not all((p,t+d) in pose and np.isfinite(pose[p,t+d]).all() for d in OFFSETS):members['last_complete_history_incomplete']+=1;continue
            members['all_five_complete']+=1
            if t-6<=bounds.get(p,-2**32):continue
            bounds[p]=t+42;selected[p]+=1;keys.append((s,p,t));past.append([xy[p,t-6],xy[p,t]])
            future.append([xy[p,t+d] for d in FUTURE]);joints.append([pose[p,t+d] for d in OFFSETS])
        counts={m:members.get(m,0) for m in ('ground_eligible_pre_greedy','last_pose_absent','last_pose_present_incomplete','last_complete_history_incomplete','all_five_complete')}
        assert counts['ground_eligible_pre_greedy']==sum(list(counts.values())[1:])
        assert sum(selected.values())==(123,68,6)[i]
        support[s]={**counts,'selected':sum(selected.values()),'recording_local_people':len(selected),'people_window_minmax':[min(selected.values()),max(selected.values())]}
    support_id=sha(json.dumps(keys,separators=(',',':')).encode())
    assert len(keys)==197 and len(set(keys))==len(keys)
    return keys,np.asarray(past),np.asarray(joints),np.asarray(future),dict(ordered_keys_sha256=support_id,by_recording=support,
          pose_offsets=list(OFFSETS),future_offsets=FUTURE.tolist(),image_timestamp_source='nominal 15 Hz native file indices, no exposure timestamps',member_sha256=SHA)

def design(history,joints):
    n=len(history);assert history.shape==(n,2,2) and joints.shape==(n,5,33,3)
    v=(history[:,1]-history[:,0])/.4
    motion=np.column_stack((v,np.linalg.norm(v,axis=1)))
    finite=np.isfinite(joints).all(-1)
    def centered(frame,mask):
        clean=np.where(mask[:,:,None],joints[:,frame],0.)
        mu=clean.sum(1)/np.maximum(1,mask.sum(1))[:,None]
        return np.where(mask[:,:,None],clean-mu[:,None],0.)
    last=np.column_stack((motion,centered(4,finite[:,4]).reshape(n,-1),finite[:,4]))
    def difference(frame):
        mask=finite[:,frame]&finite[:,4];delta=(centered(4,mask)-centered(frame,mask))/(-PAST[frame]/15)
        return np.column_stack((delta.reshape(n,-1),mask))
    ends=np.column_stack((last,difference(0)))
    five=np.column_stack((ends,*(difference(i) for i in (1,2,3))))
    return {'motion':motion,'last_pose':last,'first_last_pose':ends,'five_pose':five}

def fit_all(keys,h,j,y):
    scene=np.array([k[0] for k in keys]);x=design(h,j)
    anchor=(h[:,1]-h[:,0])[:,None,:]*np.arange(1,8)[None,:,None]
    truth=y-h[:,1:2];residual=truth-anchor
    paths={arm:np.full_like(truth,np.nan) for arm in ARMS};paths[CV]=anchor.copy()
    hidden={};folds={}
    for i,s in enumerate(SCENES):
        te=np.flatnonzero(scene==s);tr=np.flatnonzero(scene!=s)
        assert set(scene[tr])==set(SCENES)-{s}
        hidden[f'fold_{i}_train']=tr;hidden[f'fold_{i}_test']=te;folds[s]={'train':sorted(set(scene[tr]).difference({s})),'train_windows':len(tr),'test_windows':len(te)}
        for a in ARMS:
            fit=residual_fit(x[a][tr],residual[tr])
            for field,val in fit.items():hidden[f'fold_{i}_{a}_{field}']=val
            paths[a][te]=infer(x[a][te],fit,anchor[te])
    errors={}
    for arm,p in paths.items():
        assert np.isfinite(p).all()
        delta=np.linalg.norm(p-truth,axis=2);errors[arm]=np.column_stack((delta.mean(1),delta[:,-1]))
    return x,paths,errors,hidden,folds

def summarize(keys,errors):
    s=np.array([k[0] for k in keys]);p=np.array([k[1] for k in keys])
    rng=np.random.default_rng(2026);fold={};b={};points={}
    for name in ('window_equal_within_recording','person_equal_within_recording'):
        b[name]={arm:np.zeros((20000,2)) for arm in errors};points[name]={arm:np.zeros(2) for arm in errors}
    for site in SCENES:
        ids=np.flatnonzero(s==site);persons=sorted(set(p[ids]));batches=[ids[p[ids]==pid] for pid in persons]
        size=np.array(list(map(len,batches)));m=len(persons);pick=rng.multinomial(m,np.full(m,1/m),size=20000)
        by={}
        for arm,err in errors.items():
            sums=np.stack([err[ix].sum(0) for ix in batches]);means=sums/size[:,None]
            scores=dict(window_equal_within_recording=err[ids].mean(0),person_equal_within_recording=means.mean(0))
            res=dict(window_equal_within_recording=(pick@sums)/(pick@size)[:,None],person_equal_within_recording=(pick@means)/m)
            by[arm]={n:metric(v) for n,v in scores.items()}
            for n in scores:points[n][arm]+=scores[n]/3;b[n][arm]+=res[n]/3
        fold[site]={'windows':len(ids),'people':len(persons),'models':by}
    all_scores={}
    for n,v in points.items():
        contrasts={}
        for a,c in PAIRS:
            boot=b[n][a]-b[n][c]
            contrasts[a+'_minus_'+c]={'mean':metric(v[a]-v[c]),'paired_person_95':metric(np.percentile(boot,[2.5,97.5],axis=0).T)}
        all_scores[n]={'models':{a:metric(vals) for a,vals in v.items()},'paired':contrasts}
    return {'by_recording':fold,'three_recording_equal':all_scores}

def run(root,output,check):
    assert sha((D/'PROTOCOL.md').read_bytes())==LOCK
    keys,h,j,y,support=stage(root)
    expected=json.loads((D/'input_support.json').read_bytes());assert json.loads(json.dumps(support))==expected
    if check:print(json.dumps(support));return
    if not Path('/kaggle/working').is_dir():raise RuntimeError('Real fitting/evaluation is restricted to Kaggle only')
    if output.exists():raise FileExistsError(output)
    x,paths,errs,weights,folds=fit_all(keys,h,j,y)
    agg=summarize(keys,errs);agg.update(scope='Three previous development recordings. HST 33 raw-ID 3D pose estimates, two ST historical centres, seven future centres through 2.8 s. No independent ST scale proof.',primary='five_pose_minus_first_last_pose ADE, window-equal within each recording',folds=folds,support=support,alpha=1.0,protocol_sha256=LOCK,software={'python':sys.version,'numpy':np.__version__,'platform':platform.platform()})
    store=dict(keys_json=np.array(json.dumps(keys)),history_xy=h,history_joints=j,target_xy=y,history_native_frames=np.array([k[2] for k in keys])[:,None]+PAST,future_native_frames=np.array([k[2] for k in keys])[:,None]+FUTURE,**weights)
    for arm in paths:store[arm+'_paths']=paths[arm];store[arm+'_errors']=errs[arm]
    for arm in x:store[arm+'_input']=x[arm]
    arr=io.BytesIO();np.savez_compressed(arr,**store)
    files={'aggregate.json':dump(agg),'private.npz':arr.getvalue()}
    for n in ('PROTOCOL.md','run_five_source_poses.py','prior_native_residual.py','check_intersection.py','input_support.json'):files['sources/'+n]=(D/n).read_bytes()
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
        for n,raw in files.items():z.writestr(n,raw)
        z.writestr('sha256.json',dump({n:sha(r) for n,r in files.items()}))
    print('Verified',output,sha(output.read_bytes()))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path);p.add_argument('--check-only',action='store_true');a=p.parse_args();run(a.root,a.output,a.check_only)
