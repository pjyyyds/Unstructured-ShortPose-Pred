# Kaggle CPU with Internet; no raw dataset attachment required. Keep the notebook and ZIP private.
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile

RUNNER = r'''"""Three leave-recording-out JRDB development folds. NumPy only. See the locked protocol."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import tempfile
import zipfile

import numpy as np

ARMS = ('last_two_cv', 'motion', 'last_pose', 'ordered_pose')
PAIRS = (('ordered_pose', 'motion'), ('last_pose', 'motion'), ('ordered_pose', 'last_pose'),
         ('motion', 'last_two_cv'), ('last_pose', 'last_two_cv'), ('ordered_pose', 'last_two_cv'))
F = ('ade_m', 'fde_m')
EXPECTED = (91,42,5)
INPUT_HASH = '8a7bdd1223cece22fc2f622cc6a744040cd9e7458bc413108ba26727fe097d01'
LOCK = '19b9d9364f671934d5a673835c2af8953f88a4f92908b6aef473cee2e93e78a3'
RIDGE = 1.0
HELPERS = {
    'evaluate_outdoor_motion.py':'782f4125e5500ddc3170e30f7a1c85da6cf6cb6d953562279c6dfd65c174d089',
    'check_outdoor_projection.py':'1acc686949350614117a96113f56c61114e0bf29e2eecc3fae8742f534ffc4fa',
    'check_training_coordinates.py':'d0ed11f6cc1a2c8256e8b4eca4ceceed11c2ba798f2b5c95c9fd9a1b6a57233c'}

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def export(obj):
    return (json.dumps(obj, indent=2, allow_nan=False) + '\n').encode()

def joint_records(raw):
    result = {}
    person_fields, frame_suffixes = set(),set()
    for name, people in json.loads(raw)['labels'].items():
        frame=int(Path(name).stem)
        frame_suffixes.add(Path(name).suffix)
        for person in people:
            kind,ident=person['label_id'].split(':')
            if kind!='pedestrian':
                continue
            person_fields.update(person)
            key=frame,int(ident)
            coord=np.asarray(person['keypoints'],dtype=np.float64)
            if key in result or coord.shape!=(99,):
                raise ValueError('Duplicate frame/person or malformed 33×3 keypoints')
            result[key]=coord.reshape(33,3)
    if person_fields!={'label_id','keypoints'} or frame_suffixes!={'.jpg'}:
        raise ValueError('Joint records differ from the frozen time/ID schema')
    return result

def windows(tracks, pose):
    rows=[]
    candidates=0
    for person,xy in sorted(tracks.items()):
        times=np.array(sorted(xy),dtype=np.int64)
        for run in np.split(times, np.flatnonzero(np.diff(times)!=6)+1):
            for start in range(0, len(run)-8, 9):
                frames=run[start:start+9]
                candidates+=1
                hist=[pose.get((int(t),person)) for t in frames[:2]]
                if any(q is None or not np.isfinite(q).all(1).any() for q in hist):
                    continue
                rows.append((person,frames,np.stack([xy[t] for t in frames]),np.stack(hist)))
    return rows,candidates

def stage(root, ref):
    remotes={} if root else {g:ref.TrainingArchive(*address) for g,address in ref.ARCHIVES.items()}
    combined=[];support={};digests={}
    for index,scene in enumerate(ref.SCENES):
        blobs={}
        for group,pattern in ref.MEMBERS.items():
            name=pattern.format(scene)
            raw=(root/name).read_bytes() if root else remotes[group].member(name)
            digests[name]=sha(raw)
            if digests[name]!=ref.DIGESTS[group][index]:
                raise ValueError('Invalid public member '+name)
            blobs[group]=raw
        tracks,_=ref.read_tracks(blobs['tracks'])
        timing=ref.odometry(blobs['odom'],{f for ts in tracks.values() for f in ts})
        records=joint_records(blobs['poses'])
        rows,n=windows(tracks,records)
        if len(rows)!=EXPECTED[index]:
            raise ValueError('Unexpected joint support in '+scene)
        people={r[0] for r in rows}
        for p in people:
            t=[int(f[1]) for owner,f,_,_ in rows if owner==p]
            if len(t)>1 and min(np.diff(t))<54:
                raise ValueError('Two windows from the same person share native frames')
        q=np.array([r[3] for r in rows]);mask=np.isfinite(q).all(3)
        support[scene]={'motion_blocks_before_pose_join':n,'pose_windows':len(rows),
                        'recording_local_people':len(people),'historical_person_frames':2*len(rows),
                        'finite_joint_triplets':int(mask.sum()),'possible_joint_triplets':int(mask.size),
                        'people_window_count_min_max':[min(sum(r[0]==p for r in rows) for p in people),max(sum(r[0]==p for r in rows) for p in people)],
                        'native_odometry':timing}
        combined.extend((scene,*r) for r in rows)
    scenes,owners,frames,xy,joints=map(np.asarray,zip(*combined))
    assert len(set(zip(scenes,owners,frames[:,1])))==len(frames)
    assert np.all(np.diff(frames,axis=1)==6) and np.isfinite(xy).all()
    keys=list(zip(scenes.tolist(),owners.tolist(),frames[:,1].tolist()))
    key_hash=sha(json.dumps(keys,separators=(',',':')).encode())
    if key_hash!=INPUT_HASH:
        raise ValueError('Ordered key identity: '+key_hash)
    return {'site':scenes,'person':owners,'frame':frames,'xy':xy,'joints':joints}, {'sites':support,'member_sha256':digests,'ordered_keys_sha256':key_hash}

def centered(q,valid):
    clean=np.where(valid[...,None],q,0.)
    center=clean.sum(axis=1)/np.maximum(valid.sum(axis=1)[:,None],1)
    return np.where(valid[...,None],clean-center[:,None,:],0.)

def features(xy_history,q):
    if xy_history.ndim!=3 or xy_history.shape[1:]!=(2,2) or q.shape!=(len(xy_history),2,33,3):
        raise ValueError('Only two aligned historical frames are accepted')
    velocity=(xy_history[:,1]-xy_history[:,0])/.4
    motion=np.column_stack((velocity,np.linalg.norm(velocity,axis=1)))
    mask=np.isfinite(q).all(axis=3)
    last=centered(q[:,1],mask[:,1]);common=mask.all(axis=1)
    change=(centered(q[:,1],common)-centered(q[:,0],common))/.4
    last_features=np.column_stack((motion,last.reshape(len(q),-1),mask[:,1]))
    ordered=np.column_stack((last_features,change.reshape(len(q),-1),common))
    return {'motion':motion,'last_pose':last_features,'ordered_pose':ordered}

def residual_fit(x,residual):
    mean=x.mean(0); scale=x.std(0)
    active=scale>1e-12
    inv=np.divide(1.,scale,out=np.zeros_like(scale),where=active)
    z=(x-mean)*inv; target=residual.reshape(len(x),-1)
    intercept=target.mean(0); y=target-intercept
    # Dual form of mean squared loss + RIDGE*||B||^2 separately for the 14 outputs.
    weights=z.T@np.linalg.solve(z@z.T+len(x)*RIDGE*np.eye(len(x)),y)
    return {'mean':mean,'scale':scale,'inv_scale':inv,'intercept':intercept,'weights':weights}

def infer(x,fit,anchor):
    return anchor+(((x-fit['mean'])*fit['inv_scale'])@fit['weights']+fit['intercept']).reshape(anchor.shape)

def forecast(data):
    site,xy=data['site'],data['xy']
    x=features(xy[:,:2],data['joints'])
    anchor=np.arange(1,8)[None,:,None]*(xy[:,1]-xy[:,0])[:,None,:]
    true=xy[:,2:]-xy[:,1,None,:]
    predictions={name:np.full_like(true,np.nan) for name in ARMS}
    predictions['last_two_cv']=anchor.copy();private={};folds={}
    for fold,held in enumerate(sorted(set(site))):
        train,test=np.flatnonzero(site!=held),np.flatnonzero(site==held)
        if len(set(site[train]))!=2 or len(set(site[test]))!=1:
            raise ValueError('Expected three distinct recordings')
        key=lambda k:(site[k],int(data['person'][k]))
        if set(map(key,train))&set(map(key,test)):
            raise ValueError('Recording-local identity in both partitions')
        prefix='fold_'+str(fold)
        private[prefix+'_train_indices']=train;private[prefix+'_test_indices']=test
        folds[held]={'training_sites':sorted(set(site[train])),'train_windows':len(train),'test_windows':len(test),'prefix':prefix}
        for name in ARMS[1:]:
            fit=residual_fit(x[name][train],true[train]-anchor[train])
            for field,arr in fit.items():private[prefix+'_'+name+'_'+field]=arr
            predictions[name][test]=infer(x[name][test],fit,anchor[test])
    errors={}
    for name,pred in predictions.items():
        if not np.isfinite(pred).all():raise ValueError('Unscored rows')
        d=np.linalg.norm(pred-true,axis=-1)
        errors[name]=np.column_stack((d.mean(1),d[:,-1]))
        private[name+'_pred_relative_xy']=pred;private[name+'_errors_m']=errors[name]
    return errors,private,folds

def metrics(row):return dict(zip(F,np.asarray(row).tolist()))

def summarize(site,people,errors,draws=20000):
    rng=np.random.default_rng(2026)
    points={};boot={}
    for held in sorted(set(site)):
        rows=np.flatnonzero(site==held)
        clusters=[rows[people[rows]==p] for p in sorted(set(people[rows]))]
        count=len(clusters)
        counts=rng.multinomial(count,np.full(count,1/count),size=draws)
        denom=counts@np.array([len(g) for g in clusters])
        points[held]={name:values[rows].mean(0) for name,values in errors.items()}
        boot[held]={n:counts@np.stack([a[g].sum(0) for g in clusters])/denom[:,None] for n,a in errors.items()}
    agg={n:metrics(np.mean([p[n] for p in points.values()],axis=0)) for n in errors}
    pairs={}
    for a,b in PAIRS:
        observed={s:points[s][a]-points[s][b] for s in points}
        simulation=np.mean([p[a]-p[b] for p in boot.values()],axis=0)
        ci=np.quantile(simulation,[.025,.975],axis=0)
        pairs[a+'_minus_'+b]={'three_recording_equal_m':metrics(np.mean(list(observed.values()),axis=0)),
                              'paired_person_percentile95_m':dict(zip(F,ci.T.tolist())),
                              'by_recording_m':{s:metrics(v) for s,v in observed.items()}}
    return {'three_recording_equal_m':agg,'by_recording_m':{s:{n:metrics(v) for n,v in p.items()} for s,p in points.items()},'contrasts':pairs}

def save(path,data,support,private,folds,scores):
    path.parent.mkdir(parents=True,exist_ok=True)
    report={'experiment':'JRDB released stationary outdoor training recordings, native two-observation pose residual prototype',
            'frozen_helper_sha256':HELPERS,
            'public_source_datasets':{'HST':'https://github.com/google-research/human-scene-transformer/tree/f60bc7fd63487f7135f70b7ed4a1f25a1f32c5bd/human_scene_transformer/data',
                                      'ST':'https://github.com/vita-epfl/social-transmotion/releases/tag/ckpt_data'},
            'scope':'Three recordings already examined during development. Identities are local to a recording. Estimated 33-joint 3D HST poses, ST native 2.5 Hz tracks; no sensor milliseconds, ST manual/interpolated distinction, calibrated independent distance reference or shared identities across recordings.',
            'timing':'Native frame number / nominal 15 Hz. History 0.4s, seven future ST points through nominal 2.8s. Frame and ID alignment is exact; time gaps are nominal.',
            'protocol_sha256':LOCK,'runner_sha256':sha(Path(__file__).read_bytes()),'ridge_mean_loss_penalty':RIDGE,
            'primary':'ordered_pose_minus_motion ADE, equal recording weights','interval':'Marginal 95% percentiles, 20,000 within-recording recording-local-person cluster draws, RNG 2026. Tressider has five windows from just one person, giving zero within-recording person-resampling variance there. One Hewlett person contributes 27 windows. Same draws across arms, conditional on these fits, subjects and recordings. Interactions and uncertainty from unseen sites/training are outside this interval.',
            'site_support':support['sites'],'source_member_sha256':support['member_sha256'],'ordered_keys_sha256':support['ordered_keys_sha256'],
            'folds':folds,'features_per_arm':{'motion':3,'last_pose':135,'ordered_pose':267},
            'software':{'python':platform.python_version(),'numpy':np.__version__},**scores}
    with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
        tmp=Path(tmp);npz=tmp/'private.npz'
        relative_ms=(data['frame']-data['frame'][:,1,None])*1000//15
        arrays={**data,**private,'nominal_ms_relative_to_anchor':relative_ms,'joint_index':np.arange(33),
                'joint_finite_mask':np.isfinite(data['joints']).all(axis=3)}
        np.savez_compressed(npz,**arrays)
        parts={'aggregate.json':export(report),'private.npz':npz.read_bytes(),'run_jrdb_pose_residual.py':Path(__file__).read_bytes()}
        parts['sha256.json']=export({k:sha(v) for k,v in parts.items()})
        target=tmp/'result.zip'
        with zipfile.ZipFile(target,'w',zipfile.ZIP_STORED) as z:
            for k,v in parts.items():z.writestr(k,v)
        with zipfile.ZipFile(target) as z:
            if z.testzip():raise ValueError('Result ZIP failed CRC')
        target.replace(path)
    print('Saved',path,path.stat().st_size,sha(path.read_bytes()),flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--helpers',type=Path,required=True)
    parser.add_argument('--members-root',type=Path)
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--support',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    for name,digest in HELPERS.items():
        if sha((args.helpers/name).read_bytes())!=digest:
            raise ValueError('Wrong frozen helper '+name)
    sys.path.insert(0,str(args.helpers.resolve()))
    import evaluate_outdoor_motion as ref
    data,support=stage(args.members_root,ref)
    features(data['xy'][:,:2],data['joints'])
    if args.support:args.support.write_bytes(export(support))
    print(json.dumps(support['sites']),flush=True)
    if args.check_only:return
    if not Path('/kaggle/working').is_dir() or args.output is None:
        raise RuntimeError('Fit/score is restricted to Kaggle; use --check-only elsewhere')
    errors,private,folds=forecast(data)
    save(args.output,data,support,private,folds,summarize(data['site'],data['person'],errors))

if __name__=='__main__':main()
'''
RUNNER_SHA = 'e6f20edef20130590a41f581ccac1f2fcf95e76eab3eeb33f6af84d608e3a575'
REV = '032800e94bb0828c45d3f5195ac5cc17aa93bc4c'

def digest(payload):return hashlib.sha256(payload).hexdigest()

def request(url,headers,span=None):
    last=None
    for attempt in range(3):
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'JRDB-two-frame-release-research',**headers})
            with urllib.request.urlopen(req,timeout=60) as r:
                if span and (r.status!=206 or r.headers.get('Content-Range')!=span):
                    raise ValueError('Author archive did not return the requested exact range')
                return r.read()
        except (TimeoutError,urllib.error.URLError,ValueError) as exc:
            last=exc
            if attempt<2:time.sleep(1+attempt)
    raise RuntimeError('Failed to stage fixed public source: '+str(last)) from last

def get_members(location,helpers):
    sys.path.insert(0,str(helpers))
    import evaluate_outdoor_motion as ref
    # This only opens the nine pinned training members. Any full-response fallback is rejected.
    class SourceZip(ref.TrainingArchive):
        def read(self,start,stop):
            sep='&' if '?' in self.url else '?'
            url=self.url+sep+'jrdb_byte_range='+str(start)+'-'+str(stop)
            raw=request(url,{'Range':f'bytes={start}-{stop}','Accept-Encoding':'identity'},
                        f'bytes {start}-{stop}/{self.size}')
            if len(raw)!=stop-start+1:raise ValueError('Truncated author archive range')
            return raw
    # A mounted authorized extraction can be used if all nine member hashes match.
    mount=Path('/kaggle/input')
    for candidate in sorted(mount.rglob(ref.MEMBERS['poses'].format(ref.SCENES[0]))):
        root=candidate.parents[2]
        if all((root/p.format(s)).is_file() and digest((root/p.format(s)).read_bytes())==ref.DIGESTS[g][i]
               for g,p in ref.MEMBERS.items() for i,s in enumerate(ref.SCENES)):
            return root
    for group,address in ref.ARCHIVES.items():
        archive=SourceZip(*address)
        for i,scene in enumerate(ref.SCENES):
            member=ref.MEMBERS[group].format(scene)
            raw=archive.member(member)
            if digest(raw)!=ref.DIGESTS[group][i]:raise ValueError('Incorrect training member '+member)
            path=location/member;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
        print('Verified three author members',group,flush=True)
    return location

def validate(path):
    with zipfile.ZipFile(path) as z:
        manifest=json.loads(z.read('sha256.json'))
        assert z.testzip() is None and len(z.namelist())==4 and set(z.namelist())=={'aggregate.json','private.npz','run_jrdb_pose_residual.py','sha256.json'}
        assert set(manifest)==set(z.namelist())-{'sha256.json'} and all(digest(z.read(n))==h for n,h in manifest.items())
        report=json.loads(z.read('aggregate.json'))
        assert manifest['run_jrdb_pose_residual.py']==report['runner_sha256']==RUNNER_SHA
        assert sum(s['pose_windows'] for s in report['site_support'].values())==138
        assert report['ordered_keys_sha256']=='8a7bdd1223cece22fc2f622cc6a744040cd9e7458bc413108ba26727fe097d01'

def run():
    import importlib.util
    out=Path('/kaggle/working')
    if not out.is_dir():raise RuntimeError('Choose a Kaggle CPU worker with Internet enabled')
    final=out/'jrdb_native_pose_residual.zip'
    if final.exists():
        validate(final);print('Reusing validated prior result',final,digest(final.read_bytes()),flush=True);return
    if digest(RUNNER.encode())!=RUNNER_SHA:raise ValueError('Embedded runner hash mismatch')
    with tempfile.TemporaryDirectory(prefix='jrdb_two_frames_') as tmp:
        root=Path(tmp);helpers=root/'helpers';helpers.mkdir()
        script=root/'run_jrdb_pose_residual.py';script.write_text(RUNNER)
        source=importlib.util.spec_from_file_location('frozen_jrdb',script)
        model=importlib.util.module_from_spec(source);source.loader.exec_module(model)
        for name,expected in model.HELPERS.items():
            uri=f'https://raw.githubusercontent.com/pjyyyds/Unstructured-ShortPose-Pred/{REV}/reviews/jrdb_development/{name}'
            raw=request(uri,{})
            if digest(raw)!=expected:raise ValueError('Incorrect published helper '+name)
            (helpers/name).write_bytes(raw)
        data=get_members(root/'training_members',helpers)
        common=[sys.executable,'-u',str(script),'--helpers',str(helpers),'--members-root',str(data)]
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
        subprocess.run(common+['--check-only'],check=True,env=env)
        subprocess.run(common+['--output',str(final)],check=True,env=env)
        validate(final)
    print('Verified result archive',final,final.stat().st_size,digest(final.read_bytes()),flush=True)

if __name__=='__main__':run()
