"""Frozen M1/M0 under the preregistered full DUT detection/pose protocol."""
import argparse,io,json,tempfile,zipfile
from pathlib import Path
import numpy as np
from motion_cloud import D,SITES,WEIGHTS,ANGLES,sha,read,build,grouped
M1=['8ce6d093d1dc34dda0f57da008464f1ca66c9ce7b4638347bfa33111f020026b','d73d488a7b37b98d5b2e786a402ca61c3d14c7c35fd00cba3dfd2e57796a832d','0a2734ddbc1f5a1208552d0784fba6bc6ff0b56a40875d473295e0fd91fc46a1','aeaf2b9a8ce966c65f22013604a14c051f3116a18ada7c97715782f41ae1452b']
DETECTOR_SOURCES={'4862962a9e6454583aa2ae7593fc232422c428aa251e7b1f2e8a53cc59ff77db','99215e89b42eeb1589da103306bc02efe408b0d0e6a1d7a1ebc9c30b7212ba73'}
def load_pose_cache(cache,keys,frames,expected_keys_sha,expected_protocol_sha):
    n=len(keys)
    with zipfile.ZipFile(cache) as z:
        assert z.testzip() is None;m=json.loads(z.read('sha256.json'));assert set(m)==set(z.namelist())-{'sha256.json'} and all(sha(z.read(k))==v for k,v in m.items())
        detector=json.loads(z.read('aggregate.json'));assert detector['windows']==n and detector['mode']=='all_windows' and detector['script_sha256'] in DETECTOR_SOURCES and sha(z.read('source.py'))==detector['script_sha256'] and detector['full_cohort_keys_sha256']==expected_keys_sha
        assert detector['score_protocol_sha256']==expected_protocol_sha
        with np.load(io.BytesIO(z.read('private.npz')),allow_pickle=False) as dump:
            assert json.loads(dump['keys_json'].item())==[list(k) for k in keys];np.testing.assert_array_equal(dump['source_frame_1based'],frames[:,:5]);j=dump['kpts'];b=dump['bbox'];mask=dump['matched']
            tags=json.loads(dump['association_json'].item());lookup={(s,int(p),int(f)):t for s,p,f,t in tags}
            if len(tags)!=len(lookup) or any(t not in {'unique','missing','multiple_boxes','multiple_people_in_box','multiple_boxes_and_people'} for t in lookup.values()):raise ValueError('Duplicate or unknown association status')
            wanted={(s,p,int(f)) for (s,p,t),seq in zip(keys,frames) for f in seq[:5]}
            if wanted!=set(lookup):raise ValueError('Cache association records do not match requested histories')
            key_status=np.array([[lookup[s,p,int(f)] for f in seq[:5]] for (s,p,t),seq in zip(keys,frames)])
    assert j.shape==(n,5,17,3) and b.shape==(n,5,4) and mask.shape==(n,5) and mask.dtype==bool
    assert np.isfinite(j).all() and np.isfinite(b).all() and np.all(j[~mask]==0) and np.all(b[~mask]==0) and np.all(b[mask][:,2:]>0)
    assert np.bincount(mask.sum(1),minlength=6).tolist()==detector['matched_frames_per_window_0_to_5']
    if detector['matched_observations']!=int(mask.sum()) or detector['requested_historical_person_observations']!=n*5:raise ValueError('Cached pose counts disagree with the observations')
    expected_shape=(n,5)
    if key_status.shape!=expected_shape or not np.array_equal(key_status=='unique',mask):raise ValueError('Audit statuses differ from stored pose mask')
    return detector,j,b,mask
def run(weights,cache,out,root):
    if not Path('/kaggle/working').is_dir():raise RuntimeError('Use Kaggle CPU')
    import torch
    from src.models.predictor import UnstructuredShortPosePredictor
    from src.data.features import build_motion_features,build_pose_and_quality_features
    keys,frames,ticks,xy=build(root);n=len(keys);assert n==3066
    detector,j,b,mask=load_pose_cache(cache,keys,frames,read(D/'coverage.json')['key_manifest_sha256'],sha((D/'motion_and_detector_freeze.json').read_bytes()))
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    models={}
    for g,site in enumerate(SITES):
        for name,h in [('M0',WEIGHTS[g]),('M1',M1[g])]:
            if weights.is_file():
                with zipfile.ZipFile(weights) as z:
                    matches={f'folds/{site}/{name}.pt',f'seed_42/{site}/{name}.pt'}&set(z.namelist());assert len(matches)==1;blob=z.read(matches.pop())
            else:
                paths=[p for p in weights.rglob(name+'.pt') if p.parent.name==site and sha(p.read_bytes())==h];assert paths;blob=paths[0].read_bytes()
            assert sha(blob)==h;ck=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=True);assert ck['identity']['model']==name and ck['identity']['test_site']==site and ck['identity']['protocol']['seed']==42
            net=UnstructuredShortPosePredictor(name).eval();net.load_state_dict(ck['model_state'],strict=True);models[g,name]=net
    scene_meta={r['clip']:r for r in read(D/'coverage.json')['clips']};ones=np.ones(5,np.float32);poses=[]
    for i,(s,p,t) in enumerate(keys):
        x,q,_,_=build_pose_and_quality_features(j[i],b[i],ones,float(scene_meta[s]['frame_size'][1]));poses.append((x,q))
    xp,quality=map(np.stack,zip(*poses));history=xy[:,:5]-xy[:,4:5]
    modes=('M0','M1');paths=np.empty((n,4,4,2,6,30,2),np.float32);prob=np.empty((n,4,4,2,6),np.float32)
    for a,(c,ss) in enumerate(((1,0),(0,1),(-1,0),(0,-1))):
        rot=np.array([[c,-ss],[ss,c]],np.float32);motion=np.stack([build_motion_features(h@rot.T,ones,dt=.1) for h in history]);ego=np.zeros((n,8),np.float32);ego[np.arange(n),np.where(motion[:,-1,4]<.20,2,0)]=1
        with torch.inference_mode():
            for g in range(4):
                for v,m in enumerate(modes):
                    for l in range(0,n,128):
                        sl=slice(l,min(l+128,n));batch={k:torch.from_numpy(arr[sl]) for k,arr in [('x_m',motion),('ego_type_oh',ego),('x_p',xp),('q_vec',quality)]};o=models[g,m](batch)
                        paths[sl,a,g,v]=o['trajs'].numpy()@rot;prob[sl,a,g,v]=o['mode_probs'].numpy()
        print('Frozen angle',ANGLES[a],flush=True)
    assert np.isfinite(paths).all() and np.isfinite(prob).all();np.testing.assert_allclose(prob.sum(-1),1,atol=3e-6)
    truth=xy[:,5:]-xy[:,4:5];d=history[:,-1]-history[:,-2];ratio=(ticks[:,5:]-ticks[:,4,None])/(ticks[:,4]-ticks[:,3])[:,None]
    real_cv=d[:,None]*ratio[:,:,None];nom_cv=d[:,None]*np.arange(1,31)[None,:,None]
    chosen=np.take_along_axis(paths,np.argmax(prob,axis=4)[:,:,:,:,None,None,None],axis=4).squeeze(4)
    dist=np.linalg.norm(chosen.astype(float)-truth[:,None,None,None],axis=-1);errs=np.stack([dist.mean(-1),dist[:,:,:,:,-1]],axis=-1)
    def score(pred):
        diff=np.linalg.norm(pred-truth,axis=-1);return np.column_stack((diff.mean(1),diff[:,-1]))
    real,nom=score(real_cv),score(nom_cv);sup={'all':np.ones(n,bool),'complete_five_associations':mask.all(1),'zero_associations':~mask.any(1),'complete_five_frames_9_confident_joints':((j[:,:,:,2]>=.2).sum(2)>=9).all(1)}
    for k in range(6):sup[f'matched_{k}_frames']=mask.sum(1)==k
    scene=np.array([s for s,p,t in keys]);byclip={};tables={}
    for a,ang in enumerate(ANGLES):
        m0=errs[:,a,:,0].mean(1);m1=errs[:,a,:,1].mean(1);values={'M0':m0,'M1':m1,'M1_minus_M0':m1-m0,'cv_measured':real,'cv_nominal':nom,'M1_minus_cv_measured':m1-real}
        tables[str(ang)]={k:{m:grouped(v,sel,keys) for m,v in values.items()} for k,sel in sup.items()}
        if not a:byclip={s:{m:grouped(v,scene==s,keys) for m,v in values.items()} for s in sorted(set(scene))}
    report={'study':'DUT native shared-space fixed PINNS forecasts; one campus location, true image detector boxes, unassociated observations preserved as unavailable. Ground truth joints are unavailable.','protocol_sha256':sha((D/'motion_and_detector_freeze.json').read_bytes()),'keys_sha256':detector['full_cohort_keys_sha256'],'pose_cache_sha256':sha(cache.read_bytes()),'upstream_detector':detector,'weights_sha256':{'M0':dict(zip(SITES,WEIGHTS)),'M1':dict(zip(SITES,M1))},'scorer_sha256':sha(Path(__file__).read_bytes()),'by_axis_and_availability':tables,'by_clip':byclip,'runtime':{'torch':str(torch.__version__),'device':'cpu','actual_model_parameter_device':str(next(models[0,'M1'].parameters()).device),'requested_forecast_device':'cpu','threads':torch.get_num_threads()}}
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        p=Path(tmp);np.savez_compressed(p/'private.npz',paths=paths,probs=prob,top1_errors=errs,truth=truth,keys_json=np.array(json.dumps(keys)),source_frame_1based=frames,pts=ticks,quality=quality,matched=mask,cv_measured=real_cv,cv_nominal=nom_cv)
        members={'aggregate.json':json.dumps(report,indent=2,allow_nan=False).encode(),'private.npz':(p/'private.npz').read_bytes(),'source.py':Path(__file__).read_bytes()};members['sha256.json']=json.dumps({k:sha(b) for k,b in members.items()}).encode()
        zpath=p/'result.zip'
        with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
            for k,b in members.items():z.writestr(k,b)
        with zipfile.ZipFile(zpath) as z:assert z.testzip() is None
        zpath.replace(out)
    print(out,sha(out.read_bytes()),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('weights','cache','out'):p.add_argument(name,type=Path)
    p.add_argument('--source-root',type=Path,default=D/'private_source');a=p.parse_args();run(a.weights,a.cache,a.out,a.source_root)
