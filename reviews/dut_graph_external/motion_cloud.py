"""Fixed DUT shared-space M0/CV study. No fit. Only run actual external forecasts on Kaggle."""
import argparse, concurrent.futures, csv, hashlib, io, itertools, json, math, os, tempfile, urllib.request, zipfile
from collections import defaultdict
from pathlib import Path
import numpy as np
D=Path(__file__).resolve().parent
COMMIT='80b8c746833664cd1e5244fccad79c7f2a7cbe31'
SITES=['america_crossroad','italy_park','japan_crossroad','thailand_street']
WEIGHTS=['b58e909e67ce3b94f34cf6c5933d1c75d39184207107cba17da79291bff6a743','39bdc069331a19d1480e00ce519ec7d8076ade70ddda75225ff914e75a200e84','328c15620a68c7070fe82d0b742f7dfcb01fb22c9011ee022ae832ed249db4f4','028204da29c49d1c4efd4fa3d071ce40218e77fdc4a4d228c28dd9956721e542']
ANGLES=(0,90,180,270)
def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(p.read_bytes())
def get_w(path,site,h):
    if path.is_file():
        with zipfile.ZipFile(path) as z:
            matches={f'folds/{site}/M0.pt',f'seed_42/{site}/M0.pt'}&set(z.namelist());assert len(matches)==1;b=z.read(matches.pop())
    else:
        choices=[p for p in path.rglob('M0.pt') if p.parent.name==site and sha(p.read_bytes())==h];assert choices;b=choices[0].read_bytes()
    assert sha(b)==h;return b
def fetch(root,name,expected):
    p=root/name
    if not p.is_file():
        u=f'https://raw.githubusercontent.com/dongfang-steven-yang/vci-dataset-dut/{COMMIT}/{name}';b=urllib.request.urlopen(u,timeout=60).read();assert sha(b)==expected;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
    assert sha(p.read_bytes())==expected;return p
def build(root):
    coverage=read(D/'coverage.json');original=read(D/'shared_space_freeze.json');frozen=read(D/'motion_and_detector_freeze.json');records=[];m=coverage['members']
    assert sha((D/'shared_space_freeze.json').read_bytes())==frozen['coverage_protocol'] and frozen['n']==3066
    jobs=[(root,k,h) for k,h in m.items() if k.startswith('data/trajectories/') and k.endswith('_ped.csv') or k.startswith('data/ratios/')]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(lambda a:fetch(*a),jobs))
    for meta in coverage['clips']:
        s=meta['clip'];assert s in original['clips'] and meta['time_base']=='1/23976' and meta['pts_step_ticks']==[1000]
        scale=float((root/('data/ratios/'+s+'_ratio_pixel2meter.txt')).read_text());assert scale==meta['pixels_per_metre'] and scale>0
        rows=defaultdict(dict)
        with (root/('data/trajectories/'+s+'_traj_ped.csv')).open() as f:
            for r in csv.DictReader(f):
                actor=int(r['id']);fr=int(r['frame']);assert r['label']=='ped' and fr not in rows[actor] and 1<=fr<=meta['frame_count']
                rows[actor][fr]=(float(r['x']),float(r['y']))
        # All 5,069 inspected presentation timestamps have constant increments of 1000 ticks,
        # and zero starting PTS in each of the eleven official source streams. No sensor exposure timestamps provided.
        for actor,track in sorted(rows.items()):
            for _,group in itertools.groupby(enumerate(sorted(track)),lambda z:z[1]-z[0]):
                frames=[f for _,f in group];start=0
                native=np.array(frames,np.int64)
                while start<len(frames):
                    # nearest to origin+0.1*j, exact integer arithmetic, tie chooses earlier native frame.
                    offsets=np.array([(j*23976+4999)//10000 for j in range(35)],np.int64);ix=start+offsets
                    if ix[-1]>=len(frames):break
                    f=native[ix];assert np.array_equal(f-f[0],offsets)
                    xy=np.array([track[int(k)] for k in f]);w,h=meta['frame_size']
                    assert ((xy[:5,0]>=0)&(xy[:5,0]<w)&(xy[:5,1]>=0)&(xy[:5,1]<h)).all()
                    records.append(((s,actor,int(f[4])),f,(f-1)*1000,xy/scale));start=int(ix[-1])+1
    keys=[r[0] for r in records];assert len(keys)==3066 and sha(json.dumps(keys,separators=(',',':')).encode())==frozen['keys_sha256']
    f,t,xy=map(np.stack,zip(*[r[1:] for r in records]));return keys,f,t,xy
def grouped(e,sel,keys):
    ix=np.flatnonzero(sel)
    if not len(ix):return {'windows':0,'people':0,'clips':0,'window_equal':None,'person_equal':None,'clip_equal':None}
    people=defaultdict(list);clips=defaultdict(list)
    for i in ix:
        s,p,t=keys[i];people[s,p].append(i);clips[s].append(i)
    def avg(parts):return dict(zip(('ade_m','fde_m'),np.mean([e[j].mean(0) for j in parts],axis=0).tolist()))
    return {'windows':len(ix),'people':len(people),'clips':len(clips),'window_equal':avg([ix]),'person_equal':avg(people.values()),'clip_equal':avg(clips.values())}
def run(weights,out,root,check):
    from src.data.features import build_motion_features
    keys,frames,ticks,xy=build(root);files=[get_w(weights,s,h) for s,h in zip(SITES,WEIGHTS)]
    if check:print('Verified all 3066 frozen IDs and original clocks; four weight hashes match. No forecast.',flush=True);return
    if not Path('/kaggle/working').is_dir():raise RuntimeError('Actual evaluation requires Kaggle')
    import torch
    from src.models.predictor import UnstructuredShortPosePredictor
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    models=[]
    for site,blob in zip(SITES,files):
        ck=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=True);assert ck['identity']['model']=='M0' and ck['identity']['test_site']==site and ck['identity']['protocol']['seed']==42
        net=UnstructuredShortPosePredictor('M0').cpu().eval();net.load_state_dict(ck['model_state'],strict=True);models.append(net)
    k=len(keys);paths=np.empty((k,4,4,6,30,2),np.float32);prob=np.empty((k,4,4,6),np.float32);ones=np.ones(5,np.float32)
    hist=xy[:,:5]-xy[:,4:5]
    for a,(c,s) in enumerate(((1,0),(0,1),(-1,0),(0,-1))):
        rot=np.array([[c,-s],[s,c]],np.float32);motion=np.stack([build_motion_features(h@rot.T,ones,dt=.1) for h in hist]);ego=np.zeros((k,8),np.float32);ego[np.arange(k),np.where(motion[:,-1,4]<.20,2,0)]=1
        with torch.inference_mode():
            for fold,net in enumerate(models):
                for lo in range(0,k,128):
                    sl=slice(lo,min(k,lo+128));batch={'x_m':torch.from_numpy(motion[sl]),'ego_type_oh':torch.from_numpy(ego[sl])}
                    o=net(batch);paths[sl,a,fold]=o['trajs'].numpy()@rot;prob[sl,a,fold]=o['mode_probs'].numpy()
        print('Evaluated registered angle',ANGLES[a],flush=True)
    # Future points are accessed by the scorer after all learned predictions have been frozen.
    truth=xy[:,5:]-xy[:,4:5];diff=hist[:,-1]-hist[:,-2];elapsed=ticks[:,5:]-ticks[:,4,None]
    measured=diff[:,None]*elapsed[:,:,None]/(ticks[:,4]-ticks[:,3])[:,None,None];nominal=diff[:,None]*np.arange(1,31)[None,:,None]
    assert np.isfinite(paths).all() and np.isfinite(prob).all() and (prob>=0).all();np.testing.assert_allclose(prob.sum(-1),1,atol=3e-6)
    best=np.take_along_axis(paths,np.argmax(prob,axis=3)[:,:,:,None,None,None],axis=3).squeeze(3)
    d=np.linalg.norm(best.astype(float)-truth[:,None,None],axis=-1);errs=np.stack([d.mean(-1),d[:,:,:,-1]],axis=-1)
    def score(v):
        d=np.linalg.norm(v-truth,axis=-1);return np.column_stack((d.mean(1),d[:,-1]))
    m,n=score(measured),score(nominal);scenes=sorted({i[0] for i in keys});source=np.array([i[0] for i in keys])
    alltrue=np.ones(k,dtype=bool);axis={};byclip={};grid=np.arange(-4,31)*.1;rel=(ticks-ticks[:,4,None])/23976
    masks={'all':alltrue,'history_1ms':(np.max(abs(np.diff(rel[:,:5])-.1),axis=1)<=.001),'endpoint_1ms':abs(rel[:,-1]-3)<=.001,'full_35_grid_1ms':np.max(abs(rel-grid),axis=1)<=.001}
    for a,ang in enumerate(ANGLES):
        net=errs[:,a].mean(1);vals={'M0':net,'cv_measured':m,'cv_nominal':n,'M0_minus_cv_measured':net-m}
        axis[str(ang)]={k:{name:grouped(e,mask,keys) for name,e in vals.items()} for k,mask in masks.items()}
        if a==0:byclip={s:{name:grouped(e,source==s,keys) for name,e in vals.items()} for s in scenes}
    report={'study':'DUT one natural campus shared space, 11 clips, native source video presentation times, all enrolled people. Tracked pixel XY divided by author-supplied per-clip px/m. No independent sensor exposure clock or joint ground truth.','weights_sha256':dict(zip(SITES,WEIGHTS)),'features':'Original PINNS M0 translation-only five-observation motion features, original seed42 PINNS fold weights frozen, most probable of six modes. Whole cohort model-free constant velocity comparators, no fitting.','protocol_sha256':sha((D/'motion_and_detector_freeze.json').read_bytes()),'source_coverage_sha256':sha((D/'coverage.json').read_bytes()),'scorer_sha256':sha(Path(__file__).read_bytes()),'keys_sha256':read(D/'coverage.json')['key_manifest_sha256'],'runtime':{'torch':str(torch.__version__),'hardware':'Kaggle CPU','threads':2},'axis_and_timing':axis,'by_clip':byclip}
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        p=Path(tmp);np.savez_compressed(p/'private.npz',paths=paths,prob=prob,errors=errs,keys_json=np.array(json.dumps(keys)),source_frame_1based=frames,pts=ticks,truth_relative=truth,xy_history=hist,cv_measured=measured,cv_nominal=nominal,fold_order=np.array(SITES))
        members={'aggregate.json':json.dumps(report,indent=2,allow_nan=False).encode(),'private.npz':(p/'private.npz').read_bytes(),'cloud_scorer.py':Path(__file__).read_bytes()}
        members['sha256.json']=json.dumps({k:sha(v) for k,v in members.items()},sort_keys=True).encode()
        artifact=p/'result.zip'
        with zipfile.ZipFile(artifact,'w',zipfile.ZIP_DEFLATED) as z:
            for k,v in members.items():z.writestr(k,v)
        with zipfile.ZipFile(artifact) as z:assert z.testzip() is None
        artifact.replace(out)
    print('Created',out,out.stat().st_size,sha(out.read_bytes()),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('weights',type=Path);p.add_argument('out',type=Path);p.add_argument('--source-root',type=Path,default=D/'private_source');p.add_argument('--check-only',action='store_true')
    a=p.parse_args();run(a.weights,a.out,a.source_root,a.check_only)
