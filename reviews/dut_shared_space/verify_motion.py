"""Rebuild 3066 original clip-local trajectories from raw pixel records and source MP4 clocks, then audit all saved mode selection/statistics. No model inference."""
import argparse,collections,csv,hashlib,io,json,zipfile
from pathlib import Path
import av,numpy as np
D=Path(__file__).resolve().parent
def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(p.read_bytes())
def verify(bundle,expected,out):
    assert sha(bundle.read_bytes())==expected
    with zipfile.ZipFile(bundle) as z:
        assert z.testzip() is None;hs=json.loads(z.read('sha256.json'));assert set(hs)==set(z.namelist())-{'sha256.json'} and all(sha(z.read(k))==v for k,v in hs.items())
        a=json.loads(z.read('aggregate.json')); assert z.read('cloud_scorer.py')==(D/'motion_cloud.py').read_bytes()
        with np.load(io.BytesIO(z.read('private.npz')),allow_pickle=False) as v:saved=dict(v)
    assert a['protocol_sha256']==sha((D/'motion_and_detector_freeze.json').read_bytes())
    sources=read(D/'coverage.json');assert a['source_coverage_sha256']==sha((D/'coverage.json').read_bytes())
    cases=read(D/'private_cohort.json');keys=[c['key'] for c in cases];assert len(keys)==3066 and keys==json.loads(saved['keys_json'].item())
    streams={};raw={};scales={};base=D/'private_source'
    for record in sources['clips']:
        s=record['clip'];files=[base/('vci-raw-dut/'+s+'_roi.mp4'),base/('data/trajectories/'+s+'_traj_ped.csv'),base/('data/ratios/'+s+'_ratio_pixel2meter.txt')]
        for f in files:assert sha(f.read_bytes())==sources['members'][str(f.relative_to(base))]
        with av.open(str(files[0])) as c:
            stream=c.streams.video[0];assert stream.time_base.denominator==23976 and stream.time_base.numerator==1;streams[s]=np.array(sorted(q.pts for q in c.demux(stream) if q.pts is not None))
        with files[1].open() as f:
            for r in csv.DictReader(f):raw[s,int(r['id']),int(r['frame'])]=(float(r['x']),float(r['y']))
        scales[s]=float(files[2].read_text())
    hist=[];future=[];clock=[];fra=[]
    for k,c in zip(keys,cases):
        s,p,t=k;f=np.array(c['frames_1based']);fra.append(f);ticks=streams[s][f-1];clock.append(ticks);np.testing.assert_allclose(ticks/23976,c['time_s'],rtol=0,atol=1e-14);assert f[4]==t
        pts=np.array([raw[s,p,int(n)] for n in f]);xy=(pts-pts[4])/scales[s];hist.append(xy[:5]);future.append(xy[5:])
    hist=np.array(hist);future=np.array(future);clock=np.array(clock);np.testing.assert_array_equal(saved['source_frame_1based'],fra);np.testing.assert_array_equal(saved['pts'],clock)
    np.testing.assert_allclose(saved['truth_relative'],future,rtol=0,atol=1e-12);np.testing.assert_allclose(saved['xy_history'],hist,rtol=0,atol=1e-12)
    last=-hist[:,3:4];dt=(clock[:,5:]-clock[:,4,None])/(clock[:,4]-clock[:,3])[:,None];cm=last*dt[:,:,None];cn=last*np.arange(1,31)[None,:,None]
    np.testing.assert_allclose(saved['cv_measured'],cm,rtol=0,atol=2e-12);np.testing.assert_allclose(saved['cv_nominal'],cn,rtol=0,atol=2e-12)
    path=saved['paths'];pr=saved['prob'];assert path.shape==(3066,4,4,6,30,2) and pr.shape==path.shape[:4]
    assert np.isfinite(path).all() and np.isfinite(pr).all() and (pr>=0).all();np.testing.assert_allclose(pr.sum(-1),1,atol=3e-6)
    def score(pred):
        d=pred-future;err=np.hypot(d[:,:,0],d[:,:,1]);return np.column_stack([err.mean(1),err[:,-1]])
    cv_m=score(cm);cv_n=score(cn);ind=np.arange(len(keys));err=np.empty((3066,4,4,2));angles=['0','90','180','270'];clips=np.array([k[0] for k in keys]);people=np.array([k[0]+'/'+str(k[1]) for k in keys]);rows=0
    for i in range(4):
        for fold in range(4):err[:,i,fold]=score(path[ind,i,fold,np.argmax(pr[:,i,fold],axis=1)].astype(float))
    np.testing.assert_allclose(saved['errors'],err,rtol=0,atol=2e-12)
    def audit(r,values,ix):
        nonlocal rows
        assert int(ix.sum())==r['windows'] and len(set(people[ix]))==r['people'] and len(set(clips[ix]))==r['clips']
        if not ix.any():assert all(r[k] is None for k in ('window_equal','person_equal','clip_equal'));return
        for name,avg in [('window_equal',values[ix].mean(0)),('person_equal',np.mean([values[ix & (people==s)].mean(0) for s in np.unique(people[ix])],axis=0)),('clip_equal',np.mean([values[ix&(clips==s)].mean(0) for s in np.unique(clips[ix])],axis=0))]:
            np.testing.assert_allclose([r[name]['ade_m'],r[name]['fde_m']],avg,atol=2e-10,rtol=0);rows+=2
    mask={'all':np.ones(len(keys),bool),'history_1ms':abs(np.diff(clock[:,:5])/23976-.1).max(1)<=.001,'endpoint_1ms':abs((clock[:,-1]-clock[:,4])/23976-3)<=.001,'full_35_grid_1ms':abs((clock-clock[:,4,None])/23976-np.arange(-4,31)/10).max(1)<=.001}
    for i,ang in enumerate(angles):
        net=err[:,i].mean(1);values={'M0':net,'cv_measured':cv_m,'cv_nominal':cv_n,'M0_minus_cv_measured':net-cv_m}
        for t,sel in mask.items():
            for m,v in values.items():audit(a['axis_and_timing'][ang][t][m],v,sel)
        if i==0:
            for s in set(clips):
                for m,v in values.items():audit(a['by_clip'][s][m],v,clips==s)
    res={'sha256':expected,'source_people':sources['source_people'],'clips':len(streams),'raw_pts_frames':sum(map(len,streams.values())),'held_windows':len(keys),'matched_trajectory_steps':len(keys)*30,'candidate_paths':len(keys)*4*4*6,'summary_numeric_checks':rows,'timing':{k:int(v.sum()) for k,v in mask.items()},'native_axis':a['axis_and_timing']['0']['all'],'rotation_primary':{angle:a['axis_and_timing'][angle]['all']['M0_minus_cv_measured'] for angle in angles}};out.write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res['native_axis'],indent=1))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('expected_sha');p.add_argument('out',type=Path);a=p.parse_args();verify(a.archive,a.expected_sha,a.out)
