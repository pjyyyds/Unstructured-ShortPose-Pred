"""Audit the frozen returned pilot using registered sources, all returned candidate paths, and all decision-time clocks. No inference or fitting here."""
import argparse, hashlib, io, json, zipfile
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
SITES=['america_crossroad','italy_park','japan_crossroad','thailand_street']
KEYS=[['0058',3999,10785],['0058',3999,10852],['0058',3999,10903],['0058',4001,10798]]
def digest(b):return hashlib.sha256(b).hexdigest()
def raw(p):return json.loads(p.read_bytes())
def verify(bundle,expected,out):
    blob=bundle.read_bytes()
    assert digest(blob)==expected
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        assert z.testzip() is None
        names=z.namelist()
        assert len(names)==len(set(names)) and set(names)=={'aggregate.json','private.npz','sha256.json','source_members.json','cloud_runner.py'}
        member=json.loads(z.read('sha256.json'));assert set(member)==set(names)-{'sha256.json'} and all(digest(z.read(k))==v for k,v in member.items())
        assert z.read('cloud_runner.py')==(ROOT/'score_cloud.py').read_bytes()
        result=json.loads(z.read('aggregate.json'));member_sources=json.loads(z.read('source_members.json'))
        with np.load(io.BytesIO(z.read('private.npz')),allow_pickle=False) as n: saved={k:n[k] for k in n.files}
    assert result['protocol_sha256']==digest((ROOT/'coverage_freeze.txt').read_bytes()) and result['expansion_freeze_sha256']==digest((ROOT/'expansion_freeze.txt').read_bytes())
    assert result['source_member_manifest_sha256']==digest(json.dumps(member_sources,sort_keys=True,separators=(',',':')).encode())
    source=ROOT/'private_source'/'V2X-Seq-SPD-Example'/'infrastructure-side'
    assert len(member_sources)==430 and all(digest((source/k).read_bytes())==v for k,v in member_sources.items())
    frames={(r['sequence_id'],int(r['frame_id'])):r for r in raw(source/'data_info.json')}
    ranks=[];measured=[];nom=[];times=[]
    for s,p,t0 in KEYS:
        period=[];world=[];anchor_cal=None
        for f in range(t0-4,t0+31):
            frm=frames[s,f];ts=int(frm['image_timestamp']);period.append(ts)
            obs=[o for o in raw(source/frm['label_camera_std_path']) if o['type']=='Pedestrian' and int(o['track_id'])==p]
            assert len(obs)==1;xyz=np.array([obs[0]['3d_location'][k] for k in 'xyz'])
            c=raw(source/frm['calib_virtuallidar_to_world_path']);R=np.array(c['rotation']);d=np.array(c['translation']).reshape(3)
            world.append(np.matmul(R,xyz)+d)
            if f==t0:anchor_cal=R
        xy=np.linalg.solve(anchor_cal,(world-np.asarray(world[4])).T).T[:,:2]
        ticks=np.array(period,dtype=np.int64);times.append(ticks);ranks.append(xy[5:]);measured.append((ticks[5:]-ticks[4])[:,None]/(ticks[4]-ticks[3])*(xy[4]-xy[3]));nom.append(np.arange(1,31)[:,None]*(xy[4]-xy[3]))
    y=np.array(ranks);clock=np.array(times);np.testing.assert_array_equal(saved['timestamps_us'],clock)
    np.testing.assert_allclose(saved['gt_relative'],y,atol=2e-8,rtol=0)
    np.testing.assert_allclose(saved['measured_cv'],measured,atol=2e-7,rtol=0);np.testing.assert_allclose(saved['nominal_cv'],nom,atol=2e-7,rtol=0)
    assert json.loads(saved['keys_json'].item())==KEYS and saved['pinnings_fold_order'].tolist()==SITES and saved['model_order'].tolist()==['M0','M1']
    p=saved['paths_relative_native_axes'];q=saved['probs'];assert p.shape==(4,4,4,2,6,30,2) and q.shape==(4,4,4,2,6)
    assert np.isfinite(p).all() and np.isfinite(q).all() and (q>=0).all();np.testing.assert_allclose(q.sum(-1),1,atol=2e-6)
    j=saved['rtmpose_kpts'];assert j.shape==(4,5,17,3) and np.isfinite(j).all() and j[:,:,:,2].max()>0
    assert saved['quality'].tolist()==result['quality']
    by=np.zeros((4,4,4,2,2))
    for idx in np.ndindex((4,4,4,2)):
        err=np.sqrt(np.square(p[idx][np.argmax(q[idx])]-y[idx[0]]).sum(-1));by[idx]=[sum(err)/30,err[-1]]
    np.testing.assert_allclose(by,saved['top1_errors_m'],rtol=0,atol=3e-8)
    def cv_err(c):
        d=np.linalg.norm(c-y,axis=-1);return np.column_stack((d.mean(1),d[:,-1]))
    c1=cv_err(measured);c2=cv_err(nom);ages=clock-clock[:,4,None];grid=np.arange(-4,31)*100000
    masks={'all_nominal_nearest': np.max(abs(ages-grid),axis=1)<=50000,'all_35_grid_within_1ms':np.max(abs(ages-grid),axis=1)<=1000,'history_spacing_within_1ms':np.max(abs(np.diff(clock[:,:5],axis=1)-100000),axis=1)<=1000,'endpoint_within_1ms':abs(ages[:,-1]-3000000)<=1000}
    actors=np.array([k[1] for k in KEYS]);np.testing.assert_array_equal(saved['actors'],actors)
    tests=0;means={}
    for angle_idx,angle in enumerate(('0','90','180','270')):
        b=result['summary_by_rotation'][angle];n0=by[:,angle_idx,:,0].mean(1);n1=by[:,angle_idx,:,1].mean(1)
        values={'M0':n0,'M1':n1,'cv_measured':c1,'cv_nominal':c2,'M1_minus_M0':n1-n0,'M0_minus_cv_measured':n0-c1,'M1_minus_cv_measured':n1-c1}
        means[angle]={}
        for sub,mask in masks.items():
            assert set(b[sub])==set(values)
            for model,v in values.items():
                ref=b[sub][model];persons=set(actors[mask]);assert ref['windows']==int(mask.sum()) and ref['people']==len(persons)
                if not sum(mask):assert ref['person_equal'] is None and ref['window_equal'] is None;continue
                for mode,avg in [('window_equal',np.mean(v[mask],axis=0)),('person_equal',np.mean([v[mask & (actors==a)].mean(0) for a in persons],axis=0))]:
                    np.testing.assert_allclose([ref[mode]['ade_m'],ref[mode]['fde_m']],avg,atol=2e-7,rtol=0);tests+=2
                if sub=='all_nominal_nearest':means[angle][model]={'window_equal':ref['window_equal'],'person_equal':ref['person_equal']}
    full={'archive_sha256':expected,'checks':{'source_members':len(member_sources),'historical_observations':20,'chronological_axis_combinations':4*4*4*2,'candidate_forecasts':4*4*4*2*6,'native_trajectory_steps_per_window':30,'checked_finite_summary_numbers':tests,'masks':{k:int(v.sum()) for k,v in masks.items()},'future_timestamp_endpoints_s':(ages[:,-1]/1e6).tolist(),'fixed_source_code_match':True,'hardware_model_forward_replayed_here':False},'native_axis_and_sensitivity':means}
    if out:out.write_text(json.dumps(full,indent=2,allow_nan=False)+'\n')
    return full
if __name__=='__main__':
    c=argparse.ArgumentParser();c.add_argument('bundle',type=Path);c.add_argument('expected_sha256');c.add_argument('output',type=Path);a=c.parse_args();checks=verify(a.bundle,a.expected_sha256,a.output);print(json.dumps(checks['checks'],indent=1));print(json.dumps(checks['native_axis_and_sensitivity']['0'],indent=1))
