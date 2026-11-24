"""Independent third cohort verification using compact released labels, calibrations and returned six candidates. No inference."""
import argparse, collections, hashlib,io,itertools,json,zipfile
from pathlib import Path
import numpy as np
from full_range import BASE,directory,ranges
def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(p.read_bytes())
def check(bundle,expected,out):
    assert sha(bundle.read_bytes())==expected
    with zipfile.ZipFile(bundle) as z:
        names=z.namelist();assert len(names)==len(set(names)) and z.testzip() is None
        manifest=json.loads(z.read('sha256.json'));assert set(manifest)==set(names)-{'sha256.json'} and all(sha(z.read(k))==v for k,v in manifest.items())
        assert z.read('cloud_runner.py')==(BASE/'third_score_cloud.py').read_bytes()
        report=json.loads(z.read('aggregate.json'));source_hash=json.loads(z.read('source_members.json'))
        with np.load(io.BytesIO(z.read('private.npz')),allow_pickle=False) as f:n=dict(f)
    cohort=read(BASE/'third_enrollment.json');keys=read(BASE/'third_enrollment_keys.json')
    assert report['coverage']==cohort and json.loads(n['keys_json'].item())==keys and len(keys)==54
    assert report['protocol_sha256']==sha((BASE/'third_cohort_freeze.json').read_bytes())
    assert report['source_member_manifest_sha256']==sha(json.dumps(source_hash,separators=(',',':'),sort_keys=True).encode())
    assert n['pinnings_fold_order'].tolist()==['america_crossroad','italy_park','japan_crossroad','thailand_street'] and n['model_order'].tolist()==['M0','M1']
    # Only source metadata are needed: fetch individual world calibration files for independent truth verification.
    member_prefix='V2X-Seq-SPD/infrastructure-side/';d=directory('metadata')
    info_name=member_prefix+'data_info.json';data=ranges('metadata',d,[info_name]);dicts=read(data[info_name]);frames={(r['sequence_id'],int(r['frame_id'])):r for r in dicts}
    allnames={info_name}
    for s,p,t0 in keys:
        for t in range(t0-4,t0+31):
            r=frames[s,t];allnames.update(member_prefix+r[k] for k in ('label_camera_std_path','calib_virtuallidar_to_world_path'))
    m=ranges('metadata',d,allnames);assert all(sha(path.read_bytes())==source_hash[k] for k,path in m.items())
    trajectories=[];times=[];sites=[]
    for s,p,t0 in keys:
        g=[];ts=[];world_to_t0=None
        for t in range(t0-4,t0+31):
            row=frames[s,t];poses=[r for r in read(m[member_prefix+row['label_camera_std_path']]) if r['type']=='Pedestrian' and int(r['track_id'])==p];assert len(poses)==1
            c=read(m[member_prefix+row['calib_virtuallidar_to_world_path']]);R=np.array(c['rotation']);tr=np.array(c['translation']).reshape(3)
            g.append(R@np.array([poses[0]['3d_location'][axis] for axis in 'xyz'])+tr);ts.append(int(row['image_timestamp']))
            if t==t0:world_to_t0=R
        trajectories.append(np.linalg.solve(world_to_t0,(g-np.asarray(g[4])).T).T[:,:2]);times.append(ts);sites.append(row['intersection_loc'])
    traj=np.array(trajectories);times=np.array(times);target=traj[:,5:];v=traj[:,4]-traj[:,3]
    np.testing.assert_array_equal(n['timestamps_us'],times);np.testing.assert_allclose(target,n['gt_relative'],atol=2e-8,rtol=0)
    ages=times-times[:,4,None];real_cv=ages[:,5:,None]/(times[:,4]-times[:,3])[:,None,None]*v[:,None,:];nom_cv=np.arange(1,31)[None,:,None]*v[:,None,:]
    np.testing.assert_allclose(n['measured_cv'],real_cv,atol=2e-7,rtol=0);np.testing.assert_allclose(n['nominal_cv'],nom_cv,atol=2e-7,rtol=0)
    y=n['paths_relative_native_axes'];q=n['probs'];assert y.shape==(54,4,4,2,6,30,2) and q.shape==y.shape[:5]
    assert np.isfinite(y).all() and np.isfinite(q).all() and (q>=0).all();np.testing.assert_allclose(q.sum(-1),1,atol=3e-6)
    assert n['rtmpose_kpts'].shape==(54,5,17,3) and np.isfinite(n['rtmpose_kpts']).all() and n['quality'].tolist()==report['quality']
    err=np.zeros((54,4,4,2,2))
    for idx in np.ndindex(err.shape[:-1]):
        selected=y[idx][np.argmax(q[idx])];dist=np.hypot(*(selected-target[idx[0]]).T);err[idx]=[dist.mean(),dist[-1]]
    np.testing.assert_allclose(err,n['top1_errors_m'],rtol=0,atol=2e-8)
    def cv_error(pred):
        dist=np.linalg.norm(pred-target,axis=-1);return np.column_stack([dist.mean(1),dist[:,-1]])
    c1=cv_error(real_cv);c2=cv_error(nom_cv);ids=np.array([str(k[0])+':'+str(k[1]) for k in keys]);np.testing.assert_array_equal(ids,n['actors']);seqs=np.array([k[0] for k in keys]);sites=np.array(sites)
    groups={'all_nominal_nearest':np.max(abs(ages-np.arange(-4,31)*100000),axis=1)<=50000,'all_35_grid_within_1ms':np.max(abs(ages-np.arange(-4,31)*100000),axis=1)<=1000,'history_spacing_within_1ms':np.max(abs(np.diff(times[:,:5])-100000),axis=1)<=1000,'endpoint_within_1ms':abs(ages[:,-1]-3000000)<=1000}
    sums=0;tables=report['summary_by_rotation']
    def assert_group(rows,mask,values):
        people=sorted(set(ids[mask]));assert set(rows)==set(values)
        count=0
        for model,v in values.items():
            r=rows[model];assert r['windows']==int(mask.sum()) and r['people']==len(people)
            if not people:assert r['window_equal'] is None and r['person_equal'] is None;continue
            for k,avg in [('window_equal',v[mask].mean(0)),('person_equal',np.mean([v[mask & (ids==p)].mean(0) for p in people],axis=0))]:
                np.testing.assert_allclose(avg,[r[k]['ade_m'],r[k]['fde_m']],rtol=0,atol=2e-7);count+=2
        return count
    for i,deg in enumerate(('0','90','180','270')):
        m0=err[:,i,:,0,:].mean(1);m1=err[:,i,:,1,:].mean(1)
        v={'M0':m0,'M1':m1,'cv_measured':c1,'cv_nominal':c2,'M1_minus_M0':m1-m0,'M0_minus_cv_measured':m0-c1,'M1_minus_cv_measured':m1-c1}
        for g,mask in groups.items():sums+=assert_group(tables[deg][g],mask,v)
        if i==0:
            for g,field in [('by_sequence',seqs),('by_intersection',sites)]:
                levels=cohort['seqs'] if g=='by_sequence' else {row['intersection'] for row in cohort['seqs'].values()};assert set(tables[g])==set(levels)
                for level,results in tables[g].items():sums+=assert_group(results,field==level,v)
    result={'archive_sha256':expected,'scoring_code_sha256':report['script_sha256'],'source_calibrations_and_labels_checked':len(m),'source_target_steps':54*30,'frozen_forecast_windows':54,'six_mode_candidate_paths':54*4*4*2*6,'finite_summary_values_verified':sums,'meters':tables['0']['all_nominal_nearest'],'masks':{k:int(v.sum()) for k,v in groups.items()},'intersection_primary':{k:r['M1_minus_M0'] for k,r in tables['by_intersection'].items()}}
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result['meters'],indent=1))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('bundle',type=Path);p.add_argument('expected_sha');p.add_argument('out',type=Path);a=p.parse_args();check(a.bundle,a.expected_sha,a.out)
