"""Replay official JRDB development members and attest the stored ridge optimum without fitting."""
from collections import defaultdict
from pathlib import Path
import argparse,hashlib,io,itertools,json,zipfile
import numpy as np

BUNDLE='2d7a2ff1cbc6a7cff6dd87b10be01ffff413abf685150df6f3e65a34540d451a'
RUNNER='e6f20edef20130590a41f581ccac1f2fcf95e76eab3eeb33f6af84d608e3a575'
LOCK='19b9d9364f671934d5a673835c2af8953f88a4f92908b6aef473cee2e93e78a3'
KEYS='8a7bdd1223cece22fc2f622cc6a744040cd9e7458bc413108ba26727fe097d01'
SUPPORT='1564c72e7130506e4e4ce751ea5a5b2123d96899a81b6d00941e9cd951d5231f'
SITES=('hewlett-packard-intersection-2019-01-24_0','huang-lane-2019-02-12_0','tressider-2019-03-16_1')
ARMS=('last_two_cv','motion','last_pose','ordered_pose')
PAIRS=(('ordered_pose','motion'),('last_pose','motion'),('ordered_pose','last_pose'),('motion','last_two_cv'),('last_pose','last_two_cv'),('ordered_pose','last_two_cv'))
FIELDS=('ade_m','fde_m')
def h(v):return hashlib.sha256(v).hexdigest()
def near(a,b,tol=1e-10):
    a,b=np.asarray(a),np.asarray(b);assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    gap=float(np.abs(a-b).max(initial=0));assert gap<=tol,(a.shape,gap,tol)
    return gap
def metric(x):return np.array([x[k] for k in FIELDS])
def source(root,site,prior):
    def read(n):
        b=(root/n).read_bytes();assert h(b)==prior[n];return b
    odom=json.loads(read('odometry/train/'+site+'.json'))['odometry']
    ordered=sorted((int(Path(k).stem),v) for k,v in odom.items())
    assert [i for i,_ in ordered]==list(range(len(ordered)))
    xyz=np.array([[o['position'][k] for k in 'xyz'] for _,o in ordered]);q=np.array([[o['orientation'][k] for k in 'xyzw'] for _,o in ordered])
    q=q/np.linalg.norm(q,axis=1,keepdims=True)
    distance=np.max(np.linalg.norm((xyz-xyz[0])[:,:2],axis=1))
    angle=float(np.degrees(2*np.arccos(np.clip(np.abs(q@q[0]),0,1))).max())
    rows=defaultdict(dict);fps=set()
    for line in read('jrdb/data/train/'+site+'_shift_0.ndjson').splitlines():
        rec=json.loads(line)
        if 'scene' in rec:fps.add(rec['scene']['fps']);continue
        r=rec['track'];f,p=int(r['f']),int(r['p']);pt=np.array([r['x'],r['y']],dtype=float)
        assert f%6==0 and f not in rows[p]
        if np.isfinite(pt).all() and np.any(pt!=0):rows[p][f]=pt
    assert fps=={2.5}
    kp={}
    for f,row in json.loads(read('labels_3d_keypoints/train/'+site+'.json'))['labels'].items():
        for p in row:
            if p['label_id'].startswith('pedestrian:'):
                key=(int(p['label_id'].split(':')[1]),int(Path(f).stem))
                assert set(p)=={'label_id','keypoints'} and key not in kp
                kp[key]=np.asarray(p['keypoints']).reshape(33,3)
    blocks=0;samples=[]
    for person in sorted(rows):
        occupied=set()
        for _,g in itertools.groupby(enumerate(sorted(rows[person])),key=lambda a:a[1]-6*a[0]):
            times=[t for j,t in g]
            for start in range(0,len(times)-8,9):
                t=times[start:start+9];blocks+=1
                obs=[kp.get((person,ti)) for ti in t[:2]]
                if any(o is None or not np.any(np.isfinite(o).all(1)) for o in obs):continue
                assert not (occupied&set(t));occupied.update(t)
                samples.append((site,person,t,np.stack([rows[person][i] for i in t]),np.stack(obs)))
    return samples,(blocks,len(ordered),distance,angle)

def make_features(xy,joints):
    motion=(xy[:,1]-xy[:,0])/0.4
    pure=np.column_stack((motion,np.hypot(motion[:,0],motion[:,1])))
    last_rows=[];change_rows=[]
    for pair in joints:
        p,q=pair
        mp=np.isfinite(p).all(1);mq=np.isfinite(q).all(1);joint=mp&mq
        centered=np.zeros_like(q);delta=np.zeros_like(q);centered[mq]=q[mq]-q[mq].mean(0)
        if joint.any():
            displacement=q[joint]-p[joint]
            delta[joint]=(displacement-displacement.mean(0))/.4
        last_rows.append(np.r_[centered.ravel(),mq]);change_rows.append(np.r_[delta.ravel(),joint])
    last=np.c_[pure,last_rows];return {'motion':pure,'last_pose':last,'ordered_pose':np.c_[last,change_rows]}

def verify(archive,root,registration,output):
    assert h(archive.read_bytes())==BUNDLE and h((registration/'PROTOCOL.md').read_bytes())==LOCK
    assert h((registration/'source_support.json').read_bytes())==SUPPORT
    pre=json.loads((registration/'source_support.json').read_text())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and len(z.namelist())==4 and set(z.namelist())=={'run_jrdb_pose_residual.py','aggregate.json','private.npz','sha256.json'}
        m=json.loads(z.read('sha256.json'));assert set(m)==set(z.namelist())-{'sha256.json'} and all(h(z.read(k))==v for k,v in m.items())
        assert m['run_jrdb_pose_residual.py']==RUNNER
        d=json.loads(z.read('aggregate.json'))
        with np.load(io.BytesIO(z.read('private.npz')),allow_pickle=False) as a:zdata={k:a[k] for k in a.files}
    assert d['runner_sha256']==RUNNER and d['protocol_sha256']==LOCK and d['ridge_mean_loss_penalty']==1
    assert d['source_member_sha256']==pre['member_sha256'] and len(pre['member_sha256'])==9
    sample=[];counts={}
    for s in SITES:
        rows,aux=source(root,s,pre['member_sha256']);sample.extend(rows)
        cnt,frames,frpos,frangle=aux;support=d['site_support'][s]
        assert support==pre['sites'][s];assert cnt==support['motion_blocks_before_pose_join'] and len(rows)==support['pose_windows']
        assert support['recording_local_people']==len({x[1] for x in rows})
        near(frpos,support['native_odometry']['max_horizontal_displacement_m']);near(frangle,support['native_odometry']['max_3d_rotation_deg'],3e-6)
        assert frames==support['native_odometry']['native_frames'] and frpos<.001 and frangle<.01
        counts[s]=(cnt,len(rows),len({x[1] for x in rows}))
    site,people,times,xy,joints=map(np.asarray,zip(*sample))
    assert counts==dict(zip(SITES,[(755,91,21),(683,42,25),(239,5,1)])) and len(sample)==138
    k=list(zip(site.tolist(),people.tolist(),times[:,1].tolist()))
    assert len(set(k))==138 and h(json.dumps(k,separators=(',',':')).encode())==d['ordered_keys_sha256']==KEYS
    for f,a in [('site',site),('person',people),('frame',times),('xy',xy),('joints',joints),('joint_finite_mask',np.isfinite(joints).all(3)),
                ('joint_index',np.arange(33)),('nominal_ms_relative_to_anchor',(times-times[:,1:2])*1000//15)]:
        np.testing.assert_array_equal(a,zdata[f])
    assert np.all(np.diff(times,axis=1)==6) and np.isfinite(joints).all() and joints.shape==(138,2,33,3)
    delta=xy[:,1]-xy[:,0];cv=delta[:,None,:]*np.arange(1,8)[None,:,None];gt=xy[:,2:]-xy[:,1:2]
    x=make_features(xy[:,:2],joints);assert {k:v.shape[1] for k,v in x.items()}==d['features_per_arm']
    predicted={name:np.full_like(gt,np.nan) for name in ARMS};predicted['last_two_cv']=cv.copy()
    resid=(gt-cv).reshape(138,-1);grad_max=[];loss_curvature=[]
    for j,s in enumerate(SITES):
        prefix='fold_'+str(j);tr=np.flatnonzero(site!=s);te=np.flatnonzero(site==s)
        config=d['folds'][s];assert config=={'training_sites':sorted(set(SITES)-{s}),'train_windows':len(tr),'test_windows':len(te),'prefix':prefix}
        assert not set(zip(site[tr],people[tr]))&set(zip(site[te],people[te]))
        for ff,sel in [('train_indices',tr),('test_indices',te)]:np.testing.assert_array_equal(sel,zdata[prefix+'_'+ff])
        for arm in ARMS[1:]:
            def field(name):return zdata[prefix+'_'+arm+'_'+name]
            mean=x[arm][tr].mean(0);sd=x[arm][tr].std(0);inv=np.zeros_like(sd);np.divide(1,sd,out=inv,where=sd>1e-12)
            near(mean,field('mean'));near(sd,field('scale'));near(inv,field('inv_scale'),2e-9)
            centered=(x[arm][tr]-mean)*inv;intercept=resid[tr].mean(0);near(intercept,field('intercept'))
            coef=field('weights');assert coef.shape==(x[arm].shape[1],14) and np.isfinite(coef).all()
            # Strictly convex because alpha=1. Check the first-order optimum directly, without solving or refitting.
            grad=centered.T@(centered@coef+intercept-resid[tr])/len(tr)+coef
            grad_max.append(float(abs(grad).max()));assert grad_max[-1]<1e-10
            relative=(((x[arm][te]-mean)*inv)@coef+intercept).reshape(len(te),7,2)
            predicted[arm][te]=cv[te]+relative
    err={};path_gaps=[];err_gaps=[]
    for arm,y in predicted.items():
        path_gaps.append(near(y,zdata[arm+'_pred_relative_xy']))
        dist=np.sqrt(((y-gt)**2).sum(-1));err[arm]=np.stack([dist.mean(1),dist[:,-1]],axis=1)
        err_gaps.append(near(err[arm],zdata[arm+'_errors_m']))
        for s in SITES:near(err[arm][site==s].mean(0),metric(d['by_recording_m'][s][arm]))
        near(np.mean([err[arm][site==s].mean(0) for s in SITES],axis=0),metric(d['three_recording_equal_m'][arm]))
    rng=np.random.default_rng(2026);sim=defaultdict(list);ci_gap=[]
    for s in SITES:
        ids=np.flatnonzero(site==s);actors=np.unique(people[ids]);G=len(actors);indicator=np.equal.outer(actors,people[ids]).astype(float)
        weights=rng.multinomial(G,np.full(G,1/G),size=20000)
        denominator=weights@indicator.sum(1)
        for a,b in PAIRS:
            difference=err[a][ids]-err[b][ids];rep=weights@(indicator@difference)/denominator[:,None]
            if G==1:near(rep,np.broadcast_to(difference.mean(0),rep.shape))
            sim[a,b].append(rep);near(difference.mean(0),metric(d['contrasts'][a+'_minus_'+b]['by_recording_m'][s]))
    for (a,b),reps in sim.items():
        item=d['contrasts'][a+'_minus_'+b];delta=err[a]-err[b]
        near(np.mean([delta[site==s].mean(0) for s in SITES],axis=0),metric(item['three_recording_equal_m']))
        ci_gap.append(near(np.quantile(np.mean(reps,axis=0),[.025,.975],axis=0).T,metric(item['paired_person_percentile95_m'])))
    d['provenance']={'kaggle_version':373627793,'reported_seconds':34.4,'raw_result_sha256':BUNDLE,
                     'raw_result_bytes':archive.stat().st_size,'public_cloud_aggregate_sha256':m['aggregate.json'],
                     'project_helper_source_commit':'032800e94bb0828c45d3f5195ac5cc17aa93bc4c',
                     'ST_authors_release':'https://github.com/vita-epfl/social-transmotion/releases/tag/ckpt_data',
                     'HST_authors_format':'https://github.com/google-research/human-scene-transformer/blob/f60bc7fd63487f7135f70b7ed4a1f25a1f32c5bd/human_scene_transformer/data/utils.py#L68-L80'}
    d['independent_verification']={'auditor_sha256':h(Path(__file__).read_bytes()),'verified_source_members':9,'matched_recording_local_people':47,
           'verified_windows':138,'matched_input_person_frames':276,'finite_joint_triplets':9108,'verified_training_solutions':9,
           'max_abs_regularized_normal_equation_residual':max(grad_max),'verified_method_windows':552,'forecast_points':3864,
           'ADE_FDE_values':1104,'max_abs_forecast_coordinate_difference_m':max(path_gaps),'max_abs_ADE_FDE_difference_m':max(err_gaps),
           'all_six_contrasts_verified':True,'max_abs_interval_endpoint_difference_m':max(ci_gap),
           'method':'Original source members independently parsed; same-person tracks tiled before pose filtering. Shapes, raw frame numbers, nominal relative time, landmarks and XY entries checked against stored inputs. Independently reconstructed training-only standardization and attested saved coefficients via the strict ridge optimum without local fitting. All paths, errors, per-recording/equal-recording means, paired contrasts and 20,000 person-cluster intervals replayed.'}
    output.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
    print(json.dumps(d['independent_verification']))
if __name__=='__main__':
    arg=argparse.ArgumentParser()
    for k in ('archive','member_root','registration','output'):arg.add_argument(k,type=Path)
    a=arg.parse_args();verify(a.archive,a.member_root,a.registration,a.output)
