"""Independent scalar geometry, augmented least-squares and full permutation replay."""
import argparse
import hashlib
import io
import itertools
import json
from pathlib import Path
import zipfile

import numpy as np
from scripts.benchmark_heldout_sites import load_site_windows
from src.data.actions import _estimate_causal_velocities
from src.data.features import build_motion_features

D=Path(__file__).resolve().parent
METHODS=('last_two','motion','last_geometry','ordered_geometry','order_blind')
CONTRASTS=((3,2),(3,4),(3,0),(3,1),(2,1),(4,2))
PERM=tuple(itertools.permutations(range(4)))


def h(b):return hashlib.sha256(b).hexdigest()


def key_hash(keys):
    return h(json.dumps(keys, separators=(',', ':'), sort_keys=True).encode())


def descriptor_support(valid, increments):
    complete = np.logical_and.reduce(valid, axis=1)
    different = np.max(increments, axis=1) - np.min(increments, axis=1) > 1e-12
    return dict(windows=len(valid),
                valid_frames_by_descriptor=valid.sum((0, 1)).tolist(),
                valid_last_by_descriptor=valid[:, -1].sum(0).tolist(),
                valid_all_five_by_descriptor=complete.sum(0).tolist(),
                joint_support_complete_windows=int(complete.all(1).sum()),
                any_complete_descriptor_windows=int(complete.any(1).sum()),
                different_step_values_by_descriptor=different.sum(0).tolist(),
                windows_where_permutation_can_change_features=int(different.any(1).sum()))


def confirm(cond,why):
    if not cond:raise ValueError(why)


def close(a,b,tol=1e-11):
    x,y=np.asarray(a),np.asarray(b)
    confirm(x.shape==y.shape and np.isfinite(x).all() and np.isfinite(y).all(),'Shape or finite mismatch')
    np.testing.assert_allclose(x,y,rtol=0,atol=tol)
    return float(np.max(abs(x-y)))


def metrics(p,truth):
    d=p-truth
    norm=np.hypot(d[...,0],d[...,1])
    return np.column_stack((norm.mean(1),norm[:,-1]))


def example(joints,boxes):
    vals=np.zeros((5,6));valid=np.zeros((5,6),bool)
    for t,(j,box) in enumerate(zip(joints,boxes)):
        height=float(box[3])
        if not np.isfinite(height) or height<=1.:continue
        ok=(j[:,2]>=.20)&np.isfinite(j).all(1)
        z=j[:,:2].astype(float)
        for i,ids in enumerate([(15,16),(11,13,15),(12,14,16),(5,6),(5,6,11,12),(5,6,11,12)]):
            if not all(ok[k] for k in ids):continue
            if len(ids)==2:
                dx,dy=z[ids[0]]-z[ids[1]];value=np.hypot(dx,dy)/height
            elif len(ids)==3:
                a=(z[ids[0]]-z[ids[1]])/height;b=(z[ids[2]]-z[ids[1]])/height
                lhs=np.hypot(*a);rhs=np.hypot(*b)
                if min(lhs,rhs)<=1e-6:continue
                value=np.clip((a/lhs)@(b/rhs),-1.,1.)
            else:
                sl=(z[5]+z[6])/2;hp=(z[11]+z[12])/2
                value=(sl[0]-hp[0])/height if i==4 else (hp[1]-sl[1])/height
            vals[t,i]=value;valid[t,i]=True
    return vals,valid


def fit(x,y):
    mu=x.mean(0);sd=x.std(0);sd[sd<=1e-10]=1.
    t=y.reshape(len(y),60);offset=t.mean(0)
    z=(x-mu)/sd
    design=np.concatenate((z,np.eye(x.shape[1])*np.sqrt(.1*len(x))))
    rhs=np.concatenate((t-offset,np.zeros((x.shape[1],60))))
    beta=np.linalg.lstsq(design,rhs,rcond=None)[0]
    return mu,sd,offset,beta


def verify(archive,metadata,pose,plan_path,out):
    plan=json.loads(plan_path.read_bytes())
    confirm(h(plan_path.read_bytes())=='cd39946f8b9bab092e9517cd10d538e8f04eec8af5ea7d04fae80f858515b314','Wrong register')
    with zipfile.ZipFile(archive) as z:
        confirm(z.testzip() is None and set(z.namelist())=={'private_keyed.npz','geometry_scores.json','geometry.py','run_geometry.py','plan.json'},'Bad archive')
        confirm(json.loads(z.read('plan.json'))==plan,'Changed plan')
        for n in ('geometry.py','run_geometry.py'):confirm(z.read(n)==(D/n).read_bytes(),'Changed '+n)
        public=z.read('geometry_scores.json');r=json.loads(public);raw=z.read('private_keyed.npz')
        confirm(r['private_sha256']==h(raw),'Private bytes differ')
        with np.load(io.BytesIO(raw),allow_pickle=False) as q:priv={k:q[k] for k in q.files}
    groups,_,labels=load_site_windows(metadata)
    confirm(labels==plan['source']['labels_and_calibration_sha256'] and h(pose.read_bytes())==plan['source']['pose_sha256'],'Wrong core data')
    keys=[(a.clip,a.person_id,a.t0) for s in sorted(groups) for a in groups[s]]
    confirm(len(keys)==1482 and keys==list(zip(priv['clip'].tolist(),priv['person_id'].tolist(),priv['t0'].tolist())),'Heldout rows')
    confirm(key_hash(keys)==plan['keys_sha256'],'Registered full roster')
    sites=np.array([s for s in sorted(groups) for _ in groups[s]]);np.testing.assert_array_equal(sites,priv['site'])
    xy=np.stack([a.xy for s in sorted(groups) for a in groups[s]])
    hist=xy[:,:5];truth=xy[:,5:];last=hist[:,-1]-hist[:,-2]
    anchor=hist[:,-1:]+last[:,None]*np.arange(1,31)[None,:,None]
    f=np.zeros_like(last);f[:,0]=1.
    norm=np.hypot(last[:,0],last[:,1]);active=norm/.1>=.20
    f[active]=last[active]/norm[active,None]
    rot=np.stack((f,np.stack((-f[:,1],f[:,0]),axis=1)),axis=-1)
    local=hist@rot
    motion=[]
    for row in local:
        x=build_motion_features(row,np.ones(5,np.float32))[:,:5]
        speed=_estimate_causal_velocities(row,np.ones(5,bool))[2][-3:].mean()
        motion.append(np.r_[x.ravel(),speed,speed<=.20,speed>=.60])
    motion=np.array(motion,dtype=float)
    with np.load(pose,allow_pickle=False) as cache:
        ix={k:i for i,k in enumerate(zip(cache['clip'].tolist(),cache['person_id'].tolist(),cache['t0'].tolist()))}
        kp,bb=cache['hist_kpts'],cache['hist_bbox'];records=[example(kp[ix[k]],bb[ix[k]]) for k in keys]
    geo=np.stack([v for v,m in records]);mask=np.stack([m for v,m in records])
    np.testing.assert_array_equal(mask,priv['history_descriptor_valid']);complete=mask.all(1)
    increments=np.where(complete[:,None],np.diff(geo,axis=1),0.)
    support=descriptor_support(mask,increments)
    confirm(support==plan['support_overall']==r['history_support'],'Global geometry support')
    for site,roster in plan['sites'].items():
        test=sites==site
        confirm(roster['history_support']==descriptor_support(mask[test],increments[test]),'Site geometry support')
        confirm(sorted(set(sites[~test]))==roster['source_sites'],'Training sites changed')
        for name,rows in [('train',~test),('test',test)]:
            confirm(key_hash([k for k,keep in zip(keys,rows) if keep])==roster[name+'_keys_sha256'],'Registered fold keys')
        confirm(roster['windows']==int(test.sum()) and roster['clips']==len(set(priv['clip'][test])),'Fold denominator')
    basis=np.column_stack([motion,geo[:,-1],mask[:,-1],complete])
    xstatic=np.column_stack((basis,np.zeros((len(keys),24))))
    xordered=np.column_stack((basis,increments.reshape(len(keys),24)))
    maxgeom=max(close(xstatic,priv['feature_last']),close(xordered,priv['feature_ordered']))
    est={k:np.zeros_like(anchor) for k in METHODS};est['last_two'][:]=anchor
    anchor_out=0.;maxcoef=max_orbit=maxblind=0.;max_sens=0.
    for s in sorted(groups):
        train=sites!=s;test=~train;g=plan['sites'][s]['published_training_only_multiplier']
        confirm(not(set(priv['clip'][train])&set(priv['clip'][test])),'Clip leakage')
        target=(truth[train]-anchor[train])@rot[train]
        for name,x in [('motion',motion),('last_geometry',xstatic),('ordered_geometry',xordered)]:
            mu,sd,b,coef=fit(x[train],target);cloud={}
            for field,value in [('mean',mu),('scale',sd),('intercept',b),('weights',coef)]:
                cloud[field]=priv[s+'_'+name+'_'+field]
                maxcoef=max(maxcoef,close(value,cloud[field],2e-9))
            def fwd(raw):
                local_res=((raw-mu)/sd)@coef+b
                return anchor[test]+g*(local_res.reshape(-1,30,2)@rot[test].transpose(0,2,1))
            est[name][test]=fwd(x[test])
            if name=='ordered_geometry':
                windows=np.flatnonzero(test);work=x[test].copy()
                orbit=[];maxstep=np.zeros(len(windows))
                for perm in PERM:
                    work[:,-24:]=increments[windows][:,perm,:].reshape(len(windows),-1)
                    one=fwd(work);orbit.append(one)
                    maxstep=np.maximum(maxstep,np.linalg.norm(one-est[name][test],axis=2).max(1))
                avg=np.mean(orbit,axis=0);est['order_blind'][test]=avg
                canonical=x[test].copy();canonical[:,-24:]=np.tile(np.mean(increments[windows],axis=1)[:,None,:],(1,4,1)).reshape(len(windows),-1)
                max_orbit=max(max_orbit,close(avg,fwd(canonical)))
                # Changing the starting order permutes only the orbit and preserves its mean.
                swap=(2,0,3,1)
                reindexed=[PERM.index(tuple(swap[i] for i in perm)) for perm in PERM]
                maxblind=max(maxblind,close(avg,np.mean(np.array(orbit)[reindexed],axis=0)))
                sen=r['order_sensitivity'][s]
                confirm(int((maxstep>1e-6).sum())==sen['ordered_paths_changed_under_any_permutation_above_1um'],'Wrong sensitivity count')
                max_sens=max(max_sens,close(maxstep.max(),sen['max_ordered_slot_permutation_shift_m'],1e-10),
                    close(np.linalg.norm(est[name][test]-avg,axis=2).mean(),sen['mean_ordered_vs_blind_path_shift_m'],1e-10))
    errors={};maxpath=maxerr=maxsummary=0.
    fields=('ade_m','fde_m')
    for m in METHODS:
        vals=metrics(est[m],truth);errors[m]=vals
        maxpath=max(maxpath,close(est[m],priv['prediction_'+m],1e-9))
        maxerr=max(maxerr,close(vals,priv['error_'+m],1e-9))
        for s in sorted(groups):
            maxsummary=max(maxsummary,close(vals[sites==s].mean(0),[r['sites'][s][m][k] for k in fields]))
        maxsummary=max(maxsummary,close(np.mean([vals[sites==s].mean(0) for s in sorted(groups)],0),[r['four_site_macro_m'][m][k] for k in fields]))
    rng=np.random.default_rng(2026);draw={}
    for s in sorted(groups):
        mask=sites==s;u,idx=np.unique(priv['clip'][mask],return_inverse=True)
        rr=rng.integers(len(u),size=(20000,len(u)))
        mult=np.stack([(rr==i).sum(1) for i in range(len(u))],axis=1)
        draw[s]=(mult,idx,mult@np.bincount(idx))
    for ai,bi in CONTRASTS:
        a,b=METHODS[ai],METHODS[bi];rec=r['paired'][a+'_minus_'+b];allmeans=[];allrep=[]
        for s in sorted(groups):
            dm=errors[a][sites==s]-errors[b][sites==s];mu=dm.mean(0);mult,idx,den=draw[s]
            rep=(mult@np.stack([dm[idx==j].sum(0) for j in range(mult.shape[1])]))/den[:,None]
            allmeans.append(mu);allrep.append(rep)
            maxsummary=max(maxsummary,close(mu,[rec['by_site'][s]['delta_m'][k] for k in fields]),
                close(np.percentile(rep,[2.5,97.5],axis=0).T,[rec['by_site'][s]['interval95_m'][k] for k in fields]))
        maxsummary=max(maxsummary,close(np.mean(allmeans,0),[rec['four_site_delta_m'][k] for k in fields]),
            close(np.percentile(np.mean(allrep,0),[2.5,97.5],axis=0).T,[rec['four_site_interval95_m'][k] for k in fields]))
    verified=dict(archive_sha256=h(archive.read_bytes()),original_public_sha256=h(public),verifier_sha256=h(Path(__file__).read_bytes()),
                  windows=len(keys),fits_reconstructed=12,candidate_forecasts_compared=5*len(keys),explicit_permuted_forecasts=24*len(keys),
                  max_abs_feature_difference=maxgeom,max_abs_fit_parameter_difference=maxcoef,max_abs_trajectory_difference_m=maxpath,
                  max_abs_ADE_FDE_difference_m=maxerr,max_abs_site_macro_or_interval_difference_m=maxsummary,
                  max_abs_explicit_blind_vs_linear_averaged_difference_m=max_orbit,max_abs_blind_order_check_m=maxblind,
                  max_abs_sensitivity_summary_difference_m=max_sens,
                  kaggle_version=373635870,kaggle_elapsed_seconds=68.2,registration_commit='054a55225cf789f2128baca64f0d29bb6c739279',
                  source_site_ids_and_geometry_masks_match_registration=True,
                  method='Scalar geometry, observed-only inputs and native motion features; independently standardized and solved augmented ridge by SVD least squares with common source-only shrinkage; fully enumerated 24 permutations of heldout transition vectors. Rebuilt clip bootstrap with multiplicities.')
    out.mkdir(parents=True,exist_ok=True)
    (out/'geometry_scores.json').write_bytes(public);(out/'verified_provenance.json').write_text(json.dumps(verified,indent=2)+'\n')
    print(json.dumps(verified))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('archive','metadata','pose','plan','output'):p.add_argument(k,type=Path)
    a=p.parse_args();verify(a.archive,a.metadata,a.pose,a.plan,a.output)
