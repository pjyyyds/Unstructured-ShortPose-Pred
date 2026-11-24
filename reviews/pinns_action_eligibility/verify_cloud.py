"""Rebuild every published cell from source-label velocities and the frozen joint outputs."""
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from scripts.benchmark_heldout_sites import load_site_windows
from src.data.actions import _estimate_causal_velocities,compute_action_labels

D=Path(__file__).resolve().parent
W=D.parent
RETURN_SHA='8a5a2c909814dae53573f5c18897c07e1d1f82901d52284e585766be83618756'
PARENT_SHA='b84eaaadcaab4518ded15c83e9d746030e03693e89dfe9e23b15ef2f1fc8587e'
ARMS=('auxiliary','end_to_end','detached')
EVENTS=('start','stop')


def check(ok,msg):
    if not ok:raise ValueError(msg)


def sha(x):return hashlib.sha256(x).hexdigest()


def classify(labels,scores):
    y=np.asarray(labels,int);p=np.asarray(scores,float)
    counts=np.bincount(2*y+(p>=.5),minlength=4)
    tn,fp,fn,tp=map(int,counts)
    return dict(n=len(y),positives=tp+fn,tp=tp,fn=fn,fp=fp,tn=tn,
                recall=None if tp+fn==0 else tp/(tp+fn),precision=None if tp+fp==0 else tp/(tp+fp),
                f1=None if tp+fn==0 else 2*tp/(2*tp+fp+fn),
                brier=float(np.dot(p-y,p-y)/len(y)))


def same_tree(a,b):
    if isinstance(a,dict):
        check(set(a)==set(b),f'Key mismatch {set(a)^set(b)}')
        return max([same_tree(v,b[k]) for k,v in a.items()] or [0.])
    if a is None:check(b is None,'Undefined cell differs');return 0.
    x,y=np.asarray(a),np.asarray(b);np.testing.assert_allclose(x,y,rtol=0,atol=3e-14)
    return float(np.max(abs(x-y)))


def compare():
    returned=W/'action_eligibility_projection.zip';parent=W/'joint_actions_all.zip'
    check(sha(returned.read_bytes())==RETURN_SHA and sha(parent.read_bytes())==PARENT_SHA,'Wrong archives')
    with zipfile.ZipFile(returned) as z:
        check(z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==4,'Bad return archive')
        cloud_raw=z.read('projection_scores.json');report=json.loads(cloud_raw);fields=json.loads(z.read('field_hashes.json'))
        plancloud=json.loads(z.read('plan.json'));plan=json.loads((D/'plan.json').read_bytes())
        check(plancloud==plan and z.read('audit_projection.py')==(D/'audit_projection.py').read_bytes(),'Changed registration/code')
        expected_hash=sha(json.dumps(plan,sort_keys=True,separators=(',',':')).encode())
        check(report['plan_sha256']==expected_hash and report['source_sha256']==PARENT_SHA,'Wrong summary identity')
    groups,_,source=load_site_windows(W/'pinns_metadata')
    check(source==plan['labels_calibration_sha256'],'Source data differ')
    bysite={};counts={};arms_seen=0;trajectory_values=0;unchanged_count=0;decomp_err=0.;score_err=0.;hashes_ok=0
    site_names=sorted(groups);rng=np.random.Generator(np.random.PCG64(2026))
    with zipfile.ZipFile(parent) as old:
        check(old.testzip() is None,'Parent CRC')
        manifest=json.loads(old.read('sha256.json'))
        check(set(old.namelist())==set(manifest)|{'sha256.json'},'Parent membership')
        for k,h in manifest.items():check(sha(old.read(k))==h,'Parent file differs')
        for s in site_names:
            windows=groups[s];keys=[(w.clip,w.person_id,w.t0) for w in windows]
            xy=np.array([w.xy for w in windows])
            otherkeys=[(w.clip,w.person_id,w.t0) for other in site_names if other!=s for w in groups[other]]
            fold=json.loads(old.read(f'{s}/fold_keys.json'))
            check(list(map(tuple,fold['test_keys']))==keys and list(map(tuple,fold['train_keys']))==otherkeys,'Fold identity')
            check(not ({k[0] for k in keys}&{k[0] for k in otherkeys}),'Clip leakage')
            npz=io.BytesIO(old.read(f'{s}/targets.npz'))
            with np.load(npz,allow_pickle=False) as t:
                y=t['y_act'].astype(bool)
                check(list(zip(t['clip'].tolist(),t['person_id'].tolist(),t['t0'].tolist()))==keys,'Wrong saved keys')
                check(t['m_act'].all() and t['valid_by_class'].all(),'Invalid labels')
            vs=[];m=np.ones(5,dtype=bool)
            for w,gt in zip(xy,y):
                _,_,v=_estimate_causal_velocities(w[:5],m,dt=.1)
                vs.append(np.nanmean(v[-3:]))
                ynew,valid=compute_action_labels(w[:5],m,w[5:],np.ones(30),dt=.1)
                check(valid and np.array_equal(ynew.astype(bool),gt),'Different truth')
            pre=np.array(vs)
            allow=np.stack((pre<=.20,pre>=.60),axis=1)
            check(not (y[:,:2]&~allow).any(),'Impossible source positive')
            counts[s]={'windows':len(y),'eligible_start_stop':allow.sum(0).tolist(),'positive_start_stop':y[:,:2].sum(0).tolist()}
            bysite[s]={}
            clip=np.array([k[0] for k in keys]);names,inv=np.unique(clip,return_inverse=True)
            random=rng.integers(len(names),size=(20000,len(names)))
            multiplicity=np.stack([(random==i).sum(1) for i in range(len(names))],axis=1)
            size=np.bincount(inv);denom=multiplicity@size
            for arm in ('no_head',*ARMS):
                ck=io.BytesIO(old.read(f'{s}/{arm}.npz'))
                with np.load(ck,allow_pickle=False) as f:
                    for k,val in fields[s][arm].items():
                        check(sha(f[k].tobytes())==val,k+' mismatch');hashes_ok+=1
                    path=f['trajs'];prob=f['mode_probs'];logits=f['mode_logits']
                    check(path.shape==(len(y),6,30,2),'Invalid path shape')
                    np.testing.assert_array_equal(np.argmax(logits,axis=1),np.argmax(prob,axis=1))
                    trajectory_values+=path.size;arms_seen+=1
                    if arm=='no_head':check('act_probs' not in f.files,'Spurious action output');continue
                    raw=f['act_probs'].astype(float)
                    calc=np.exp(-np.logaddexp(0,-f['act_logits'].astype(float)))
                    score_err=max(score_err,float(abs(raw-calc).max()))
                    np.testing.assert_allclose(raw,calc,atol=2e-7,rtol=0)
                    revised=raw.copy()
                    for j in range(2):revised[~allow[:,j],j]=0.
                    check(np.array_equal(revised[:,2:],raw[:,2:]),'Unrelated action changed')
                    unchanged_count+=(revised[:,2:]==raw[:,2:]).sum()
                    delta=(revised[:,:2]-y[:,:2])**2-(raw[:,:2]-y[:,:2])**2
                    expected=np.where(~allow,-raw[:,:2]**2,0.)
                    decomp_err=max(decomp_err,float(np.abs(delta-expected).max()))
                    check(np.array_equal(delta,expected),'Pointwise squared-error identity fails')
                    sums=np.stack([delta[inv==i].sum(0) for i in range(len(names))])
                    rep=(multiplicity@sums)/denom[:,None]
                    bysite[s][arm]=(y,allow,raw,revised,rep)
    reconstructed={}
    for arm in ARMS:
        reconstructed[arm]={}
        for j,event in enumerate(EVENTS):
            per={};truth=[];raw_all=[];new_all=[];reps=[]
            for name in site_names:
                y,allow,raw,new,rep=bysite[name][arm]
                a,b=classify(y[:,j],raw[:,j]),classify(y[:,j],new[:,j])
                check(a['tp']==b['tp'] and a['fn']==b['fn'],'Unexpected event recall change')
                per[name]=dict(original=a,projected=b,eligible=int(allow[:,j].sum()),eligible_positives=int(y[:,j].sum()),
                               removed_false_positives=a['fp']-b['fp'],changed_probabilities=int((raw[:,j]!=new[:,j]).sum()),
                               paired_brier_difference_95=np.percentile(rep[:,j],[2.5,97.5]).tolist())
                truth.extend(y[:,j]);raw_all.extend(raw[:,j]);new_all.extend(new[:,j]);reps.append(rep[:,j])
            total={k:classify(truth,scores) for k,scores in [('original',raw_all),('projected',new_all)]}
            macro={k:np.mean([v[k]['brier'] for v in per.values()]) for k in total}
            reconstructed[arm][event]=dict(total_windows=len(truth),pooled=total,equal_site_brier=macro,
                 projected_minus_original_equal_site_brier=macro['projected']-macro['original'],
                 paired_site_macro_brier95=np.percentile(np.mean(reps,axis=0),[2.5,97.5]).tolist(),
                 removed_false_positives=total['original']['fp']-total['projected']['fp'],by_site=per)
    maxerr=same_tree(reconstructed,report['summary'])
    check(report['identity_audit']=={'frozen_trajectories_checked':35568,'six_mode_forecasts':5928,'trajectory_or_probability_mutations':0,'threshold_selection_labels':0,'new_fits':0,'trajectory_feedback_used':False},'Different identity audit')
    provenance=dict(kaggle_version=373631270,kaggle_duration_seconds=60.5,parent_kaggle_version=373610725,
                    archive_sha256=RETURN_SHA,parent_archive_sha256=PARENT_SHA,public_scores_sha256=sha(cloud_raw),
                    preregistered_plan_file_sha256=sha((D/'plan.json').read_bytes()),plan_sha256=expected_hash,
                    independent_verifier_sha256=sha(Path(__file__).read_bytes()),known_before_scoring='Five-frame eligibility rules and previously published low stop recall.',
                    plans_and_cloud_code_byte_or_structurally_equal=True,site_counts=counts,frozen_folds_heads=arms_seen,
                    frozen_trajectory_field_hashes_verified=hashes_ok,trajectory_coordinate_values_checked=trajectory_values,
                    per_model_per_window_candidate_paths_checked=trajectory_values//(30*2),
                    other_class_head_window_probabilities_unchanged=int(unchanged_count),
                    max_abs_raw_sigmoid_vs_frozen_score=score_err,max_abs_brier_identity_error_per_window=decomp_err,
                    max_abs_aggregates_counts_and_interval_difference=maxerr,
                    verification='Original labeler causal velocities calculated on history alone, labels recalculated from ground truth. Threshold 0.5, exact paired confusion reconstruction and per-window Brier identity; bootstrap independently expressed via clip multiplicities. Unmodified frozen forecast and mode fields identified by checksums. No model refitted or re-forwarded.')
    output=D/'public';output.mkdir(exist_ok=True)
    (output/'action_eligibility_scores.json').write_bytes(cloud_raw)
    (output/'verified_provenance.json').write_text(json.dumps(provenance,indent=2,allow_nan=False)+'\n')
    print(json.dumps(provenance,indent=2))


if __name__=='__main__':compare()
