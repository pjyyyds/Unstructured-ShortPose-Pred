"""Four-site held-out feasibility check and source-only fits; fixed six geometric features."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from geometry import NAMES,PERMUTATIONS,blind_input,descriptors,design_features,permuted
from scripts import evaluate_pinns_pose_residuals as old
from scripts.analyze_heldout_pairs import intervals,metric_dict,paired_clip_draws
from scripts.run_heldout_seed_sensitivity import json_digest
from scripts.train_pinns_heldout_sites import ROOT,atomic_write,file_hash,save_json

D=Path(__file__).resolve().parent
LEGACY_SOURCE='930c255b240233dad3dab6cba6d9448443f681436c03669a64bed488af84a6f6'
G={'america_crossroad':.125,'italy_park':.25,'japan_crossroad':.125,'thailand_street':.125}
ARMS=('last_two','motion','last_geometry','ordered_geometry','order_blind')
PAIRS=(('ordered_geometry','last_geometry'),('ordered_geometry','order_blind'),('ordered_geometry','last_two'),
       ('ordered_geometry','motion'),('last_geometry','motion'),('order_blind','last_geometry'))


def check(condition,why):
    if not condition:raise ValueError(why)


def preflight(data,pose):
    check(file_hash(Path(old.__file__))==LEGACY_SOURCE,'Different frozen ridge evaluator')
    reference=json.loads((ROOT/'results/pinns_pose_residuals.json').read_bytes())
    check({s:v['selected_multiplier'] for s,v in reference['sites'].items()}==G,'Unverified old inner shrinkage factors')
    inp,identity=old.load_inputs(data,pose)
    keys=inp['keys'];sites=inp['sites']
    with np.load(pose,allow_pickle=False) as z:
        ck=list(zip(z['clip'].tolist(),z['person_id'].tolist(),z['t0'].tolist()))
        order={k:i for i,k in enumerate(ck)}
        ids=[order[k] for k in keys]
        geo,valid=descriptors(z['hist_kpts'][ids],z['hist_bbox'][ids])
    stat,ordered,inc=design_features(inp['features']['motion'],geo,valid)
    common=valid.all(1)
    def support(m):
        changed=np.ptp(inc[m],axis=1)>1e-12
        return dict(windows=int(m.sum()),valid_frames_by_descriptor=valid[m].sum((0,1)).tolist(),
            valid_last_by_descriptor=valid[m,-1].sum(0).tolist(),valid_all_five_by_descriptor=common[m].sum(0).tolist(),
            joint_support_complete_windows=int(common[m].all(1).sum()),any_complete_descriptor_windows=int(common[m].any(1).sum()),
            different_step_values_by_descriptor=changed.sum(0).tolist(),windows_where_permutation_can_change_features=int(changed.any(1).sum()))
    roster={}
    for s in sorted(G):
        rows=np.flatnonzero(sites==s);tr=np.flatnonzero(sites!=s)
        check(not ({keys[i][0] for i in rows}&{keys[i][0] for i in tr}),'A clip crosses folds')
        roster[s]=dict(windows=len(rows),clips=len(set(inp['clips'][rows])),source_sites=sorted(set(sites[tr])),
                       train_keys_sha256=json_digest([keys[i] for i in tr]),test_keys_sha256=json_digest([keys[i] for i in rows]),
                       published_training_only_multiplier=G[s],history_support=support(sites==s))
    plan=dict(name='Same-cohort five-frame 2D geometry ridge and permutation-blind heldout replay',source=identity,
              legacy_public_sha256=file_hash(ROOT/'results/pinns_pose_residuals.json'),driver_sha256=file_hash(Path(__file__)),
              geometry_sha256=file_hash(D/'geometry.py'),keys_sha256=json_digest(keys),sites=roster,
              geometry_names=NAMES,geometry_definition='Ankle distance, left/right internal knee cosines, shoulder width, midpoint shoulders minus hips signed x and hips minus shoulders signed y. Euclidean distances/offsets are normalized by each measured box height. Cosine requires both projected segment lengths >1e-6 box heights. All constituent joints finite with confidence>=0.20 and box height>1. No 3D or heading inference from images.',
              prior_comparison='Published 28/79/283-dimensional linear motion/final raw joints/ordered joint-delta models; new six handcrafted image-space geometry values include nonlinear distances and cosine operations. This specifies a new representation, not new observations or independent sites.',
              dimensions={'motion':28,'common':46,'static_and_dynamic':70,'transition_slots':4,'geometry_per_slot':6},
              support_overall=support(np.ones(len(sites),bool)),
              common_inputs='28 historical heading-aligned motion features; six final geometry values, six terminal-valid flags, six all-five-common masks (same in all geometry arms). Static model sees these shared historical availability flags; earlier geometry values are unavailable to it. Invalids zero, no interpolation.',
              contrast='Same ridge and 70 columns: final geometry + zeroed 24 dynamic slots vs final geometry + four signed adjacent changes of descriptors valid in all five frames.',
              blind_control='Use the same already-fitted ordered model. Uniform mean of all 24 permutations of the four transition vectors at test, moving all six columns together. Holds terminal values, masks, unordered set and total increments fixed. Canonical averaged input produces a bitwise order-invariant output; compare to explicit 24-forecast mean. Not a separately retrained control.',
              fits='From source three sites only fit 3 ridge models: motion, last_geometry, ordered_geometry (12 final fits). Same calibrated fold multiplier from the already published source-only nested grid for all arms; no new tuning on geometry or held-out scores. Validates reconstructed motion and CV heldout metrics against the old report for each site.',
              objective='Published train-only standardization and local-heading residual ridge, mean squared error plus 0.1 times squared coefficients, unpenalized intercept, 30 target x/y offsets about unchanged last-two velocity. No new hyperparameter choices after scoring.',
              primary='Ordered minus final geometry: equal-site mean single-path ADE (metres), paired whole-clip conditional 95% interval. Negative favors ordered geometry.',
              secondary_contrasts=PAIRS[1:],membership='All 1482 windows, no exclusions. Four previously analysed development sites, no independent external generalization. Prediction depends on projected 2D measurements and visibility.',
              interval='20,000 joint clip bootstrap draws with replacement within site, PCG64 seed2026, average four site means. FDE at 3.0 seconds. No finding sign guarantees.',
              frozen_status='Source-only shrinkage previously known; no fits/outcomes from this geometry design were used in specifying it.')
    inp.update(static=stat,ordered=ordered,increments=inc,valid_geometry=valid)
    return inp,reference,json.loads(json.dumps(plan))


def evaluate(inp,ref,plan,out):
    out.mkdir(parents=True,exist_ok=True)
    sites=inp['sites'];frames=inp['frames'];anchor=inp['anchor'];truth=inp['truth']
    scores={a:np.full((len(sites),2),np.nan) for a in ARMS}
    preds={a:np.empty_like(anchor) for a in ARMS}
    preds['last_two'][:]=anchor
    priv={};sensitivity={}
    for s in sorted(G):
        tr=np.flatnonzero(sites!=s);te=np.flatnonzero(sites==s);g=G[s]
        target=(truth[tr]-anchor[tr])@frames[tr]
        for arm,x in [('motion',inp['features']['motion']),('last_geometry',inp['static']),('ordered_geometry',inp['ordered'])]:
            fit=old.ridge_fit(x[tr],target)
            for k,v in fit.items():priv[f'{s}_{arm}_{k}']=v
            def predict_x(f):
                return anchor[te]+g*old.predict_world_residual(fit,f,frames[te])
            preds[arm][te]=predict_x(x[te])
            if arm=='ordered_geometry':
                blinded=blind_input(x[te]);blind=predict_x(blinded);preds['order_blind'][te]=blind
                orbit=[];peak=np.zeros(len(te));max_blind_repeat=0.
                for perm in PERMUTATIONS:
                    alt=permuted(x[te],perm)
                    repeat=blind_input(alt)
                    np.testing.assert_array_equal(repeat,blinded)
                    control=predict_x(repeat)
                    np.testing.assert_array_equal(control,blind)
                    max_blind_repeat=max(max_blind_repeat,float(np.max(abs(blind-control))))
                    candidate=predict_x(alt);orbit.append(candidate)
                    peak=np.maximum(peak,np.linalg.norm(candidate-preds[arm][te],axis=-1).max(1))
                orbit_mean=np.mean(orbit,axis=0)
                np.testing.assert_allclose(blind,orbit_mean,rtol=0,atol=5e-12)
                shift=np.linalg.norm(blind-preds[arm][te],axis=-1).mean(1)
                sensitivity[s]=dict(windows=len(te),ordered_paths_changed_under_any_permutation_above_1um=int((peak>1e-6).sum()),
                    mean_ordered_vs_blind_path_shift_m=float(shift.mean()),max_ordered_slot_permutation_shift_m=float(peak.max()),
                    max_blind_path_change_across_permutations_m=max_blind_repeat,
                    max_abs_explicit_orbit_mean_vs_invariant_prediction_m=float(np.max(abs(blind-orbit_mean))))
    for a in ARMS:
        check(np.isfinite(preds[a]).all(),'Bad forecast')
        scores[a]=old.errors(preds[a],truth)
    by_site={}
    for s in sorted(G):
        mask=sites==s
        aggregates={k:metric_dict(v[mask].mean(0)) for k,v in scores.items()}
        for k in ('motion','last_two'):
            for metric,value in aggregates[k].items():
                np.testing.assert_allclose(value,ref['sites'][s]['mean_m'][k][metric],rtol=0,atol=1e-10)
        by_site[s]=aggregates
    macro={k:metric_dict(np.mean([scores[k][sites==s].mean(0) for s in sorted(G)],axis=0)) for k in ARMS}
    paired={}
    for a,b in PAIRS:
        rng=np.random.default_rng(2026);each={};draws=[]
        for s in sorted(G):
            ids=sites==s;delta=scores[a][ids]-scores[b][ids]
            d=paired_clip_draws(delta,inp['clips'][ids],rng);draws.append(d)
            each[s]=dict(delta_m=metric_dict(delta.mean(0)),interval95_m=intervals(d))
        paired[a+'_minus_'+b]=dict(by_site=each,four_site_delta_m={k:macro[a][k]-macro[b][k] for k in macro[a]},
                                    four_site_interval95_m=intervals(np.mean(draws,axis=0)))
    keys=inp['keys']
    priv.update(clip=inp['clips'],site=sites,person_id=np.array([k[1] for k in keys]),t0=np.array([k[2] for k in keys]),
                history_descriptor_valid=inp['valid_geometry'],feature_last=inp['static'],feature_ordered=inp['ordered'])
    for a in ARMS:priv['prediction_'+a]=preds[a];priv['error_'+a]=scores[a]
    packed=out/'private_keyed.npz'
    atomic_write(packed,lambda p:old.write_arrays(p,priv))
    public=out/'geometry_scores.json';save_json(public,dict(plan_sha256=json_digest(plan),protocol=plan['name'],
        primary=plan['primary'],history_support=plan['support_overall'],sites=by_site,four_site_macro_m=macro,
        paired=paired,order_sensitivity=sensitivity,private_sha256=file_hash(packed)))
    result=out/'ordered_pose_geometry.zip'
    with zipfile.ZipFile(result,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in (packed,public,Path(__file__),D/'geometry.py'):z.write(p,p.name)
        z.writestr('plan.json',json.dumps(plan,indent=2)+'\n')
    print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('data','pose','plan'):p.add_argument(k,type=Path)
    p.add_argument('--mode',choices=('check','register','score'),required=True)
    p.add_argument('--out',type=Path,default=Path('/kaggle/working/ordered_pose_geometry'))
    a=p.parse_args();inputs,ref,plan=preflight(a.data,a.pose)
    if a.mode=='register':
        if a.plan.exists():check(json.loads(a.plan.read_bytes())==plan,'Conflicting registration')
        else:save_json(a.plan,plan)
    elif a.mode=='score':
        check(json.loads(a.plan.read_bytes())==plan,'Registration differs')
        evaluate(inputs,ref,plan,a.out)
    print(json_digest(plan))
