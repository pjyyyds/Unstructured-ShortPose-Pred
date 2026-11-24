"""Frozen five-observation M1 pose-order perturbations. Default registration performs no model inference."""
import argparse
from collections import defaultdict
from contextlib import contextmanager, nullcontext
import hashlib
import io
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import torch

from scripts.analyze_heldout_pairs import BOOTSTRAP_SEED, REPLICATES, intervals, paired_clip_draws
from scripts.benchmark_heldout_sites import load_site_windows
from scripts.audit_pinns_observations import source_frames
from scripts.run_heldout_seed_sensitivity import FIXED_SEEDS, SITE_COUNTS, json_digest
from scripts.train_pinns_heldout_sites import ROOT, file_hash, source_hash, tensorize
from scripts.verify_heldout_bundle import ATOL_M, RTOL
from src.models.predictor import UnstructuredShortPosePredictor

CONDITIONS = ("recorded", "interior_order", "static_by_joint_and_root", "disabled_residual")
STATS = ("ade_m", "fde_m", "mean_top_path_shift_m", "top_mode_changed", "absolute_gate_change")
MODEL_KEYS = ("x_m", "ego_type_oh", "x_p", "q_vec")
BUNDLE_SHA = "0bf2745ab2d5ddcf5f5fbb5d330e6c6cc800625cec301f4c7105d031bdb49d2d"


def change_geometry(pose, condition):
    """Transform only observed normalized XY; all confidence, masks, roots, scale and slots stay fixed."""
    if pose.ndim != 4 or tuple(pose.shape[1:]) != (5,17,6) or not torch.isfinite(pose).all():
        raise ValueError("Expected finite five-frame COCO17 pose features")
    if condition not in CONDITIONS:
        raise ValueError(condition)
    if condition in ("recorded","disabled_residual"):
        return pose
    altered = pose.clone()
    valid = pose[...,3] > .5
    root = pose[...,4]
    xy = pose[...,:2]
    if condition == "interior_order":
        # Both endpoints and the middle stay fixed. Swap only joints sharing a valid coordinate system.
        eligible = valid[:,1] & valid[:,3] & (root[:,1]==root[:,3])
        altered[:,1,:,:2] = torch.where(eligible[...,None],xy[:,3],xy[:,1])
        altered[:,3,:,:2] = torch.where(eligible[...,None],xy[:,1],xy[:,3])
    else:
        times = torch.arange(5,device=pose.device)[None,:,None]
        for ref in (0.,1.):
            eligible = valid & (root == ref)
            latest = torch.where(eligible,times,-1).amax(1)
            source = xy.gather(1,latest.clamp_min(0)[:,None,:,None].expand(-1,1,-1,2))
            altered[...,:2] = torch.where(eligible[...,None],source,altered[...,:2])
    return altered


@contextmanager
def disabled_residual(gate):
    # The frozen gate's h_p=None branch returns LayerNorm(h_m) and an exact zero gate.
    def clear_pose(module, args, kwargs):
        return args,{**kwargs,"h_p":None}
    handle = gate.register_forward_pre_hook(clear_pose,with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()


@torch.inference_mode()
def predict_inputs(model, historical_inputs, condition):
    """There is deliberately no target or time-after-trigger argument."""
    if set(historical_inputs) != set(MODEL_KEYS):
        raise ValueError("Predictor accepts exactly four observed input tensors")
    batch = {**historical_inputs,"x_p":change_geometry(historical_inputs["x_p"],condition)}
    ctx = disabled_residual(model.quality_gate) if condition=="disabled_residual" else nullcontext()
    with ctx:
        out = model(batch)
    candidates, prob = out["trajs"],out["mode_probs"]
    if not torch.isfinite(candidates).all() or not torch.isfinite(prob).all():
        raise ValueError("Invalid network output")
    modes = prob.argmax(-1)
    path = candidates[torch.arange(len(modes)),modes].numpy().astype(np.float64)
    gate = out["gate_weights"].reshape(len(modes)).numpy().astype(np.float64)
    if np.any(~np.isfinite(gate)) or np.any((gate < 0)|(gate > 1)) or (condition=="disabled_residual" and np.any(gate)):
        raise ValueError("Unexpected gate value")
    return path, modes.numpy(), gate


def cohort(data, archive):
    replay = json.loads((ROOT/"results/pinns_heldout_replay.json").read_text())
    if file_hash(archive) != BUNDLE_SHA or source_hash()!=replay["frozen_source_sha256"]:
        raise ValueError("Unexpected frozen bundle or feature/model source")
    with tempfile.TemporaryDirectory() as tmp,zipfile.ZipFile(archive) as z:
        pose = Path(tmp)/"pose.npz"; pose.write_bytes(z.read("pose.npz"))
        if file_hash(pose)!=replay["pose_sha256"]:
            raise ValueError("Unexpected pose archive")
        groups,_,source = load_site_windows(data)
        tensors,index,keys,_,_,_ = tensorize(groups,pose)
        if (json_digest(keys)!=replay["ordered_window_ids_sha256"] or len(set(keys))!=1482
                or {s:len(g) for s,g in index.items()}!=SITE_COUNTS or source!=replay["label_and_calibration_sha256"]):
            raise ValueError("Wrong source, windows or sites")
        if not tensors["fut_mask"].bool().all():
            raise ValueError("Incomplete future in registered cohort")
        manifest = json.loads(z.read("sha256.json"))
        checksums = {}
        for seed in FIXED_SEEDS:
            for site, ids in index.items():
                key=f"seed_{seed}/{site}/M1.pt"; raw=z.read(key); ck=torch.load(io.BytesIO(raw),map_location="cpu",weights_only=True)
                checksums[key]=hashlib.sha256(raw).hexdigest(); ident=ck["identity"]
                other = [i for s,indices in sorted(index.items()) if s!=site for i in indices]
                if (checksums[key]!=manifest[key] or list(map(tuple,ck["window_keys"]))!=[keys[i] for i in ids]
                        or set(ident["training_sites"])!=set(index)-{site} or ident["test_site"]!=site or ident["model"]!="M1"
                        or ident["protocol"]["seed"]!=seed or ident["protocol"]["epochs"]!=28
                        or ident["pose_sha256"]!=replay["pose_sha256"] or ident["source_sha256"]!=replay["frozen_source_sha256"]
                        or ident["data_sha256"]!=source or ident["ordered_ids_sha256"]!=json_digest(keys)
                        or {keys[i][0] for i in other}&{keys[i][0] for i in ids}):
                    raise ValueError(f"Invalid held-out checkpoint {key}")
                if seed!=42 and (ident["train_ids_sha256"]!=json_digest([keys[i] for i in other]) or ident["test_ids_sha256"]!=json_digest([keys[i] for i in ids])):
                    raise ValueError(f"Invalid training or evaluation membership in {key}")
                UnstructuredShortPosePredictor("M1").load_state_dict(ck["model_state"],strict=True)
    return tensors,index,keys,{"source_sha256":source_hash(),"model_archive_sha256":BUNDLE_SHA,"pose_sha256":replay["pose_sha256"],
            "labels_and_calibration_sha256":source,"window_keys_sha256":json_digest(keys),"checkpoint_hashes":checksums}


def first_sampled_flags(data, keys):
    groups=defaultdict(list)
    for i,(clip,pid,t0) in enumerate(keys):
        groups[clip].append((i,pid,t0))
    flags=np.zeros(len(keys),dtype=bool)
    for clip,rows in groups.items():
        times,*_=source_frames(data/"labels"/(clip+".txt"))
        for i,pid,t0 in rows:
            flags[i]=t0-4==min(times[pid])
    ref=json.loads((ROOT/"results/pinns_matched_histories.json").read_text())
    expected=ref["cohorts"]["removed_for_short_history"]["ordered_window_ids_sha256"]
    if flags.sum()!=449 or json_digest([k for k,f in zip(keys,flags) if f])!=expected:
        raise ValueError("The predeclared 449 first-sampled windows changed")
    return flags


def geometry_counts(pose):
    valid=pose[...,3]>.5
    eligible=valid[:,1] & valid[:,3] & (pose[:,1,:,4]==pose[:,3,:,4])
    result={"windows":len(pose),"window_joints_total":len(pose)*17,"joint_time_slots_total":len(pose)*5*17,
            "joint_time_slots_valid":int(valid.sum()),"joints_swappable_in_both_interior_slots":int(eligible.sum()),
            "fraction_all_window_joints_swappable":float(eligible.double().mean()),
            "windows_with_swappable_joint":int(eligible.any(1).sum())}
    for cond in CONDITIONS[1:3]:
        other=change_geometry(pose,cond)
        changed=(pose[...,:2]!=other[...,:2]).any(-1)
        if (not torch.equal(pose[...,2:],other[...,2:]) or not torch.equal(pose[:,-1],other[:,-1])
                or (changed & ~valid).any() or (cond=="interior_order" and changed[:,[0,2,4]].any())):
            raise ValueError("Quality, anchor, untouched slots or missingness changed")
        joints=changed.any(1)
        affected=joints.any(1)
        result[cond]={"changed_joint_time_slots":int(changed.sum()),"fraction_all_joint_time_slots":float(changed.double().mean()),
            "fraction_valid_joint_time_slots":float(changed.sum()/valid.double().sum().clamp_min(1)),
            "changed_window_joints":int(joints.sum()),"fraction_all_window_joints":float(joints.double().mean()),
            "changed_windows":int(affected.sum()),"fraction_all_windows":float(affected.double().mean())}
    return result


def row_summary(values):
    return {name:dict(zip(STATS,row.tolist())) for name,row in zip(CONDITIONS,values)}


def calculate(bundle, tensors,index,keys):
    totals=np.full((len(FIXED_SEEDS),len(keys),len(CONDITIONS),len(STATS)),np.nan)
    ref_max=0.
    with zipfile.ZipFile(bundle) as z:
        for si,seed in enumerate(FIXED_SEEDS):
            for site,ids in index.items():
                ck=torch.load(io.BytesIO(z.read(f"seed_{seed}/{site}/M1.pt")),map_location="cpu",weights_only=True)
                net=UnstructuredShortPosePredictor("M1").cpu().eval(); net.load_state_dict(ck["model_state"],strict=True)
                for first in range(0,len(ids),64):
                    batch=ids[first:first+64]; observed={n:tensors[n][batch] for n in MODEL_KEYS}
                    forecasts=[predict_inputs(net,observed,c) for c in CONDITIONS]
                    true=tensors["fut_xy"][batch].numpy().astype(np.float64)
                    p0,m0,g0=forecasts[0]
                    for ci,(path,mode,gate) in enumerate(forecasts):
                        e=np.linalg.norm(path-true,axis=2); shifts=np.linalg.norm(path-p0,axis=2)
                        totals[si,batch,ci]=np.column_stack((e.mean(1),e[:,-1],shifts.mean(1),mode!=m0,np.abs(gate-g0)))
                cloud=ck["errors"].numpy()
                ref=totals[si,ids,0,:2]
                np.testing.assert_allclose(ref,cloud,atol=ATOL_M,rtol=RTOL)
                ref_max=max(ref_max,float(np.abs(ref-cloud).max()))
    if not np.isfinite(totals).all():
        raise ValueError("Incomplete held-out replay")
    return totals,ref_max


def aggregate(totals,index,keys):
    clips=np.array([k[0] for k in keys])
    by_site={};by_seed={str(s):{} for s in FIXED_SEEDS}; all_draws=[]; all_deltas=[]; macros=[]
    rng=np.random.default_rng(BOOTSTRAP_SEED)
    def contrast(center,draws):
        return {k:{"mean":dict(zip(("ade_m","fde_m"),center[j].tolist())),
                   "paired_clip_percentile95":intervals(draws[:,2*j:2*j+2])} for j,k in enumerate(CONDITIONS[1:])}
    for site,ids in sorted(index.items()):
        v=np.take(totals,ids,axis=1)
        means=v.mean(axis=1);macros.append(means)
        for k,seed in enumerate(FIXED_SEEDS):by_seed[str(seed)][site]=row_summary(means[k])
        dv=(v[:,:,1:,:2]-v[:,:,:1,:2]).mean(0)
        draws=paired_clip_draws(dv.reshape(len(ids),-1),clips[ids],rng)
        all_draws.append(draws);all_deltas.append(dv.mean(0))
        by_site[site]={"windows":len(ids),"clips":len(set(clips[ids])),"five_seed_means":row_summary(means.mean(0)),
                       "control_minus_recorded_m":contrast(dv.mean(0),draws)}
    per_seed=np.mean(macros,axis=0)
    return {"by_seed_site":by_seed,"five_seed_by_site":by_site,
            "four_site_equal_by_seed":{str(s):row_summary(v) for s,v in zip(FIXED_SEEDS,per_seed)},
            "four_site_five_seed_equal":row_summary(per_seed.mean(0)),
            "four_site_control_minus_recorded_m":contrast(np.mean(all_deltas,axis=0),np.mean(all_draws,axis=0))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("five_seed_archive",type=Path);p.add_argument("data_root",type=Path);p.add_argument("--lock",required=True,type=Path)
    p.add_argument("--score",action="store_true");p.add_argument("--output-dir",type=Path)
    args=p.parse_args()
    torch.set_num_threads(2)
    tensors,index,keys,identity=cohort(args.data_root,args.five_seed_archive)
    first=first_sampled_flags(args.data_root,keys)
    index_first={s:i[first[i]] for s,i in index.items()}
    counts={s:geometry_counts(tensors["x_p"][i]) for s,i in index.items()}
    registration={**identity,"script_sha256":file_hash(Path(__file__)),"conditions":list(CONDITIONS),
            "seeds":list(FIXED_SEEDS),"all_windows_coverage":geometry_counts(tensors["x_p"]),"frozen_window_coverage":counts,
            "first_annotation_secondary_coverage":geometry_counts(tensors["x_p"][first]),
            "first_annotation_counts_by_site":{s:len(i) for s,i in index_first.items()},
            "registration_stage":"Before any outcomes from the new order/static/disabled conditions; earlier seed-42 missing and full reversal results were known.",
            "primary_metric":"interior_order minus recorded ADE in metres",
            "scope":"Exploratory fixed-weight intervention on four previously used physical sites. Per-site window means, "
                    "then four-site and five-seed equal means; 20,000 whole-clip paired resamples within each site, conditional on these fits.",
            "bootstrap_rng_seed":BOOTSTRAP_SEED,"replicates":REPLICATES}
    if args.lock.exists():
        if json.loads(args.lock.read_text())!=registration:raise ValueError("Different preregistered source, cohort or manipulation")
    elif args.score:raise ValueError("An existing independently fixed check-only registration is required")
    else:args.lock.write_text(json.dumps(registration,indent=2)+"\n")
    if not args.score:
        print("Checked all 1,482 inputs and 20 frozen M1 checkpoints. No prediction or score was calculated.")
        return
    out=args.output_dir
    if out is None or out.exists():raise ValueError("Supply a new output directory")
    values,ref_max=calculate(args.five_seed_archive,tensors,index,keys)
    result={"primary_all_windows":aggregate(values,index,keys),"secondary_first_sampled_449":aggregate(values,index_first,keys),
            "recorded_largest_absolute_error_discrepancy_m":ref_max}
    out.mkdir(parents=True)
    pairs=out/"private_paired_pose_order.npz"
    np.savez_compressed(pairs,values=values,clip=np.array([k[0] for k in keys]),person_id=np.array([k[1] for k in keys]),
        t0=np.array([k[2] for k in keys]),first_sampled=first,seeds=np.array(FIXED_SEEDS),conditions=np.array(CONDITIONS),fields=np.array(STATS))
    public=out/"pose_order_scores.json"
    public.write_text(json.dumps({"registration":registration,"private_pair_sha256":file_hash(pairs),**result},indent=2,allow_nan=False)+"\n")
    with zipfile.ZipFile(out/"pose_order_audit.zip","w",zipfile.ZIP_DEFLATED) as pack:
        for path in (pairs,public,args.lock):pack.write(path,path.name)
if __name__=="__main__":main()
