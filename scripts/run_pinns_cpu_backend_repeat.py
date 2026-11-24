"""Matched-duration CPU backend repeat of the frozen 623-window PINNS protocol."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import zipfile

REV = "e8aafb53a74c0e0dcc10610ed9e74064c34bfaad"
PILOT = "b493a47536664ddb4fd66603c1bbf6b2c2e33f553520382027b3ae96b53447d0"
CACHE = "59e59e3dfa6a2b5583ce4c3fbcb010866e793a3595e43cf691dbefbfcddc64f1"
RUNNER = "d6aa83473a11740848430ca0701d6b73b0eda8f4eef3d9fb354faf5066e6d1ec"
COHORT = "11ec9713c26fdd3db6a9a887b4ac8b79fbbf13c6389b5924fcd61c315a38aa34"
DURATIONS = (5,10,20)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def item(root, name, fingerprint):
    for p in sorted(root.rglob(name)):
        if digest(p) == fingerprint:
            return p
    raise FileNotFoundError(f"The mounted inputs lack {name} with SHA-256 {fingerprint}")


def main(args):
    repo = args.repo.resolve()
    rev = subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
    if rev != REV or subprocess.check_output(["git","-C",str(repo),"status","--porcelain","--untracked-files=no"]):
        raise ValueError("Frozen full repository revision is required")
    sys.path.insert(0,str(repo))
    from scripts.stage_pinns_metadata import stage_metadata
    from scripts.run_pinns_matched_durations import (read_arrays,load_reference,validate_cache,tensors_for,
        fit_predict,Protocol,split_indices,json_digest,MODELS,save_json,atomic_write,mean_metrics,run_cpu_forward,
        UnstructuredShortPosePredictor,ATOL_M,RTOL)
    import numpy as np
    import torch

    if len(args.durations) != len(set(args.durations)) or not set(args.durations) <= set(DURATIONS):
        raise ValueError("Specify distinct lengths selected from 5, 10 and 20")
    ncpu = len(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else os.cpu_count() or 1
    quota = Path("/sys/fs/cgroup/cpu.max")
    if quota.is_file():
        limit, period = quota.read_text().split()
        if limit != "max":
            ncpu=min(ncpu,max(1,int(limit)//int(period)))
    threads = min(4,ncpu)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    if torch.cuda.is_available():
        raise RuntimeError("The CPU-repeat process must not see an active GPU")
    if args.data is None:
        args.data = stage_metadata(Path("/tmp/matched_cpu_metadata"))
    cache = args.cache or item(Path("/kaggle/input"),"matched_observations.npz",CACHE)
    pilot = args.pose or item(Path("/kaggle/input"),"pose.npz",PILOT)
    if digest(cache) != CACHE or digest(pilot) != PILOT or digest(repo/"scripts/run_pinns_matched_durations.py") != RUNNER:
        raise ValueError("Original script, observation cache or pilot cache failed its identity check")
    ref=load_reference(args.data,pilot)
    arrays=read_arrays(cache)
    validate_cache(arrays,ref)
    keys=ref.keys
    assert len(keys) == 623 and json_digest(keys) == COHORT
    groups=ref.index
    folds={}
    for site in groups:
        sites,train,test=split_indices(groups,keys,site)
        folds[site]={"train_sites":sites, "train_ids":[keys[i] for i in train],"test_ids":[keys[i] for i in test]}
    protocol=Protocol()
    base={"protocol":asdict(protocol),"cache_sha256":CACHE,"keys_sha256":COHORT,"runner_sha256":RUNNER}
    sources={name:digest(repo/name) for name in ["scripts/run_pinns_matched_durations.py","scripts/train_pinns_heldout_sites.py",
        "scripts/verify_heldout_bundle.py","scripts/run_heldout_seed_sensitivity.py","src/models/predictor.py",
        "src/data/features.py","src/losses/losses.py","src/engine/trainer.py"]}
    cpuinfo=Path("/proc/cpuinfo")
    cpu_model=next((s.split(":",1)[1].strip() for s in cpuinfo.read_text().splitlines() if s.startswith("model name")),"") if cpuinfo.is_file() else platform.machine()
    backend={"experiment":"matched_durations_cpu_backend_repeat","device":"cpu","revision":REV,
        "protocol_implementation":"frozen fit_predict and tensors_for; same seed and last epoch as the GPU design",
        "driver_sha256":digest(Path(__file__)),"source_sha256":sources,"torch":str(torch.__version__),
        "numpy":np.__version__,"python":platform.python_version(),"platform":platform.platform(),"cpu":cpu_model,
        "threads":torch.get_num_threads(),"interop_threads":torch.get_num_interop_threads(),
        "deterministic_algorithms":torch.are_deterministic_algorithms_enabled(),"mkldnn":torch.backends.mkldnn.enabled,
        "float32_matmul_precision":torch.get_float32_matmul_precision(),"torch_build_config":torch.__config__.show(),
        "env":{k:os.environ.get(k) for k in ("MKL_NUM_THREADS","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","PYTHONHASHSEED","CUDA_VISIBLE_DEVICES")}}
    backend_id={"device":"cpu","repeat":"matched_durations_cpu_backend_repeat","backend_sha256":json_digest(backend)}
    for t in args.durations:
        data=tensors_for(arrays,t)
        assert data['x_m'].shape==(623,t,9) and data['x_p'].shape==(623,t,17,6)
        assert data['fut_xy'].shape==(623,30,2) and torch.isfinite(data['fut_xy']).all() and data['fut_mask'].all()
        assert all(torch.isfinite(data[k]).all() for k in ('x_m','x_p','q_vec','ego_type_oh'))
    checks={"check_only":args.check_only,"backend":backend_id,"windows":len(keys),"poses":int(arrays["pose_inferred"].sum()),
            "counts":{s:len(i) for s,i in groups.items()},"durations":args.durations,"source_sha256":sources,
            "threads":threads,"cohort_sha256":COHORT}
    print(json.dumps(checks),flush=True)
    if args.check_only:
        return
    if not Path("/kaggle/working").is_dir() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Fitting this backend repeat is enabled on Kaggle only")
    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    def emit_copy(path,source):
        atomic_write(path,lambda tmp:tmp.write_bytes(source.read_bytes()))
    save_json(out/"backend.json",backend); save_json(out/"fold_ids.json",folds)
    emit_copy(out/"cpu_driver.py",Path(__file__))
    models={}; differences={}; macros={}; maxerr={}
    def pack(lengths,target):
        rows={str(t):models[t] for t in lengths}
        report={**base,"backend_repeat":backend_id,"windows":len(keys),"site_counts":checks["counts"],
                "pose_inferred_by_duration":{str(t):int(arrays["pose_inferred"][:,-t:].sum()) for t in lengths},
                "models_replayed":8*len(lengths),"max_cpu_vs_stored_abs_m":max(maxerr[t] for t in lengths),
                "site_metrics":rows,"four_site_macro":{str(t):macros[t] for t in lengths},
                "paired_m1_minus_m0":{str(t):differences[t] for t in lengths}}
        payload={"matched_observations.npz":cache,**{key:out/key for key in ("backend.json","fold_ids.json","cpu_driver.py")}}
        for t in lengths:
            for site in groups:
                for model in MODELS:
                    name=f"T{t}/{site}/{model}.pt"
                    payload[name]=out/name
        manifest={name:digest(path) for name,path in payload.items()}
        report_bytes=(json.dumps(report,indent=2,allow_nan=False)+"\n").encode()
        manifest["report.json"]=hashlib.sha256(report_bytes).hexdigest()
        def write(tmp):
            with zipfile.ZipFile(tmp,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
                for name,path in payload.items(): z.write(path,name)
                z.writestr("report.json",report_bytes)
                z.writestr("sha256.json",json.dumps(manifest,sort_keys=True))
            with zipfile.ZipFile(tmp) as z:
                if z.testzip() or set(z.namelist()) != set(manifest)|{"sha256.json"}:
                    raise ValueError("Incomplete archive")
        atomic_write(out/target,write)
        print(f"Completed durations {lengths}, verified {len(manifest)} members in {target}",flush=True)
    for t in args.durations:
        tensors=tensors_for(arrays,t); models[t]={}; differences[t]={}; maxerr[t]=0.
        for site in groups:
            sites,train,test=split_indices(groups,keys,site)
            models[t][site]={}; err={}
            for model in MODELS:
                path=out/f"T{t}/{site}/{model}.pt"
                fold=folds[site]
                ident={**base,"backend_repeat":backend_id,"duration":t,"site":site,"model":model,"train_sites":sites,
                       "train_ids_sha256":json_digest(fold["train_ids"]),"test_ids_sha256":json_digest(fold["test_ids"])}
                if not path.exists():
                    print(f"CPU fit T{t} {site} {model} ({len(train)} train, {len(test)} held out)",flush=True)
                    state,errors=fit_predict(model,tensors,train,test,torch.device("cpu"),protocol)
                    atomic_write(path,lambda tmp:torch.save({"identity":ident,"model_state":state,"errors":torch.from_numpy(errors)},tmp))
                ck=torch.load(path,map_location="cpu",weights_only=True)
                if ck["identity"]!=ident: raise ValueError(f"Unexpected state for {path}")
                values=ck["errors"].numpy().astype(np.float64)
                replay=run_cpu_forward(UnstructuredShortPosePredictor,model,ck["model_state"],tensors,test,protocol.batch_size)
                if not np.isfinite(replay).all() or not np.isfinite(values).all() or values.shape!=(len(test),2): raise ValueError("Invalid errors")
                np.testing.assert_allclose(replay,values,atol=ATOL_M,rtol=RTOL)
                maxerr[t]=max(maxerr[t],float(np.max(np.abs(replay-values))))
                models[t][site][model]=mean_metrics(values);err[model]=values
            differences[t][site]=mean_metrics(err["M1"]-err["M0"])
        macros[t]={v:{k:float(np.mean([row[v][k] for row in models[t].values()])) for k in ("ade_m","fde_m")} for v in MODELS}
        pack([t],f"cpu_T{t}.zip")
    if set(args.durations)==set(DURATIONS):
        pack(DURATIONS,"matched_cpu_repeat.zip")

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo",type=Path,required=True)
    for name in ["data","pose","cache"]: p.add_argument("--"+name,type=Path)
    p.add_argument("--out",type=Path,default=Path("/kaggle/working/matched_cpu_repeat"))
    p.add_argument("--durations",type=int,nargs="+",default=list(DURATIONS),choices=DURATIONS)
    p.add_argument("--check-only",action="store_true")
    main(p.parse_args())
