"""Same-worker CPU seed extension of matched PINNS histories. Check locally, train on Kaggle only."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import time
import zipfile

import numpy as np
import torch

from scripts.run_pinns_matched_fusion import (ROOT, DURATIONS, CACHE_SHA, KEY_SHA, LEGACY_TREE, PARENT_SHA,
    ATOL_M, RTOL, load_inputs, tensors_for, fit_variant, make_model, split_indices, check_predictions, parameter_digest)
from scripts.train_pinns_heldout_sites import Protocol, fit_predict, file_hash, atomic_write, save_json
from scripts.run_heldout_seed_sensitivity import json_digest
from src.data.constants import FIXED_SEEDS
from src.models.predictor import UnstructuredShortPosePredictor

REV='a2eeaf01568d74f7a01a0f74a077632e33e41b47'
FUSION_SHA='251145ae0e33f47cfae858ddfe787595844dea23beb3455b607139e81c594606'
FAMILIES=('M0','M1','M1-concat','M1b')
IMU=('x_m','ego_type_oh')
POSE=IMU+('x_p','q_vec')
SITES={'america_crossroad':342,'italy_park':32,'japan_crossroad':186,'thailand_street':63}


def need(condition,label):
    if not condition: raise ValueError(label)


def parameters(seed,lengths):
    need(tuple(FIXED_SEEDS)==(42,1024,2025,3407,4096) and seed in FIXED_SEEDS,'Unregistered seed')
    need(len(set(lengths))==len(lengths)>0 and set(lengths)<=set(DURATIONS),'Invalid or repeated duration')
    return replace(Protocol(),seed=seed)


def check_sources(data,pilot,cache):
    rev=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    need(rev==REV and not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain','--untracked-files=no']),'Wrong frozen source checkout')
    need(file_hash(ROOT/'scripts/run_pinns_matched_fusion.py')==FUSION_SHA,'Fusion factory or fit implementation changed')
    ref,arr,prior=load_inputs(data,pilot,cache)
    # These are the provenance fields provided by the frozen loader. Its original seed-42/CUDA defaults
    # describe its previous experiment and are deliberately omitted from the current CPU identity.
    sources={k:prior[k] for k in ('data_sha256','pilot_pose_sha256','matched_pose_sha256','ordered_keys_sha256','core_source_sha256','dependency_sha256','runner_and_concat_sha256')}
    need({s:len(x) for s,x in ref.index.items()}==SITES and prior['core_source_sha256']==LEGACY_TREE,'Unrecognized cohort')
    return ref,arr,sources


def backend(training):
    threads=min(4,len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count() or 1)
    quota=Path('/sys/fs/cgroup/cpu.max')
    if quota.is_file():
        max_cpu,period=quota.read_text().split()
        if max_cpu!='max':threads=min(threads,max(1,int(max_cpu)//int(period)))
    torch.set_num_threads(threads);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    if training:
        need(Path('/kaggle/working').is_dir() and Path('/kaggle/input').is_dir(),'Kaggle-only training')
        need(not torch.cuda.is_available() and os.environ.get('CUDA_VISIBLE_DEVICES')=='','Expected CPU execution')
        need(str(torch.__version__)=='2.11.0+cpu','Different torch training backend from the seed-42 CPU studies')
    cp=Path('/proc/cpuinfo');boot=Path('/proc/sys/kernel/random/boot_id')
    cpu=next((l.split(':',1)[1].strip() for l in cp.read_text().splitlines() if l.startswith('model name')),platform.machine()) if cp.exists() else platform.machine()
    worker=hashlib.sha256((socket.gethostname()+'|'+(boot.read_text().strip() if boot.exists() else platform.machine())).encode()).hexdigest()
    return {'device':'cpu','worker_fingerprint':worker,'cpu_model':cpu,'torch':str(torch.__version__),'numpy':np.__version__,
            'python':platform.python_version(),'platform':platform.platform(),'threads':threads,'interop_threads':1,
            'deterministic':torch.are_deterministic_algorithms_enabled(),'mkldnn':torch.backends.mkldnn.enabled,
            'float32_matmul_precision':torch.get_float32_matmul_precision(),'torch_config':torch.__config__.show(),
            'thread_environment':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONHASHSEED')}}


def make(name):
    return UnstructuredShortPosePredictor(name) if name in ('M0','M1') else make_model(name)


@torch.inference_mode()
def forward(name,state,tensors,test,batch_size):
    model=make(name).eval();model.load_state_dict(state,strict=True)
    use=IMU if name=='M0' else POSE
    paths=[];prob=[]
    for i in range(0,len(test),batch_size):
        idx=test[i:i+batch_size]
        pred=model({k:tensors[k][idx] for k in use})
        paths.append(pred['trajs']);prob.append(pred['mode_probs'])
    paths=torch.cat(paths);prob=torch.cat(prob)
    winner=paths[torch.arange(len(test)),prob.argmax(1)].double()
    d=torch.linalg.vector_norm(winner-tensors['fut_xy'][test].double(),dim=2)
    return {'trajs':paths,'probs':prob,'errors':torch.stack([d.mean(1),d[:,-1]],dim=1)}


def fit_arm(name,tensors,train,test,protocol):
    dev=torch.device('cpu')
    if name in ('M0','M1'):
        state,scores=fit_predict(name,tensors,train,test,dev,protocol)
        detail=forward(name,state,tensors,test,protocol.batch_size)
        np.testing.assert_allclose(detail['errors'].numpy(),scores,atol=ATOL_M,rtol=RTOL)
        detail['errors']=torch.from_numpy(scores)
    else:
        state,detail=fit_variant(name,tensors,train,test,dev,protocol)
    return state,detail


def check_detail(name,state,detail,tensors,test):
    need(set(detail)=={'trajs','probs','errors'},'Invalid prediction fields')
    scores=check_predictions(detail,tensors['fut_xy'][test].numpy())
    replay=forward(name,state,tensors,test,31)
    delta={}
    for k in replay:
        a,b=replay[k].numpy(),detail[k].numpy()
        np.testing.assert_allclose(a,b,atol=2e-5 if k=='probs' else ATOL_M,rtol=RTOL)
        delta[k]=float(np.max(np.abs(a-b)))
    return np.asarray(scores,dtype=np.float64),delta


def immutable_json(path,obj):
    if path.exists():need(json.loads(path.read_text())==json.loads(json.dumps(obj)),f'Stale file {path.name}')
    else:save_json(path,obj)


def write_truth(path,arrays):
    if not path.exists():
        def writer(tmp):
            with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
        atomic_write(path,writer)
    with np.load(path,allow_pickle=False) as z:
        need(set(z.files)==set(arrays),'Unexpected ground-truth array fields')
        for k,value in arrays.items():np.testing.assert_array_equal(z[k],value)


def describe(a):
    return dict(zip(('ade_m','fde_m'),np.mean(a,axis=0).tolist()))


def pack(out,lengths,seed,base,results):
    chosen={str(t):results[t] for t in lengths}
    summary={**base,'seed':seed,'lengths':list(lengths),'checkpoints':len(lengths)*16,
             'results':chosen,'scope':'623 matched windows across four previously studied sites. Macro is equally weighted across sites, then uniform over windows within a site. These are per-seed fixed last-epoch results. Multiple model families from the same seed are paired, not separate random-seed repetitions.'}
    filename='multiseed_s'+str(seed)+'_'+('all' if len(lengths)==3 else 'T'+str(lengths[0]))+'.zip'
    paths=[out/'folds.json',out/'backend.json',out/'driver.py']+[out/'truth'/f'{s}.npz' for s in SITES]
    paths+= [out/f'T{t}'/s/(a+'.pt') for t in lengths for s in SITES for a in FAMILIES]
    def write(tmp):
        with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
            manifest={}
            for p in paths:
                name=p.relative_to(out).as_posix();manifest[name]=file_hash(p);z.write(p,name)
            rb=(json.dumps(summary,indent=2,allow_nan=False)+'\n').encode()
            z.writestr('report.json',rb);manifest['report.json']=hashlib.sha256(rb).hexdigest()
            z.writestr('sha256.json',json.dumps(manifest,sort_keys=True))
        with zipfile.ZipFile(tmp) as z:
            need(z.testzip() is None and len(z.namelist())==len(manifest)+1,'ZIP validation failed')
    atomic_write(out/filename,write)
    print('Verified',filename,flush=True)


def run(args):
    protocol=parameters(args.seed,args.durations)
    back=backend(not args.check_only)
    ref,arrays,sources=check_sources(args.data,args.pilot,args.cache)
    keys=ref.keys;indices=ref.index
    folds={}
    for s in sorted(indices):
        others,train,test=split_indices(indices,keys,s)
        folds[s]={'train_sites':others,'train_keys':[keys[i] for i in train],'test_keys':[keys[i] for i in test]}
    base={'experiment':'matched_623_fusion_multiseed_cpu','fixed_seed_set':list(FIXED_SEEDS),'protocol':asdict(protocol),
          'source_revision':REV,'driver_sha256':file_hash(Path(__file__)),'source_identity':sources,'folds_sha256':json_digest(folds),
          'backend_sha256':json_digest(back),'site_counts':SITES,'window_count':len(keys),'clips':len(set(k[0] for k in keys)),
          'families':list(FAMILIES),'history_fps':10,'observed_span_s':{'5':0.4,'10':0.9,'20':1.9},'future_steps_10hz':30,
          'pose_frames':int(arrays['pose_inferred'].sum()),'processor_controls':{'original_video_controls':47,'zero_offset_passed':True,
            'pose_parent_sha256':PARENT_SHA,'matched_cache_sha256':CACHE_SHA,'ordered_keys_sha256':KEY_SHA}}
    sources_meta=json.loads(arrays['metadata_json'].item()); control=sources_meta['zero_offset_controls']['clips']
    base['processor_controls'].update(onnx_sha256=sources_meta['onnx_sha256'],providers=sources_meta['providers'],
        frame_mapping=sources_meta['frame_mapping'],prefix_sha256=sources_meta['prefix_sha256'],control_crops=sum(v['crops'] for v in control.values()),
        max_xy_px=max(v['xy_max_abs_px'] for v in control.values()),max_conf=max(v['confidence_max_abs'] for v in control.values()))
    if args.check_only:
        for t in args.durations:
            tensors=tensors_for(arrays,t)
            need(tensors['x_m'].shape==(623,t,9) and tensors['x_p'].shape==(623,t,17,6),'Incorrect observed suffix')
            np.testing.assert_array_equal(tensors['fut_xy'].numpy(),arrays['xy'][:,20:].astype(np.float32))
            need(tensors['fut_mask'].all() and all(torch.isfinite(v).all() for v in tensors.values()),'Invalid real inputs')
        print(json.dumps({'checked':True,'seed':args.seed,'durations':args.durations,**base},indent=2));return
    out=args.out/f'seed{args.seed}';out.mkdir(parents=True,exist_ok=True)
    immutable_json(out/'backend.json',back);immutable_json(out/'folds.json',folds)
    driver=out/'driver.py'
    if driver.exists():need(file_hash(driver)==base['driver_sha256'],'Unexpected existing driver')
    else:atomic_write(driver,lambda tmp:tmp.write_bytes(Path(__file__).read_bytes()))
    for s,rows in indices.items():
        write_truth(out/'truth'/f'{s}.npz',{'clip':arrays['clip'][rows],'person_id':arrays['person_id'][rows],'t0':arrays['t0'][rows],
                   'history_xy_20':arrays['xy'][rows,:20],'fut_xy':arrays['xy'][rows,20:].astype(np.float32),'fut_mask':np.ones((len(rows),30),dtype=bool)})
    results={}
    for t in args.durations:
        tensors=tensors_for(arrays,t);means={};deltas={};timing={};verification={}
        for s in sorted(indices):
            others,train,test=split_indices(indices,keys,s);means[s]={};vals={};timing[s]={};verification[s]={}
            for a in FAMILIES:
                path=out/f'T{t}'/s/(a+'.pt')
                identity={'experiment':base['experiment'],'seed':args.seed,'duration':t,'heldout_site':s,'model':a,'protocol':asdict(protocol),
                          'train_keys_sha256':json_digest(folds[s]['train_keys']),'test_keys_sha256':json_digest(folds[s]['test_keys']),
                          'sources_sha256':json_digest(sources),'driver_sha256':base['driver_sha256'],'backend_sha256':base['backend_sha256']}
                if not path.exists():
                    print(f'Fit s{args.seed} T{t} {s} {a}; {len(train)} train/{len(test)} test',flush=True)
                    start=time.perf_counter()
                    state,detail=fit_arm(a,tensors,train,test,protocol)
                    record={'identity':identity,'model_state':state,'predictions':detail,'fit_and_collect_seconds':time.perf_counter()-start,
                            'parameter_sha256':parameter_digest(state)}
                    atomic_write(path,lambda tmp:torch.save(record,tmp))
                ck=torch.load(path,map_location='cpu',weights_only=True)
                need(ck['identity']==identity and ck['parameter_sha256']==parameter_digest(ck['model_state']),'Stale checkpoint')
                vals[a],verification[s][a]=check_detail(a,ck['model_state'],ck['predictions'],tensors,test)
                means[s][a]=describe(vals[a]);timing[s][a]=float(ck['fit_and_collect_seconds'])
            deltas[s]={f'M1_minus_{a}':describe(vals['M1']-vals[a]) for a in FAMILIES if a!='M1'}
        macro={a:{k:float(np.mean([means[s][a][k] for s in indices])) for k in ('ade_m','fde_m')} for a in FAMILIES}
        results[t]={'site_means':means,'four_site_macro':macro,'paired_within_site':deltas,'fit_and_collect_seconds':timing,
                    'cpu_replay_max_abs_by_field':verification}
        pack(out,[t],args.seed,base,results)
    if set(args.durations)==set(DURATIONS):pack(out,sorted(args.durations),args.seed,base,results)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','pilot','cache'):p.add_argument(name,type=Path)
    p.add_argument('--seed',required=True,type=int,choices=FIXED_SEEDS)
    p.add_argument('--durations',type=int,nargs='+',choices=DURATIONS,default=list(DURATIONS))
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--out',type=Path,default=Path('/kaggle/working/matched_fusion_multiseed'))
    run(p.parse_args())
