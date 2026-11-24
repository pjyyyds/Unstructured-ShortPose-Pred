"""Rebuild seed-42 graph study initial weights on a CPU, without data, models fits or forecasts."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import zipfile

import numpy as np
import torch
from src.models.predictor import UnstructuredShortPosePredictor


def sha(b):return hashlib.sha256(b).hexdigest()


def describe(state):
    all_bytes=hashlib.sha256();info={}
    for k,v in sorted(state.items()):
        arr=v.detach().cpu().numpy()
        framed=np.ascontiguousarray(arr)
        header=b'\0'.join((k.encode(),framed.dtype.str.encode(),str(framed.shape).encode()))
        all_bytes.update(header+b'\0'+framed.tobytes())
        info[k]={'shape':list(arr.shape),'hash_shape':list(framed.shape),'dtype':framed.dtype.str,'sha256':sha(framed.tobytes())}
    return all_bytes.hexdigest(),info


def initialize(name):
    torch.manual_seed(42)
    reference=UnstructuredShortPosePredictor('M1').cpu()
    common=reference.state_dict()
    if name=='M1':return common,common
    torch.manual_seed(42)
    graph=UnstructuredShortPosePredictor('M2b').cpu()
    graph_state=graph.state_dict()
    assert set(common)<=set(graph_state)
    assert all(k.startswith('interaction.') for k in set(graph_state)-set(common))
    # Explicitly copy each shared tensor; this code does not call the study's initializer.
    for k,v in common.items():graph_state[k].copy_(v)
    assert all(torch.equal(graph_state[k],v) for k,v in common.items())
    return common,graph_state


def run(repo,reference,output):
    expected=json.loads(reference.read_bytes())
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
    files=[repo/'scripts/train_pinns_heldout_sites.py',repo/'scripts/benchmark_heldout_sites.py',*sorted((repo/'src').rglob('*.py'))]
    tree=sha(json.dumps([(p.relative_to(repo).as_posix(),sha(p.read_bytes())) for p in files]).encode())
    assert tree==expected['frozen_tree_sha256']
    machine=platform.machine()
    cpu=next((l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),'') if Path('/proc/cpuinfo').exists() else machine
    backend={'torch':str(torch.__version__),'numpy':np.__version__,'torch_build':torch.__config__.show(),
             'python':platform.python_version(),'cpu':cpu,'threads':torch.get_num_threads(),
             'interop':torch.get_num_interop_threads(),'determinism':torch.are_deterministic_algorithms_enabled(),'mkldnn':torch.backends.mkldnn.enabled}
    comparison={k:backend[k]==expected['backend'][k] for k in backend}
    allinfo={};allchecks={};samples={}
    for arm in ('M1','M2b_real','M2b_empty'):
        common,full=initialize(arm);c,_=describe(common);f,one=describe(full)
        allinfo[arm]=one
        if arm in ('M1','M2b_real'):samples[arm]=full
        else:assert all(torch.equal(v,samples['M2b_real'][k]) for k,v in full.items())
        allchecks[arm]={s:dict(common_match=c==v[arm]['shared_initial_sha256'],full_match=f==v[arm]['full_initial_sha256'],
                               common_rebuilt=c,full_rebuilt=f) for s,v in expected['expected_by_fold'].items()}
    exact=all(a['common_match'] and a['full_match'] for rec in allchecks.values() for a in rec.values())
    report={'diagnostic':'Initialization only; no training or PINNS observations. Exact equality is required to verify.',
            'reference_sha256':sha(reference.read_bytes()),'source_tree_sha256':tree,'weights_archive_sha256':expected['weights_archive_sha256'],
            'platform':machine,'environment':backend,'same_backend_fields':comparison,'checks':allchecks,
            'all_12_shared_and_full_digests_exact':exact,'tensor_meta':allinfo}
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        z.writestr('initialization.json',json.dumps(report,indent=2))
        z.writestr('reference.json',reference.read_bytes())
        import io
        memory=io.BytesIO()
        arrays={arm+'::'+k:v.detach().cpu().numpy() for arm,st in samples.items() for k,v in st.items()}
        np.savez_compressed(memory,**arrays)
        z.writestr('initial_tensors.npz',memory.getvalue())
    print('Exact 12-check result:',exact,'environment fields:',comparison,'file:',output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('repo','reference','output'):p.add_argument(n,type=Path)
    a=p.parse_args();run(a.repo,a.reference,a.output)
