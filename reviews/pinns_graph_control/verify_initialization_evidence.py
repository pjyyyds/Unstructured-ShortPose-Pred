"""Certify a reconstructed initial state against original checkpoint hashes, with platform diagnostics."""
import hashlib
import io
import json
import zipfile

import numpy as np
import torch
from src.models.predictor import UnstructuredShortPosePredictor


def sha(b):return hashlib.sha256(b).hexdigest()


def encoded(state):
    """Match upstream np.ascontiguousarray framing, including scalar shape (1,)."""
    h=hashlib.sha256()
    for name,v in sorted(state.items()):
        a=np.ascontiguousarray(v)
        h.update(name.encode()+b'\0'+a.dtype.str.encode()+b'\0'+str(a.shape).encode()+b'\0'+a.tobytes())
    return h.hexdigest()


def seed_states():
    torch.manual_seed(42);m=UnstructuredShortPosePredictor('M1').cpu().state_dict()
    torch.manual_seed(42);g=UnstructuredShortPosePredictor('M2b').cpu().state_dict()
    assert set(m)<=set(g)
    g={k:m[k] if k in m else v for k,v in g.items()}
    return {net:{k:v.detach().numpy() for k,v in st.items()} for net,st in [('M1',m),('M2b_real',g)]}


def certification(A,manifest,bundle,sidecar,partial):
    local=seed_states()
    actual={k:encoded(v) for k,v in local.items()}
    fields=A['folds']
    shared={row['shared_initial_sha256'] for arms in fields.values() for row in arms.values()}
    full_graph={arms[a]['full_initial_sha256'] for arms in fields.values() for a in ('M2b_real','M2b_empty')}
    assert len(fields)==4 and len(shared)==len(full_graph)==1
    common=shared.pop();graph=full_graph.pop()
    assert all(arms['M1']['full_initial_sha256']==common for arms in fields.values())
    record={'local_shared':actual['M1'],'local_full_graph':actual['M2b_real'],'local_torch':str(torch.__version__),
            'training_shared':common,'training_full_graph':graph,'local_all_digests_match':(common,graph)==(actual['M1'],actual['M2b_real'])}
    if sidecar is None:
        if partial:
            record['status']='initialization_not_certified';return common,graph,record
        assert record['local_all_digests_match'],'Cross-platform initialization hashes differ; supply an exact independent seed-state reconstruction'
        record['status']='independent_local_bitwise_reconstruction';return common,graph,record
    dg=sha(sidecar.read_bytes());diff={}
    with zipfile.ZipFile(sidecar) as z:
        assert set(z.namelist())=={'initialization.json','reference.json','initial_tensors.npz'} and len(z.namelist())==3 and z.testzip() is None
        j=json.loads(z.read('initialization.json'));r=json.loads(z.read('reference.json'))
        assert j['reference_sha256']==sha(z.read('reference.json')) and r['weights_archive_sha256']==j['weights_archive_sha256']==bundle
        assert j['source_tree_sha256']==r['frozen_tree_sha256']==A['source']['frozen_tree_sha256'] and r['protocol_sha256']==A['protocol_sha256']
        assert r['backend']==A['backend'] and j['all_12_shared_and_full_digests_exact'] is True
        cross={k:v==r['backend'][k] for k,v in j['environment'].items()}
        assert cross==j['same_backend_fields'] and all(cross[k] for k in ('torch','torch_build','numpy','threads','interop','determinism','mkldnn'))
        with np.load(io.BytesIO(z.read('initial_tensors.npz')),allow_pickle=False) as npz:
            remote={a:{k:npz[a+'::'+k] for k in vals} for a,vals in local.items()}
            assert set(npz.files)=={a+'::'+k for a,st in local.items() for k in st}
        assert encoded(remote['M1'])==common and encoded(remote['M2b_real'])==graph
        for k,v in remote['M1'].items():np.testing.assert_array_equal(v,remote['M2b_real'][k])
        assert j['tensor_meta']['M2b_real']==j['tensor_meta']['M2b_empty']
        for a in local:
            assert set(j['tensor_meta'][a])==set(local[a])
            changed={}
            for k,here in local[a].items():
                there=remote[a][k];meta=j['tensor_meta'][a][k]
                assert here.shape==there.shape and here.dtype==there.dtype
                assert meta==dict(shape=list(there.shape),hash_shape=list(np.ascontiguousarray(there).shape),dtype=there.dtype.str,sha256=sha(there.tobytes()))
                if not np.array_equal(here,there):
                    changed[k]={'differing_elements':int(np.count_nonzero(here!=there)),'total_elements':int(here.size),
                                'max_abs_diff':float(np.max(np.abs(here.astype(float)-there)))}
            diff[a]={'tensor_count':len(local[a]),'differing_tensors':changed}
        assert set(r['expected_by_fold'])==set(fields)
        for s,models in fields.items():
            for arm,meta in models.items():
                rv=r['expected_by_fold'][s][arm];target=common if arm=='M1' else graph
                assert rv['shared_initial_sha256']==meta['shared_initial_sha256']==common
                assert rv['full_initial_sha256']==meta['full_initial_sha256']==target
                assert rv['checkpoint_sha256']==manifest[meta['weights']]
                assert j['checks'][arm][s]==dict(common_match=True,full_match=True,common_rebuilt=common,full_rebuilt=target)
    record.update(status='independent_bitwise_seed_state_corroborated_from_same_torch_build',sidecar_sha256=dg,
                  all_12_full_and_shared_hashes_match=True,source_environment_match=cross,
                  local_vs_reconstruction_per_tensor=diff)
    return common,graph,record
