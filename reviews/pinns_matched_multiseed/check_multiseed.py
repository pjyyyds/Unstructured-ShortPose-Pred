"""Synthetic no-update tests for family dispatch, seed identity, six-mode outputs and ZIPs."""
from pathlib import Path
import json,tempfile,zipfile
from unittest.mock import patch
import numpy as np
import torch
import run_multiseed as r

torch.set_num_threads(2)
n=7
torch.manual_seed(188)
x={'x_m':torch.randn(n,5,9),'ego_type_oh':torch.eye(8)[:n],'x_p':torch.randn(n,5,17,6),'q_vec':torch.rand(n,4),
   'fut_xy':torch.randn(n,30,2),'fut_mask':torch.ones(n,30)}
x['x_m'][:,:,5:7]=1.;x['x_p'][:,:,:,3]=1.
train,test=np.arange(3),np.arange(3,7)
seen=[]
def mock_fitter(a,tensors,tr,te,dev,p):
    assert str(dev)=='cpu' and np.array_equal(tr,train) and np.array_equal(te,test) and p.epochs==28 and p.batch_size==64
    seen.append((p.seed,a))
    torch.manual_seed(p.seed)
    state=r.make(a).state_dict()
    d=r.forward(a,state,tensors,te,64)
    return state, d['errors'].numpy() if a in ('M0','M1') else d

assert r.parameters(42,[5,10,20]).seed==42
for seed,durs in ((0,[5]),(42,[]),(42,[5,5]),(42,[4])):
    try:r.parameters(seed,durs)
    except ValueError:pass
    else:raise AssertionError('Invalid request accepted')
hashes={}
with patch.object(r,'fit_predict',side_effect=mock_fitter) as base,patch.object(r,'fit_variant',side_effect=mock_fitter) as fusion:
    for seed in r.FIXED_SEEDS:
        p=r.parameters(seed,[5]);hashes[seed]={}
        for a in r.FAMILIES:
            state,detail=r.fit_arm(a,x,train,test,p)
            errors,_=r.check_detail(a,state,detail,x,test)
            assert len(test)==len(errors) and detail['trajs'].shape==(4,6,30,2)
            hashes[seed][a]=r.parameter_digest(state)
    assert base.call_count==10 and fusion.call_count==10
    assert seen==[(seed,a) for seed in r.FIXED_SEEDS for a in r.FAMILIES]
assert all(len({hashes[s][a] for s in r.FIXED_SEEDS})==5 for a in r.FAMILIES)
with tempfile.TemporaryDirectory() as tmp:
    out=Path(tmp)
    base={'protocol':{'seed':1024}}
    for name in ('driver.py','folds.json','backend.json'):(out/name).write_text('{}')
    for s in r.SITES:
        r.write_truth(out/'truth'/f'{s}.npz',{'x':np.zeros((1,))})
        for t in (5,10,20):
            for a in r.FAMILIES:
                path=out/f'T{t}'/s/f'{a}.pt';path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'synthetic')
    for lengths in ([5],[10],[20],[5,10,20]):
        r.pack(out,lengths,1024,base,{t:{'synthetic':True} for t in lengths})
        name='all' if len(lengths)==3 else 'T'+str(lengths[0])
        with zipfile.ZipFile(out/f'multiseed_s1024_{name}.zip') as z:
            m=json.loads(z.read('sha256.json'));assert len([k for k in m if k.endswith('.pt')])==16*len(lengths)
            assert z.testzip() is None and json.loads(z.read('report.json'))['lengths']==lengths
print('PASS: five seed identities, exact dispatch of protocol to both frozen fit implementations (optimizer mocked), all four six-mode forwards, each duration archive and combined archive. No fits.')
