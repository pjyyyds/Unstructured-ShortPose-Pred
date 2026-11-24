"""Exercise the actual official loss, both fit conditions and interrupted resume on synthetic tracks."""
from dataclasses import dataclass
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

from check_protocol import ROOT,build_observed,model_class
import run_pinns as r


def test():
    cpu=torch.device('cpu');torch.set_num_threads(2)
    cls,_=model_class(ROOT/'source/model_jrdb.py',True)
    spec=importlib.util.spec_from_file_location('loss',ROOT/'source/utils/metrics.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    rng=np.random.default_rng(17)
    h=rng.normal(size=(6,5,2));boxes=rng.normal(size=(6,5,4)).astype('f')
    boxes[:,:,2:]=abs(boxes[:,:,2:])+2
    y=torch.tensor(rng.normal(size=(6,30,1,2)),dtype=torch.float32)
    a,b=[torch.from_numpy(build_observed(h,boxes,flag)) for flag in [True,False]]
    assert torch.equal(a[:,:,0],b[:,:,0]) and b[:,:,1].abs().sum()==0 and a[:,:,1].abs().sum()>0
    with TemporaryDirectory() as tmp:
        folder=Path(tmp)
        states={}
        for arm,x in [('traj',b),('traj_box',a)]:
            with patch.object(r,'EPOCHS',2):
                state=r.fit(cls,mod.MSE_LOSS,cpu,x,y,{'arm':arm},folder/(arm+'_twice.pt'))
            with patch.object(r,'EPOCHS',1):
                r.fit(cls,mod.MSE_LOSS,cpu,x,y,{'arm':arm},folder/(arm+'_resume.pt'))
            with patch.object(r,'EPOCHS',2):
                resumed=r.fit(cls,mod.MSE_LOSS,cpu,x,y,{'arm':arm},folder/(arm+'_resume.pt'))
            assert r.state_digest(state['model_state'])==r.state_digest(resumed['model_state'])
            assert state['epochs']==resumed['epochs']
            net=r.create(cls,cpu).eval()
            net.load_state_dict(state['model_state'])
            path=r.predict(net,x)
            assert path.shape==(6,30,2)
            per_window=r.metrics(path,y[:,:,0].numpy())
            official=mod.MSE_LOSS(torch.from_numpy(path)[:,:,None],y)
            np.testing.assert_allclose(per_window[:,0].mean()*100.,official.item(),rtol=1e-6)
            states[arm]=state
        assert states['traj']['initial_sha256']==states['traj_box']['initial_sha256']
        assert r.state_digest(states['traj']['model_state'])!=r.state_digest(states['traj_box']['model_state'])
    print('Verified CPU paired initialization, exact resume, native loss identity, one-path metrics and input separation.')


if __name__=='__main__':test()
