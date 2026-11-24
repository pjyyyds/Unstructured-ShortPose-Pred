"""Synthetic-only fitting, exact reference errors, archive and interval smoke test."""
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

import run_anchor as r
from test_anchored import fixture, six_offsets


class TestDriver(unittest.TestCase):
    def test_four_arms_and_external_reference_archive(self):
        inp=fixture()
        truth=six_offsets()[:,None]*torch.linspace(.1,1,30)[None,:,None]
        tensors=dict(inp,truth=truth,fut_mask=torch.ones(6,30))
        idx=np.arange(6)
        keys=[('sim_a' if i<2 else 'sim_b',i,100+i) for i in idx.tolist()]
        models={}
        with patch.object(r,'P',r.Protocol(epochs=2,batch_size=6)):
            for arm in r.ARMS:
                state,logs=r.fit(arm,tensors)
                self.assertEqual(len(logs),2)
                self.assertEqual(logs[0]['tied_windows'],6)
                self.assertAlmostEqual(sum(logs[0]['mode_credit']),6)
                net=r.SixCandidates(*r.ARMS[arm]).eval()
                net.load_state_dict(state)
                models[arm]=net
            with self.assertRaisesRegex(ValueError,'observed tensors only'):
                r.forecast(net,tensors,idx)
            with TemporaryDirectory() as tmp:
                out=Path(tmp)
                original=io.BytesIO()
                with zipfile.ZipFile(original,'w') as archive:
                    for m in ('M0','M1'):
                        net=r.UnstructuredShortPosePredictor(m)
                        path,prob=r.forecast(net,inp,idx,m)
                        score,*_=r.measures(path,prob,truth.numpy())
                        payload=io.BytesIO()
                        torch.save({'model_state':net.state_dict(),'window_keys':keys,
                                    'errors':torch.from_numpy(score[:,:2].copy())},payload)
                        archive.writestr('seed_42/synthetic/'+m+'.pt',payload.getvalue())
                with zipfile.ZipFile(original) as archive:
                    result=r.score_site(archive,'synthetic',idx,keys,tensors,models,out)
                aggregate=r.compare({'sim_s1':result,'sim_s2':result,'sim_s3':result,'sim_s4':result})
                self.assertEqual(set(aggregate['four_site_macro']),set(r.ARMS)|set(r.ORIGINAL)|{'last_two'})
                primary=aggregate['paired']['cv_M0_minus_zero_M0']['equal_site_difference_ade_fde_m']
                np.testing.assert_allclose(primary,(result[1]['cv_M0']-result[1]['zero_M0']).mean(0))
                private=out/'heldout_private.npz'
                with np.load(private,allow_pickle=False) as data:
                    self.assertEqual(data['cv_M1_paths'].shape,(6,6,30,2))
                    self.assertEqual(data['original_M0_probabilities'].shape,(6,6))
                r.write_archive(out,[private])
                with zipfile.ZipFile(out/'anchored_six_candidates.zip') as archive:
                    self.assertEqual(json.loads(archive.read('sha256.json'))[private.name],r.file_hash(private))


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
