"""Compare independent math against the frozen architecture on generated inputs only."""
import unittest
import numpy as np
import torch
import verify_graph_control as v
from test_graph_control import tensor_fixture
import run_graph_control as reference

class Independent(unittest.TestCase):
    def test_synthetic_attention_and_readout(self):
        torch.set_num_threads(1)
        t=tensor_fixture()
        base={k:x for k,x in t.items() if k in reference.COMMON}
        src=np.full((8,12),-1,np.int8);src[:4,:2]=[0,1]
        graph={'source_class':src,'nbr_edges':t['nbr_edges'].numpy(),'nbr_type_oh':t['nbr_type_oh'].numpy()}
        rows=np.arange(8)
        for arm in reference.ARMS:
            model,*_=reference.initialize(arm);model.eval()
            for method in (arm,*reference.CONTROLS) if arm=='M2b_real' else (arm,):
                actual,prob=v.forward(model,method,base,graph,rows)
                expected,pr,e=reference.predict(model,method,t,rows)
                v.near(actual,expected,3e-5,2e-5);v.near(prob,pr,1e-6)
                _,re=v.scores(actual,prob,t['fut_xy'].numpy());v.near(e,re,3e-5)

    def test_independent_bootstrap_and_slice_statistics(self):
        rng=np.random.default_rng(31)
        sites={name:np.arange(i*7,(i+1)*7) for i,name in enumerate(v.SITES)}
        keys=[(s+'_'+str(i%3),1,20) for s in v.SITES for i in range(7)]
        src=rng.choice([-1,0,1],size=(28,12));e={k:rng.uniform(size=(28,2)) for k in v.METHODS}
        summary=reference.stats(sites,keys,src,e)
        v.summary(keys,sites,src,e,summary)
        summary['contrasts']['M2b_real_minus_M1']['four_site_m']['ade_m']+=.01
        with self.assertRaises(AssertionError):v.summary(keys,sites,src,e,summary)
if __name__=='__main__':unittest.main()
