"""Synthetic regression tests for the locked perturbations, outcome isolation, gating and aggregation."""
import unittest
import numpy as np
import torch
from pose_order_audit import CONDITIONS,FIXED_SEEDS,STATS,aggregate,change_geometry,disabled_residual,geometry_counts,predict_inputs
from src.models.predictor import UnstructuredShortPosePredictor


def sample_pose(n=2):
    x=torch.zeros((n,5,17,6))
    for t in range(5):
        x[:,t,:,0]=t+torch.arange(17)*.05
        x[:,t,:,1]=-t
        x[:,t,:,2]=.6+.02*t
        x[:,t,:,3:5]=1
        x[:,t,:,5]=-.2*t
    return x


class GeometryTests(unittest.TestCase):
    def test_primary_is_involutive_keeps_both_endpoints_quality_and_noneligible_joints(self):
        p=sample_pose()
        p[0,1,1,:4]=0
        p[0,3,2,4]=0
        source=p.clone()
        a=change_geometry(p,'interior_order')
        self.assertTrue(torch.equal(a[:,[0,2,4]],p[:,[0,2,4]]))
        self.assertTrue(torch.equal(a[...,2:],p[...,2:]))
        self.assertTrue(torch.equal(a[0,:,1],p[0,:,1]))
        self.assertTrue(torch.equal(a[0,:,2],p[0,:,2]))
        self.assertTrue(torch.equal(a[:,1,0,:2],p[:,3,0,:2]))
        self.assertTrue(torch.equal(a[:,3,0,:2],p[:,1,0,:2]))
        self.assertTrue(torch.equal(change_geometry(a,'interior_order'),p))
        self.assertTrue(torch.equal(p,source))

    def test_static_uses_latest_valid_in_each_same_root_and_no_imputation(self):
        p=sample_pose(1)
        p[0,2:,0,4]=0
        p[0,4,0,:4]=0
        a=change_geometry(p,'static_by_joint_and_root')
        for t,expected in enumerate((1,1,3,3,4)):
            self.assertTrue(torch.equal(a[0,t,0,:2],p[0,expected,0,:2]))
        self.assertTrue(torch.equal(a[...,2:],p[...,2:]))
        self.assertTrue(torch.equal(a[:,-1],p[:,-1]))
        self.assertTrue(torch.equal(change_geometry(a,'static_by_joint_and_root'),a))

    def test_no_geometry_and_single_observation_do_not_generate_evidence(self):
        p=torch.zeros_like(sample_pose(1));p[:,1,0]=torch.tensor([.1,.2,.3,1,0,-1])
        for c in CONDITIONS:self.assertTrue(torch.equal(p,change_geometry(p,c)))
        counts=geometry_counts(p)
        self.assertEqual(counts['interior_order']['changed_windows'],0)
        self.assertEqual(counts['static_by_joint_and_root']['changed_window_joints'],0)

    def test_disabled_residual_preserves_real_quality_inputs_and_removes_hook_on_exception(self):
        torch.manual_seed(2);model=UnstructuredShortPosePredictor('M1').eval()
        hm=torch.randn(3,128);hp=torch.randn(3,128);q=torch.full((3,4),.8)
        a=model.quality_gate(hm,h_p=hp,q_vec=q)
        with self.assertRaises(RuntimeError):
            with disabled_residual(model.quality_gate):
                b,g=model.quality_gate(hm,h_p=hp,q_vec=q)
                self.assertTrue(torch.equal(b,model.quality_gate.out_norm(hm)))
                self.assertFalse(g.any());raise RuntimeError('force cleanup')
        out=model.quality_gate(hm,h_p=hp,q_vec=q)
        self.assertTrue(all(torch.equal(v,w) for v,w in zip(a,out)))

    def test_forward_excludes_future_and_does_not_modify_observations(self):
        net=UnstructuredShortPosePredictor('M1').eval(); data={'x_m':torch.ones((2,5,9)),
            'x_p':sample_pose(),'q_vec':torch.full((2,4),.6),'ego_type_oh':torch.eye(8)[:2]}
        original={k:v.clone() for k,v in data.items()}
        for c in CONDITIONS:
            paths,indices,gates=predict_inputs(net,data,c)
            self.assertEqual(paths.shape,(2,30,2));self.assertEqual(indices.shape,(2,));self.assertEqual(gates.shape,(2,))
        for k,v in data.items():self.assertTrue(torch.equal(v,original[k]))
        with self.assertRaises(ValueError):predict_inputs(net,{**data,'fut_xy':torch.empty(2,30,2)},'recorded')

    def test_constant_paired_effect_and_equal_site_weight(self):
        ns=(2,4,5,9);n=sum(ns);f=len(FIXED_SEEDS);values=np.zeros((f,n,len(CONDITIONS),len(STATS)))
        index={};keys=[];start=0
        for j,k in enumerate(ns):
            ids=np.arange(start,start+k);index[str(j)]=ids
            keys+= [(f'{j}/clip_{i%2}',i,i*10) for i in range(k)]
            values[:,ids,0,:2]=2*j
            for c in range(1,len(CONDITIONS)):values[:,ids,c,:2]=2*j+c
            start+=k
        s=aggregate(values,index,keys)
        self.assertEqual(s['four_site_five_seed_equal']['recorded']['ade_m'],3.)
        d=s['four_site_control_minus_recorded_m']['interior_order']
        self.assertEqual(d['paired_clip_percentile95']['ade_m'],[1.,1.])
        self.assertEqual(d['mean']['ade_m'],1.)
        self.assertNotEqual(values[:,:,0,0].mean(),3.)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
