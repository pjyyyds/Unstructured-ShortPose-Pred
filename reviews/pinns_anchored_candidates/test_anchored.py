import inspect
import math
import unittest

import numpy as np
import torch

from anchored_model import SixCandidates, observed_base, responsibilities, training_loss
from scripts.rerank_pinns_velocity_modes import last_two_future
from src.losses.losses import JointPredictionLoss


def fixture():
    generator = torch.Generator().manual_seed(105)
    x = torch.randn(6, 5, 9, generator=generator)
    x[:, :, 5:7] = 1.
    xy = x[:, :, :2].double()
    xy = xy - xy[:, -1:].clone()
    xy[:, -2] = torch.tensor([-1., 0.])
    x[:, :, :2] = xy.float()
    x[:, -1, 2:4] = torch.tensor([10., 0.])
    pose = torch.randn(6, 5, 17, 6, generator=generator)
    pose[..., 2:4] = 1.
    return dict(history_xy=xy, x_m=x, ego_type_oh=torch.eye(8)[:6], x_p=pose,
                q_vec=torch.ones(6, 4))


def six_offsets():
    s = math.sqrt(3.) / 2
    return torch.tensor([[1,0],[.5,s],[-.5,s],[-1,0],[-.5,-s],[.5,-s]], dtype=torch.float64)


class TestAnchored(unittest.TestCase):
    def test_no_target_input_and_paired_initial_states(self):
        self.assertEqual(set(inspect.signature(SixCandidates.forward).parameters),
                         {'self','history_xy','x_m','ego_type_oh','x_p','q_vec'})
        models = {(m,b): SixCandidates(m,b).eval() for m in ('M0','M1') for b in (False,True)}
        shared = models['M0',False].state_dict()
        for model in models.values():
            for k,v in shared.items():
                self.assertTrue(torch.equal(v, model.state_dict()[k]), k)
        for variant in ('M0','M1'):
            a, b = models[variant,False], models[variant,True]
            for k,v in a.state_dict().items():
                self.assertTrue(torch.equal(v,b.state_dict()[k]), k)
            inputs = fixture()
            with torch.no_grad():
                p0, l0, *_ = a(**inputs)
                p1, l1, base, _ = b(**inputs)
            np.testing.assert_array_equal(base.numpy(), last_two_future(inputs['history_xy'].numpy()))
            self.assertTrue(torch.equal(p0,torch.zeros_like(p0)))
            self.assertTrue(torch.equal(p1,base[:,None].expand_as(p1)))
            self.assertTrue(torch.equal(l0,l1))

    def test_sector_support_and_model_breaks_symmetry(self):
        for variant in ('M0','M1'):
            for anchored in (False,True):
                net=SixCandidates(variant, anchored).eval()
                inp=fixture()
                paths, logits, base, last = net(**inp)
                paths.retain_grad()
                logits.retain_grad()
                truth=base + six_offsets()[:,None] * torch.linspace(.1,1.,30)[None,:,None]
                loss, weight, tied = training_loss(paths,logits,base,last,truth,torch.ones(6,30))
                self.assertTrue(torch.equal(weight, torch.eye(6,dtype=weight.dtype)))
                self.assertTrue(tied.all())
                loss.backward()
                grad=paths.grad.abs().sum((2,3))
                self.assertTrue(torch.equal(grad > 0,torch.eye(6,dtype=torch.bool)))
                self.assertTrue((logits.grad.diag().abs() > 1e-4).all())
                self.assertTrue((net.network.decoder.mode_queries.grad.norm(dim=1)>1e-6).all())
                self.assertGreater(net.network.decoder.pi_head[-1].weight.grad.norm().item(),1e-6)
                self.assertGreater(net.network.decoder.delta_out.weight.grad.norm().item(),1e-6)
                with torch.no_grad():
                    for p in net.parameters():
                        if p.grad is not None: p.add_(p.grad,alpha=-.01)
                    paths,*_=net(**inp)
                    sep=torch.pdist(paths[0].reshape(6,-1))
                    self.assertTrue((sep > 1e-6).all())

    def test_moving_stationary_tiny_partial_and_zero_residual(self):
        target=torch.zeros((4,30,2),dtype=torch.float64)
        base=torch.zeros_like(target)
        # The last row has a masked spurious extreme excursion.
        target[:3,0,1]=1; target[0,1,0]=1; target[2,2,1]=-100
        heading=torch.tensor([[0., 1e-250],[0.,0.],[0.,1.],[.1,0.]],dtype=torch.float64)
        ade=torch.zeros(4,6,dtype=torch.float64)
        mask=torch.ones(4,30,dtype=torch.bool);mask[2,2]=False
        weights, _=responsibilities(ade,target,base,heading,mask)
        self.assertEqual(weights[0,0],1.)  # tiny northward speed, earliest equal excursion
        self.assertEqual(weights[1,1],.5)  # stationary world y bisects sectors 1, 2
        self.assertEqual(weights[1,2],.5)
        self.assertEqual(weights[2,0],1.)
        torch.testing.assert_close(weights[3],torch.full((6,),1/6,dtype=torch.float64))
        ade[0,0]=2; ade[0,2:]=2
        weights,_=responsibilities(ade,target,base,heading,mask)
        self.assertEqual(weights[0,1],1.)  # outside best-ADE set never receives a label
        p=base[:,None].expand(-1,6,-1,-1).clone().requires_grad_()
        z=torch.zeros(4,6,requires_grad=True)
        exact,_,_=training_loss(p,z,base,heading,base,mask)
        exact.backward()
        self.assertEqual(p.grad.abs().sum(),0.)
        self.assertLess(float(z.grad.abs().max()),1e-8)

    def test_unique_winner_same_as_original_loss(self):
        rng=torch.Generator().manual_seed(211)
        paths=torch.randn(5,6,30,2,generator=rng)
        logits=torch.randn(5,6,generator=rng)
        target=torch.randn(5,30,2,generator=rng)
        mask=torch.rand(5,30,generator=rng)>.15
        base,last=observed_base(torch.zeros(5,5,2,dtype=torch.float64),True)
        a,_,tied=training_loss(paths.double(),logits,base,last,target,mask)
        b=JointPredictionLoss(use_action_loss=False)({'trajs':paths,'mode_logits':logits},
                                                     target,mask.float())['loss']
        self.assertFalse(tied.any())
        torch.testing.assert_close(a,b.double(),atol=3e-7,rtol=2e-7)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
