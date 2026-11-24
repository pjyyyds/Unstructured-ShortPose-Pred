import inspect
import math
import unittest
import numpy as np
from selector import candidate_events, eligibility, rank, select


class SelectorTests(unittest.TestCase):
    def setUp(self):
        self.allow=np.array([[True,False]])
        self.pi=np.array([[.44,.30,.10,.08,.05,.03]])
        self.bits=np.zeros((1,6,5),bool)
        self.bits[0,1,0]=True;self.bits[0,2,2]=True
        self.bits[:,:,4]=~self.bits[:,:,:4].any(2)
        self.p=np.array([[.999,.9,.05,.05,.02]])

    def test_independent_scalar_formula(self):
        orig=self.p.copy()
        choice,score=rank(self.bits,self.allow,self.pi,self.p)
        p=[.999,1e-6,.05,.05]
        expected=[math.log(pi)+sum(math.log(float(v) if yes else 1-float(v))
                                    for v,yes in zip(p,flags[:4]))/4 for pi,flags in zip(self.pi[0],self.bits[0])]
        np.testing.assert_allclose(score[0],expected,rtol=0,atol=1e-14)
        self.assertEqual(choice.item(),1)
        np.testing.assert_array_equal(self.p,orig)

    def test_qualification_unused_class_and_no_head(self):
        first=rank(self.bits,self.allow,self.pi,self.p)
        self.p[0,1]=0;self.p[0,4]=1
        second=rank(self.bits,self.allow,self.pi,self.p)
        for x,y in zip(first,second):np.testing.assert_array_equal(x,y)
        self.assertEqual(rank(self.bits,self.allow,self.pi,None)[0].item(),0)
        self.bits[0,1,1]=True
        with self.assertRaises(ValueError):rank(self.bits,self.allow,self.pi,self.p)

    def test_ties_zero_one_and_initial_fields(self):
        self.pi[:]=1/6;self.p[:]=.5
        self.assertEqual(rank(self.bits,self.allow,self.pi,self.p)[0].item(),0)
        self.pi[:]=0;self.pi[0,0]=1;self.p[:]=1
        self.assertTrue(np.isfinite(rank(self.bits,self.allow,self.pi,self.p)[1]).all())
        self.assertEqual(list(inspect.signature(select).parameters),['history','paths','probabilities','action_probabilities'])

    def test_real_rule_synthetic_trajectories(self):
        h=np.array([[[0.,0.]]*5,[[i,0.] for i in [-.4,-.3,-.2,-.1,0.]]])
        p=np.zeros((2,6,30,2));p[:,0,:,0]=np.arange(1,31)*.1
        allow=eligibility(h);np.testing.assert_array_equal(allow,[[True,False],[False,True]])
        flags=candidate_events(h,p)
        self.assertTrue(flags[0,0,0] and flags[1,1,1])
        self.assertFalse(flags[0,1,0] or flags[1,0,1])
        choices,_,repeat,_=select(h,p,np.full((2,6),1/6),None)
        np.testing.assert_array_equal(choices,[0,0]);np.testing.assert_array_equal(repeat,flags)


if __name__=='__main__':unittest.main()
