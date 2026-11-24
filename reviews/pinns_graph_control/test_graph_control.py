"""Only generated inputs in these model/edge regressions."""
import unittest
import numpy as np
import torch
import run_graph_control as g
from scripts.train_pinns_heldout_sites import Protocol

def tensor_fixture():
    gen=torch.Generator().manual_seed(500)
    n=8
    out={k:torch.rand(shape,generator=gen) for k,shape in {'x_m':(n,5,9),'x_p':(n,5,17,6),'q_vec':(n,4),
          'nbr_edges':(n,12,9),'fut_xy':(n,30,2)}.items()}
    out['x_m'][:,:,5:7]=1;out['x_p'][:,:,:,3]=1;out['q_vec'][:,0]=1
    out['ego_type_oh']=torch.nn.functional.one_hot(torch.zeros(n,dtype=torch.long),8).float()
    out['fut_mask']=torch.ones(n,30);out['nbr_mask']=torch.zeros(n,12);out['nbr_mask'][:4,:2]=1
    out['nbr_type_oh']=torch.zeros(n,12,8);out['nbr_type_oh'][:4,0,0]=1;out['nbr_type_oh'][:4,1,6]=1
    return out

class GraphTests(unittest.TestCase):
    def test_typed_history_causality_velocity_and_cap(self):
        ego=('Person',1)
        history=[{ego:np.array([i*.1,0.])} for i in range(5)]
        history[0][('Car',1)]=np.array([2.4,0.]);history[-1][('Car',1)]=np.array([2.4,0.])
        history[-1][('Person',2)]=np.array([1.4,0.])
        history[2][('Person',3)]=np.array([4.,4.])
        expected=np.array([[-.4+i*.1,0] for i in range(5)])
        e,t,m,p,src,gap,count,_=g.graph(history,1,expected)
        self.assertEqual(count,2);np.testing.assert_equal(p[:2],[2,1]);np.testing.assert_equal(src[:2],[0,1])
        self.assertEqual(m.sum(),2);self.assertEqual(t[0].argmax(),0);self.assertEqual(t[1].argmax(),6)
        np.testing.assert_equal(e[0,2:4],0);self.assertEqual(e[0,6],0)
        self.assertEqual(gap[1],4);self.assertAlmostEqual(e[1,5],1,places=5)
        with self.assertRaises(ValueError):g.graph(history+[{}],1,expected)
        for i in range(20):
            history[-1][('Car',10+i)]=np.array([.5+i*.25,0])
        self.assertEqual(g.graph(history,1,expected)[2].sum(),12)

    def test_shared_init_and_masks(self):
        torch.set_num_threads(1);t=tensor_fixture();ids=np.arange(8)
        base,x,_=g.initialize('M1');graph,y,h=g.initialize('M2b_real');empty,z,h2=g.initialize('M2b_empty')
        self.assertEqual(x,y);self.assertEqual(x,z);self.assertEqual(h,h2)
        for k,v in base.state_dict().items():self.assertTrue(torch.equal(v,graph.state_dict()[k]))
        for mask,expected in [('real_drop_car',0),('real_drop_people',6)]:
            batch=g.inputs(t,ids,mask);valid=batch['nbr_mask'].bool()
            self.assertTrue(torch.all(batch['nbr_type_oh'].argmax(2)[valid]==expected))
        graph.eval();blank=g.inputs(t,ids,'M2b_empty')
        with torch.no_grad():
            zero=graph(blank);junk={**blank,'nbr_edges':torch.randn_like(blank['nbr_edges'])*10,'nbr_type_oh':torch.rand_like(blank['nbr_type_oh'])}
            dirty=graph(junk)
            self.assertTrue(torch.equal(zero['mode_probs'],dirty['mode_probs']))
            self.assertTrue(torch.equal(zero['trajs'],dirty['trajs']));self.assertTrue(torch.equal(zero['nbr_attn'],torch.zeros_like(zero['nbr_attn'])))
            true=graph(g.inputs(t,ids,'M2b_real'))
            self.assertTrue(torch.equal(zero['trajs'][4:],true['trajs'][4:]))
            self.assertGreater((zero['trajs'][:4]-true['trajs'][:4]).abs().max().item(),0)

    def test_optimizer_order_and_replay(self):
        torch.set_num_threads(1);t=tensor_fixture();seq=np.arange(8);p=Protocol(epochs=1,batch_size=4)
        draws=[];saved=[]
        for arm in g.ARMS:
            model,_,_=g.initialize(arm);loss,order=g.train(model,arm,t,seq,p)
            self.assertEqual(len(loss),2);draws.append(order)
            paths,prob,err=g.predict(model,arm,t,seq)
            direct=np.linalg.norm(paths[np.arange(8),prob.argmax(1)].astype(float)-t['fut_xy'].numpy(),axis=-1)
            np.testing.assert_allclose(err[:,0],direct.mean(1));np.testing.assert_allclose(err[:,1],direct[:,-1]);saved.append(paths)
        self.assertEqual(len(set(draws)),1)
        self.assertGreater(np.max(abs(saved[1]-saved[2])),0)
if __name__=='__main__':unittest.main()
