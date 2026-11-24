"""Synthetic causal, clock, edge intervention, coordinate and aggregation checks."""
import numpy as np
import torch
from observed_graph import edges
from run_transfer import graph_input,motion,paired,metrics,MATRICES,RESERVE,predict
from src.models.predictor import UnstructuredShortPosePredictor

def main():
    fs=np.array([2,4,7,9,12]);ts=(fs-fs[-1])*1000/23976
    hist=[{('Person',3):(float(t),0.),('Car',3):(4.+2.*t,0.)} for t in ts]
    hist[-1][('Person',9)]=(0.,1.)
    # Distance exactly 10 and (distinct identity) exactly coincident.
    hist[-1].update({('Car',4):(10.,0.),('Car',5):(1e-6,0.),('Car',6):(10.01,0.)})
    e,k,ident,typ,lag,gap,pre=edges(3,hist,fs)
    np.testing.assert_equal(pre,[1,2,1]);assert ident[:3].tolist()==[9,3,4]
    np.testing.assert_allclose(e[1],[4.,0.,1.,0.,4.,-4/(4+1e-6),1,1,1],rtol=2e-6)
    np.testing.assert_allclose(e[0],[0,1,0,0,1,0,0,.2,1]);assert (lag[1],gap[1])==(1,3)
    crowded=[{('Person',0):(0.,0.),('Car',2):(8.,0.),('Car',1):(8.,0.),**{('Person',j):(.1*j,0.) for j in range(1,15)}} for _ in fs]
    before=edges(0,crowded,fs);after=edges(0,crowded,fs,True)
    assert before[2][:12].tolist()==list(range(1,13)) and after[2][-1]==1 and after[1][-1]==1
    assert after[0][-1,4]==8. and before[0][-1,4]!=after[0][-1,4]
    for x,y in zip(before[:-1],after[:-1]):np.testing.assert_equal(x[:11],y[:11])
    # The caller supplies only registered snapshots. Remove a future row without changing requested history.
    all_snap={int(f):v for f,v in zip(fs,hist)};all_snap[int(fs[-1])+3]={('Person',0):(10.,20.)}
    assert all(np.array_equal(v,w) for v,w in zip(edges(3,[all_snap[int(f)] for f in fs],fs),edges(3,hist,fs)))
    for bad in (fs[::-1],fs[:4]):
        try:edges(3,hist,bad)
        except ValueError:pass
        else:raise AssertionError('Invalid five-frame time axis accepted')
    g={'edges':np.stack((before[0],e)),'source':np.stack((before[1],k)),'encoded_type':np.stack((before[3],typ)),
       'reserve_changed':np.array([True,False]),'reserve_last_edge':np.stack((after[0][-1],e[-1]))}
    for a,r in MATRICES.items():
        raw=graph_input(g,'M2b_real',r);res=graph_input(g,RESERVE,r)
        np.testing.assert_equal(raw['nbr_edges'][:,:,:2].numpy()@r,g['edges'][:,:,:2])
        np.testing.assert_equal(raw['nbr_edges'][:,:,2:4].numpy()@r,g['edges'][:,:,2:4])
        np.testing.assert_equal(raw['nbr_edges'][:,:,4:].numpy(),g['edges'][:,:,4:])
        assert res['nbr_type_oh'][0,-1,6] and not raw['nbr_type_oh'][0,-1,6]
        np.testing.assert_equal(res['nbr_edges'][1].numpy(),raw['nbr_edges'][1].numpy())
        assert graph_input(g,'real_drop_car',r)['nbr_mask'].sum()==13
        assert graph_input(g,'real_drop_people',r)['nbr_mask'].sum()==2
        assert graph_input(g,'M2b_empty',r)['nbr_mask'].sum()==0
    np.testing.assert_equal(g['edges'][0],before[0])
    vals=np.array([[10.,1.],[0.,2.],[0.,3.]])
    z=paired(vals,np.ones(3,bool),np.array(['one','two','two']))
    assert z['clip_equal']=={'ade_m':5.,'fde_m':1.75} and z['clips']==2
    # Synthetic random model paths exercise the API without using any real DUT observation or learned source state.
    torch.set_num_threads(2);torch.manual_seed(12)
    h=np.stack([np.column_stack((ts,ts*0)),np.zeros((5,2))]);r=MATRICES['0']
    xm,ego=motion(h,r);shared=dict(x_m=xm,ego_type_oh=ego,x_p=torch.zeros(2,5,17,6),q_vec=torch.zeros(2,4))
    for arm,m in [('M1','M1'),('M2b_real','M2b'),(RESERVE,'M2b')]:
        model=UnstructuredShortPosePredictor(m).eval()
        p,q,idx=predict(model,shared,graph_input(g,arm,r),r);assert p.shape==(2,6,30,2) and q.shape==(2,6)
        assert metrics(p[np.arange(2),idx],np.zeros((2,30,2))).shape==(2,2)
    print('Synthetic checks passed')
if __name__=='__main__':main()
