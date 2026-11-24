"""No-optimizer synthetic property tests using arbitrary network weights, paths and image poses."""
import numpy as np
import torch
from equivariant_model import GroundForecaster, frame, loss_for

def rotation(angle):
    a=np.deg2rad(angle);return torch.tensor([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]],dtype=torch.float64)
def make(local,pose):
    torch.manual_seed(42);return GroundForecaster(local,pose).eval()
@torch.no_grad()
def check():
    torch.set_num_threads(2);gen=torch.Generator().manual_seed(193)
    hist=torch.randn(12,5,2,generator=gen,dtype=torch.float64).cumsum(1)
    hist[0]=3.;hist[1]=0.;hist[1,:,0]=torch.arange(5,dtype=torch.float64)*.01
    hist[2]=0.;hist[2,0,0]=.050001;hist[3]=0.;hist[3,0,0]=.049999
    hist[4,-2]=hist[4,-1];hist[5]=0.;hist[5,0,0]=.05
    pose=torch.randn(12,5,17,6,generator=gen);pose[:,:,:,3]=1.;pose[6:8]=0
    q=torch.rand(12,4,generator=gen)
    target=torch.randn(12,30,2,dtype=torch.float64,generator=gen)
    origins,axes,valid=frame(hist,True)
    assert valid[:6].tolist()==[False,False,True,False,True,True]
    cv=hist[:,-1:]+(hist[:,-1:]-hist[:,-2:-1])*torch.arange(1,31,dtype=hist.dtype)[None,:,None]
    largest=[0.,0.];rng=np.random.default_rng(195)
    initial={n:v for n,v in make(False,False).state_dict().items()}
    for use_pose in (False,True):
        mod=make(True,use_pose)
        assert all(torch.equal(initial[k],v) for k,v in mod.state_dict().items())
        p,l,ok=mod(hist,pose,q); probs=l.softmax(1)
        torch.testing.assert_close(p[:,0],cv,atol=0,rtol=0)
        torch.testing.assert_close(p[~valid],cv[~valid,None].expand(-1,6,-1,-1),atol=0,rtol=0)
        assert (probs[~valid]==probs[~valid,:1]).all()
        if not use_pose:
            pp,lp,_=mod(hist,torch.zeros_like(pose),torch.zeros_like(q))
            assert torch.equal(pp,p) and torch.equal(lp,l)
        else:
            base=make(True,False)(hist)
            masked=mod(hist,torch.zeros_like(pose),q)
            assert torch.equal(base[0],masked[0]) and torch.equal(base[1],masked[1])
            assert torch.equal(base[0][6:8],p[6:8]) and not torch.equal(base[0][-1],p[-1])
        for angle in [0.,37.,90.,173.,*rng.uniform(-180,180,20)]:
            rot=rotation(angle);shift=torch.as_tensor(rng.normal(0,12,2),dtype=torch.float64)
            idx=torch.ones(len(hist),dtype=torch.bool)
            pt,lt,vt=mod((hist@rot.T+shift)[idx],pose[idx],q[idx]);assert torch.equal(vt,ok[idx])
            pred=p[idx]@rot.T+shift
            largest[0]=max(largest[0],float((pred-pt).abs().max()));largest[1]=max(largest[1],float((probs[idx]-lt.softmax(1)).abs().max()))
            torch.testing.assert_close(pt,pred,atol=1e-5,rtol=0);torch.testing.assert_close(lt.softmax(1),probs[idx],atol=2e-6,rtol=0)
            assert torch.equal(probs[idx].argsort(dim=1,descending=True),lt.softmax(1).argsort(dim=1,descending=True))
            actual=loss_for(pt,lt,(target@rot.T+shift)[idx],vt)
            expected=loss_for(p[idx],l[idx],target[idx],ok[idx]);torch.testing.assert_close(actual,expected,atol=1e-6,rtol=0)
    g=make(False,True);a,la,_=g(hist,pose,q);r=rotation(37);b,lb,_=g(hist@r.T,pose,q)
    assert (a[valid]@r.T-b[valid]).abs().max()>1e-4
    assert loss_for(a[~valid],la[~valid],target[~valid],valid[~valid]) is None
    with torch.enable_grad():
        differentiable=make(True,True)
        p,l,v=differentiable(hist,pose,q);obj=loss_for(p,l,target,v);obj.backward()
        assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in differentiable.parameters())
        assert differentiable.residual.weight.grad.abs().sum()>0 and differentiable.scores.weight.grad.abs().sum()>0
    print('PASS: ordered six paths, probabilities, ranks, invariant loss, rotation and translation, absent pose, degenerate, near-threshold and fixed CV',largest)
if __name__=='__main__':check()
