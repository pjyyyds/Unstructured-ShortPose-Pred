"""Five-frame coordinate-equivariant residual trajectories with a protected velocity candidate."""
import torch
from torch import nn

AXIS_THRESHOLD_M=.05

def frame(history, local):
    if history.dtype!=torch.float64 or history.ndim!=3 or history.shape[1:]!=(5,2):
        raise ValueError('Expected five float64 metric xy measurements')
    origin=history[:,-1:]
    back=(origin-history[:,:-1]).flip(1)
    lengths=torch.linalg.vector_norm(back,dim=2)
    eligible=lengths >= AXIS_THRESHOLD_M - 1e-12  # Allow round-off at the physical 5 cm boundary.
    valid=eligible.any(1)
    i=eligible.long().argmax(1);row=torch.arange(len(history),device=history.device)
    unit=back[row,i]/lengths[row,i,None].clamp_min(1e-12)
    left=torch.stack((-unit[:,1],unit[:,0]),1)
    basis=torch.stack((unit,left),2) if local else torch.eye(2,device=history.device,dtype=history.dtype).expand(len(history),2,2)
    basis=torch.where(valid[:,None,None],basis,torch.zeros_like(basis))
    return origin,basis,valid

class GroundForecaster(nn.Module):
    def __init__(self,local,use_pose):
        super().__init__()
        self.local=local;self.use_pose=use_pose
        # Allocate parameters in exactly the same order in all arms. Only pose arms read the pose branch.
        self.motion=nn.GRU(5,48,batch_first=True)
        self.pose_input=nn.Linear(17*6,24)
        self.pose=nn.GRU(24,24,batch_first=True)
        self.pose_out=nn.Linear(24+4,48)
        self.context=nn.Sequential(nn.LayerNorm(48),nn.Linear(48,48),nn.SiLU())
        self.residual=nn.Linear(48,5*3*2)
        self.scores=nn.Linear(48,6)
        nn.init.normal_(self.residual.weight,mean=0.,std=.01)
        nn.init.zeros_(self.residual.bias)

    def forward(self,history,x_p=None,q_vec=None):
        origin,basis,valid=frame(history,self.local)
        body=(history-origin)@basis
        step=torch.diff(body,dim=1,prepend=body[:,:1])/.1
        feat=torch.cat((body,step,torch.linalg.vector_norm(step,dim=2,keepdim=True)),2)
        _,hm=self.motion(feat.float());hidden=hm[0]
        if self.use_pose:
            if x_p is None or q_vec is None:raise ValueError('Pose arm requires explicit measured pose features or all-zero missing arrays')
            if x_p.shape!=(len(history),5,17,6) or q_vec.shape!=(len(history),4):raise ValueError('Invalid pose input shape')
            present=(x_p[:,:,:,3]>.5).any(2).any(1)
            _,hp=self.pose(torch.nn.functional.silu(self.pose_input(x_p.flatten(2))))
            hidden=hidden+torch.where(present[:,None],self.pose_out(torch.cat((hp[0],q_vec),1)),torch.zeros_like(hidden))
        hidden=self.context(hidden)
        logits=self.scores(hidden)
        knots=self.residual(hidden).view(len(history),5,3,2).double()
        knots=torch.cat((torch.zeros_like(knots[:,:,:1]),knots),2)
        times=torch.arange(1,31,dtype=history.dtype,device=history.device)
        left=(torch.arange(30,device=history.device)//10).long()
        phase=(times/10-left)[None,None,:,None]
        delta=knots[:,:,left]*(1-phase)+knots[:,:,left+1]*phase
        world=torch.einsum('bkts,bws->bktw',delta,basis)
        cv=origin+(history[:,-1:]-history[:,-2:-1])*times[None,:,None]
        paths=torch.cat((cv[:,None],cv[:,None]+world),1)
        logits=torch.where(valid[:,None],logits,torch.zeros_like(logits))
        return paths,logits,valid

def loss_for(paths,logits,target,valid):
    if not torch.any(valid):return None
    dist=torch.linalg.vector_norm(paths[valid]-target[valid,None],dim=-1)
    ade=dist.detach().mean(2)
    winner=ade[:,1:]==ade[:,1:].amin(1,keepdim=True)
    weight=winner.double()/winner.sum(1,keepdim=True)
    reg=((torch.sqrt(dist[:,1:].square()+.25)-.5).mean(2)*weight).sum(1).mean()
    soft=(-ade/.15).softmax(1).to(logits.dtype)
    cls=-(soft*logits[valid].log_softmax(1)).sum(1).mean()
    return reg+.25*cls
