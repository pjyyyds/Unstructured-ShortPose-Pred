"""Frozen decision path; all inputs are available at prediction time."""
import numpy as np
from src.data.actions import compute_action_labels

EPS=1e-6


def eligibility(history):
    h=np.asarray(history,dtype=np.float64)
    if h.ndim!=3 or h.shape[1:]!=(5,2) or not np.isfinite(h).all():
        raise ValueError('Expected five observed finite ground positions')
    v=np.diff(h,axis=1)/.1
    avg=np.stack((v[:,:2].mean(1),v[:,:3].mean(1),v[:,1:].mean(1)),1)
    speed=np.hypot(avg[:,:,0],avg[:,:,1]).mean(1)
    return np.stack((speed<=.2,speed>=.6),1)


def candidate_events(history,paths):
    h=np.asarray(history);p=np.asarray(paths)
    if h.ndim!=3 or h.shape[1:]!=(5,2) or p.shape!=(len(h),6,30,2) or not np.isfinite(p).all() or not np.isfinite(h).all():
        raise ValueError('Invalid observations or six candidate paths')
    events=np.empty((len(h),6,5),bool)
    for i in range(len(h)):
        for k in range(6):
            flags,valid=compute_action_labels(h[i],np.ones(5),p[i,k],np.ones(30))
            if valid!=1:raise ValueError('Invalid complete candidate')
            events[i,k]=flags
    if np.any(events[:,:,4]!=~events[:,:,:4].any(2)):
        raise ValueError('Maintain must be the complement')
    return events


def rank(events,eligible,probabilities,action_probabilities):
    bits=np.asarray(events);allow=np.asarray(eligible);pi=np.asarray(probabilities,dtype=float)
    if bits.shape!=(len(pi),6,5) or bits.dtype!=bool or allow.shape!=(len(pi),2) or allow.dtype!=bool:
        raise ValueError('Incorrect candidate event/qualification array')
    if pi.shape!=(len(bits),6) or not np.isfinite(pi).all() or np.any((pi<0)|(pi>1)) or not np.allclose(pi.sum(1),1,rtol=0,atol=2e-6):
        raise ValueError('Invalid archived mode probabilities')
    if np.any(bits[:,:,:2]&~allow[:,None,:]):
        raise ValueError('Event contradicts observable history')
    log_pi=np.log(np.clip(pi,EPS,1-EPS))
    if action_probabilities is None:return pi.argmax(1),log_pi
    raw=np.asarray(action_probabilities,dtype=float)
    if raw.shape!=(len(bits),5) or not np.isfinite(raw).all() or np.any((raw<0)|(raw>1)):
        raise ValueError('Invalid action probabilities')
    p=raw[:,:4].copy()
    p[:,:2]=np.where(allow,p[:,:2],0.0)
    p=np.clip(p,EPS,1-EPS)
    score=log_pi+np.where(bits[:,:,:4],np.log(p[:,None,:]),np.log1p(-p[:,None,:])).mean(-1)
    by_score=score==score.max(1,keepdims=True)
    choice=np.where(by_score,pi,-1).argmax(1)
    return choice,score


def select(history,paths,probabilities,action_probabilities=None):
    flags=candidate_events(history,paths)
    allow=eligibility(history)
    selected,score=rank(flags,allow,probabilities,action_probabilities)
    return selected,score,flags,allow
