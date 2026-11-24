import inspect
import numpy as np

from audit_projection import eligible,project,class_metrics
from src.data.actions import _estimate_causal_velocities,compute_action_labels


rng=np.random.default_rng(49)
h=np.cumsum(rng.normal(size=(200,5,2))*.057,axis=1)
special=np.zeros((6,5,2))
for i,s in enumerate([0.,.1999,.2001,.5999,.6001,1.5]):special[i,:,0]=np.arange(5)*.1*s
h=np.concatenate((special,h))
allow=eligible(h)
assert allow[:6].tolist()==[[True,False],[True,False],[False,False],[False,False],[False,True],[False,True]]
boundaries=np.stack([np.array([[0.,0.],[0.,0.],[0.,0.],[0.,0.],[0.,0.]])]*2)
for i,s in enumerate([.20,.60]):boundaries[i,:,0]=np.arange(5)*s*.1
for histories in (boundaries,h):
    for history,poss in zip(histories,eligible(histories)):
        _,_,speed=_estimate_causal_velocities(history,np.ones(5,dtype=bool))
        v=float(np.mean(speed[-3:]));np.testing.assert_array_equal(poss,[v<=.2,v>=.6])
        for mult in (0.,.08,.4):
            future=history[-1]+np.cumsum(mult*rng.normal(size=(30,2)),axis=0)
            y,valid=compute_action_labels(history,np.ones(5),future,np.ones(30))
            assert valid and not (y[:2].astype(bool)&~poss).any()
raw=rng.uniform(size=(len(h),5)).astype('f')
keep=raw.copy(); revised=project(raw,allow)
np.testing.assert_array_equal(raw,keep);np.testing.assert_array_equal(revised[:,2:],raw[:,2:])
np.testing.assert_array_equal(revised[:,:2][allow],raw[:,:2][allow]);assert not revised[:,:2][~allow].any()
assert set(inspect.signature(eligible).parameters)=={'histories','dt'}
assert set(inspect.signature(project).parameters)=={'original_probs','eligible_events'}
for j in range(2):
    ref=(rng.uniform(size=len(h))<.15)&allow[:,j]
    a,b=[class_metrics(ref,p[:,j]) for p in (raw,revised)]
    assert a['tp']==b['tp'] and a['fn']==b['fn'] and b['fp']<=a['fp']
    np.testing.assert_allclose(a['brier']-b['brier'],np.where(allow[:,j],0,raw[:,j].astype(float)**2).mean(),rtol=0,atol=1e-14)
print('Passed causal eligibility, thresholds, 624 independent future continuations, Brier identity and reporting-only projection.')
