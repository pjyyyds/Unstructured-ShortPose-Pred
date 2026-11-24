import numpy as np
from evaluate_visible_boundaries import choose,frame,transverse,evaluate,RULES,READERS

def line(points,inside=None):
    return {'xy':points,'inside':inside if inside is not None else [True]*len(points)}
edge=[line([[-4.,0.],[0.,0.],[4.,0.]])]
assert frame(np.array([0.,1.]),edge,True)[0]['reason']=='supported'
assert frame(np.array([4.,1.]),edge,True)[0]['reason']=='insufficient_visible_arc'
assert frame(np.array([0.,1.]),edge,False)[0]['reason']=='anchor_outside_control_hull'
assert frame(np.array([0.,6.]),edge,True)[0]['reason']=='beyond_5m'
assert frame(np.array([0.,1.]),[line([[-4.,0.],[4.,0.]],[False,True])],True)[0]['reason']=='local_curve_outside_control_hull'
history=np.array([[0,.6],[0,.5],[0,.4],[0,.3],[0,.2]])
paths=np.zeros((6,30,2));steps=np.arange(1,31)*.1
paths[1,:,1]=-steps;paths[2,:,1]=-steps;paths[1,:,0]=1.
prob=np.array([.4,.2,.3,.05,.03,.02])
first=choose(history,paths,prob,edge,True)
assert first['modes']=={'probability':0,'normal':2,'full_velocity':2,'tangent':0}
assert all(v==0 for v in choose(history,paths,prob,edge,False)['modes'].values())
assert abs(first['angle_to_edge_deg']-90)<1e-9
crossing=np.array([[.2,.2],[.2,-.2]])
assert transverse(crossing,edge,False)
assert not transverse(crossing+np.array([10.,0.]),edge,False)
assert not transverse(np.array([[.2,.2],[.2,0.]]),edge,False)
assert not transverse(np.array([[0.,0.],[.2,0.]]),edge,False)
assert not transverse(crossing,[line([[-4.,0.],[4.,0.]],[False,True])],True)
# Zero-support sites have unchanged choices, and moving toward an edge is permitted.
sites=np.repeat(['a','b','c','d'],[5,13,28,13])
bank={'clip':sites,'hist':np.repeat(history[None],59,axis=0),'paths':np.repeat(paths[None],59,axis=0),'person':np.arange(59)}
mp={s:{r:edge for r in READERS} for s in set(sites)}
entries=[{r:choose(history,paths,prob,edge,i==0 and r=='first') for r in READERS} for i in range(59)]
truth=np.repeat((paths[2]+history[-1])[None],59,axis=0)
a=evaluate(bank,mp,entries,truth)
assert a['first']['scores']['supported']['windows']==1
assert a['independent']['scores']['supported']['contributing_clip_equal']['normal'] is None
prior=a['first']['scores']['all_including_fallback']['normal_minus_contributing_clip_equal']['probability']['ade_m']
np.testing.assert_allclose(prior,-np.mean(steps)/5/4,atol=1e-12)
assert a['first']['scores']['supported']['contributing_clip_equal']['normal']['ade_m']==0.
print('PASS finite geometry, support/fallback, angle, probability ties, legal crossings, held-clip means and zero-support output')
