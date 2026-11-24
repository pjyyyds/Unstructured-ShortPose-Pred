"""Synthetic checks of geometry, common-support handling, temporal blinding and full ridge entry."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import warnings

import numpy as np

from geometry import PERMUTATIONS,blind_input,descriptors,design_features,permuted
import run_geometry as driver


def test_geometry():
    p=np.zeros((1,5,17,3));p[:,:,:,2]=1.
    p[:,:,[5,6],:2]=[[-2.,0.],[2.,0.]]
    p[:,:,[11,12],:2]=[[-1.,4.],[1.,4.]]
    p[:,:,[13,14],:2]=[[-1.,7.],[1.,7.]]
    p[:,:,[15,16],:2]=[[-1.,10.],[4.,7.]]
    box=np.zeros((1,5,4));box[:,:,3]=10
    g,m=descriptors(p,box)
    np.testing.assert_allclose(g[0,0],[np.sqrt(34)/10,-1.,0.,.4,0.,.4])
    assert m.all()
    pp=p.copy();pp[...,:2]=pp[...,:2]*3+[41.,-50.]
    gg,mm=descriptors(pp,box*3)
    np.testing.assert_allclose(gg,g,atol=1e-13);np.testing.assert_array_equal(mm,m)
    pp=p.copy();pp[:,1,13,2]=.19;pp[:,3,16,:2]=pp[:,3,14,:2]
    h,v=descriptors(pp,box);a,b,steps=design_features(np.zeros((1,28)),h,v)
    assert a.shape==b.shape==(1,70) and not a[:,-24:].any() and not steps[:,:,1:3].any()
    assert v[0,-1,1] and v[0,-1,2] and not v[0,1,1] and not v[0,3,2]
    np.testing.assert_array_equal(a[:,:46],b[:,:46])


def synthetic():
    rng=np.random.default_rng(64)
    sites=np.repeat(sorted(driver.G),8);n=len(sites)
    pose=rng.normal(size=(n,5,17,3));pose[...,2]=rng.uniform(size=(n,5,17))
    box=np.ones((n,5,4))*10
    geom,valid=descriptors(pose,box);motion=rng.normal(size=(n,28))
    static,ordered,inc=design_features(motion,geom,valid)
    ground=rng.normal(size=(n,30,2));anchor=rng.normal(size=ground.shape)
    frames=np.tile(np.eye(2),(n,1,1))
    keys=[(s+str(i//4),i,100+i) for i,s in enumerate(sites)]
    inp=dict(features={'motion':motion},frames=frames,truth=ground,anchor=anchor,sites=sites,clips=np.array([k[0] for k in keys]),
             keys=keys,static=static,ordered=ordered,increments=inc,valid_geometry=valid)
    return inp


def test_order():
    rng=np.random.default_rng(13)
    x=rng.normal(size=(13,70));f=rng.normal(size=(70,60));loc=rng.normal(size=70);sd=rng.uniform(.02,4,size=70)
    forward=lambda v:((v-loc)/sd)@f
    pool=blind_input(x)
    all_predictions=[]
    for perm in PERMUTATIONS:
        z=permuted(x,perm);np.testing.assert_array_equal(blind_input(z),pool)
        np.testing.assert_array_equal(z[:,:46],x[:,:46]);all_predictions.append(forward(z))
    assert np.ptp(all_predictions,axis=0).max()>1.
    np.testing.assert_allclose(forward(pool),np.mean(all_predictions,0),rtol=0,atol=1e-12)


def test_full():
    inp=synthetic();ref={'sites':{}}
    for s,g in driver.G.items():
        tr=inp['sites']!=s;te=~tr
        fit=driver.old.ridge_fit(inp['features']['motion'][tr],inp['truth'][tr]-inp['anchor'][tr])
        forecasts={'last_two':inp['anchor'][te],'motion':inp['anchor'][te]+g*driver.old.predict_world_residual(fit,inp['features']['motion'][te],inp['frames'][te])}
        ref['sites'][s]={'mean_m':{k:driver.metric_dict(driver.old.errors(p,inp['truth'][te]).mean(0)) for k,p in forecasts.items()}}
    plan={'name':'synthetic only','primary':'test','support_overall':{}}
    with TemporaryDirectory() as tmp:
        out=Path(tmp)
        driver.evaluate(inp,ref,plan,out);r=json.loads((out/'geometry_scores.json').read_bytes())
        for v in r['order_sensitivity'].values():assert v['max_blind_path_change_across_permutations_m']==0.
        with np.load(out/'private_keyed.npz') as z:
            assert z['prediction_order_blind'].shape==(32,30,2)
            check=driver.old.errors(z['prediction_ordered_geometry'],inp['truth'])
            np.testing.assert_array_equal(z['error_ordered_geometry'],check)


if __name__=='__main__':
    test_geometry();test_order();test_full()
    print('Six descriptors, scale/translation invariance, partial/degenerate joints, blind permutation identity, complete four-fold toy fit/score passed.')
