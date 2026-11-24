"""Synthetic end-to-end test. No access to personal data or real forecast targets."""
import argparse
from pathlib import Path
import numpy as np
from compare_axes import VERSIONS, METHODS, build_features, compare_scores, fits, frozen_math, map_support


def test(path):
    funcs=frozen_math(path.read_bytes())
    rng=np.random.default_rng(12)
    n=32; eligible=np.arange(n)%4!=0; hull=eligible & (np.arange(n)%3!=0)
    image0=eligible.copy();image0[-1]=False;image1=eligible.copy()
    motion=rng.normal(size=(n,14))
    h=rng.normal(size=(n,6))*hull[:,None]
    old=rng.normal(size=(n,6))*image0[:,None];new=rng.normal(size=(n,6))*image1[:,None]
    avail={};ff={}
    for name,geom,mask in zip(VERSIONS,(old,new),(image0,image1)):
        avail[name],ff[name]=build_features(motion,eligible,geom,h,mask,hull)
        np.testing.assert_array_equal(ff[name]['image_geometry'][~mask,-6:],0)
    test=np.arange(n)>=24;val=(np.arange(n)>=18)&~test;train=~(val|test)
    groups=np.where(test,'test',np.where(val,'val','train'))
    support=map_support(test,avail)
    assert support['image_new_only']==1 and support['image_old_only']==0 and support['windows']==8
    cvel=rng.normal(size=(n,2));cv=cvel[:,None]*(.1*np.arange(1,31))[None,:,None]
    truth=cv+rng.normal(size=cv.shape);valid=np.ones((n,30),bool);valid[-1,-1]=False
    # Deliberately perturb all withheld targets and labels; the fitted estimates cannot change.
    changed=truth.copy();changed[~train]+=900.0
    p={};params={}
    for version in VERSIONS:
        p[version],params[version]=fits(funcs,ff[version],cv,truth,valid,train)
        altered,_=fits(funcs,ff[version],cv,changed,valid,train)
        for name in ('ols_five',*METHODS):
            assert np.isfinite(p[version][name]).all()
            np.testing.assert_array_equal(altered[name],p[version][name])
    for m in ('ols_five','motion'):np.testing.assert_array_equal(p[VERSIONS[0]][m],p[VERSIONS[1]][m])
    for m in ('availability','occupancy_geometry'):
        for key in ('weights','intercept','mean','scale'):
            np.testing.assert_array_equal(params[VERSIONS[0]][m+'_'+key],params[VERSIONS[1]][m+'_'+key])
    s=np.array(['site_'+str(i%2) for i in range(n)]);c=np.array(['clip_'+str(i%4) for i in range(n)])
    out=compare_scores(p,funcs,s,c,avail,valid,groups,truth)
    pooled=out['test']['pooled'];first=pooled['paired']['corrected_image_minus_corrected_availability']['ade_m']
    delta_old=pooled['paired']['legacy_image_minus_legacy_availability']['ade_m']['mean_m']
    assert pooled['errors'][VERSIONS[0]]['motion']['fde_exact_3s_m']['windows']==7
    assert pooled['errors'][VERSIONS[0]]['motion']['fde_last_valid_m']['windows']==8
    per_clip=sum(row['paired']['corrected_image_minus_corrected_availability']['ade_m']['sum_m'] for row in out['test']['by_clip'].values())
    np.testing.assert_allclose(per_clip,first['sum_m'])
    np.testing.assert_allclose(first['mean_m']-delta_old,out['registered_test_intervals']['difference_in_differences']['estimate_m'])
    print('Passed synthetic fits, zero geometry, independent hull, paired endpoints and per-clip reconstruction')


if __name__=='__main__':
    arg=argparse.ArgumentParser();arg.add_argument('original_evaluator',type=Path);test(arg.parse_args().original_evaluator)
