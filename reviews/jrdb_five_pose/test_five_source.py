"""Synthetic feature-isolation, normalization, split and paired-mean checks."""
import numpy as np
from run_five_source_poses import SCENES,PAST,ARMS,design,fit_all,summarize
from prior_native_residual import features as prior_features

def run():
    rng=np.random.default_rng(13);n=18
    h=rng.normal(size=(n,2,2));j=rng.normal(size=(n,5,33,3));y=rng.normal(size=(n,7,2));f=design(h,j)
    old=prior_features(h,j[:,[0,4]])
    for arm,a in [('motion','motion'),('last_pose','last_pose'),('first_last_pose','ordered_pose')]:
        np.testing.assert_array_equal(f[arm],old[a])
    # Shifting all joints in an intermediate frame leaves its centered shape unchanged.
    shift=rng.normal(size=(n,5,1,3));fj=design(h,j+shift)
    for a in ARMS:np.testing.assert_allclose(f[a],fj[a],atol=2e-13,rtol=0)
    altered=j.copy();altered[:,2,9]+=5;new=design(h,altered)
    for a in ARMS[:3]:np.testing.assert_array_equal(new[a],f[a])
    assert not np.array_equal(new['five_pose'],f['five_pose'])
    missing=j.copy();missing[:,4]=np.nan;x=design(h,missing);assert np.isfinite(x['five_pose']).all()
    # All pose columns, including validity flags, vanish when the last frame is absent.
    assert not np.any(x['five_pose'][:,3:])
    keys=[(SCENES[i//6],i%3,60+i*54) for i in range(n)]
    feats,preds,errs,coeff,folds=fit_all(keys,h,j,y)
    y2=y.copy();y2[:6]+=10000
    _,other,_,_,_=fit_all(keys,h,j,y2)
    for a in ARMS:np.testing.assert_allclose(preds[a][:6],other[a][:6],atol=1e-12)
    for i,s in enumerate(SCENES):
        tr=coeff[f'fold_{i}_train'];te=coeff[f'fold_{i}_test']
        assert sorted(set(tr)|set(te))==list(range(n)) and not set(tr)&set(te)
        for arm in ARMS:np.testing.assert_allclose(coeff[f'fold_{i}_{arm}_mean'],feats[arm][tr].mean(0))
    stats=summarize(keys,errs);assert len(stats['by_recording'])==3 and PAST.tolist()==[-6,-5,-3,-2,0]
    for nm in stats['three_recording_equal']:
        pair=stats['three_recording_equal'][nm]['paired']['five_pose_minus_first_last_pose']
        assert all(np.isfinite(v).all() for v in pair['paired_person_95'].values())
    print('Synthetic tests passed: native spacing, earlier three-arm identity, intermediate-only perturbation, missing pose, held future isolation, train-only scaling, both weightings.')
if __name__=='__main__':run()
