"""Synthetic cases only. No source trajectory is fitted or scored."""
import unittest
import numpy as np
import run_jrdb_pose_residual as m

class Alignment(unittest.TestCase):
    def test_tiles_first_then_filters(self):
        tracks={7:{f:(f,1.) for f in range(0,108,6)}}
        joints=np.ones((33,3))
        pose={(f,7):joints.copy() for f in (6,12,54,60)}
        rows,blocks=m.windows(tracks,pose)
        self.assertEqual(blocks,2)
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0][1].tolist(),list(range(54,108,6)))

    def test_masks_center_rotation_free_differences(self):
        rng=np.random.default_rng(5)
        xy=rng.normal(size=(2,2,2));q=rng.normal(size=(2,2,33,3))
        q[0,0,0]=np.nan;q[0,1,1]=np.nan;q[1,0]=np.nan
        expected=m.features(xy,q)
        translated=q+np.array([[[[8,3,-7]],[[6,-5,2]]]])
        actual=m.features(xy+19,translated)
        for n,x in actual.items():
            np.testing.assert_allclose(x,expected[n],atol=1e-12)
            self.assertTrue(np.isfinite(x).all())
        self.assertEqual([s.shape[-1] for s in actual.values()],[3,135,267])
        ordered=actual['ordered_pose']
        np.testing.assert_equal(ordered[0,-33:-31],[0,0])
        np.testing.assert_equal(ordered[1,135:],0)
        np.testing.assert_equal(actual['last_pose'][0,6:9],0)
        with self.assertRaises(ValueError):m.features(np.zeros((2,3,2)),q)

    def test_ridge_primal_and_frozen_fold(self):
        rng=np.random.default_rng(22);x=rng.normal(size=(8,6));x[:,1]=0.
        y=rng.normal(size=(8,7,2))
        fit=m.residual_fit(x,y);z=(x-x.mean(0))*fit['inv_scale']
        beta=np.linalg.solve(z.T@z+len(x)*np.eye(6),z.T@(y.reshape(8,-1)-y.reshape(8,-1).mean(0)))
        np.testing.assert_allclose(beta,fit['weights'],atol=1e-12)
        self.assertEqual(fit['inv_scale'][1],0)
        data={'site':np.repeat(['a','b','c'],4),'person':np.tile([1,2,2,3],3),
              'xy':rng.normal(size=(12,9,2)),'joints':rng.normal(size=(12,2,33,3))}
        before,params,folds=m.forecast(data)
        # Absolute targets and input for a different held subject cannot contaminate fold parameters.
        change={**data,'xy':data['xy'].copy(),'joints':data['joints'].copy()}
        change['xy'][:4,2:]+=100.;change['joints'][0]+=rng.normal(size=(2,33,3))
        after,new,_=m.forecast(change)
        for k,x in params.items():
            if k.startswith('fold_0'):np.testing.assert_array_equal(x,new[k])
        for n in m.ARMS:
            np.testing.assert_array_equal(params[n+'_pred_relative_xy'][1:4],new[n+'_pred_relative_xy'][1:4])
        summary=m.summarize(data['site'],data['person'],before,31)
        ref=np.mean([before['ordered_pose'][s:s+4].mean(0)-before['motion'][s:s+4].mean(0) for s in (0,4,8)],0)
        np.testing.assert_allclose(list(summary['contrasts']['ordered_pose_minus_motion']['three_recording_equal_m'].values()),ref)
        cv=m.forecast({**data, 'xy':np.cumsum(np.repeat(data['xy'][:,:1],9,axis=1),axis=1)})[0]
        for value in cv.values():np.testing.assert_allclose(value,0,atol=1e-12)

if __name__=='__main__': unittest.main()
