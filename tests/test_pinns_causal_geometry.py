import unittest
from unittest.mock import patch

import numpy as np

from scripts import evaluate_pinns_causal_geometry as exp


class CausalGeometryTests(unittest.TestCase):
    def test_nearest_projection_on_segment_and_axial_orientation(self):
        lines = np.zeros((3, 2, 3, 11), dtype=float)
        lines[0, 0, :2, :2] = [[-4., 2.], [6., 2.]]
        lines[0, 0, :2, 10] = 1.
        lines[1] = lines[0]
        lines[1, 0, :2] = lines[0, 0, 1::-1]
        # Third window has no data; zero geometry alone must not signal open space.
        features, available = exp.geometry_features(lines, np.array([[0., 3.]]*3))
        np.testing.assert_allclose(features[:2], [[0., 2., 2., 1., 0., 3.]]*2)
        np.testing.assert_array_equal(available, [True, True, False])
        np.testing.assert_array_equal(features[2], np.zeros(6))

    def test_masked_ols_uses_actual_times_and_last_valid_fde_differs_from_three_seconds(self):
        history = np.array([[[-.8, 0.], [70., -81.], [-.4, 0.], [-.2, 0.], [0., 0.]]])
        observed = np.array([[True, False, True, True, True]])
        np.testing.assert_allclose(exp.mask_velocity(history, observed), [[2., 0.]])
        pred = np.zeros((2, 30, 2)); truth = np.zeros_like(pred)
        pred[:, 0, 0], pred[:, 17, 0], pred[:, 29, 0] = 2., 4., 12.
        valid = np.zeros((2, 30), dtype=bool)
        valid[0, [0,17]] = True
        valid[1, [0,17,29]] = True
        errors = exp.errors(pred, truth, valid)
        np.testing.assert_allclose(errors[:,:2], [[3., 4.], [6., 12.]])
        self.assertTrue(np.isnan(errors[0,-1]))
        self.assertEqual(errors[1,-1], 12.)

    def test_per_lead_scaling_and_coefficients_exclude_held_and_missing_labels(self):
        x = np.arange(6, dtype=float)[:,None]
        y = np.repeat(np.array([0.,1.,4.,9.,16.,25.])[:,None,None], 2, axis=1)
        y = np.repeat(y, 2, axis=2)
        train = np.array([True]*4+[False]*2)
        mask = np.ones((6,2), dtype=bool); mask[3,1] = False
        fitted = exp.fit_horizons(x,y,mask,train)
        np.testing.assert_allclose(fitted["mean"], [[1.5],[1.]])
        np.testing.assert_allclose(fitted["intercept"], [[3.5,3.5],[5./3,5./3]])
        x[~train]=9e8; y[~train]=1e5; y[3,1]=3e5
        altered = exp.fit_horizons(x,y,mask,train)
        for name in fitted: np.testing.assert_array_equal(fitted[name],altered[name])

    def test_bootstrap_preserves_fixed_site_mix(self):
        sites=np.array(["a","a","a","b"]); clips=np.array(["x","x","x","y"])
        interval=exp.clip_interval(np.array([1.,1.,1.,5.]),sites,clips)
        self.assertEqual(interval["estimate_m"],2.)
        np.testing.assert_array_equal(interval["percentile_95_m"],[2.,2.])

    def test_design_only_skips_fitting_and_scores(self):
        n=4; splits=np.array(["train","train","val","test"])
        c=exp.Cohort([("x",i,4) for i in range(n)], np.array(["a"]*n), np.array(["x"]*n),splits,
            np.ones((n,5),bool),np.zeros((n,30,2)),np.ones((n,30),bool),np.zeros((n,2)),
            np.zeros((n,3),bool),{m:np.zeros((n,4)) for m in exp.METHODS},{})
        with patch.object(exp,"fit_horizons",side_effect=AssertionError("design tried to fit")):
            r,private=exp.evaluate(c,True)
        self.assertIsNone(private)
        self.assertNotIn("means",r["test"]["all"])
        self.assertEqual(r["test"]["all"]["fde_exact_3s_windows"],1)
        # A fully synthetic end-to-end run exercises the serialization payload and the primary comparison.
        result, archive = exp.evaluate(c,False)
        self.assertEqual(result["primary_test_paired_ade"]["percentile_95_m"],[0.,0.])
        self.assertEqual(archive["prediction_image_geometry"].shape,(n,30,2))
        self.assertEqual(archive["image_geometry_weights"].shape,(30,4,2))
        self.assertFalse(result["test"]["both_maps"]["windows"])
